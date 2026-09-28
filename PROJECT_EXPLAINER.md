# LLM Gateway — Complete Project Explainer
> Written for interview prep. Covers what it is, how it works, how to test it manually, and code-level details.

---

# PART 1 — WHAT IS THIS & WHAT PROBLEM DOES IT SOLVE?

## The Simple Explanation

Imagine you're a company using AI (ChatGPT, Gemini, etc.) in your product. You face these problems:

- Same question asked 1000 times = 1000 API calls = 1000x cost
- Too many requests from one user crashes your system
- OpenAI goes down = your entire product breaks
- You use 3 different AI providers = 3 different codebases
- No visibility into what's happening inside your system

This project is a **gateway** — it sits between your app and all AI providers. Your app talks to ONE place, the gateway handles everything else.

```
Your App
    ↓
LLM Gateway  ←  Rate Limiter + Cache + Router + Circuit Breaker
    ↓
OpenAI / Gemini / Perplexity / Ollama / Mock
```

It's OpenAI-compatible — meaning any app already using OpenAI just changes one URL and gets all these features for free.

---

# PART 2 — ALL FEATURES EXPLAINED

---

## FEATURE 1: RATE LIMITER

**Problem:** A buggy script or malicious user sends 10,000 requests/sec. Costs explode, server crashes.

**Solution:** Each tenant gets a quota. Exceed it → HTTP 429 "Too Many Requests".

### 4 Algorithms (all hot-swappable at runtime):

---

### 1. Token Bucket (default, config: `strategy: "token_bucket"`)
Think of a bucket that holds tokens. Each request consumes 1 token. Bucket refills at 2 tokens/second, max capacity 10.
- Burst of 10 requests? Fine — bucket had 10 tokens.
- 11th request immediately? Blocked. Wait ~0.5s for refill.
- **Good for:** Allowing short bursts while enforcing average rate.

**Key code:** `src/rate_limiter/token_bucket.py`
- `_refill()`: `elapsed * refill_rate` tokens added since last call (lazy refill, no background thread).
- `is_allowed(key, cost)`: returns `(allowed: bool, remaining: float, retry_after: float)`.
- `retry_after = deficit / refill_rate` — tells the client exactly when to retry.

---

### 2. Sliding Window Log (config: `strategy: "sliding_window_log"`)
Keeps a log of exact timestamps for every request. If more than N timestamps exist in the last 60 seconds → block.
- 100% accurate, no boundary exploit (token bucket can be gamed at window edges).
- Uses more memory: stores every timestamp.

**Key code:** `src/rate_limiter/sliding_window_log.py`
- Deque of float timestamps per tenant.
- On each request: `popleft()` all timestamps older than `now - window_seconds`, then check length.

---

### 3. Sliding Window Counter (config: `strategy: "sliding_window_counter"`)
Two integer counters: current window + previous window. Uses a weighted estimate:
```
estimated = current_count + previous_count × (1 - time_elapsed_into_current_window / window_size)
```
As you move through the current window, the previous window's contribution fades to zero.
- O(1) memory, ~99.95% accurate.
- Best for high-scale systems with millions of tenants.

**Key code:** `src/rate_limiter/sliding_window_counter.py`

---

### 4. Redis Token Bucket (config: `strategy: "redis_token_bucket"`)
Same as Token Bucket but state lives in Redis, updated via an **atomic Lua script**.
- Works across multiple gateway instances (horizontal scaling).
- Lua script runs atomically server-side — no distributed locks needed.
- If Redis dies → automatically falls back to in-memory token bucket.

**Key code:** `src/rate_limiter/redis_token_bucket.py`

---

### HOW TO TEST RATE LIMITER MANUALLY:

Open a second terminal (keep uvicorn running in the first):

```powershell
# Fire 12 requests fast — first 10 pass, 11th and 12th get 429
for ($i = 1; $i -le 12; $i++) {
    $body = '{"messages":[{"role":"user","content":"hello"}],"tenant_id":"test_user","cache_bypass":true}'
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
        Write-Host "Request $i : OK -> $($r.provider_id)"
    } catch {
        Write-Host "Request $i : BLOCKED -> $($_.Exception.Response.StatusCode)"
    }
}
```

Wait 5 seconds and try again — the bucket refills and you can send more.

Switch algorithm live (no restart needed):
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/gateway/reconfigure" -Method Post `
  -Body '{"rate_limiter_strategy": "sliding_window_log"}' -ContentType "application/json"
```

---

## FEATURE 2: CACHING

**Problem:** Users ask the same (or very similar) questions repeatedly. Every call costs money and takes time.

**Solution:** Cache responses. On a hit, return instantly without touching any LLM provider.

### 3 Cache Implementations:

---

### 1. LRU Cache — Least Recently Used (config: `strategy: "lru_exact"`)
A classic doubly-linked list + hashmap. Most recently used items are at the head. When the cache is full, the item at the tail (least recently used) is evicted.

**Key code:** `src/cache/lru_cache.py`
- Two sentinel nodes `__HEAD__` and `__TAIL__` eliminate null-check edge cases.
- `get()`: O(1) — hash lookup + move node to head.
- `set()`: O(1) — insert at head, evict tail if over capacity.
- Cache key = SHA-256 hash of the prompt string.

**Interview point:** This is the same algorithm used in CPU caches and browser caches.

---

### 2. LFU Cache — Least Frequently Used (config: `strategy: "lfu_exact"`)
Evicts the item accessed the fewest times overall. Better than LRU for workloads where some prompts are asked very frequently (hot prompts stay forever).

**Key code:** `src/cache/lfu_cache.py`
- `_key_to_node`: dict of key → node (with frequency counter).
- `_freq_to_dll`: dict of frequency → doubly-linked list of nodes at that frequency.
- `_min_freq`: integer. Points to the DLL where the eviction candidate lives.
- All operations O(1).

**Interview point:** When you access a node, it moves from `freq[n]` DLL to `freq[n+1]` DLL. If `freq[n]` DLL becomes empty and `n == _min_freq`, increment `_min_freq`.

---

### 3. Semantic Vector Cache (default, config: `strategy: "semantic_vector"`)
The most sophisticated one. Handles paraphrases — "What is machine learning?" and "Can you explain ML to me?" should return the same cached answer.

**Two-tier architecture:**
- **Tier 1 (Exact):** SHA-256 hash match → O(1), no AI needed.
- **Tier 2 (Semantic):** Converts the prompt to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`. Compares against all stored vectors using **cosine similarity**. If similarity ≥ threshold (default 0.75) → cache hit.

**Key code:** `src/cache/semantic_cache.py`, `src/cache/embeddings.py`

**Cosine similarity** = dot product of two normalized vectors. Measures the angle between them. 1.0 = identical meaning, 0.0 = unrelated.

**Eviction:** LRU by `last_accessed` timestamp when cache hits max_size.

**Fallback:** If `sentence-transformers` isn't installed, uses a custom subword n-gram hash vectorizer (deterministic, works offline).

---

### HOW TO TEST CACHING MANUALLY:

```powershell
# First request — cache MISS, calls the LLM
$body = '{"messages":[{"role":"user","content":"Explain what is a hash map"}],"tenant_id":"test_user"}'
$r1 = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
Write-Host "Cache hit: $($r1.cache_hit) | Latency: $($r1.latency_ms)ms"

# Second request — EXACT cache HIT, no LLM call
$r2 = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
Write-Host "Cache hit: $($r2.cache_hit) | Type: $($r2.cache_type) | Latency: $($r2.latency_ms)ms"

# Third request — paraphrase — SEMANTIC cache HIT
$body3 = '{"messages":[{"role":"user","content":"What is a hashmap and how does it work"}],"tenant_id":"test_user"}'
$r3 = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body3 -ContentType "application/json"
Write-Host "Cache hit: $($r3.cache_hit) | Type: $($r3.cache_type) | Latency: $($r3.latency_ms)ms"
```

You'll see Tier 1 "EXACT" hit on r2, and "SEMANTIC" hit on r3. Latency drops from ~60ms to ~5ms.

Clear the cache:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/gateway/cache/clear" -Method Post
```

---

## FEATURE 3: ROUTER / LOAD BALANCER

**Problem:** You have multiple LLM providers. How do you decide which one handles each request?

**Solution:** 4 routing strategies, all hot-swappable.

---

### 1. Round Robin (config: `strategy: "round_robin"`)
Circular distribution. Request 1 → Provider A, Request 2 → Provider B, Request 3 → Provider A...
- Dead simple, uniform distribution.
- Doesn't care about performance or cost.

---

### 2. Latency-Aware EWMA (default, config: `strategy: "latency_aware_ewma"`)
Routes to the provider with the lowest **Exponential Weighted Moving Average** latency.
```
new_EWMA = alpha × new_latency + (1 - alpha) × old_EWMA
```
- `alpha = 0.2`: 20% weight to the newest measurement, 80% to history. Smooths out spikes.
- On failure: treats effective latency as `max(5000ms, actual × 2)` — a big penalty that quickly degrades the failing provider's score.
- New provider starts at 200ms until real data comes in.

**Interview point:** EWMA is also used in TCP congestion control (RTT estimation).

---

### 3. Cost-Aware (config: `strategy: "cost_aware"`)
Picks the cheapest provider every time.
```
score = cost_per_1k_input + cost_per_1k_output
```
Selects provider with minimum blended token price. Pure static comparison, O(N).

---

### 4. Consistent Hashing (config: `strategy: "consistent_hashing"`)
Same tenant always goes to the same provider (sticky routing). Built on a hash ring.

**How the ring works:**
- Each provider gets 50 "virtual nodes" on a circular ring (hash space 0 to 2^32).
- Virtual node positions = MD5("{provider_id}#vnode_{i}") as 32-bit int.
- To route: hash the `session_id` or `tenant_id`, do binary search (`bisect_right`) on the sorted ring, assign to the nearest clockwise node's provider.
- If a provider is removed, only ~1/N keys remap (unlike simple modulo hashing where everything remaps).

**Interview point:** This is how DynamoDB, Cassandra, and Chord DHT work.

---

### HOW TO TEST ROUTER MANUALLY:

```powershell
# See which provider is selected
$body = '{"messages":[{"role":"user","content":"test"}],"tenant_id":"user1","cache_bypass":true}'
1..5 | ForEach-Object {
    $r = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Request $_ -> Provider: $($r.provider_id)"
}
```

Switch to round robin and watch it alternate:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/gateway/reconfigure" -Method Post `
  -Body '{"router_strategy": "round_robin"}' -ContentType "application/json"
```

Force a specific provider (bypass router entirely):
```powershell
$body = '{"messages":[{"role":"user","content":"test"}],"tenant_id":"user1","cache_bypass":true,"preferred_provider":"mock-slow"}'
$r = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
Write-Host "Provider: $($r.provider_id)"
```

---

## FEATURE 4: CIRCUIT BREAKER

**Problem:** Provider X starts failing (timeout, 500 errors). Every request still tries it → all users wait for timeouts → cascading failure brings down everything.

**Solution:** Circuit Breaker pattern. After enough failures, "trip" the circuit — stop sending requests to that provider. After a cooldown, test it with a few probe requests. If those pass, close the circuit and resume normal traffic.

### 3 States:

```
CLOSED  ──(too many failures)──▶  OPEN  ──(cooldown expires)──▶  HALF_OPEN
  ▲                                                                    │
  └────────────(N probe successes)────────────────────────────────────┘
                         ▲ (any probe fails → back to OPEN)
```

- **CLOSED:** Normal operation. All requests pass through.
- **OPEN:** All requests fast-fail immediately with 503. No waiting. Failover kicks in.
- **HALF_OPEN:** Lets through a max of 3 probe requests to test recovery. If they succeed → CLOSED. If any fails → back to OPEN.

### 2 Trip Strategies:

**Consecutive Failure:** Trip after 5 errors in a row. Simple.

**Sliding Window Error Rate (default):** Track success/failure over the last 30 seconds. If error rate ≥ 50% AND at least 5 requests seen → trip. More nuanced, handles occasional blips without tripping.

**Thundering Herd Prevention:** In HALF_OPEN, only 3 probes allowed in-flight simultaneously. Without this, when the circuit re-opens, hundreds of queued requests would all rush in at once and overwhelm the recovering provider.

**Key code:** `src/circuit_breaker/breaker.py`

---

### HOW TO TEST CIRCUIT BREAKER MANUALLY:

```powershell
# Check current state
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json

# Manually trip mock-fast's circuit breaker to OPEN
Invoke-RestMethod -Uri "http://localhost:8000/gateway/circuit-breaker/trip?provider_id=mock-fast" -Method Post

# Check state again — mock-fast should show OPEN
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json

# Send a request — gateway will automatically failover to mock-slow
$body = '{"messages":[{"role":"user","content":"test failover"}],"tenant_id":"user1","cache_bypass":true}'
$r = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
Write-Host "Used provider: $($r.provider_id)"   # Should say mock-slow

# Reset all circuit breakers
Invoke-RestMethod -Uri "http://localhost:8000/gateway/circuit-breaker/reset" -Method Post
```

---

## FEATURE 5: PROVIDER ADAPTERS

**Problem:** Every AI provider has a different API format, auth method, response structure.

**Solution:** Adapter pattern. Each provider gets its own adapter class that speaks the provider's native API but exposes a standard `generate_completion(request)` interface.

| Provider | Config ID | Notes |
|---|---|---|
| Mock | `mock-fast`, `mock-slow` | Built-in fake provider. Simulates latency and failure rates. Used for testing. |
| OpenAI | `openai-gpt4o-mini` | Standard `POST /v1/chat/completions`. Set `OPENAI_API_KEY` env var. |
| Gemini | `gemini-1.5-flash` | Google's API. Messages translated: `assistant` → `model`, `system` → `user`. Set `GEMINI_API_KEY`. |
| Perplexity | `perplexity-sonar` | OpenAI-compatible API, nearly identical adapter. Set `PERPLEXITY_API_KEY`. |
| Ollama | `ollama-local` | Local model server at `localhost:11434`. No API key. Different payload format. |

To enable a real provider, set `enabled: true` in `config/gateway_config.yaml` and set the env var.

---

## FEATURE 6: OBSERVABILITY

### Prometheus Metrics (`/metrics` endpoint)

| Metric | What It Tracks |
|---|---|
| `gateway_requests_total` | Total requests, labeled by HTTP status and tenant |
| `gateway_rate_limited_total` | How many requests got throttled (429) |
| `gateway_cache_hits_total` | Cache hits, labeled EXACT or SEMANTIC |
| `gateway_cache_misses_total` | Cache misses |
| `gateway_cache_dollars_saved_total` | Estimated USD saved by not calling LLMs |
| `gateway_request_duration_seconds` | Histogram of E2E latency |
| `gateway_circuit_breaker_state` | 0=CLOSED, 1=HALF_OPEN, 2=OPEN per provider |
| `gateway_circuit_breaker_failovers_total` | How many times failover was triggered |
| `gateway_tokens_processed_total` | Input/output tokens per provider |

Prometheus scrapes `/metrics` every 15 seconds. Grafana reads Prometheus and shows dashboards.

### Structured Logging
Every log line includes `[cid:XXXX]` — a correlation ID that traces a single request through all log lines, even across async tasks. Set automatically by the HTTP middleware.

---

## FEATURE 7: DYNAMIC RECONFIGURATION

You can swap any algorithm at runtime with zero downtime:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/gateway/reconfigure" -Method Post `
  -Body '{
    "rate_limiter_strategy": "sliding_window_counter",
    "cache_strategy": "lfu_exact",
    "router_strategy": "cost_aware",
    "circuit_breaker_strategy": "consecutive_failure"
  }' -ContentType "application/json"
```

This rebuilds the entire pipeline internally. The old pipeline is replaced atomically.

---

# PART 3 — THE REQUEST JOURNEY (End to End)

When `POST /v1/chat/completions` arrives:

```
1. HTTP Middleware
   → Generate/extract X-Correlation-ID
   → Start timer

2. Rate Limiter
   → is_allowed(tenant_id)?
   → NO → HTTP 429, Retry-After header
   → YES → continue

3. Cache Lookup
   → cache_bypass=false?
   → Tier 1: SHA-256 hash exact match → HIT → return immediately
   → Tier 2: embed prompt → cosine similarity scan → HIT → return immediately
   → MISS → continue

4. Router
   → Filter out OPEN circuit breaker providers
   → select_provider(healthy_providers) using active strategy
   → Returns primary_provider

5. Execute with Circuit Breaker + Failover
   → build execution_order = [primary] + [failover chain]
   → For each provider in order:
      a. circuit_breaker.allow_request() → if OPEN, skip
      b. provider.generate_completion(request)
      c. SUCCESS → record_success(), record_feedback(), store in cache async, return
      d. FAILURE → record_failure(), record_feedback(), try next provider
   → All failed → HTTP 502

6. HTTP Middleware
   → Inject X-Response-Time-Ms header
   → Return response
```

---

# PART 4 — CODE ARCHITECTURE (Design Patterns Used)

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | Rate limiters, Caches, Routers | Swap algorithms without changing calling code |
| **Factory** | `RateLimiterFactory`, `CacheFactory`, `RouterFactory`, `ProviderFactory` | Centralized object creation from config strings |
| **Adapter** | All provider classes | Unified interface over different external APIs |
| **State Machine** | Circuit Breaker | Clean modeling of CLOSED/OPEN/HALF_OPEN transitions |
| **Null Object** | `_NoOpCache` for `strategy: "disabled"` | Avoid null checks everywhere |
| **Dependency Injection** | FastAPI `Depends(get_pipeline)` | Inject shared pipeline into all route handlers |
| **Decorator** | HTTP middleware | Cross-cutting concerns (timing, correlation ID) without touching business logic |

---

# PART 5 — COMMON INTERVIEW CROSS-QUESTIONS

**Q: Why not just use a Redis cache instead of building LRU/LFU?**
A: Redis is a great production choice. Building LRU/LFU from scratch demonstrates understanding of the underlying data structures — doubly linked list + hashmap for O(1) operations. In production you'd use Redis with TTL, but the interview point is the algorithm.

**Q: What's the difference between LRU and LFU? When would you use each?**
A: LRU evicts by recency — the item not used for the longest time. LFU evicts by frequency — the item used the fewest times total. Use LRU when access patterns shift over time (old prompts become irrelevant). Use LFU when some prompts are permanently hot (popular FAQs) and should never be evicted.

**Q: Why does the semantic cache need two tiers?**
A: Embedding computation is expensive (~5-20ms). For exact duplicate prompts, you don't need to waste time on embeddings — a hash lookup is O(1) and deterministic. The two-tier design gives you the best of both worlds.

**Q: What's the difference between Token Bucket and Sliding Window?**
A: Token Bucket allows bursts (up to bucket capacity) then enforces an average rate. Sliding Window Log is precise — it counts exact requests in the last N seconds with no boundary exploit. Token Bucket is better for normal API clients that occasionally burst. Sliding Window is stricter for financial or billing-critical systems.

**Q: Why does Circuit Breaker have a HALF_OPEN state? Why not just go CLOSED after the cooldown?**
A: If you went straight to CLOSED, you'd immediately send full traffic to a provider that might still be partially down. HALF_OPEN sends just 1-3 probe requests first. If they succeed, close the circuit. This prevents hammering a recovering service.

**Q: What is thundering herd and how does the circuit breaker prevent it?**
A: When a CLOSED circuit transitions to HALF_OPEN after a cooldown, thousands of queued requests might all rush in simultaneously. The breaker limits in-flight HALF_OPEN probes to 3 (`_max_half_open_probes`). Everything else fast-fails until the circuit decides to close or reopen.

**Q: How does consistent hashing help with caching?**
A: If the same tenant always routes to the same provider, and that provider maintains its own cache, you get higher cache hit rates. With round robin, the same tenant might hit 5 different providers, each with a cold cache. Consistent hashing also means adding/removing a provider only remaps ~1/N of the traffic, not everything.

**Q: How does EWMA work and why use it over a simple average?**
A: EWMA = alpha × new_value + (1 - alpha) × old_EWMA. It gives more weight to recent measurements and fades out old ones exponentially. A simple average treats a 10-second-old measurement the same as a 1ms-old one. With alpha=0.2, a single bad reading doesn't immediately destroy a provider's score — it adapts gradually.

**Q: What happens if all providers' circuit breakers are OPEN?**
A: The pipeline checks all circuit breaker states before routing. If `healthy_providers` list is empty → HTTP 503 "All providers degraded".

**Q: How is this OpenAI-compatible?**
A: The `/v1/chat/completions` endpoint accepts the same JSON schema as OpenAI's API (`messages`, `model`, `temperature`, `max_tokens`). Any SDK or app already using OpenAI can point to `http://localhost:8000` instead and get rate limiting, caching, failover for free.

**Q: How does the gateway scale horizontally?**
A: Use `redis_token_bucket` strategy — rate limit state is shared in Redis across all instances. The in-memory strategies (token_bucket, sliding_window_*) only work for a single instance. The cache is currently in-memory too — for multi-instance, you'd replace it with a Redis-backed cache.

---

# PART 6 — USEFUL ENDPOINTS CHEAT SHEET

| Endpoint | Method | What It Does |
|---|---|---|
| `http://localhost:8000` | GET | Interactive visual dashboard |
| `http://localhost:8000/docs` | GET | Swagger API docs |
| `http://localhost:8000/health` | GET | System health + circuit breaker states |
| `http://localhost:8000/metrics` | GET | Prometheus metrics |
| `http://localhost:8000/v1/chat/completions` | POST | Send a chat request |
| `http://localhost:8000/v1/models` | GET | List enabled providers |
| `http://localhost:8000/gateway/reconfigure` | POST | Hot-swap algorithms |
| `http://localhost:8000/gateway/cache/clear` | POST | Clear cache |
| `http://localhost:8000/gateway/circuit-breaker/trip?provider_id=mock-fast` | POST | Force-open a circuit |
| `http://localhost:8000/gateway/circuit-breaker/reset` | POST | Reset all circuits to CLOSED |
| `http://localhost:9090` | GET | Prometheus UI |
| `http://localhost:3000` | GET | Grafana (admin/admin) |

---

# PART 7 — RUNNING TESTS

```powershell
# All unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# All tests with coverage
python -m pytest tests/ -v --cov=src

# Benchmark demo (runs against live gateway — make sure uvicorn is running)
python benchmark_demo.py
```
