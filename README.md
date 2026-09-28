# LLM Gateway

A production-grade, asynchronous API gateway that routes, rate-limits, semantically caches, and load-balances requests across multiple LLM providers — built from first principles with hand-rolled implementations of every core algorithm.

> All algorithms — LRU, LFU, Token Bucket, Sliding Window, Consistent Hashing, EWMA, Circuit Breaker — are implemented from scratch. No external algorithm libraries.

---

## What It Does

You point your app at `http://localhost:8000/v1/chat/completions` instead of OpenAI directly. The gateway handles everything in between:

```
Your App
    │
    ▼
┌─────────────────────────────────────────────┐
│              LLM Gateway Pipeline           │
│                                             │
│  1. Rate Limiter   → enforce per-tenant     │
│                      request quotas         │
│                                             │
│  2. Semantic Cache → return cached answers  │
│                      for similar prompts    │
│                                             │
│  3. Router         → pick the best provider │
│                      by latency / cost      │
│                                             │
│  4. Circuit Breaker → auto-failover when    │
│                       a provider goes down  │
└─────────────────────────────────────────────┘
    │
    ▼
OpenAI / Gemini / Perplexity / Ollama / Mock
```

The API is fully OpenAI-compatible — any SDK or app already using OpenAI just needs a URL change.

---

## Features

### Rate Limiting — 4 algorithms, all interchangeable at runtime

| Algorithm | How It Works | Best For |
|---|---|---|
| **Token Bucket** | Bucket fills at fixed rate, requests consume tokens. Allows bursts. | General-purpose APIs |
| **Sliding Window Log** | Stores exact timestamps, 100% accurate, no boundary exploits | Billing-critical systems |
| **Sliding Window Counter** | Two integers + weighted estimate. O(1) memory. ~99.95% accurate | High-scale, millions of tenants |
| **Redis Token Bucket** | Atomic Lua script executed server-side. Shared across all instances | Horizontally scaled deployments |

### Caching — 3 backends, semantic matching included

| Backend | How It Works |
|---|---|
| **LRU Exact** | Doubly-linked list + hashmap. O(1) get/set. Evicts least recently used. |
| **LFU Exact** | Frequency buckets + hashmap. O(1) get/set. Evicts least frequently used. |
| **Semantic Vector** | Two-tier: exact SHA-256 match first, then cosine similarity on 384-dim embeddings. Handles paraphrased prompts. |

The semantic cache uses `sentence-transformers/all-MiniLM-L6-v2` locally (no external calls). Falls back to a subword n-gram hash vectorizer if the model isn't installed.

### Load Balancing — 4 routing strategies

| Strategy | How It Works |
|---|---|
| **Round Robin** | Circular distribution across healthy providers |
| **Latency-Aware EWMA** | Routes to the provider with the lowest exponentially-smoothed latency. Penalises failures with 5000ms effective latency |
| **Cost-Aware** | Picks the provider with the lowest blended $/1k token price |
| **Consistent Hashing** | Virtual-node ring (50 vnodes/provider). Same tenant always routes to the same provider for cache locality |

### Circuit Breaker — automatic failover, zero code changes

Two trip strategies:
- **Consecutive Failure** — trips after N sequential errors
- **Sliding Window Error Rate** — trips when error % exceeds threshold in a rolling time window

Three states: `CLOSED` → `OPEN` → `HALF_OPEN`. Thundering herd prevention limits concurrent probe requests during recovery.

When a provider's circuit trips open, the gateway silently fails over to the next healthy provider in the configured failover chain. The caller receives a successful response.

### Provider Adapters

| Provider | Notes |
|---|---|
| **Mock** | Built-in fake provider with configurable latency and failure rate. No API key needed. |
| **OpenAI** | GPT-4o, GPT-4o-mini. Set `OPENAI_API_KEY`. |
| **Google Gemini** | Gemini 1.5 Flash/Pro. Message format translation handled automatically. Set `GEMINI_API_KEY`. |
| **Perplexity** | Sonar model. OpenAI-compatible API. Set `PERPLEXITY_API_KEY`. |
| **Ollama** | Local models (Llama 3, Mistral, etc). Runs at `localhost:11434`. No API key. |

### Observability

- **Prometheus metrics** at `/metrics` — request counts, latency histograms, cache hit/miss rates, circuit breaker states, token counts, estimated cost saved
- **Structured logging** with per-request correlation IDs (`X-Correlation-ID`) that trace through every log line
- **Grafana dashboard** via Docker Compose

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM GATEWAY PIPELINE                         │
│                                                                     │
│  1. RATE LIMITER                                                    │
│     ├── Token Bucket          (lazy refill, O(1), burst-friendly)  │
│     ├── Sliding Window Log    (exact timestamps, O(N) memory)      │
│     ├── Sliding Window Counter(weighted estimate, O(1) memory)     │
│     └── Redis Token Bucket    (atomic Lua script, distributed)     │
│                                                                     │
│  2. TWO-TIER CACHE                                                  │
│     ├── Tier 1: SHA-256 exact hash match       → O(1) fast path    │
│     ├── Tier 2: Cosine similarity on embeddings→ semantic match    │
│     └── Eviction: LRU (DLL+Map) or LFU (freq buckets)             │
│                                                                     │
│  3. ROUTER / LOAD BALANCER                                          │
│     ├── Round Robin           (circular, O(1))                     │
│     ├── Latency-Aware EWMA    (min EWMA latency, O(K))             │
│     ├── Cost-Aware            (min $/token, O(K))                  │
│     └── Consistent Hashing   (virtual-node ring, sticky sessions)  │
│                                                                     │
│  4. CIRCUIT BREAKER + FAILOVER                                      │
│     ├── CLOSED → OPEN → HALF_OPEN state machine                    │
│     ├── Sliding Window Error Rate or Consecutive Failure           │
│     └── Automatic failover cascade to next healthy provider        │
│                                                                     │
│  5. PROVIDER ADAPTERS                                               │
│     ├── OpenAI, Gemini, Perplexity, Ollama, Mock                   │
│     └── Unified interface via Adapter pattern                      │
│                                                                     │
│  6. OBSERVABILITY                                                   │
│     ├── Prometheus metrics (/metrics)                              │
│     └── Structured JSON logs with correlation ID tracing           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/llm_gateway_example.git
cd llm_gateway_example
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Redis, Prometheus and Grafana (optional but recommended)

```bash
docker compose up -d
```

### 3. Start the gateway

```bash
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the interactive control plane dashboard.

### 4. Send a request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain consistent hashing in distributed systems."}],
    "tenant_id": "user_1"
  }'
```

### 5. Check system health

```bash
curl http://localhost:8000/health
```

Returns circuit breaker states, active algorithm strategies, cache entry count, and configured providers.

### 6. Hot-swap algorithms at runtime (no restart)

```bash
curl -X POST http://localhost:8000/gateway/reconfigure \
  -H "Content-Type: application/json" \
  -d '{
    "rate_limiter_strategy": "sliding_window_log",
    "cache_strategy": "lfu_exact",
    "router_strategy": "cost_aware"
  }'
```

---

## Configuration

All algorithms are configured in `config/gateway_config.yaml`:

```yaml
rate_limiter:
  strategy: "token_bucket"       # token_bucket | sliding_window_log | sliding_window_counter | redis_token_bucket
  default_capacity: 10
  default_refill_rate: 2.0

cache:
  strategy: "semantic_vector"    # semantic_vector | lru_exact | lfu_exact | disabled
  max_size: 1000
  ttl_seconds: 3600
  similarity_threshold: 0.75

router:
  strategy: "latency_aware_ewma" # latency_aware_ewma | round_robin | cost_aware | consistent_hashing
  ewma_alpha: 0.2

circuit_breaker:
  strategy: "sliding_window_error_rate"   # sliding_window_error_rate | consecutive_failure
  failure_rate_threshold: 0.5
  window_size_seconds: 30
  minimum_requests: 5
  recovery_time_seconds: 15
```

To enable a real provider, set `enabled: true` in the `providers` section and export the corresponding API key as an environment variable.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive control plane dashboard |
| `/docs` | GET | Swagger / OpenAPI documentation |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/models` | GET | List all enabled providers |
| `/health` | GET | System health and circuit breaker states |
| `/metrics` | GET | Prometheus metrics scrape endpoint |
| `/gateway/reconfigure` | POST | Hot-swap algorithms at runtime |
| `/gateway/cache/clear` | POST | Purge all cache entries |
| `/gateway/circuit-breaker/trip` | POST | Force-open a provider's circuit breaker |
| `/gateway/circuit-breaker/reset` | POST | Reset all circuit breakers to CLOSED |

---

## Running Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Integration tests (full pipeline, failover)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --tb=short

# Load testing with Locust
locust -f tests/load/locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089

# Benchmark demo (runs against live gateway)
python benchmark_demo.py
```

---

## Design Patterns Used

| Pattern | Where Applied |
|---|---|
| **Strategy** | Rate limiters, cache backends, routers — all interchangeable at runtime |
| **Factory** | `RateLimiterFactory`, `CacheFactory`, `RouterFactory`, `ProviderFactory` |
| **Adapter** | Each provider wraps a different external API behind a uniform interface |
| **State Machine** | Circuit breaker CLOSED / OPEN / HALF_OPEN transitions |
| **Null Object** | `_NoOpCache` for `strategy: disabled` — no null checks in pipeline |
| **Dependency Injection** | FastAPI `Depends(get_pipeline)` injects shared pipeline into route handlers |

---

## Monitoring

| Service | URL | Notes |
|---|---|---|
| Gateway Dashboard | `http://localhost:8000` | Interactive control plane |
| Prometheus | `http://localhost:9090` | Raw metrics and query interface |
| Grafana | `http://localhost:3000` | Dashboards (default login: admin / admin) |

---

## Project Structure

```
llm_gateway/
├── config/
│   └── gateway_config.yaml        # All algorithm and provider config
├── src/
│   ├── gateway/
│   │   ├── app.py                 # FastAPI app, all HTTP endpoints
│   │   ├── pipeline.py            # Core 4-step request pipeline
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── config.py              # Pydantic config loader
│   │   └── static/index.html      # Control plane dashboard
│   ├── rate_limiter/              # Token Bucket, Sliding Window, Redis
│   ├── cache/                     # LRU, LFU, Semantic Vector, Embeddings
│   ├── router/                    # Round Robin, EWMA, Cost-Aware, Consistent Hash
│   ├── circuit_breaker/           # State machine, dual strategies
│   ├── providers/                 # OpenAI, Gemini, Perplexity, Ollama, Mock
│   └── observability/             # Prometheus metrics, structured logger
├── tests/
│   ├── unit/                      # Per-component algorithm tests
│   ├── integration/               # Full pipeline + failover tests
│   └── load/                      # Locust load test scenarios
├── docker-compose.yml             # Redis + Prometheus + Grafana
├── prometheus.yml                 # Prometheus scrape config
└── requirements.txt
```
