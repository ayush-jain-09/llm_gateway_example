# LLD & HLD Algorithms Deep Dive
> Complete ground-up explanation of every algorithm used in this project.
> Covers the idea, the data structure, the math, the code, the trade-offs, and interview questions.

---

# TABLE OF CONTENTS

1. [Rate Limiting](#1-rate-limiting)
   - 1.1 Token Bucket
   - 1.2 Sliding Window Log
   - 1.3 Sliding Window Counter
   - 1.4 Redis Token Bucket (Distributed)

2. [Caching](#2-caching)
   - 2.1 LRU Cache (Least Recently Used)
   - 2.2 LFU Cache (Least Frequently Used)
   - 2.3 Semantic Vector Cache
   - 2.4 Vector Embeddings & Cosine Similarity

3. [Load Balancing / Routing](#3-load-balancing--routing)
   - 3.1 Round Robin
   - 3.2 Latency-Aware EWMA
   - 3.3 Cost-Aware
   - 3.4 Consistent Hashing with Virtual Nodes

4. [Circuit Breaker](#4-circuit-breaker)
   - 4.1 State Machine (CLOSED / OPEN / HALF_OPEN)
   - 4.2 Consecutive Failure Strategy
   - 4.3 Sliding Window Error Rate Strategy
   - 4.4 Thundering Herd Prevention

5. [System Design Concepts Used](#5-system-design-concepts-used)
   - Design Patterns
   - Concurrency & Async Safety
   - HLD Concepts

---

---

# 1. RATE LIMITING

## What Problem Does It Solve?

Without rate limiting:
- A buggy script sends 100,000 requests/minute → server dies
- One free-tier user hogs all the resources → paying users suffer
- A single compromised API key causes infinite billing

Rate limiting enforces a quota per tenant/user. Exceed it → HTTP 429 "Too Many Requests".

---

## 1.1 Token Bucket

### The Idea
Imagine a physical bucket that holds up to 10 tokens. Every second, 2 new tokens are dropped in (but never more than 10). Each request takes 1 token out. If the bucket is empty → request is blocked.

```
capacity = 10 tokens
refill_rate = 2 tokens/second

[ 10 tokens full ]
→ Request 1: take 1 → [9 tokens]
→ Request 2: take 1 → [8 tokens]
...
→ Request 10: take 1 → [0 tokens]
→ Request 11: BLOCKED. retry_after = 0.5s (bucket needs 0.5s to refill 1 token)

After 5 seconds of idle: [10 tokens] (full again)
```

### Why "Lazy Refill"?
You don't need a background thread adding tokens every second. Instead, when a request arrives, you calculate "how many tokens should have accumulated since the last request" using elapsed time.

```python
elapsed = now - bucket.last_updated
new_tokens = elapsed * refill_rate
bucket.tokens = min(capacity, bucket.tokens + new_tokens)
```

This is **lazy evaluation** — compute only when needed. O(1), no threads.

### The Math
```
tokens_now = min(capacity, tokens_at_last_request + (elapsed_seconds × refill_rate))

if tokens_now >= cost:
    tokens_now -= cost
    ALLOW

else:
    deficit = cost - tokens_now
    retry_after = deficit / refill_rate
    BLOCK
```

### Data Structure
```python
@dataclass
class _BucketState:
    tokens: float        # Current tokens in the bucket
    last_updated: float  # time.monotonic() of last update

_buckets: Dict[str, _BucketState]  # One per tenant
```

One dict entry per tenant. Each entry is just 2 floats. Memory = O(K) where K = unique tenants.

### Code Walkthrough
```python
def _refill(self, bucket, now):
    elapsed = now - bucket.last_updated        # time since last request
    new_tokens = elapsed * self.refill_rate    # tokens earned
    bucket.tokens = min(self.capacity, bucket.tokens + new_tokens)  # cap at max
    bucket.last_updated = now

async def is_allowed(self, key, cost=1):
    now = time.monotonic()
    async with self._lock:                      # asyncio lock for thread safety
        if key not in self._buckets:
            self._buckets[key] = _BucketState(tokens=self.capacity, last_updated=now)
        bucket = self._buckets[key]
        self._refill(bucket, now)              # lazy refill first
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, bucket.tokens, 0.0   # allowed
        else:
            deficit = cost - bucket.tokens
            retry_after = deficit / self.refill_rate
            return False, bucket.tokens, retry_after  # blocked
```

### Properties
| Property | Value |
|---|---|
| Time Complexity | O(1) |
| Space Complexity | O(K) — K unique tenants |
| Burst Handling | YES — allows up to `capacity` in a burst |
| Accuracy | 100% exact |
| Distributed | NO — in-memory only |

### When to Use
- Default choice for most APIs
- When you want to allow occasional bursts but enforce a long-term average
- Example: user can send 10 messages at once, but not more than 2/second sustained

---

## 1.2 Sliding Window Log

### The Problem With Fixed Windows
Say you allow 10 requests per minute (fixed window: 0-60s, 60-120s...).

A user sends 10 requests at 11:59:55 (all valid in window 0-60) and 10 more at 12:00:05 (all valid in window 60-120). That's **20 requests in 10 seconds** — double the intended limit. This is the **boundary burst** problem.

### The Fix: Remember Every Timestamp
Store the exact timestamp of every request. On each new request, look at the last 60 seconds only.

```
Window = 60 seconds, max = 3 requests

Time 0s:   [ ]                    → Allow, log [0s]
Time 1s:   [0s]                   → Allow, log [0s, 1s]
Time 2s:   [0s, 1s]               → Allow, log [0s, 1s, 2s]
Time 3s:   [0s, 1s, 2s]           → BLOCK (3 in last 60s)
Time 61s:  [1s, 2s]   ← 0s expired → Allow, log [1s, 2s, 61s]
```

### The Algorithm
```
boundary = now - window_seconds

1. Remove all timestamps from the front of the deque where timestamp <= boundary
2. If len(log) + cost <= max_requests → ALLOW, append `now` to deque
3. Else → BLOCK, retry_after = oldest_timestamp + window_seconds - now
```

### Data Structure
```python
_logs: Dict[str, Deque[float]]
# One deque of float timestamps per tenant
```

`deque` is used (not list) because `popleft()` is O(1). List `pop(0)` is O(N).

### Code Walkthrough
```python
async def is_allowed(self, key, cost=1):
    now = time.time()
    boundary = now - self.window_seconds

    async with self._lock:
        log = self._logs.setdefault(key, deque())

        # Purge expired timestamps from the LEFT (oldest)
        while log and log[0] <= boundary:
            log.popleft()          # O(1) each

        if len(log) + cost <= self.max_requests:
            for _ in range(cost):
                log.append(now)    # O(1) each
            return True, float(self.max_requests - len(log)), 0.0
        else:
            oldest = log[0]
            retry_after = (oldest + self.window_seconds) - now
            return False, 0.0, retry_after
```

### Properties
| Property | Value |
|---|---|
| Time Complexity | O(M) amortized — M = expired entries purged |
| Space Complexity | O(N) per tenant — N = requests in window |
| Boundary Burst | NO — 100% precise |
| Accuracy | 100% exact |
| Memory | HIGH for busy tenants |

### When to Use
- When accuracy is critical (billing, financial APIs)
- Low-traffic systems where memory isn't a concern
- When you can't tolerate any over-permission

---

## 1.3 Sliding Window Counter

### The Idea
Sliding Window Log is accurate but memory-hungry (stores every timestamp). What if you only kept 2 integers and estimated?

Divide time into fixed windows (e.g., each window = 60 seconds). Keep:
- `current_count`: requests in the current window
- `previous_count`: requests in the previous window

Then estimate how many requests happened in the "virtual" sliding window that ends right now:

```
                 previous window        current window
                 [      60s      ] [         60s         ]
                 |----------------|---30s----*-->
                                           ↑ now
                                   time_into_current = 30s

overlap_ratio = 30 / 60 = 0.5
weight_prev   = 1 - 0.5 = 0.5   ← previous window contributes 50%

estimated = current_count + (previous_count × weight_prev)
          = current_count + (previous_count × 0.5)
```

As you move through the current window, the previous window's contribution fades linearly from 100% to 0%.

### Why This Works
The assumption: previous window's requests were spread uniformly. So only the 50% that falls inside your sliding window should count. This assumption is slightly wrong in edge cases (hence ~0.05% error) but works great in practice.

### The Math
```
window_id = floor(timestamp / window_seconds)
time_into_current = timestamp - (window_id × window_seconds)
weight_prev = 1.0 - (time_into_current / window_seconds)
estimated = current_count + (previous_count × weight_prev)

if estimated + cost <= max_requests → ALLOW, current_count += cost
else → BLOCK
```

### Data Structure
```python
@dataclass
class _WindowCounter:
    current_window_id: int    # which 60s window are we in
    current_count: int        # requests this window
    previous_count: int       # requests last window
```

Just 3 integers per tenant. Dramatically less memory than Sliding Window Log.

### Window Transition
When `window_id` changes (we rolled into a new minute):
```python
if curr_window_id > counter.current_window_id:
    if curr_window_id == counter.current_window_id + 1:
        counter.previous_count = counter.current_count  # shift current to previous
    else:
        counter.previous_count = 0                       # gap > 1 window, discard
    counter.current_count = 0
    counter.current_window_id = curr_window_id
```

### Properties
| Property | Value |
|---|---|
| Time Complexity | O(1) |
| Space Complexity | O(1) per tenant — just 3 integers |
| Boundary Burst | ~0.05% error possible |
| Accuracy | ~99.95% |
| Memory | VERY LOW — best for millions of tenants |

### When to Use
- High-scale systems with many tenants
- When memory efficiency matters more than perfect accuracy
- Used by Nginx, Cloudflare, and most CDNs in production

---

## 1.4 Redis Token Bucket (Distributed)

### The Problem
The above 3 algorithms are all in-memory. If you run 5 copies of the gateway (horizontal scaling), each instance has its own separate counter. User could hit 10 requests on each instance = 50 total. The rate limit doesn't work.

### The Solution
Store token bucket state in Redis (a shared external store). All instances share the same state.

### The Challenge: Atomicity
Consider two gateway instances (A and B) both serving the same tenant simultaneously:

```
Instance A: reads tokens = 1
Instance B: reads tokens = 1
Instance A: tokens >= 1, allow, writes tokens = 0
Instance B: tokens >= 1, allow, writes tokens = 0  ← RACE CONDITION! Both allowed.
```

You can't just read-then-write without protection. Options:
1. Redis transactions (WATCH/MULTI/EXEC) — complex
2. Distributed locks (SETNX) — adds latency
3. **Lua script — atomic execution server-side** ✅

### Lua Script Approach
Redis guarantees that a Lua script runs **atomically** — no other command can execute between lines of the script. So you do the entire read-compute-write in a single Lua script that Redis executes as one unit.

```lua
-- This entire script runs atomically in Redis
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Read current state from Redis hash
local state = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(state[1])
local last_updated = tonumber(state[2])

-- First time this key is seen
if not tokens then
    tokens = capacity
    last_updated = now
else
    -- Lazy refill: compute tokens since last update
    local delta = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + (delta * refill_rate))
    last_updated = now
end

-- Check and deduct
if tokens >= cost then
    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, 3600)   -- auto-cleanup after 1 hour of inactivity
    return {1, tokens, 0.0}           -- allowed
else
    local retry_after = (cost - tokens) / refill_rate
    return {0, tokens, retry_after}   -- blocked
end
```

### Graceful Degradation
```python
try:
    result = await client.eval(LUA_SCRIPT, ...)
    return parse(result)
except Exception:
    # Redis down → fall back to in-memory token bucket
    return await self._fallback_limiter.is_allowed(key, cost)
```

If Redis goes down, the gateway doesn't crash. It falls back to per-instance in-memory limiting. Rate limiting becomes approximate but the service stays alive.

### Properties
| Property | Value |
|---|---|
| Time Complexity | O(1) Lua script execution |
| Space Complexity | O(K) in Redis |
| Distributed | YES — shared across all instances |
| Atomicity | YES — Lua script runs as one unit |
| Fault Tolerant | YES — fallback to in-memory |

---

---

# 2. CACHING

## What Problem Does It Solve?

Every call to OpenAI/Gemini costs money and takes 1-10 seconds. If the same question is asked 1000 times, you make 1000 API calls. A cache stores the first response and returns it for all subsequent identical (or similar) questions instantly.

---

## 2.1 LRU Cache — Least Recently Used

### The Idea
Keep a fixed-size cache. When it's full and you need to add something new, evict the item that was accessed the **longest time ago** (least recently used).

**Intuition:** If you haven't needed something in a long time, you probably won't need it again soon.

```
Cache size = 3

SET A  →  [A]
SET B  →  [B, A]      (B is most recent)
SET C  →  [C, B, A]   (full)
GET A  →  [A, C, B]   (A moves to front — recently used)
SET D  →  [D, A, C]   (B evicted — B was LRU)
```

### The Data Structure Problem
You need:
- `get(key)` in O(1) — a hashmap gives this
- `put(key, value)` in O(1)
- "Move to front" in O(1) — a linked list gives this
- "Remove from back" in O(1) — a doubly linked list gives this

**The solution: Doubly Linked List + Hashmap**

```
hashmap:   { "A": Node_A, "B": Node_B, "C": Node_C }

doubly linked list:
HEAD ↔ [C] ↔ [A] ↔ [B] ↔ TAIL
         ↑ most recent    ↑ least recent
```

The hashmap gives O(1) lookup. The linked list gives O(1) move-to-front and O(1) pop-from-tail.

### Sentinel Nodes
```
HEAD ↔ [real nodes...] ↔ TAIL
```

`HEAD` and `TAIL` are dummy nodes that never hold data. They eliminate edge cases:
- Without them: inserting into an empty list requires null checks
- With them: always just insert between HEAD and HEAD.next

```python
self._head = _Node("__HEAD__", None)
self._tail = _Node("__TAIL__", None)
self._head.next = self._tail
self._tail.prev = self._head
```

### Operations

**get(key):**
1. Hash the prompt → look up in `_map`
2. If not found → return None (miss)
3. If found but expired → remove from list and map → return None
4. Move node to head (most recently used)
5. Return item

**set(key, value):**
1. If key exists → update node + move to head
2. If new key:
   a. Create new node, insert at head
   b. If `len(map) > max_size` → pop from tail (evict LRU), delete from map

**_remove_node(node):**
```python
# Unlink from wherever it currently is
prev_node.next = node.next
node.next.prev = prev_node
```

**_add_to_head(node):**
```python
# Insert right after HEAD sentinel
node.next = self._head.next
node.prev = self._head
self._head.next.prev = node
self._head.next = node
```

**_pop_tail():**
```python
# The node just before TAIL is the LRU
lru_node = self._tail.prev
self._remove_node(lru_node)
return lru_node
```

### Visual Step by Step
```
Initial (capacity=3):
HEAD ↔ TAIL

SET "python" (value=resp1):
HEAD ↔ [python] ↔ TAIL

SET "java" (value=resp2):
HEAD ↔ [java] ↔ [python] ↔ TAIL
       ^MRU      ^LRU

SET "rust" (value=resp3):
HEAD ↔ [rust] ↔ [java] ↔ [python] ↔ TAIL
       ^MRU               ^LRU

GET "python":  ← python accessed, move to front
HEAD ↔ [python] ↔ [rust] ↔ [java] ↔ TAIL
       ^MRU               ^LRU

SET "go" (capacity full):  ← evict java (LRU)
HEAD ↔ [go] ↔ [python] ↔ [rust] ↔ TAIL
       ^MRU               ^LRU
```

### Properties
| Property | Value |
|---|---|
| get() | O(1) |
| set() | O(1) |
| Eviction | O(1) |
| Space | O(N) — N = max_size |
| Best for | Recency-based patterns |

---

## 2.2 LFU Cache — Least Frequently Used

### The Idea
Evict the item that has been accessed the **fewest times total**, not just the least recently.

**Why it's better than LRU sometimes:**
- A popular prompt asked 1000 times should never be evicted
- LRU would evict it if it hasn't been asked in the last hour
- LFU keeps it because it has `freq=1000`

**LFU use case:** FAQ systems, popular document caches, anything with a stable hot set.

### The Challenge
A naive LFU (scan all items to find min frequency) is O(N) for eviction. You need O(1).

### The Data Structure
Three components:

```
1. key_to_node: Dict[key → LFUNode]
   Direct lookup by key

2. freq_to_dll: Dict[frequency → DoublyLinkedList of nodes with that freq]
   Groups nodes by frequency count

3. min_freq: int
   Points to the DLL where the eviction candidate lives
```

```
key_to_node = {
    "python": Node(freq=3),
    "java":   Node(freq=1),
    "rust":   Node(freq=2),
}

freq_to_dll = {
    1: DLL → [java]
    2: DLL → [rust]
    3: DLL → [python]
}

min_freq = 1  ← java will be evicted next
```

### Operations

**get(key):**
1. Look up in `key_to_node`
2. Call `_increment_freq(node)`:
   - Remove node from `freq_to_dll[old_freq]`
   - If old_freq == min_freq AND that DLL is now empty → `min_freq += 1`
   - Add node to `freq_to_dll[old_freq + 1]` (create DLL if needed)
   - `node.freq += 1`
3. Return item

**set(key, value):**
- If key exists: update value + `_increment_freq`
- If new key:
  1. If at capacity: evict from `freq_to_dll[min_freq].pop_tail()` (the LFU item)
  2. Insert new node into `freq_to_dll[1]`
  3. `min_freq = 1` ← new items always start at freq=1

**The key insight for O(1) eviction:**
`min_freq` always points directly to the bucket with the least-frequent items. You never scan — just go straight to `freq_to_dll[min_freq]` and pop from its tail.

### Why Within Same Frequency, Pop From Tail?
Within `freq_to_dll[1]`, there might be multiple items all accessed once. Which one to evict? The one accessed **least recently** within that frequency group. Since we always add to the head, the tail is the oldest = LRU within the same frequency. This makes it LFU+LRU tiebreaking, which is the classic O(1) LFU design.

### Visual Step by Step
```
Capacity=3, min_freq=0

SET A:  key_to_node={A:f1}  freq_to_dll={1:[A]}  min_freq=1
SET B:  key_to_node={A:f1,B:f1}  freq_to_dll={1:[B,A]}  min_freq=1
SET C:  key_to_node={A:f1,B:f1,C:f1}  freq_to_dll={1:[C,B,A]}  min_freq=1

GET A:  A.freq → 2
        freq_to_dll={1:[C,B], 2:[A]}  min_freq=1 (1-DLL still has B,C)

GET A:  A.freq → 3
        freq_to_dll={1:[C,B], 2:[], 3:[A]}  min_freq=1

SET D:  (at capacity) → evict from freq_to_dll[min_freq=1].pop_tail() → evict B
        INSERT D into freq_to_dll[1], min_freq=1
        freq_to_dll={1:[D,C], 3:[A]}
```

### Properties
| Property | Value |
|---|---|
| get() | O(1) |
| set() | O(1) |
| Eviction | O(1) |
| Space | O(N) |
| Best for | Stable hot-set workloads |

### LRU vs LFU — When to Use Which?
| Scenario | LRU | LFU |
|---|---|---|
| News feed (fresh content matters) | ✅ | ❌ |
| FAQ / docs (popular = always needed) | ❌ | ✅ |
| User sessions (recent = relevant) | ✅ | ❌ |
| Model responses to common prompts | ❌ | ✅ |

---

## 2.3 Semantic Vector Cache

### The Problem With Exact Matching
LRU and LFU use SHA-256 hash of the prompt. Two prompts must be byte-for-byte identical to hit the cache:

```
"What is machine learning?"      → hash: abc123
"What is machine learning? "     → hash: xyz789  ← MISS (trailing space!)
"Explain machine learning to me" → hash: def456  ← MISS (same meaning, different words)
```

This wastes money — the same question phrased differently hits the LLM every time.

### The Solution: Meaning-Based Matching
Convert prompts into vectors that capture **meaning**. Similar meanings → nearby vectors. Check if the incoming prompt is "close enough" to any cached prompt.

### Two-Tier Architecture

```
Tier 1: SHA-256 exact match → O(1) instant
    ↓ miss
Tier 2: Embed prompt → compare against all stored vectors → cosine similarity
    ↓ similarity >= threshold (0.75)
    → HIT (semantic match)
    ↓ similarity < threshold
    → MISS (call the LLM)
```

Tier 1 is the fast path — zero embedding overhead for exact repeats. Tier 2 handles paraphrases.

### What Is a Vector Embedding?
A sentence embedding is a list of 384 floating point numbers (a 384-dimensional vector) that represents the **meaning** of a sentence. Sentences with similar meaning produce similar vectors.

```
"What is a hash table?"       → [0.23, -0.45, 0.12, ..., 0.67]  ← 384 numbers
"Explain hash maps to me"     → [0.21, -0.44, 0.11, ..., 0.65]  ← nearby!
"How do I cook pasta?"        → [0.89,  0.34,-0.55, ..., 0.01]  ← far away
```

The model (`all-MiniLM-L6-v2`) is a 80MB neural network trained to produce these vectors. It runs locally, no API calls.

### Cosine Similarity
To measure how similar two vectors are, use cosine similarity = the cosine of the angle between them.

```
cos(θ) = (A · B) / (|A| × |B|)

Where A · B = dot product = sum of element-wise products
      |A| = magnitude (length) of vector A
```

Range: -1.0 (opposite) to 0.0 (unrelated) to 1.0 (identical meaning)

Since both vectors are already L2-normalized (magnitude = 1.0), this simplifies to:
```python
cosine_similarity = np.dot(vec1, vec2)  # dot product of normalized vectors
```

Threshold = 0.75 means: if the angle between vectors corresponds to >75% similarity → cache hit.

```
"What is a hash table?" vs "Explain hash maps"   → 0.87 → HIT  ✅
"What is a hash table?" vs "How to make pasta?"   → 0.12 → MISS ❌
```

### Embedding Generation (Async Non-Blocking)
Embedding is CPU-intensive (neural network inference). You can't do it on the async event loop thread — it would block everything.

```python
async def embed(self, text: str) -> np.ndarray:
    # Offload to thread pool — event loop stays free
    return await asyncio.to_thread(self._sync_embed, text)
```

`asyncio.to_thread` runs the CPU work in a separate OS thread from a thread pool, and the event loop continues handling other requests while waiting.

### Fallback Embedding (No sentence-transformers needed)
If `sentence-transformers` isn't installed:
1. Lowercase and clean the text
2. Remove stop words (a, the, is, are...)
3. For each remaining word: hash it → add 2.0 to that bucket in the 384-dim vector
4. For each character trigram in each word: hash it → add 0.8 to that bucket
5. L2-normalize the result

This is a deterministic "bag of subwords" approach. Not as accurate as the neural model but works well for similar phrasing and requires no downloads.

### Eviction: LRU by Last Access Time
When the cache is full and a new entry needs to be added:
```python
def _evict_lru(self):
    oldest = min(self._entries, key=lambda e: e.last_accessed)
    self._remove_entry(oldest)
```

Simple linear scan to find the least recently accessed entry. O(N) but only happens on cache write at capacity.

---

## 2.4 Vector Embeddings & Cosine Similarity — Deep Dive

### Why 384 Dimensions?
`all-MiniLM-L6-v2` produces 384-dimensional vectors. More dimensions = more expressive power but more memory and compute. 384 is the sweet spot for semantic sentence similarity tasks.

### What Does L2 Normalization Do?
```
raw_vector = [3.0, 4.0]
magnitude = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
normalized = [3/5, 4/5] = [0.6, 0.8]

magnitude of normalized = sqrt(0.6² + 0.8²) = sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
```

After normalization, every vector has length 1.0. This makes cosine similarity = dot product, which is just one line of numpy — extremely fast.

### Why Not Euclidean Distance Instead?
Euclidean distance measures absolute distance in space. Cosine similarity measures the angle between vectors. For text:
- "machine learning" and "Machine Learning" (capitalization) would have different magnitudes → Euclidean distance would say they're different
- After normalization and using cosine, they're identical → angle = 0 → similarity = 1.0

Cosine is the right metric for comparing meaning regardless of sentence length.

---

---

# 3. LOAD BALANCING / ROUTING

## What Problem Does It Solve?

You have multiple LLM providers (or multiple instances of the same provider). Each request needs to go to one of them. Which one? The router decides.

Good routing:
- Maximizes throughput (don't overload one provider while others are idle)
- Minimizes latency (prefer faster providers)
- Minimizes cost (prefer cheaper providers when speed isn't critical)
- Maintains session affinity (same user → same provider for consistent conversation)

---

## 3.1 Round Robin

### The Idea
Cycle through providers in order. Request 1 → Provider A. Request 2 → Provider B. Request 3 → Provider C. Request 4 → Provider A again.

```
providers = [A, B, C]
index = 0

Request 1: A (index → 1)
Request 2: B (index → 2)
Request 3: C (index → 0)
Request 4: A (index → 1)
```

### Implementation
```python
async def select_provider(self, available_providers, request):
    async with self._lock:
        selected = available_providers[self._index % len(available_providers)]
        self._index = (self._index + 1) % len(available_providers)
        return selected
```

`% len(providers)` handles the wrap-around. The lock ensures two concurrent requests don't pick the same index.

### Properties
- O(1) selection
- Uniform distribution
- Doesn't adapt to provider health or speed
- Best when all providers are equal and healthy

---

## 3.2 Latency-Aware EWMA

### The Problem With Round Robin
Provider A responds in 50ms. Provider B (being overloaded) responds in 3000ms. Round Robin still sends 50% of traffic to the slow provider. Users suffer.

### The Solution
Track observed latency per provider. Route to the fastest one.

But raw average latency has a problem: one unlucky 5-second spike would ruin a provider's score for a long time.

### EWMA — Exponentially Weighted Moving Average
Give more weight to **recent** measurements. Old data fades out exponentially.

```
EWMA_new = alpha × latency_new + (1 - alpha) × EWMA_old

alpha = 0.2 (20% weight to new, 80% to history)
```

Example: Provider was fast at 50ms, then got slow at 500ms
```
EWMA starts = 50ms
After 1 slow request:  0.2×500 + 0.8×50  = 100 + 40 = 140ms
After 2 slow requests: 0.2×500 + 0.8×140 = 100 + 112 = 212ms
After 3 slow requests: 0.2×500 + 0.8×212 = 100 + 169 = 269ms
...gradually rises to reflect the new reality
```

Example: Provider was slow at 500ms, then recovered to 50ms
```
EWMA starts = 500ms
After 1 fast request:  0.2×50 + 0.8×500 = 10 + 400 = 410ms
After 2 fast requests: 0.2×50 + 0.8×410 = 10 + 328 = 338ms
...gradually recovers
```

Alpha controls how fast it adapts. Higher alpha = more reactive to new data but more volatile. Lower alpha = more stable but slower to adapt.

### Failure Penalty
```python
effective_latency = latency if success else max(5000.0, latency * 2.0)
```

If a request fails, treat it as a 5000ms (5 second) latency. This quickly pushes the failing provider to the bottom of the ranking and routes traffic away from it.

### Selection
```python
async def select_provider(self, available_providers, request):
    async with self._lock:
        best_provider = min(available_providers,
                            key=lambda p: self._ewma_latencies.get(p.provider_id, 200.0))
        return best_provider
```

New providers get a default latency of 200ms until real data comes in — reasonable enough to get traffic but not so low it monopolizes everything.

### Properties
- O(K) — K = number of providers (tiny, usually 2-5)
- Self-healing: automatically shifts traffic when a provider recovers
- Works without explicit health checks

---

## 3.3 Cost-Aware Routing

### The Idea
Pick the cheapest provider every time. Simple.

```python
cheapest = min(providers, key=lambda p: p.cost_per_1k_input + p.cost_per_1k_output)
```

Blended cost = (input cost + output cost) per 1000 tokens. The provider with the lowest blended cost gets every request.

### When to Use
- Background batch processing where latency doesn't matter
- Cost optimization over user experience
- When providers are otherwise equal in quality

### Limitation
No adaptation to latency or failures. A cheap but extremely slow provider would get all traffic.

---

## 3.4 Consistent Hashing With Virtual Nodes

### The Problem
**Scenario:** You want the same user (tenant) to always go to the same provider. Why?
- If Provider A has cached that user's conversation context, sticky routing means cache hits
- Predictable, auditable routing for compliance

Simple approach: `hash(tenant_id) % num_providers` — but this breaks badly when providers are added/removed. If you go from 3 to 4 providers, almost every user gets remapped. Cache locality destroyed.

### The Hash Ring Solution
Imagine a circle (ring) with numbers 0 to 2³²-1 (about 4.3 billion points).

Place each provider at multiple points on the ring (virtual nodes). To route a request: hash the tenant_id → find its position on the ring → go clockwise until you hit a provider's virtual node.

```
Ring (simplified to 0-100):

  Provider A at: 10, 35, 72
  Provider B at: 20, 55, 85
  Provider C at: 5,  40, 90

  tenant_id hashes to 25 → go clockwise → hit 35 (Provider A) → route to A
  tenant_id hashes to 60 → go clockwise → hit 72 (Provider A) → route to A
  tenant_id hashes to 80 → go clockwise → hit 85 (Provider B) → route to B
```

### Why Virtual Nodes?
With just 1 real node per provider, the distribution on the ring could be very uneven — one provider might get 80% of the hash space. Virtual nodes (50 per provider = 150 total ring points for 3 providers) smooth out the distribution.

### When a Provider Is Removed
Only the tenants mapped to that provider's virtual nodes remap to the next clockwise provider. Everyone else stays. Compare to modulo: removing one provider remaps ~N/(N+1) of all keys. Consistent hashing remaps only ~1/N.

```
3 providers, remove 1:
- Modulo hashing: 66% of keys remap
- Consistent hashing: 33% of keys remap (only the ones that were going to removed provider)
```

### Implementation

**Build the ring:**
```python
def _build_ring(self, available_providers):
    ring = []
    ring_map = {}

    for provider in available_providers:
        for v_idx in range(50):  # 50 virtual nodes per provider
            v_key = f"{provider.provider_id}#vnode_{v_idx}"
            v_hash = int(md5(v_key)[:8], 16)  # 32-bit hash
            ring.append(v_hash)
            ring_map[v_hash] = provider.provider_id

    ring.sort()  # sorted list of hash positions
```

**Route a request:**
```python
key_hash = hash(tenant_id)

# Binary search: find first ring position >= key_hash
idx = bisect.bisect_right(ring, key_hash)
if idx == len(ring):
    idx = 0  # wrap around

provider_id = ring_map[ring[idx]]
```

`bisect.bisect_right` is O(log N) binary search on the sorted ring. Fast even for large rings.

### Properties
| Property | Value |
|---|---|
| select_provider | O(V×N + log(V×N)) — build ring + binary search |
| Sticky routing | YES — same key always maps to same provider |
| Node add/remove | Only ~1/N keys remap |
| Best for | Session affinity, cache locality |

---

---

# 4. CIRCUIT BREAKER

## What Problem Does It Solve?

Provider X is having an outage — all requests time out after 30 seconds. Without a circuit breaker:
1. Request arrives
2. Gateway tries Provider X
3. Waits 30 seconds → timeout error
4. Next request → same thing
5. 1000 users × 30 seconds = the gateway is essentially down, all threads blocked

With a circuit breaker: after enough failures, the gateway **stops trying** Provider X entirely and immediately routes to another provider. No waiting. Fail fast, recover gracefully.

---

## 4.1 State Machine: CLOSED / OPEN / HALF_OPEN

A state machine is a system that can be in exactly one state at a time, with defined transitions between states.

```
                   ┌─────────────────────────────┐
                   │                             │
    failures       │          CLOSED             │  success (normal)
   exceed          │      (normal operation)     │
   threshold       │       all reqs pass         │
       ┌───────────┘                             └───────────┐
       ▼                                                      │
    ┌─────────────────────────────┐                          │
    │          OPEN               │                          │
    │   (fast-fail, no LLM calls) │                          │
    │   all reqs rejected         │                          │
    └─────────────────────────────┘                          │
       │                                                      │
       │ cooldown timer expires                               │
       ▼                                                      │
    ┌─────────────────────────────┐  N probe successes        │
    │        HALF_OPEN            │─────────────────────────▶│
    │    (cautious probing)       │
    │  max 3 probe reqs allowed   │
    └─────────────────────────────┘
       │
       │ any probe fails
       ▼
    back to OPEN
```

### CLOSED State
Normal operation. Every request passes through. The circuit breaker records success/failure history but doesn't interfere.

### OPEN State
Tripped circuit. Every `allow_request()` call immediately throws `CircuitBreakerOpenException` with a `retry_after` time. No LLM calls are made. Fast-fail in microseconds instead of waiting 30s for a timeout.

After `recovery_time_seconds` (default: 15s), the circuit transitions to HALF_OPEN.

### HALF_OPEN State
Recovery testing. The circuit is "cautiously open" — it allows a small number of probe requests to test if the provider has recovered.

- Max 3 probes in-flight at once (`_max_half_open_probes = 3`)
- If all probes succeed → CLOSED (provider is healthy again)
- If any probe fails → immediately back to OPEN (provider still down)

---

## 4.2 Consecutive Failure Strategy

### How It Works
Trip the circuit after 5 consecutive failures (no successes in between).

```python
self._consecutive_failures += 1
if self._consecutive_failures >= 5:
    transition_to(OPEN)

# Any success resets the counter
self._consecutive_failures = 0
```

### Limitation
Single intermittent errors don't trip the circuit — you need 5 in a row. But a provider running at 40% error rate never trips (failures aren't consecutive — successes keep resetting the counter).

### When to Use
- Providers that either work perfectly or fail completely
- All-or-nothing failure modes (network partition, service down)

---

## 4.3 Sliding Window Error Rate Strategy (Default)

### How It Works
Track success/failure over a rolling time window. If the error rate exceeds 50% → trip.

```python
_history: Deque[Tuple[float, bool]]  # (timestamp, is_success)

# On each failure:
self._history.append((now, False))

# Purge events older than window_size_seconds
while history[0][0] <= now - window_size_seconds:
    history.popleft()

# Evaluate
total = len(history)
if total >= minimum_requests:  # need at least 5 data points
    failures = sum(1 for _, success in history if not success)
    error_rate = failures / total
    if error_rate >= 0.5:
        transition_to(OPEN)
```

### Example
```
window = 30 seconds, min_requests = 5, threshold = 50%

t=0s:  SUCCESS  history=[S]
t=1s:  SUCCESS  history=[S,S]
t=2s:  FAIL     history=[S,S,F]
t=3s:  FAIL     history=[S,S,F,F]
t=4s:  FAIL     history=[S,S,F,F,F]  → total=5, failures=3, rate=60% → TRIP!
```

This handles the "flapping" scenario better — a provider with 40% error rate properly trips the circuit even if the failures aren't consecutive.

### Why Minimum Requests?
Without `minimum_requests`: the very first request failing would trip the circuit (1/1 = 100%). That's way too sensitive. Requiring at least 5 requests means you have enough data to be confident about the failure rate.

---

## 4.4 Thundering Herd Prevention

### The Problem
Provider X has been in OPEN state for 15 seconds (recovery period). The circuit transitions to HALF_OPEN. At that exact moment, 500 queued requests all try to probe Provider X simultaneously. Even a healthy provider might struggle with 500 sudden concurrent requests — it could fail and trip the circuit again immediately.

### The Solution
Limit simultaneous probes to 3 during HALF_OPEN:

```python
if self._half_open_inflight_probes >= self._max_half_open_probes:
    raise CircuitBreakerOpenException(provider_id, 1.0)
self._half_open_inflight_probes += 1
```

Only 3 requests at a time get through to test the provider. All others fast-fail. If those 3 succeed and `_half_open_success_count >= half_open_success_threshold` → close the circuit. Then all traffic flows normally.

### Why "Monotonic Clock" for Timing?
The code uses `time.monotonic()` instead of `time.time()` for timing the recovery period.

`time.time()` returns wall clock time. If the system clock is adjusted (NTP sync, daylight saving, manual change), time could jump backward → the recovery period calculation would be wrong.

`time.monotonic()` is guaranteed to always move forward, regardless of system clock changes. It's the right choice for measuring elapsed time in distributed systems.

---

---

# 5. SYSTEM DESIGN CONCEPTS USED

---

## Design Patterns

### Strategy Pattern
Define a family of algorithms, encapsulate each one, and make them interchangeable.

```
RateLimiterBase (interface)
    ├── TokenBucketRateLimiter
    ├── SlidingWindowLogRateLimiter
    ├── SlidingWindowCounterRateLimiter
    └── RedisTokenBucketRateLimiter

CacheBase (interface)
    ├── LRUCache
    ├── LFUCache
    └── SemanticVectorCache
```

The pipeline holds a `rate_limiter: RateLimiterBase`. It doesn't know or care which algorithm is running. You swap strategies by replacing the object, not changing calling code.

### Factory Pattern
Centralizes object creation. The config string `"token_bucket"` maps to `TokenBucketRateLimiter(...)`.

```python
class RateLimiterFactory:
    _registry = {
        "token_bucket": TokenBucketRateLimiter,
        "sliding_window_log": SlidingWindowLogRateLimiter,
        ...
    }

    @classmethod
    def create(cls, config):
        return cls._registry[config.strategy](config.capacity, config.refill_rate)
```

### Adapter Pattern
Each LLM provider has a different API (different endpoints, auth, request/response formats). Adapters wrap each provider and expose a standard `generate_completion(request)` interface.

```
BaseProvider (standard interface)
    ├── OpenAIAdapter    (POST /v1/chat/completions with Bearer token)
    ├── GeminiAdapter    (POST /generateContent with ?key= param, different message format)
    ├── OllamaAdapter    (POST /api/chat, no auth, different response format)
    └── MockProvider     (pure in-memory, no HTTP)
```

### State Machine Pattern
Circuit Breaker is the State Machine pattern. Each state (CLOSED/OPEN/HALF_OPEN) has clearly defined:
- Entry actions (reset counters)
- Behavior while in the state (allow/block requests)
- Exit conditions (failure threshold, cooldown timer, probe success)
- Transitions (which states you can move to from here)

### Null Object Pattern
`_NoOpCache` for `strategy: "disabled"`:
```python
class _NoOpCache(CacheBase):
    async def get(self, prompt): return None
    async def set(self, prompt, value): pass
    def size(self): return 0
```

Instead of `if cache_strategy != "disabled": check cache` everywhere, just swap in a cache that does nothing. Calling code stays clean.

### Dependency Injection (via FastAPI)
```python
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    pipeline: GatewayPipeline = Depends(get_pipeline)  # injected
):
    return await pipeline.process_chat_completion(request)
```

FastAPI's `Depends()` injects the shared pipeline object into every route handler. No global variable access in business logic. Easier to test (inject a mock pipeline).

---

## Concurrency Concepts

### asyncio and Event Loop
Python's `asyncio` runs coroutines on a single thread via an event loop. When one coroutine is waiting for I/O (network request to OpenAI), the event loop runs other coroutines. This is **cooperative multitasking** — much lighter than OS threads.

### asyncio.Lock
Multiple coroutines can run "concurrently" on the same thread. If two coroutines both try to modify `_buckets[tenant_id]` at the same time, you get race conditions.

```python
async with self._lock:
    # Only one coroutine at a time can be here
    bucket = self._buckets[key]
    self._refill(bucket, now)
    bucket.tokens -= cost
```

`async with self._lock` suspends the coroutine if the lock is held by another, then resumes when released. This is an **async mutex** — non-blocking for the event loop while waiting.

### asyncio.to_thread
Runs a CPU-bound or blocking function in a thread pool without blocking the event loop:

```python
async def embed(self, text):
    return await asyncio.to_thread(self._sync_embed, text)
    # Event loop is free while the neural network runs in a background thread
```

### asyncio.create_task
Schedules a coroutine to run in the background without waiting for it:

```python
# Don't wait for cache write — return the response to the user immediately
asyncio.create_task(self.cache.set(prompt_text, response, ttl_seconds=3600))
```

The user gets the response ~5ms faster. Cache write happens asynchronously.

---

## HLD Concepts

### The Gateway Pattern
A single entry point that all clients talk to. The gateway handles cross-cutting concerns (auth, rate limiting, caching, routing) so individual services don't have to. This is the same pattern used by AWS API Gateway, Kong, and Nginx.

### Failover / High Availability
When the primary provider fails, automatically try the next one in the failover chain. The user never sees a 500 error unless **all** providers are down. This is the same principle behind multi-AZ deployments in AWS.

### Graceful Degradation
When Redis is unavailable, don't crash — fall back to in-memory rate limiting. When `sentence-transformers` isn't installed, fall back to the n-gram hash vectorizer. The system does less but stays alive.

### Observability: Metrics, Logs, Traces
- **Metrics** (Prometheus): Numbers over time. "How many requests? What latency? How many cache hits?" Used for dashboards and alerting.
- **Logs** (structured JSON): Events. "Request X from tenant Y to provider Z took 150ms." Used for debugging.
- **Correlation ID**: A unique ID per request that's injected into every log line. When debugging a problem, search for the correlation ID and see every log line for that request across all systems.

### Horizontal Scaling
Running multiple copies of the same service. In-memory state (token bucket, LRU cache) doesn't work horizontally — each instance has its own isolated state. Shared external state (Redis) enables true horizontal scaling. This is why the Redis Token Bucket exists.

---

## Complexity Summary Table

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Token Bucket | O(1) | O(K) | K = tenants |
| Sliding Window Log | O(M) amortized | O(N) | N = requests/window |
| Sliding Window Counter | O(1) | O(K) | Most memory-efficient |
| Redis Token Bucket | O(1) Lua | O(K) Redis | Distributed |
| LRU Cache get | O(1) | O(N) | N = max_size |
| LRU Cache set | O(1) | O(N) | |
| LFU Cache get | O(1) | O(N) | |
| LFU Cache set | O(1) | O(N) | |
| Semantic Cache get | O(N) tier 2 | O(N×D) | D = embedding dims |
| Round Robin | O(1) | O(1) | |
| EWMA Router | O(K) | O(K) | K = providers |
| Cost-Aware Router | O(K) | O(1) | |
| Consistent Hash | O(V×K + log(V×K)) | O(V×K) | V = virtual nodes |
| Circuit Breaker | O(W) amortized | O(W) | W = window size |

---

## The Full Request Pipeline

```
Client Request
     │
     ▼
HTTP Middleware
 - Attach/generate Correlation ID (X-Correlation-ID header)
 - Start timer
     │
     ▼
Rate Limiter.is_allowed(tenant_id)
 - Token Bucket / Sliding Window / Redis Lua
 - NO  → HTTP 429, Retry-After header
 - YES → continue
     │
     ▼
Cache.get(prompt_text)
 - Tier 1: SHA-256 hash exact lookup
 - Tier 2: Embed → cosine similarity scan
 - HIT  → return cached response (EXACT or SEMANTIC label)
 - MISS → continue
     │
     ▼
Filter: remove providers where circuit_breaker.state == OPEN
 - If 0 healthy → HTTP 503
     │
     ▼
Router.select_provider(healthy_providers)
 - Round Robin / EWMA / Cost / Consistent Hash
 - Returns primary_provider
     │
     ▼
Build execution_order = [primary] + [failover chain]
     │
     ▼
For each provider in execution_order:
 - circuit_breaker.allow_request()  → HALF_OPEN/OPEN → skip
 - provider.generate_completion()   → timeout/error   → record_failure(), try next
 - SUCCESS:
     └─ circuit_breaker.record_success()
     └─ router.record_feedback(latency, success=True)
     └─ asyncio.create_task(cache.set(...))   ← async, don't wait
     └─ return response

All failed → HTTP 502
     │
     ▼
HTTP Middleware
 - Inject X-Response-Time-Ms header
 - Return to client
```

---

*Document generated from source code analysis. Every algorithm described is directly implemented in the `src/` directory.*
