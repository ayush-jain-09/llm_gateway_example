"""
Prometheus Metrics Registry and Exporter for LLM Gateway Telemetry.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# 1. Request Counters
GATEWAY_REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total incoming chat completion requests",
    ["status", "tenant_id"],
)

# 2. Rate Limiting Counter
RATE_LIMITED_REQUESTS_TOTAL = Counter(
    "gateway_rate_limited_total",
    "Total requests throttled with HTTP 429",
    ["tenant_id", "strategy"],
)

# 3. Cache Metrics
CACHE_HITS_TOTAL = Counter(
    "gateway_cache_hits_total",
    "Total requests served directly from cache",
    ["cache_type"],  # EXACT or SEMANTIC
)

CACHE_MISSES_TOTAL = Counter(
    "gateway_cache_misses_total",
    "Total requests that missed cache and hit LLM provider",
)

CACHE_ESTIMATED_DOLLARS_SAVED = Counter(
    "gateway_cache_dollars_saved_total",
    "Total estimated USD saved via cache hits",
)

# 4. Latency Histograms
GATEWAY_REQUEST_DURATION_SECONDS = Histogram(
    "gateway_request_duration_seconds",
    "Total end-to-end gateway latency in seconds",
    ["provider_id", "cache_hit"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# 5. Circuit Breaker Metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Current state of provider circuit breaker (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["provider_id"],
)

CIRCUIT_BREAKER_FAILOVERS_TOTAL = Counter(
    "gateway_circuit_breaker_failovers_total",
    "Total requests rerouted due to tripped circuit breakers",
    ["failed_provider_id", "fallback_provider_id"],
)

# 6. Financial Token Counters
TOKENS_PROCESSED_TOTAL = Counter(
    "gateway_tokens_processed_total",
    "Total input and output tokens billed",
    ["provider_id", "token_type"],  # input or output
)


def get_metrics_output() -> tuple[bytes, str]:
    """Generates prometheus scrape output."""
    return generate_latest(), CONTENT_TYPE_LATEST
