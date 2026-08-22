"""
Self-contained asynchronous test runner for LLM Gateway test suite.
Executes all unit and integration test cases using Python's standard library.
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from tests.unit.test_rate_limiter import (
    test_token_bucket_burst_and_exhaustion,
    test_token_bucket_refill,
    test_token_bucket_tenant_isolation,
    test_sliding_window_log_precision,
    test_sliding_window_counter,
)
from tests.unit.test_lru_lfu_cache import (
    test_lru_cache_eviction_order,
    test_lru_ttl_expiration,
    test_lfu_cache_eviction_by_frequency,
)
from tests.unit.test_semantic_cache import (
    test_exact_hash_hit,
    test_semantic_similarity_hit,
    test_semantic_dissimilar_query_miss,
)
from tests.unit.test_circuit_breaker import (
    test_circuit_breaker_trips_to_open_on_failures,
    test_circuit_breaker_half_open_recovery,
)
from tests.unit.test_router import (
    test_round_robin_cycling,
    test_cost_aware_routing,
    test_latency_aware_routing,
    test_consistent_hash_stickiness,
)
from tests.integration.test_gateway_pipeline import (
    test_pipeline_normal_execution_and_cache_hit,
    test_pipeline_failover_when_primary_fails,
)


TEST_FUNCTIONS = [
    # Rate Limiter
    ("RateLimiter: Token Bucket Burst & Exhaustion", test_token_bucket_burst_and_exhaustion),
    ("RateLimiter: Token Bucket Refill Math", test_token_bucket_refill),
    ("RateLimiter: Multi-Tenant Isolation", test_token_bucket_tenant_isolation),
    ("RateLimiter: Sliding Window Log Precision", test_sliding_window_log_precision),
    ("RateLimiter: Sliding Window Counter Approximation", test_sliding_window_counter),

    # LRU / LFU Cache
    ("Cache: LRU Eviction Order (DLL + Map)", test_lru_cache_eviction_order),
    ("Cache: LRU TTL Expiration", test_lru_ttl_expiration),
    ("Cache: LFU Frequency Eviction", test_lfu_cache_eviction_by_frequency),

    # Semantic Cache
    ("SemanticCache: Exact SHA-256 Fast Path", test_exact_hash_hit),
    ("SemanticCache: Vector Cosine Similarity Hit", test_semantic_similarity_hit),
    ("SemanticCache: Dissimilar Query Miss", test_semantic_dissimilar_query_miss),

    # Circuit Breaker
    ("CircuitBreaker: CLOSED -> OPEN on Failure Threshold", test_circuit_breaker_trips_to_open_on_failures),
    ("CircuitBreaker: HALF_OPEN Probe Recovery -> CLOSED", test_circuit_breaker_half_open_recovery),

    # Router
    ("Router: Round Robin Cyclic Distribution", test_round_robin_cycling),
    ("Router: Cost Aware Financial Optimization", test_cost_aware_routing),
    ("Router: Latency Aware EWMA Dynamic Steering", test_latency_aware_routing),
    ("Router: Consistent Hashing Ring Stickiness", test_consistent_hash_stickiness),

    # Pipeline Integration
    ("Pipeline: Full Request Lifecycle & Cache Storing", test_pipeline_normal_execution_and_cache_hit),
    ("Pipeline: Circuit Breaker Trip & Cascade Failover", test_pipeline_failover_when_primary_fails),
]


async def run_all_tests():
    passed = 0
    failed = 0
    print("=" * 70)
    print("🚀 RUNNING LLM GATEWAY COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    start_total = time.monotonic()
    for name, test_fn in TEST_FUNCTIONS:
        test_start = time.monotonic()
        try:
            await test_fn()
            duration_ms = (time.monotonic() - test_start) * 1000.0
            print(f"  ✅ PASS [{duration_ms:6.1f}ms] - {name}")
            passed += 1
        except Exception as e:
            duration_ms = (time.monotonic() - test_start) * 1000.0
            print(f"  ❌ FAIL [{duration_ms:6.1f}ms] - {name}")
            print(f"     Error: {type(e).__name__}: {e}")
            failed += 1

    total_duration = (time.monotonic() - start_total) * 1000.0
    print("=" * 70)
    print(f"RESULTS: {passed} Passed, {failed} Failed out of {len(TEST_FUNCTIONS)} tests in {total_duration:.1f}ms")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
