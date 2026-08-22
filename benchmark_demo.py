"""
Interactive Demonstration & Benchmarking Script for LLM Gateway.
Shows live exact/semantic caching, rate limiting, and circuit breaker failover.
"""

import sys
import io
import asyncio
import time

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.gateway.models import ChatRequest, ChatMessage
from src.gateway.config import AppConfig
from src.gateway.pipeline import GatewayPipeline


async def main():
    print("=" * 80)
    print("[*] LLM GATEWAY LIVE DEMO & BENCHMARK")
    print("=" * 80)

    # 1. Load config and initialize pipeline
    config = AppConfig.load_from_yaml()
    pipeline = GatewayPipeline(config)
    print(f"[+] Pipeline Initialized:")
    print(f"   - Rate Limiter : {config.rate_limiter.strategy} (capacity={config.rate_limiter.default_capacity})")
    print(f"   - Cache Engine : {config.cache.strategy} (threshold={config.cache.similarity_threshold})")
    print(f"   - Router       : {config.router.strategy}")
    print(f"   - Providers    : {list(pipeline.providers.keys())}\n")

    # -------------------------------------------------------------
    # DEMO 1: Cache Miss vs Exact Cache Hit
    # -------------------------------------------------------------
    print("--- [TEST 1: Exact Match Caching] ---")
    req1 = ChatRequest(
        messages=[ChatMessage(role="user", content="Explain consistent hashing ring algorithms")],
        tenant_id="tenant_alpha",
    )
    
    t0 = time.monotonic()
    resp1 = await pipeline.process_chat_completion(req1)
    d1 = (time.monotonic() - t0) * 1000.0
    print(f"Request 1 (Cold / Miss) -> Latency: {d1:6.2f}ms | Provider: {resp1.provider_id} | CacheHit: {resp1.cache_hit}")

    # Allow async cache storage task to complete
    await asyncio.sleep(0.05)

    t0 = time.monotonic()
    resp2 = await pipeline.process_chat_completion(req1)
    d2 = (time.monotonic() - t0) * 1000.0
    print(f"Request 2 (Warm / Hit)  -> Latency: {d2:6.2f}ms | Provider: {resp2.provider_id} | CacheHit: {resp2.cache_hit} ({resp2.cache_type})")
    print(f"[*] Latency reduction: {((d1 - d2) / d1) * 100:.1f}%\n")

    # -------------------------------------------------------------
    # DEMO 2: Semantic Caching on Paraphrased Prompt
    # -------------------------------------------------------------
    print("--- [TEST 2: Semantic Vector Caching (Paraphrased)] ---")
    req_paraphrase = ChatRequest(
        messages=[ChatMessage(role="user", content="Explain consistent hashing ring algorithms and virtual nodes")],
        tenant_id="tenant_alpha",
    )
    t0 = time.monotonic()
    resp_para = await pipeline.process_chat_completion(req_paraphrase)
    d_para = (time.monotonic() - t0) * 1000.0
    print(f"Request 3 (Paraphrased) -> Latency: {d_para:6.2f}ms | CacheHit: {resp_para.cache_hit} ({resp_para.cache_type})")
    print(f"[*] Saved LLM API Cost: ${resp_para.usage.cost_usd:.6f}\n")

    # -------------------------------------------------------------
    # DEMO 3: Burst Rate Limiter Throttling
    # -------------------------------------------------------------
    print("--- [TEST 3: Token Bucket Rate Limiting & 429 Throttling] ---")
    burst_tenant = "burst_tenant_99"
    burst_req = ChatRequest(
        messages=[ChatMessage(role="user", content="Rapid burst check")],
        tenant_id=burst_tenant,
        cache_bypass=True,
    )
    
    success_count = 0
    throttled_count = 0
    for i in range(15):
        try:
            await pipeline.process_chat_completion(burst_req)
            success_count += 1
        except Exception:
            throttled_count += 1

    print(f"Fired 15 rapid requests (Limit capacity = {config.rate_limiter.default_capacity}):")
    print(f"   -> {success_count} Allowed (200 OK)")
    print(f"   -> {throttled_count} Throttled (429 Too Many Requests)\n")

    # -------------------------------------------------------------
    # DEMO 4: Circuit Breaker & Automatic Failover
    # -------------------------------------------------------------
    print("--- [TEST 4: Provider Outage & Circuit Breaker Automatic Failover] ---")
    cb = pipeline.circuit_breakers.get("mock-fast")
    if cb:
        for _ in range(25):
            await cb.record_failure()
        print(f"Simulated outage on 'mock-fast': State is now {cb.state.value}")

    req_failover = ChatRequest(
        messages=[ChatMessage(role="user", content="Failover routing test")],
        cache_bypass=True,
    )
    resp_failover = await pipeline.process_chat_completion(req_failover)
    print(f"Gateway automatically rerouted traffic to healthy provider: '{resp_failover.provider_id}'")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
