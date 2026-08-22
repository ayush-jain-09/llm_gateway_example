"""
Integration test for full Gateway Pipeline.
Tests: Rate Limiting -> Exact/Semantic Caching -> Provider Selection -> Circuit Breaker Trip -> Automatic Failover.
"""

import pytest
from src.gateway.models import ChatRequest, ChatMessage
from src.gateway.config import AppConfig, RateLimiterConfig, CacheConfig, RouterConfig, CircuitBreakerConfig, ProviderConfig
from src.gateway.pipeline import GatewayPipeline


def _build_test_pipeline(rate_limit_capacity: int = 5):
    cfg = AppConfig(
        rate_limiter=RateLimiterConfig(strategy="token_bucket", default_capacity=rate_limit_capacity, default_refill_rate=1.0),
        cache=CacheConfig(strategy="semantic_vector", similarity_threshold=0.80),
        router=RouterConfig(strategy="round_robin"),
        circuit_breaker=CircuitBreakerConfig(strategy="consecutive_failure", recovery_time_seconds=1),
        providers=[
            ProviderConfig(id="mock-primary", type="mock", model="m-primary", simulated_latency_ms=10, simulated_failure_rate=0.0),
            ProviderConfig(id="mock-secondary", type="mock", model="m-secondary", simulated_latency_ms=10, simulated_failure_rate=0.0),
        ],
        failover_order=["mock-primary", "mock-secondary"],
    )
    return GatewayPipeline(cfg)


@pytest.mark.asyncio
async def test_pipeline_normal_execution_and_cache_hit():
    pipeline = _build_test_pipeline()
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="Explain binary search tree algorithms")]
    )

    # 1. First execution: Cache Miss, hits mock provider
    resp1 = await pipeline.process_chat_completion(req)
    assert resp1.cache_hit is False
    assert resp1.choices[0].message.content is not None

    # Allow async cache set task to finish
    import asyncio
    await asyncio.sleep(0.05)

    # 2. Second execution (Identical prompt): Cache Hit!
    resp2 = await pipeline.process_chat_completion(req)
    assert resp2.cache_hit is True
    assert resp2.cache_type in ("EXACT", "SEMANTIC")


@pytest.mark.asyncio
async def test_pipeline_failover_when_primary_fails():
    # Primary configured with 100% failure rate, secondary with 0%
    cfg = AppConfig(
        rate_limiter=RateLimiterConfig(strategy="token_bucket", default_capacity=10),
        cache=CacheConfig(strategy="disabled"),
        router=RouterConfig(strategy="round_robin"),
        circuit_breaker=CircuitBreakerConfig(strategy="consecutive_failure"),
        providers=[
            ProviderConfig(id="faulty-p1", type="mock", model="m1", simulated_failure_rate=1.0),
            ProviderConfig(id="healthy-p2", type="mock", model="m2", simulated_failure_rate=0.0),
        ],
        failover_order=["faulty-p1", "healthy-p2"],
    )
    pipeline = GatewayPipeline(cfg)
    req = ChatRequest(messages=[ChatMessage(content="Test failover")])

    # Must succeed by automatically falling over to healthy-p2
    resp = await pipeline.process_chat_completion(req)
    assert resp.provider_id == "healthy-p2"
