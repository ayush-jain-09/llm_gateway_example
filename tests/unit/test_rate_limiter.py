"""
Unit tests for Rate Limiter algorithms.
Tests: Token Bucket (burst & refill), Sliding Window Log, Sliding Window Counter, and Multi-tenant isolation.
"""

import pytest
import asyncio
import time
from src.rate_limiter.token_bucket import TokenBucketRateLimiter
from src.rate_limiter.sliding_window_log import SlidingWindowLogRateLimiter
from src.rate_limiter.sliding_window_counter import SlidingWindowCounterRateLimiter


@pytest.mark.asyncio
async def test_token_bucket_burst_and_exhaustion():
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1.0)
    
    # 1. First 3 requests must succeed (burst capacity)
    for _ in range(3):
        allowed, remaining, retry_after = await limiter.is_allowed("tenant_A", cost=1)
        assert allowed is True
        assert retry_after == 0.0

    # 2. 4th request must be throttled
    allowed, remaining, retry_after = await limiter.is_allowed("tenant_A", cost=1)
    assert allowed is False
    assert retry_after > 0.0


@pytest.mark.asyncio
async def test_token_bucket_refill():
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate=10.0)  # 10 tokens/sec
    
    # Exhaust tokens
    await limiter.is_allowed("tenant_B", cost=2)
    allowed, _, _ = await limiter.is_allowed("tenant_B", cost=1)
    assert allowed is False

    # Wait 0.2s (refills ~2 tokens)
    await asyncio.sleep(0.25)
    allowed, remaining, _ = await limiter.is_allowed("tenant_B", cost=1)
    assert allowed is True


@pytest.mark.asyncio
async def test_token_bucket_tenant_isolation():
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1.0)
    
    # Exhaust Tenant 1
    await limiter.is_allowed("user_1", cost=2)
    allowed1, _, _ = await limiter.is_allowed("user_1", cost=1)
    assert allowed1 is False

    # Tenant 2 should still have full capacity
    allowed2, remaining2, _ = await limiter.is_allowed("user_2", cost=1)
    assert allowed2 is True
    assert remaining2 == 1.0


@pytest.mark.asyncio
async def test_sliding_window_log_precision():
    limiter = SlidingWindowLogRateLimiter(max_requests=2, window_seconds=1)
    
    # Allow 2 requests
    assert (await limiter.is_allowed("client_x", cost=1))[0] is True
    assert (await limiter.is_allowed("client_x", cost=1))[0] is True
    
    # Block 3rd
    assert (await limiter.is_allowed("client_x", cost=1))[0] is False
    
    # After 1.1s window expires, should allow again
    await asyncio.sleep(1.1)
    assert (await limiter.is_allowed("client_x", cost=1))[0] is True


@pytest.mark.asyncio
async def test_sliding_window_counter():
    limiter = SlidingWindowCounterRateLimiter(max_requests=3, window_seconds=1)
    
    assert (await limiter.is_allowed("client_y", cost=1))[0] is True
    assert (await limiter.is_allowed("client_y", cost=1))[0] is True
    assert (await limiter.is_allowed("client_y", cost=1))[0] is True
    assert (await limiter.is_allowed("client_y", cost=1))[0] is False
