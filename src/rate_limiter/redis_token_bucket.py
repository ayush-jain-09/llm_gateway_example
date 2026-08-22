"""
Distributed Redis Token Bucket Rate Limiter using Atomic Lua Scripting.

Architecture:
- State is held in Redis, enabling multi-instance gateway clusters.
- Atomic Lua execution guarantees zero race conditions without distributed lock contention.
- Graceful Degradation: If Redis is unavailable, falls back to a local in-memory Token Bucket.
"""

import time
import logging
from typing import Tuple, Optional
import redis.asyncio as aioredis
from src.rate_limiter.base import RateLimiterBase
from src.rate_limiter.token_bucket import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

# Atomic Redis Lua script for Token Bucket
LUA_TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local state = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(state[1])
local last_updated = tonumber(state[2])

if not tokens or not last_updated then
    tokens = capacity
    last_updated = now
else
    local delta = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + (delta * refill_rate))
    last_updated = now
end

if tokens >= cost then
    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, 3600)
    return {1, string.format("%.4f", tokens), "0.0"}
else
    local deficit = cost - tokens
    local retry_after = deficit / refill_rate
    return {0, string.format("%.4f", tokens), string.format("%.4f", retry_after)}
end
"""


class RedisTokenBucketRateLimiter(RateLimiterBase):
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "llm_gw:rate_limit:",
        capacity: int = 10,
        refill_rate: float = 2.0,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._redis_client: Optional[aioredis.Redis] = None
        self._script_sha: Optional[str] = None
        self._fallback_limiter = TokenBucketRateLimiter(capacity, refill_rate)

    async def _get_client(self) -> aioredis.Redis:
        if self._redis_client is None:
            self._redis_client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
        return self._redis_client

    async def is_allowed(self, key: str, cost: int = 1) -> Tuple[bool, float, float]:
        full_key = f"{self.key_prefix}{key}"
        now = time.time()
        try:
            client = await self._get_client()
            result = await client.eval(
                LUA_TOKEN_BUCKET_SCRIPT,
                1,
                full_key,
                str(self.capacity),
                str(self.refill_rate),
                str(cost),
                str(now),
            )
            allowed = bool(result[0] == 1)
            remaining = float(result[1])
            retry_after = float(result[2])
            return allowed, remaining, retry_after
        except Exception as e:
            logger.warning(
                f"Redis rate limiter unavailable ({type(e).__name__}: {e}). Falling back to in-memory Token Bucket."
            )
            return await self._fallback_limiter.is_allowed(key, cost)

    async def reset(self, key: Optional[str] = None) -> None:
        try:
            client = await self._get_client()
            if key is not None:
                await client.delete(f"{self.key_prefix}{key}")
            else:
                keys = await client.keys(f"{self.key_prefix}*")
                if keys:
                    await client.delete(*keys)
        except Exception as e:
            logger.warning(f"Error resetting Redis keys: {e}")
        await self._fallback_limiter.reset(key)

    async def close(self) -> None:
        if self._redis_client is not None:
            await self._redis_client.aclose()
            self._redis_client = None
