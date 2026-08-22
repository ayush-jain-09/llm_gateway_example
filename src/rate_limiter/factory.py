"""
Factory for Rate Limiter instantiation.
Implements the Factory Pattern (Creational).
"""

from typing import Dict, Type
from src.rate_limiter.base import RateLimiterBase
from src.rate_limiter.token_bucket import TokenBucketRateLimiter
from src.rate_limiter.sliding_window_log import SlidingWindowLogRateLimiter
from src.rate_limiter.sliding_window_counter import SlidingWindowCounterRateLimiter
from src.rate_limiter.redis_token_bucket import RedisTokenBucketRateLimiter
from src.gateway.config import RateLimiterConfig


class RateLimiterFactory:
    """Creates rate limiter instances based on configuration."""

    @staticmethod
    def create(config: RateLimiterConfig) -> RateLimiterBase:
        strategy = config.strategy.lower()

        if strategy == "token_bucket":
            return TokenBucketRateLimiter(
                capacity=config.default_capacity,
                refill_rate=config.default_refill_rate,
            )
        elif strategy == "sliding_window_log":
            return SlidingWindowLogRateLimiter(
                max_requests=config.default_capacity,
                window_seconds=config.sliding_window_seconds,
            )
        elif strategy == "sliding_window_counter":
            return SlidingWindowCounterRateLimiter(
                max_requests=config.default_capacity,
                window_seconds=config.sliding_window_seconds,
            )
        elif strategy == "redis_token_bucket":
            return RedisTokenBucketRateLimiter(
                redis_url=config.redis.url,
                key_prefix=config.redis.key_prefix,
                capacity=config.default_capacity,
                refill_rate=config.default_refill_rate,
            )
        else:
            raise ValueError(f"Unsupported rate limiter strategy: '{strategy}'")
