"""
In-Memory Thread-Safe Token Bucket Rate Limiter.

Algorithm:
- Bucket holds up to `capacity` tokens.
- Tokens refill continuously at `refill_rate` tokens per second.
- Lazy Refill: Evaluates token delta dynamically upon request based on time.monotonic().
- Time Complexity: O(1)
- Space Complexity: O(K) where K is number of active unique clients/tenants.
"""

import time
import asyncio
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from src.rate_limiter.base import RateLimiterBase


@dataclass
class _BucketState:
    tokens: float
    last_updated: float


class TokenBucketRateLimiter(RateLimiterBase):
    """
    High-performance in-memory Token Bucket rate limiter.
    Uses asyncio.Lock to guarantee thread safety during token updates across concurrent async tasks.
    """

    def __init__(self, capacity: int = 10, refill_rate: float = 2.0):
        """
        Args:
            capacity: Maximum burst capacity of the bucket.
            refill_rate: Tokens added per second.
        """
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._buckets: Dict[str, _BucketState] = {}
        self._lock = asyncio.Lock()

    def _refill(self, bucket: _BucketState, now: float) -> None:
        """Calculates tokens accumulated since last refill without running background tick threads."""
        elapsed = now - bucket.last_updated
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            bucket.tokens = min(self.capacity, bucket.tokens + new_tokens)
            bucket.last_updated = now

    async def is_allowed(self, key: str, cost: int = 1) -> Tuple[bool, float, float]:
        now = time.monotonic()
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _BucketState(tokens=self.capacity, last_updated=now)

            bucket = self._buckets[key]
            self._refill(bucket, now)

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, bucket.tokens, 0.0
            else:
                deficit = cost - bucket.tokens
                retry_after = deficit / self.refill_rate if self.refill_rate > 0 else 1.0
                return False, bucket.tokens, retry_after

    async def reset(self, key: Optional[str] = None) -> None:
        async with self._lock:
            if key is not None:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()
