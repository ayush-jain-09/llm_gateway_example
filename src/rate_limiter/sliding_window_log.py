"""
Sliding Window Log Rate Limiter.

Algorithm:
- Maintains a sorted double-ended queue (deque) of timestamps for each request within the window.
- On each request, discards timestamps older than (current_time - window_size).
- If remaining timestamps + cost <= max_requests, allows request and appends current timestamp.
- Time Complexity: O(log N) or O(M) where M is expired entries purged.
- Space Complexity: O(N) timestamps per client.
- Accuracy: 100% exact (eliminates boundary burst issues of fixed windows).
"""

import time
import asyncio
from collections import deque
from typing import Dict, Deque, Tuple, Optional
from src.rate_limiter.base import RateLimiterBase


class SlidingWindowLogRateLimiter(RateLimiterBase):
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._logs: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str, cost: int = 1) -> Tuple[bool, float, float]:
        now = time.time()
        boundary = now - self.window_seconds

        async with self._lock:
            if key not in self._logs:
                self._logs[key] = deque()

            log = self._logs[key]

            # Purge expired timestamps outside the rolling window
            while log and log[0] <= boundary:
                log.popleft()

            current_count = len(log)
            if current_count + cost <= self.max_requests:
                for _ in range(cost):
                    log.append(now)
                remaining = self.max_requests - len(log)
                return True, float(remaining), 0.0
            else:
                # Oldest timestamp determines when the next slot will free up
                oldest = log[0] if log else now
                retry_after = max(0.0, (oldest + self.window_seconds) - now)
                return False, float(self.max_requests - current_count), retry_after

    async def reset(self, key: Optional[str] = None) -> None:
        async with self._lock:
            if key is not None:
                self._logs.pop(key, None)
            else:
                self._logs.clear()
