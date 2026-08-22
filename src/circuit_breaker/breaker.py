"""
Production-Grade Concurrency-Safe Circuit Breaker.

Features:
- Dual Strategy: Sliding Window Error Rate % (Default) OR Consecutive Failures.
- Thundering Herd Prevention: Strict concurrency limit on probe requests during HALF_OPEN.
- Monotonic clock for time drift resistance.
- Asynchronous thread safety with asyncio.Lock.
"""

import time
import asyncio
import logging
from collections import deque
from typing import Deque, Tuple, Optional
from src.circuit_breaker.states import CircuitState, CircuitBreakerOpenException
from src.gateway.config import CircuitBreakerConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(
        self,
        provider_id: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.provider_id = provider_id
        cfg = config or CircuitBreakerConfig()
        
        self.strategy = cfg.strategy
        self.failure_rate_threshold = cfg.failure_rate_threshold
        self.window_size_seconds = cfg.window_size_seconds
        self.minimum_requests = cfg.minimum_requests
        self.recovery_time_seconds = cfg.recovery_time_seconds
        self.half_open_success_threshold = cfg.half_open_success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._last_state_change: float = time.monotonic()
        
        # Sliding window timestamps: deque of (timestamp, is_success: bool)
        self._history: Deque[Tuple[float, bool]] = deque()
        
        # Consecutive failure counter
        self._consecutive_failures: int = 0
        
        # Half-open state counters
        self._half_open_success_count: int = 0
        self._half_open_inflight_probes: int = 0
        self._max_half_open_probes: int = 3

        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _purge_expired_history(self, now: float) -> None:
        boundary = now - self.window_size_seconds
        while self._history and self._history[0][0] <= boundary:
            self._history.popleft()

    def _transition_to(self, new_state: CircuitState, now: float) -> None:
        if self._state != new_state:
            logger.warning(
                f"[CircuitBreaker:{self.provider_id}] State changed: {self._state.value} -> {new_state.value}"
            )
            self._state = new_state
            self._last_state_change = now
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_success_count = 0
                self._half_open_inflight_probes = 0
            elif new_state == CircuitState.CLOSED:
                self._consecutive_failures = 0
                self._history.clear()

    async def allow_request(self) -> bool:
        """
        Evaluates whether a request to the provider is allowed.
        Throws CircuitBreakerOpenException if OPEN.
        """
        now = time.monotonic()
        async with self._lock:
            # 1. State: OPEN -> Check if cooldown expired
            if self._state == CircuitState.OPEN:
                elapsed = now - self._last_state_change
                if elapsed >= self.recovery_time_seconds:
                    self._transition_to(CircuitState.HALF_OPEN, now)
                else:
                    retry_after = self.recovery_time_seconds - elapsed
                    raise CircuitBreakerOpenException(self.provider_id, retry_after)

            # 2. State: HALF_OPEN -> Check probe limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_inflight_probes >= self._max_half_open_probes:
                    raise CircuitBreakerOpenException(self.provider_id, 1.0)
                self._half_open_inflight_probes += 1
                return True

            # 3. State: CLOSED -> Allowed
            return True

    async def record_success(self) -> None:
        now = time.monotonic()
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_inflight_probes = max(0, self._half_open_inflight_probes - 1)
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.half_open_success_threshold:
                    self._transition_to(CircuitState.CLOSED, now)
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0
                self._history.append((now, True))
                self._purge_expired_history(now)

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        now = time.monotonic()
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_inflight_probes = max(0, self._half_open_inflight_probes - 1)
                # Any probe failure immediately trips back to OPEN
                self._transition_to(CircuitState.OPEN, now)
                return

            if self._state == CircuitState.CLOSED:
                self._consecutive_failures += 1
                self._history.append((now, False))
                self._purge_expired_history(now)

                if self.strategy == "consecutive_failure":
                    if self._consecutive_failures >= 5:
                        self._transition_to(CircuitState.OPEN, now)
                else:
                    # Sliding window error rate strategy
                    total_reqs = len(self._history)
                    if total_reqs >= self.minimum_requests:
                        failures = sum(1 for _, is_success in self._history if not is_success)
                        error_rate = failures / float(total_reqs)
                        if error_rate >= self.failure_rate_threshold:
                            logger.error(
                                f"[CircuitBreaker:{self.provider_id}] Error rate {error_rate:.1%} exceeds threshold {self.failure_rate_threshold:.1%}. Tripping circuit."
                            )
                            self._transition_to(CircuitState.OPEN, now)
