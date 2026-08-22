"""
Circuit Breaker States and Exceptions.
Implements the State Pattern (Behavioral).
"""

from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests pass through
    OPEN = "OPEN"            # Tripped: fast-fail immediately to protect downstream
    HALF_OPEN = "HALF_OPEN"  # Testing: trial probes allowed to assess recovery


class CircuitBreakerOpenException(Exception):
    """Raised when a request is blocked because the provider's circuit breaker is in OPEN state."""

    def __init__(self, provider_id: str, retry_after: float):
        super().__init__(
            f"Circuit breaker for provider '{provider_id}' is OPEN. Retry after {retry_after:.1f}s."
        )
        self.provider_id = provider_id
        self.retry_after = retry_after
