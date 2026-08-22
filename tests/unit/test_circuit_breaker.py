"""
Unit tests for Circuit Breaker state machine.
Tests: CLOSED -> OPEN transition, fast-fail rejection, recovery cooldown into HALF_OPEN, and probe success recovery.
"""

import pytest
import asyncio
from src.circuit_breaker.breaker import CircuitBreaker
from src.circuit_breaker.states import CircuitState, CircuitBreakerOpenException
from src.gateway.config import CircuitBreakerConfig


@pytest.mark.asyncio
async def test_circuit_breaker_trips_to_open_on_failures():
    config = CircuitBreakerConfig(
        strategy="consecutive_failure",
        recovery_time_seconds=1,
    )
    cb = CircuitBreaker("test-provider", config)
    assert cb.state == CircuitState.CLOSED

    # Record 5 consecutive failures
    for _ in range(5):
        await cb.record_failure()

    # Breaker should now be OPEN
    assert cb.state == CircuitState.OPEN

    # Subsequent request should raise CircuitBreakerOpenException
    with pytest.raises(CircuitBreakerOpenException):
        await cb.allow_request()


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    config = CircuitBreakerConfig(
        strategy="consecutive_failure",
        recovery_time_seconds=0.3,
        half_open_success_threshold=2,
    )
    cb = CircuitBreaker("test-provider-2", config)
    
    # Trip breaker to OPEN
    for _ in range(5):
        await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Wait for recovery cooldown
    await asyncio.sleep(0.35)

    # First request after cooldown should transition to HALF_OPEN
    allowed = await cb.allow_request()
    assert allowed is True
    assert cb.state == CircuitState.HALF_OPEN

    # Record 2 successful probe responses
    await cb.record_success()
    await cb.record_success()

    # Circuit should be CLOSED again
    assert cb.state == CircuitState.CLOSED
