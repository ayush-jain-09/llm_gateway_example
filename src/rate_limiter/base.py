"""
Abstract Base Class for Rate Limiting Strategies.
Implements the Strategy Pattern (Behavioral).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional


class RateLimiterBase(ABC):
    """
    Abstract interface for rate limiter implementations.
    All rate limiters must provide an asynchronous is_allowed method.
    """

    @abstractmethod
    async def is_allowed(self, key: str, cost: int = 1) -> Tuple[bool, float, float]:
        """
        Determines whether an action keyed by `key` is allowed under current limits.

        Args:
            key: Tenant ID, API Key, or Client IP identifying the caller.
            cost: Number of tokens or request units to consume.

        Returns:
            Tuple of:
                - allowed (bool): True if request is allowed, False if throttled (429).
                - remaining_tokens (float): Number of tokens/requests remaining in current budget.
                - retry_after (float): Estimated seconds to wait before enough tokens refill (0.0 if allowed).
        """
        pass

    @abstractmethod
    async def reset(self, key: Optional[str] = None) -> None:
        """Reset limits for a specific key or all keys (used in testing & cleanup)."""
        pass
