"""
Abstract Base Class and Models for Caching Engines.
Implements the Strategy Pattern (Behavioral).
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from dataclasses import dataclass
import time


@dataclass
class CachedItem:
    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    prompt_text: Optional[str] = None
    embedding: Optional[list] = None
    access_count: int = 1
    last_accessed: float = 0.0

    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


class CacheBase(ABC):
    """Abstract interface for all exact and semantic cache implementations."""

    @abstractmethod
    async def get(self, prompt: str, **kwargs) -> Optional[CachedItem]:
        """Retrieve cached response if hit, else None."""
        pass

    @abstractmethod
    async def set(
        self, prompt: str, value: Any, ttl_seconds: Optional[int] = None, **kwargs
    ) -> None:
        """Store item in cache with eviction if capacity exceeded."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of cached items."""
        pass
