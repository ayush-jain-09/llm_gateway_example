"""
Factory for Cache engine instantiation.
Implements the Factory Pattern (Creational).
"""

from src.cache.base import CacheBase
from src.cache.lru_cache import LRUCache
from src.cache.lfu_cache import LFUCache
from src.cache.semantic_cache import SemanticVectorCache
from src.gateway.config import CacheConfig


class _NoOpCache(CacheBase):
    """Null Object pattern for disabled cache strategy."""

    async def get(self, prompt: str, **kwargs):
        return None

    async def set(self, prompt: str, value, ttl_seconds=None, **kwargs):
        pass

    async def clear(self):
        pass

    def size(self) -> int:
        return 0


class CacheFactory:
    """Creates cache engine instances based on configuration."""

    @staticmethod
    def create(config: CacheConfig) -> CacheBase:
        strategy = config.strategy.lower()

        if strategy == "disabled":
            return _NoOpCache()
        elif strategy == "lru_exact":
            return LRUCache(
                max_size=config.max_size,
                default_ttl=config.ttl_seconds,
            )
        elif strategy == "lfu_exact":
            return LFUCache(
                max_size=config.max_size,
                default_ttl=config.ttl_seconds,
            )
        elif strategy in ("semantic_vector", "semantic_vector_cosine"):
            return SemanticVectorCache(
                max_size=config.max_size,
                default_ttl=config.ttl_seconds,
                similarity_threshold=config.similarity_threshold,
                embedding_model=config.embedding_model,
            )
        else:
            raise ValueError(f"Unsupported cache strategy: '{strategy}'")
