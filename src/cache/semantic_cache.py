"""
Two-Tier Semantic Vector Cache with Cosine Similarity and LRU Eviction.

Architecture:
- Tier 1: Exact SHA-256 Hash Match (O(1) fast path, 0ms embedding overhead).
- Tier 2: Semantic Vector Similarity Search (Cosine similarity >= threshold).
- Eviction: LRU (Least Recently Used) ordering maintained across both exact and vector stores.
"""

import time
import hashlib
import asyncio
from typing import Optional, Any, Dict, List, Tuple
import numpy as np

from src.cache.base import CacheBase, CachedItem
from src.cache.embeddings import EmbeddingEngine


class _VectorEntry:
    def __init__(
        self,
        key: str,
        prompt: str,
        embedding: np.ndarray,
        value: Any,
        created_at: float,
        ttl_seconds: int,
    ):
        self.key = key
        self.prompt = prompt
        self.embedding = embedding
        self.value = value
        self.created_at = created_at
        self.ttl_seconds = ttl_seconds
        self.access_count = 1
        self.last_accessed = created_at

    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


class SemanticVectorCache(CacheBase):
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        similarity_threshold: float = 0.88,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.max_size = max(1, max_size)
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self._embedder = EmbeddingEngine(embedding_model)

        # Exact match index (hash -> _VectorEntry)
        self._exact_map: Dict[str, _VectorEntry] = {}
        # Linear vector store list
        self._entries: List[_VectorEntry] = []
        self._lock = asyncio.Lock()

    def _hash_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

    async def get(self, prompt: str, **kwargs) -> Optional[CachedItem]:
        exact_key = self._hash_key(prompt)
        now = time.time()

        async with self._lock:
            # 1. Tier 1: Exact Match Fast Path
            if exact_key in self._exact_map:
                entry = self._exact_map[exact_key]
                if not entry.is_expired():
                    entry.access_count += 1
                    entry.last_accessed = now
                    return CachedItem(
                        key=entry.key,
                        value=entry.value,
                        created_at=entry.created_at,
                        ttl_seconds=entry.ttl_seconds,
                        prompt_text=entry.prompt,
                        access_count=entry.access_count,
                        last_accessed=entry.last_accessed,
                    )
                else:
                    self._remove_entry(entry)

        # 2. Tier 2: Vector Embedding & Semantic Similarity Search
        query_vec = await self._embedder.embed(prompt)

        async with self._lock:
            best_match: Optional[_VectorEntry] = None
            highest_sim = -1.0

            # Scan stored entries (filtering out expired ones)
            i = 0
            while i < len(self._entries):
                entry = self._entries[i]
                if entry.is_expired():
                    self._remove_entry(entry)
                    continue

                sim = self._embedder.cosine_similarity(query_vec, entry.embedding)
                if sim > highest_sim:
                    highest_sim = sim
                    best_match = entry
                i += 1

            if best_match and highest_sim >= self.similarity_threshold:
                best_match.access_count += 1
                best_match.last_accessed = now
                return CachedItem(
                    key=best_match.key,
                    value=best_match.value,
                    created_at=best_match.created_at,
                    ttl_seconds=best_match.ttl_seconds,
                    prompt_text=best_match.prompt,
                    access_count=best_match.access_count,
                    last_accessed=best_match.last_accessed,
                )

        return None

    def _remove_entry(self, entry: _VectorEntry) -> None:
        if entry.key in self._exact_map:
            del self._exact_map[entry.key]
        if entry in self._entries:
            self._entries.remove(entry)

    def _evict_lru(self) -> None:
        """Evicts the entry with the oldest last_accessed timestamp."""
        if not self._entries:
            return
        oldest_entry = min(self._entries, key=lambda e: e.last_accessed)
        self._remove_entry(oldest_entry)

    async def set(
        self, prompt: str, value: Any, ttl_seconds: Optional[int] = None, **kwargs
    ) -> None:
        key = self._hash_key(prompt)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()

        # Compute embedding
        vec = await self._embedder.embed(prompt)

        async with self._lock:
            # Check if key exists already
            if key in self._exact_map:
                existing = self._exact_map[key]
                existing.value = value
                existing.embedding = vec
                existing.last_accessed = now
                return

            # Check capacity
            if len(self._entries) >= self.max_size:
                self._evict_lru()

            entry = _VectorEntry(
                key=key,
                prompt=prompt,
                embedding=vec,
                value=value,
                created_at=now,
                ttl_seconds=ttl,
            )
            self._exact_map[key] = entry
            self._entries.append(entry)

    async def clear(self) -> None:
        async with self._lock:
            self._exact_map.clear()
            self._entries.clear()

    def size(self) -> int:
        return len(self._entries)
