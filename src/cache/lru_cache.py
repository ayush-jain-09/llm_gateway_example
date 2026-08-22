"""
Handcrafted In-Memory O(1) LRU (Least Recently Used) Cache.

Data Structure:
- Doubly-Linked List (Node with prev and next pointers) + Hash Map (key -> Node).
- Dummy Head and Tail sentinels to eliminate null-checks on edge node insertions/removals.
- Get: O(1)
- Put: O(1)
- Eviction: O(1)
"""

import time
import hashlib
import asyncio
from typing import Optional, Any, Dict
from src.cache.base import CacheBase, CachedItem


class _Node:
    def __init__(self, key: str, item: CachedItem):
        self.key = key
        self.item = item
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class LRUCache(CacheBase):
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max(1, max_size)
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._lock = asyncio.Lock()

        # Sentinel Head & Tail nodes
        self._head = _Node("__HEAD__", None)  # type: ignore
        self._tail = _Node("__TAIL__", None)  # type: ignore
        self._head.next = self._tail
        self._tail.prev = self._head

    def _hash_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from its current position in the Doubly Linked List."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after Head sentinel (marking it most recently used)."""
        node.next = self._head.next
        node.prev = self._head
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Evict the node right before Tail sentinel (least recently used)."""
        lru_node = self._tail.prev
        if lru_node and lru_node != self._head:
            self._remove_node(lru_node)
            return lru_node
        return None

    async def get(self, prompt: str, **kwargs) -> Optional[CachedItem]:
        key = self._hash_key(prompt)
        async with self._lock:
            if key not in self._map:
                return None

            node = self._map[key]
            # Check TTL
            if node.item.is_expired():
                self._remove_node(node)
                del self._map[key]
                return None

            node.item.access_count += 1
            node.item.last_accessed = time.time()
            self._move_to_head(node)
            return node.item

    async def set(
        self, prompt: str, value: Any, ttl_seconds: Optional[int] = None, **kwargs
    ) -> None:
        key = self._hash_key(prompt)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()

        item = CachedItem(
            key=key,
            value=value,
            created_at=now,
            ttl_seconds=ttl,
            prompt_text=prompt,
            last_accessed=now,
        )

        async with self._lock:
            if key in self._map:
                node = self._map[key]
                node.item = item
                self._move_to_head(node)
            else:
                new_node = _Node(key, item)
                self._map[key] = new_node
                self._add_to_head(new_node)

                if len(self._map) > self.max_size:
                    evicted = self._pop_tail()
                    if evicted and evicted.key in self._map:
                        del self._map[evicted.key]

    async def clear(self) -> None:
        async with self._lock:
            self._map.clear()
            self._head.next = self._tail
            self._tail.prev = self._head

    def size(self) -> int:
        return len(self._map)
