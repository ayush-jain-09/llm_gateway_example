"""
Handcrafted In-Memory O(1) LFU (Least Frequently Used) Cache.

Data Structure:
- `key_to_node`: Map of key -> _LFUNode
- `freq_to_dll`: Map of frequency -> DoublyLinkedList containing nodes with that exact frequency
- `min_freq`: Tracks the current minimum frequency for O(1) eviction
- Get: O(1)
- Put: O(1)
- Evict: O(1)
"""

import time
import hashlib
import asyncio
from typing import Optional, Any, Dict
from src.cache.base import CacheBase, CachedItem


class _LFUNode:
    def __init__(self, key: str, item: CachedItem):
        self.key = key
        self.item = item
        self.freq: int = 1
        self.prev: Optional["_LFUNode"] = None
        self.next: Optional["_LFUNode"] = None


class _DoublyLinkedList:
    def __init__(self):
        self.head = _LFUNode("__HEAD__", None)  # type: ignore
        self.tail = _LFUNode("__TAIL__", None)  # type: ignore
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_head(self, node: _LFUNode) -> None:
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def remove_node(self, node: _LFUNode) -> None:
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node
        self.size -= 1

    def pop_tail(self) -> Optional[_LFUNode]:
        if self.size == 0:
            return None
        node = self.tail.prev
        if node and node != self.head:
            self.remove_node(node)
            return node
        return None


class LFUCache(CacheBase):
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max(1, max_size)
        self.default_ttl = default_ttl
        self._key_to_node: Dict[str, _LFUNode] = {}
        self._freq_to_dll: Dict[int, _DoublyLinkedList] = {}
        self._min_freq: int = 1
        self._lock = asyncio.Lock()

    def _hash_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

    def _increment_freq(self, node: _LFUNode) -> None:
        old_freq = node.freq
        dll = self._freq_to_dll[old_freq]
        dll.remove_node(node)

        # Update min_freq if the list at min_freq is now empty
        if self._min_freq == old_freq and dll.size == 0:
            self._min_freq += 1

        node.freq += 1
        new_freq = node.freq
        if new_freq not in self._freq_to_dll:
            self._freq_to_dll[new_freq] = _DoublyLinkedList()
        self._freq_to_dll[new_freq].add_to_head(node)

    async def get(self, prompt: str, **kwargs) -> Optional[CachedItem]:
        key = self._hash_key(prompt)
        async with self._lock:
            if key not in self._key_to_node:
                return None

            node = self._key_to_node[key]
            if node.item.is_expired():
                # Purge expired node
                dll = self._freq_to_dll[node.freq]
                dll.remove_node(node)
                del self._key_to_node[key]
                return None

            node.item.access_count += 1
            node.item.last_accessed = time.time()
            self._increment_freq(node)
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
            if key in self._key_to_node:
                node = self._key_to_node[key]
                node.item = item
                self._increment_freq(node)
            else:
                if len(self._key_to_node) >= self.max_size:
                    min_dll = self._freq_to_dll.get(self._min_freq)
                    if min_dll:
                        evicted = min_dll.pop_tail()
                        if evicted and evicted.key in self._key_to_node:
                            del self._key_to_node[evicted.key]

                new_node = _LFUNode(key, item)
                self._key_to_node[key] = new_node
                self._min_freq = 1
                if 1 not in self._freq_to_dll:
                    self._freq_to_dll[1] = _DoublyLinkedList()
                self._freq_to_dll[1].add_to_head(new_node)

    async def clear(self) -> None:
        async with self._lock:
            self._key_to_node.clear()
            self._freq_to_dll.clear()
            self._min_freq = 1

    def size(self) -> int:
        return len(self._key_to_node)
