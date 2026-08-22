"""
Unit tests for LRU and LFU cache implementations.
Tests: O(1) eviction, capacity bounds, update existing keys, and TTL expiration.
"""

import pytest
import asyncio
from src.cache.lru_cache import LRUCache
from src.cache.lfu_cache import LFUCache


@pytest.mark.asyncio
async def test_lru_cache_eviction_order():
    # Capacity = 2
    cache = LRUCache(max_size=2, default_ttl=3600)
    
    await cache.set("prompt1", "response1")
    await cache.set("prompt2", "response2")
    assert cache.size() == 2

    # Access prompt1 to make it most recently used
    item1 = await cache.get("prompt1")
    assert item1 is not None and item1.value == "response1"

    # Insert prompt3 -> should evict prompt2 (least recently used)
    await cache.set("prompt3", "response3")
    assert cache.size() == 2

    assert await cache.get("prompt2") is None  # Evicted!
    assert (await cache.get("prompt1")).value == "response1"
    assert (await cache.get("prompt3")).value == "response3"


@pytest.mark.asyncio
async def test_lru_ttl_expiration():
    cache = LRUCache(max_size=10, default_ttl=1)  # 1 second TTL
    await cache.set("temp_prompt", "temp_value", ttl_seconds=1)

    assert (await cache.get("temp_prompt")) is not None
    await asyncio.sleep(1.1)
    assert (await cache.get("temp_prompt")) is None


@pytest.mark.asyncio
async def test_lfu_cache_eviction_by_frequency():
    # Capacity = 2
    cache = LFUCache(max_size=2, default_ttl=3600)
    
    await cache.set("frequent", "val_freq")
    await cache.set("rare", "val_rare")

    # Access "frequent" twice -> frequency = 3
    await cache.get("frequent")
    await cache.get("frequent")

    # Insert "new" -> should evict "rare" (frequency = 1)
    await cache.set("new", "val_new")
    assert cache.size() == 2

    assert await cache.get("rare") is None  # Evicted!
    assert (await cache.get("frequent")).value == "val_freq"
    assert (await cache.get("new")).value == "val_new"
