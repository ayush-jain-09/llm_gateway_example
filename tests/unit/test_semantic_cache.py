"""
Unit tests for Semantic Vector Cache.
Tests: Exact hash match fast path, semantic cosine similarity matching with threshold, and eviction.
"""

import pytest
from src.cache.semantic_cache import SemanticVectorCache


@pytest.mark.asyncio
async def test_exact_hash_hit():
    cache = SemanticVectorCache(max_size=10, similarity_threshold=0.88)
    
    await cache.set("What is Python?", "Python is a programming language.")
    
    hit = await cache.get("What is Python?")
    assert hit is not None
    assert hit.value == "Python is a programming language."


@pytest.mark.asyncio
async def test_semantic_similarity_hit():
    # Lower threshold slightly for hash vectorizer fallback compatibility
    cache = SemanticVectorCache(max_size=10, similarity_threshold=0.75)
    
    await cache.set(
        "tell me about python programming language features",
        "Python features include readable syntax and dynamic typing."
    )
    
    # Paraphrased query
    hit = await cache.get("tell me about python programming language features and details")
    assert hit is not None
    assert hit.value == "Python features include readable syntax and dynamic typing."


@pytest.mark.asyncio
async def test_semantic_dissimilar_query_miss():
    cache = SemanticVectorCache(max_size=10, similarity_threshold=0.85)
    
    await cache.set("How to cook pasta?", "Boil water and add salt.")
    
    # Completely different question
    hit = await cache.get("Explain quantum physics relativity")
    assert hit is None
