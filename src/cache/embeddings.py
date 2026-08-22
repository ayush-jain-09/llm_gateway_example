"""
Vector Embedding Engine for Semantic Caching.

Architecture:
- Asynchronous non-blocking embedding generation via `asyncio.to_thread`.
- Primary backend: `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, 384 dimensions).
- Fallback backend: Lightweight Subword N-Gram Hash Vectorizer with L2-normalization for zero-download testing & offline execution.
"""

import math
import hashlib
import asyncio
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._dimension = 384

    def _init_model(self) -> None:
        if self._initialized:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer loaded successfully (dim={self._dimension}).")
        except Exception as e:
            logger.info(
                f"SentenceTransformer not loaded ({e}). Using deterministic Subword Hash Vectorizer (dim=384)."
            )
            self._model = None
        self._initialized = True

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
        "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    }

    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Deterministic semantic subword n-gram vectorizer with stop-word suppression
        and L2-normalization.
        """
        import re
        vec = np.zeros(self._dimension, dtype=np.float32)
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [w for w in clean_text.split() if w not in self.STOP_WORDS and len(w) > 1]
        
        if not tokens:
            tokens = clean_text.split()

        for token in tokens:
            # Word unigram hash (high weight)
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dimension
            vec[idx] += 2.0

            # Subword root prefix / character trigrams
            for i in range(len(token) - 2):
                tri = token[i : i + 3]
                idx_tri = int(hashlib.md5(tri.encode()).hexdigest(), 16) % self._dimension
                vec[idx_tri] += 0.8

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _sync_embed(self, text: str) -> np.ndarray:
        self._init_model()
        if self._model is not None:
            raw_vec = self._model.encode(text, normalize_embeddings=True)
            return np.array(raw_vec, dtype=np.float32)
        else:
            return self._fallback_embed(text)

    async def embed(self, text: str) -> np.ndarray:
        """Non-blocking async embedding inference offloaded to worker thread pool."""
        return await asyncio.to_thread(self._sync_embed, text)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine similarity between two L2-normalized vectors via dot product.
        Time Complexity: O(d) using vectorized SIMD.
        """
        dot = float(np.dot(vec1, vec2))
        return max(-1.0, min(1.0, dot))
