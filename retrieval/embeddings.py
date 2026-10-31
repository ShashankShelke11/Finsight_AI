"""
Dense Embedding Engine for FinSight AI
Generates normalized 384-dimensional vectors for semantic retrieval.
Uses SentenceTransformers ('all-MiniLM-L6-v2') when available,
with a deterministic pseudo-dense fallback engine for offline and testing environments.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Optional
from core.config import settings
from core.logging import logger


class EmbeddingService:
    """Manages dense vector generation with caching, normalization, and fallback capability."""

    def __init__(self, model_name: Optional[str] = None, dimension: int = 384):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self._model = None
        self._model_load_attempted = False

    def _get_model(self):
        """Lazy loader for SentenceTransformer model."""
        if not self._model_load_attempted:
            self._model_load_attempted = True
            # Check if fast mock embeddings are enabled
            if settings.USE_MOCK_EMBEDDINGS:
                logger.info("Fast deterministic embedding engine enabled (USE_MOCK_EMBEDDINGS=true).")
                return None

            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer model '{self.model_name}' could not be loaded ({e}). "
                    "Falling back to deterministic 384-dimensional dense projection engine."
                )
                self._model = None
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generates a normalized dense vector for a single query or text."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized dense vectors for a list of texts."""
        if not texts:
            return []

        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return [arr.tolist() for arr in embeddings]
            except Exception as e:
                logger.warning(f"Error encoding with SentenceTransformer: {e}. Using deterministic fallback.")

        # Deterministic normalized dense vector generator
        return [self._generate_deterministic_vector(t) for t in texts]

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """
        Generates a deterministic 384-dimensional unit vector from text tokens.
        Preserves non-zero cosine similarity for overlapping financial terms and exact matches.
        """
        vec = [0.0] * self.dimension
        clean_text = text.lower().strip()
        tokens = re.findall(r"\b[a-z0-9_]{2,}\b", clean_text)
        if not tokens:
            vec[0] = 1.0
            return vec

        # Use token hashing into vector indices
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            weight = 1.0 + (len(token) * 0.1)
            vec[idx] += sign * weight

            # Bigram hashing for local syntax
            h2 = int(hashlib.sha1((token + "_bi").encode("utf-8")).hexdigest(), 16)
            idx2 = (h2 + idx) % self.dimension
            vec[idx2] += (sign * 0.5)

        # Normalize vector to unit length (L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [round(v / norm, 6) for v in vec]
        else:
            vec[0] = 1.0

        return vec


# Global embedding service instance
embedding_service = EmbeddingService()
