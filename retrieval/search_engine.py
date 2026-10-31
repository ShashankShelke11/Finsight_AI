"""
Baseline Vector Search Engine for FinSight AI
Coordinates query embedding generation, similarity search, metadata filtering,
and retrieval diagnostics computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from core.logging import logger
from retrieval.embeddings import embedding_service
from retrieval.vector_store import SearchResult, VectorStore


@dataclass
class RetrievalResponse:
    """Encapsulates retrieval results and diagnostic telemetry."""
    query: str
    results: List[SearchResult]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "diagnostics": self.diagnostics,
        }


class VectorSearchEngine:
    """Baseline vector similarity retrieval engine with metadata filtering."""

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        reporting_period: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        Executes dense vector search against stored document chunks.
        Applies metadata filters and computes retrieval telemetry diagnostics.
        """
        start_time = time.perf_counter()

        # 1. Embed query
        query_vector = embedding_service.embed_text(query)

        # 2. Query VectorStore
        results = VectorStore.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            reporting_period=reporting_period,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Calculate diagnostics
        scores = [r.score for r in results]
        diagnostics = {
            "retrieval_method": "dense_vector_cosine",
            "latency_ms": round(latency_ms, 2),
            "results_count": len(results),
            "top_score": round(max(scores), 4) if scores else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "filter_applied": {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "reporting_period": reporting_period,
            },
        }

        logger.info(
            f"Vector search for '{query[:40]}...' returned {len(results)} chunks in {latency_ms:.2f}ms "
            f"(top_score={diagnostics['top_score']})"
        )

        return RetrievalResponse(query=query, results=results, diagnostics=diagnostics)


# Global search engine instance
search_engine = VectorSearchEngine()
