"""
Unified Hybrid Search and Retrieval Engine for FinSight AI
Supports Vector-Only, Hybrid RRF (Dense + BM25), and Hybrid + Cross-Encoder Reranking.
Provides detailed latency breakdowns, score distributions, and rank telemetry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from core.logging import logger
from retrieval.embeddings import embedding_service
from retrieval.hybrid import HybridSearchResult, hybrid_retriever
from retrieval.keyword_search import KeywordSearchEngine
from retrieval.reranker import RerankedResult, cross_encoder_reranker
from retrieval.vector_store import SearchResult, VectorStore


@dataclass
class RetrievalResponse:
    """Encapsulates retrieved passages, retrieval mode, and diagnostic telemetry."""
    query: str
    mode: str
    results: List[SearchResult]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
            "diagnostics": self.diagnostics,
        }


class UnifiedSearchEngine:
    """Multi-stage retrieval coordinator supporting Vector, Hybrid RRF, and Cross-Encoder Reranking."""

    @classmethod
    def search(
        cls,
        query: str,
        mode: str = "hybrid_rerank",  # 'vector', 'hybrid', or 'hybrid_rerank'
        top_k: int = 5,
        candidate_pool_size: int = 15,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        reporting_period: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        Executes search under the specified retrieval mode:
        - 'vector': Dense vector similarity only
        - 'hybrid': Dense vector + BM25 keyword search fused via Reciprocal Rank Fusion
        - 'hybrid_rerank': Hybrid RRF candidate selection followed by Cross-Encoder reranking
        """
        start_total = time.perf_counter()
        breakdown: Dict[str, float] = {}

        # 1. Dense Vector Retrieval
        t0 = time.perf_counter()
        query_vector = embedding_service.embed_text(query)
        dense_results = VectorStore.search_similar(
            query_vector=query_vector,
            top_k=candidate_pool_size if mode in ("hybrid", "hybrid_rerank") else top_k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            reporting_period=reporting_period,
        )
        breakdown["dense_vector_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Mode A: Vector Only
        if mode == "vector":
            total_latency = (time.perf_counter() - start_total) * 1000.0
            scores = [r.score for r in dense_results]
            diagnostics = {
                "retrieval_method": "vector_only",
                "latency_ms": round(total_latency, 2),
                "latency_breakdown_ms": breakdown,
                "results_count": len(dense_results),
                "top_score": round(max(scores), 4) if scores else 0.0,
                "min_score": round(min(scores), 4) if scores else 0.0,
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "filter_applied": {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "reporting_period": reporting_period,
                },
            }
            return RetrievalResponse(
                query=query, mode=mode, results=dense_results[:top_k], diagnostics=diagnostics
            )

        # 2. Sparse BM25 Keyword Search
        t1 = time.perf_counter()
        sparse_results = KeywordSearchEngine.search(
            query=query,
            top_k=candidate_pool_size,
            ticker=ticker,
            fiscal_year=fiscal_year,
            reporting_period=reporting_period,
        )
        breakdown["keyword_bm25_ms"] = round((time.perf_counter() - t1) * 1000.0, 2)

        # 3. Reciprocal Rank Fusion
        t2 = time.perf_counter()
        hybrid_candidates = hybrid_retriever.fuse_rankings(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=candidate_pool_size if mode == "hybrid_rerank" else top_k,
        )
        breakdown["rrf_fusion_ms"] = round((time.perf_counter() - t2) * 1000.0, 2)

        # Mode B: Hybrid Only (RRF)
        if mode == "hybrid":
            total_latency = (time.perf_counter() - start_total) * 1000.0
            scores = [r.score for r in hybrid_candidates]
            diagnostics = {
                "retrieval_method": "hybrid_rrf",
                "latency_ms": round(total_latency, 2),
                "latency_breakdown_ms": breakdown,
                "results_count": len(hybrid_candidates),
                "dense_candidates_count": len(dense_results),
                "sparse_candidates_count": len(sparse_results),
                "top_score": round(max(scores), 4) if scores else 0.0,
                "min_score": round(min(scores), 4) if scores else 0.0,
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "filter_applied": {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "reporting_period": reporting_period,
                },
            }
            return RetrievalResponse(
                query=query, mode=mode, results=hybrid_candidates[:top_k], diagnostics=diagnostics
            )

        # 4. Mode C: Cross-Encoder Reranking
        t3 = time.perf_counter()
        reranked_results = cross_encoder_reranker.rerank(
            query=query,
            candidates=hybrid_candidates,
            top_k=top_k,
        )
        breakdown["cross_encoder_rerank_ms"] = round((time.perf_counter() - t3) * 1000.0, 2)

        total_latency = (time.perf_counter() - start_total) * 1000.0
        scores = [r.score for r in reranked_results]
        rank_movements = [
            {
                "chunk_id": r.chunk_id,
                "title": r.document_title,
                "original_rank": r.original_rank,
                "reranked_rank": r.reranked_rank,
                "rank_delta": r.rank_delta,
                "rerank_score": round(r.cross_encoder_score, 4),
            }
            for r in reranked_results
        ]

        diagnostics = {
            "retrieval_method": "hybrid_rrf_plus_cross_encoder_rerank",
            "latency_ms": round(total_latency, 2),
            "latency_breakdown_ms": breakdown,
            "results_count": len(reranked_results),
            "candidate_pool_size": candidate_pool_size,
            "top_score": round(max(scores), 4) if scores else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "rank_movements": rank_movements,
            "filter_applied": {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "reporting_period": reporting_period,
            },
        }

        logger.info(
            f"Hybrid+Rerank search for '{query[:35]}...' finished in {total_latency:.2f}ms "
            f"(top_score={diagnostics['top_score']})"
        )
        return RetrievalResponse(
            query=query, mode=mode, results=reranked_results, diagnostics=diagnostics
        )


# Backward-compatible and singleton exports
VectorSearchEngine = UnifiedSearchEngine
search_engine = UnifiedSearchEngine()
