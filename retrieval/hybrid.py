"""
Reciprocal Rank Fusion (RRF) Hybrid Search for FinSight AI
Merges and recalibrates dense semantic vector rankings and sparse BM25 lexical rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from core.config import settings
from core.logging import logger
from retrieval.keyword_search import KeywordSearchEngine
from retrieval.vector_store import SearchResult, VectorStore


@dataclass
class HybridSearchResult(SearchResult):
    """Retrieved chunk augmented with individual retriever ranks and combined RRF score."""
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["dense_rank"] = self.dense_rank
        d["sparse_rank"] = self.sparse_rank
        d["rrf_score"] = round(self.rrf_score, 6)
        return d


class HybridRetriever:
    """Combines dense vector retrieval and sparse BM25 retrieval via Reciprocal Rank Fusion."""

    def __init__(
        self,
        rrf_k: Optional[int] = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        self.rrf_k = rrf_k or settings.HYBRID_SEARCH_RRF_K
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse_rankings(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int = 5,
    ) -> List[HybridSearchResult]:
        """
        Executes Reciprocal Rank Fusion over dense and sparse ranked lists:
        RRF_Score(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))
        """
        chunk_map: Dict[int, SearchResult] = {}
        dense_ranks: Dict[int, int] = {}
        sparse_ranks: Dict[int, int] = {}

        # 1. Map dense rankings (1-indexed)
        for rank, res in enumerate(dense_results, start=1):
            chunk_map[res.chunk_id] = res
            dense_ranks[res.chunk_id] = rank

        # 2. Map sparse rankings (1-indexed)
        for rank, res in enumerate(sparse_results, start=1):
            if res.chunk_id not in chunk_map:
                chunk_map[res.chunk_id] = res
            sparse_ranks[res.chunk_id] = rank

        # 3. Calculate RRF scores
        fused_items: List[HybridSearchResult] = []
        for chunk_id, base_res in chunk_map.items():
            r_dense = dense_ranks.get(chunk_id)
            r_sparse = sparse_ranks.get(chunk_id)

            score = 0.0
            if r_dense is not None:
                score += self.dense_weight / (self.rrf_k + r_dense)
            if r_sparse is not None:
                score += self.sparse_weight / (self.rrf_k + r_sparse)

            fused_items.append(
                HybridSearchResult(
                    chunk_id=base_res.chunk_id,
                    document_id=base_res.document_id,
                    company_ticker=base_res.company_ticker,
                    document_title=base_res.document_title,
                    fiscal_year=base_res.fiscal_year,
                    reporting_period=base_res.reporting_period,
                    page_number=base_res.page_number,
                    section_name=base_res.section_name,
                    content=base_res.content,
                    clean_text=base_res.clean_text,
                    score=score,  # Combined RRF score
                    is_table=base_res.is_table,
                    source_reference=base_res.source_reference,
                    dense_rank=r_dense,
                    sparse_rank=r_sparse,
                    rrf_score=score,
                )
            )

        # 4. Sort descending by RRF score
        fused_items.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused_items[:top_k]

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        candidate_pool_size: int = 15,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        reporting_period: Optional[str] = None,
    ) -> List[HybridSearchResult]:
        """
        Executes parallel dense and sparse retrieval, followed by Reciprocal Rank Fusion.
        """
        # Fetch candidates from both search mechanisms
        dense_candidates = VectorStore.search_similar(
            query_vector=query_vector,
            top_k=candidate_pool_size,
            ticker=ticker,
            fiscal_year=fiscal_year,
            reporting_period=reporting_period,
        )

        sparse_candidates = KeywordSearchEngine.search(
            query=query,
            top_k=candidate_pool_size,
            ticker=ticker,
            fiscal_year=fiscal_year,
            reporting_period=reporting_period,
        )

        fused = self.fuse_rankings(dense_candidates, sparse_candidates, top_k=top_k)
        logger.info(
            f"Hybrid RRF fusion: merged {len(dense_candidates)} dense and {len(sparse_candidates)} "
            f"sparse candidates into {len(fused)} results."
        )
        return fused


# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()
