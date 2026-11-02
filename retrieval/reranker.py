"""
Cross-Encoder Reranker for FinSight AI
Jointly scores (query, candidate_passage) pairs to re-order hybrid retrieval candidates,
improving precision and NDCG before passing context to LLM agents.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from core.config import settings
from core.logging import logger
from retrieval.vector_store import SearchResult


@dataclass
class RerankedResult(SearchResult):
    """Result enriched with cross-encoder relevance score and rank delta."""
    original_rank: int = 1
    reranked_rank: int = 1
    rank_delta: int = 0
    cross_encoder_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["original_rank"] = self.original_rank
        d["reranked_rank"] = self.reranked_rank
        d["rank_delta"] = self.rank_delta
        d["cross_encoder_score"] = round(self.cross_encoder_score, 4)
        return d


class CrossEncoderReranker:
    """Reranks candidate chunks using a cross-encoder model with semantic fallback."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None
        self._load_attempted = False

    def _get_model(self):
        """Lazy loader for SentenceTransformers CrossEncoder."""
        if not self._load_attempted:
            self._load_attempted = True
            if settings.USE_MOCK_EMBEDDINGS:
                logger.info("Fast deterministic reranker enabled (USE_MOCK_EMBEDDINGS=true).")
                return None

            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
                logger.info("CrossEncoder model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"CrossEncoder '{self.model_name}' could not be loaded ({e}). "
                    "Using deterministic query-passage cross-attention fallback."
                )
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[RerankedResult]:
        """
        Reranks a list of candidate search results against the query.
        Returns top_k items with cross-encoder scores and rank delta telemetry.
        """
        if not candidates:
            return []

        model = self._get_model()
        scored_candidates: List[RerankedResult] = []

        if model is not None:
            try:
                pairs = [[query, c.clean_text] for c in candidates]
                scores = model.predict(pairs)
                for idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
                    # Sigmoid transform to 0-1 if logits
                    norm_score = 1.0 / (1.0 + math.exp(-float(score))) if isinstance(score, (int, float)) else float(score)
                    scored_candidates.append(
                        self._wrap_reranked(cand, original_rank=idx, score=norm_score)
                    )
            except Exception as e:
                logger.warning(f"Error during CrossEncoder prediction: {e}. Falling back to internal scorer.")
                scored_candidates = []

        if not scored_candidates:
            # Deterministic cross-attention scoring
            for idx, cand in enumerate(candidates, start=1):
                score = self._compute_cross_score(query, cand)
                scored_candidates.append(
                    self._wrap_reranked(cand, original_rank=idx, score=score)
                )

        # Sort descending by cross_encoder_score
        scored_candidates.sort(key=lambda x: x.cross_encoder_score, reverse=True)

        # Calculate new rank and delta (original_rank - reranked_rank, where positive is rank improvement)
        final_results: List[RerankedResult] = []
        for new_rank, item in enumerate(scored_candidates[:top_k], start=1):
            item.reranked_rank = new_rank
            item.rank_delta = item.original_rank - new_rank
            item.score = item.cross_encoder_score  # Update primary score
            final_results.append(item)

        logger.info(
            f"Cross-encoder reranked {len(candidates)} candidates to top-{len(final_results)} "
            f"(top score: {final_results[0].cross_encoder_score:.4f})"
        )
        return final_results

    def _wrap_reranked(self, cand: SearchResult, original_rank: int, score: float) -> RerankedResult:
        """Converts SearchResult to RerankedResult."""
        return RerankedResult(
            chunk_id=cand.chunk_id,
            document_id=cand.document_id,
            company_ticker=cand.company_ticker,
            document_title=cand.document_title,
            fiscal_year=cand.fiscal_year,
            reporting_period=cand.reporting_period,
            page_number=cand.page_number,
            section_name=cand.section_name,
            content=cand.content,
            clean_text=cand.clean_text,
            score=score,
            is_table=cand.is_table,
            source_reference=cand.source_reference,
            original_rank=original_rank,
            cross_encoder_score=score,
        )

    def _compute_cross_score(self, query: str, cand: SearchResult) -> float:
        """
        Calculates joint query-passage cross-attention interaction:
        1. Exact phrase matching.
        2. Financial terminology density.
        3. Term proximity.
        4. Section heading alignment.
        """
        clean_q = query.lower().strip()
        doc_text = (cand.clean_text or cand.content).lower()
        q_tokens = re.findall(r"\b[a-z0-9_\$%\.]+\b", clean_q)
        if not q_tokens:
            return 0.5

        # 1. Term coverage ratio
        matched_tokens = sum(1 for t in q_tokens if t in doc_text)
        coverage_score = matched_tokens / len(q_tokens)

        # 2. Exact phrase match bonus
        phrase_score = 0.0
        if len(q_tokens) > 1 and clean_q in doc_text:
            phrase_score = 0.35
        elif any(f"{q_tokens[i]} {q_tokens[i+1]}" in doc_text for i in range(len(q_tokens)-1)):
            phrase_score = 0.20

        # 3. Section name alignment
        section_score = 0.0
        sec_name = cand.section_name.lower()
        if any(t in sec_name for t in q_tokens):
            section_score = 0.15

        # 4. Term proximity bonus: shortest span containing matched tokens
        proximity_score = 0.0
        positions = [m.start() for t in q_tokens for m in re.finditer(re.escape(t), doc_text)]
        if len(positions) >= 2:
            span = max(positions) - min(positions)
            if span < 200:
                proximity_score = 0.10

        # Table bonus if query mentions ratios/numbers/metrics
        table_score = 0.05 if cand.is_table and any(k in clean_q for k in ("table", "growth", "margin", "revenue", "ratio")) else 0.0

        total_score = (coverage_score * 0.5) + phrase_score + section_score + proximity_score + table_score
        return round(min(1.0, max(0.0, total_score)), 4)


# Global reranker instance
cross_encoder_reranker = CrossEncoderReranker()
