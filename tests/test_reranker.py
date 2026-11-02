"""
Unit Tests for Cross-Encoder Reranker
"""

from __future__ import annotations

import unittest
from core.database import db
from retrieval.reranker import CrossEncoderReranker, RerankedResult
from retrieval.search_engine import UnifiedSearchEngine
from retrieval.vector_store import SearchResult


class TestCrossEncoderReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        self.reranker = CrossEncoderReranker()

    def test_cross_encoder_scoring_and_rank_delta(self):
        """Verify cross-encoder reranking boosts highly relevant passages and computes rank delta."""
        query = "free cash flow capital expenditure dividend"

        # Candidate 1: Generic discussion (initially rank 1)
        cand1 = SearchResult(
            chunk_id=101,
            document_id=1,
            company_ticker="INFY",
            document_title="Infosys Report",
            fiscal_year=2024,
            reporting_period="FY2024",
            page_number=18,
            section_name="Overview",
            content="Infosys delivered resilient financial performance amidst macroeconomic uncertainties.",
            clean_text="Infosys delivered resilient financial performance amidst macroeconomic uncertainties.",
            score=0.85,
        )

        # Candidate 2: Exact cash flow and capex passage (initially rank 2)
        cand2 = SearchResult(
            chunk_id=102,
            document_id=1,
            company_ticker="INFY",
            document_title="Infosys Report",
            fiscal_year=2024,
            reporting_period="FY2024",
            page_number=164,
            section_name="Cash Flow and Capital Allocation",
            content="Free cash flow stood at INR 22,914 crore after accounting for capital expenditure of INR 2,741 crore. The Board recommended dividend.",
            clean_text="Free cash flow stood at INR 22,914 crore after accounting for capital expenditure of INR 2,741 crore. The Board recommended dividend.",
            score=0.75,
        )

        candidates = [cand1, cand2]
        reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=2)

        self.assertEqual(len(reranked), 2)
        top_res = reranked[0]
        # Candidate 2 must be reranked to 1st place!
        self.assertEqual(top_res.chunk_id, 102)
        self.assertEqual(top_res.original_rank, 2)
        self.assertEqual(top_res.reranked_rank, 1)
        # Rank delta = 2 - 1 = +1 (improved by 1 position)
        self.assertEqual(top_res.rank_delta, 1)
        self.assertGreater(top_res.cross_encoder_score, reranked[1].cross_encoder_score)

    def test_search_engine_hybrid_rerank_mode(self):
        """Verify UnifiedSearchEngine executes in hybrid_rerank mode and returns diagnostics."""
        query = "Infosys Topaz enterprise generative AI solutions"
        response = UnifiedSearchEngine.search(
            query=query,
            mode="hybrid_rerank",
            top_k=3,
            ticker="INFY",
        )

        self.assertEqual(response.mode, "hybrid_rerank")
        self.assertGreater(len(response.results), 0)
        self.assertIn("rank_movements", response.diagnostics)
        self.assertIn("latency_breakdown_ms", response.diagnostics)
        self.assertIn("cross_encoder_rerank_ms", response.diagnostics["latency_breakdown_ms"])

        top = response.results[0]
        self.assertEqual(top.company_ticker, "INFY")
        self.assertIsInstance(top, RerankedResult)


if __name__ == "__main__":
    unittest.main()
