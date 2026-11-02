"""
Unit Tests for Reciprocal Rank Fusion (RRF) Hybrid Retrieval
"""

from __future__ import annotations

import unittest
from core.database import db
from retrieval.embeddings import embedding_service
from retrieval.hybrid import HybridRetriever, HybridSearchResult
from retrieval.vector_store import SearchResult


class TestHybridRRF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        from scripts.seed_sample_data import seed_database
        seed_database()

    def setUp(self):
        self.retriever = HybridRetriever(rrf_k=60)

    def _make_dummy_result(self, chunk_id: int, title: str, ticker: str = "INFY") -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            document_id=1,
            company_ticker=ticker,
            document_title=title,
            fiscal_year=2024,
            reporting_period="FY2024",
            page_number=1,
            section_name="Overview",
            content=f"Content for chunk {chunk_id}",
            clean_text=f"Clean text for chunk {chunk_id}",
            score=0.8,
        )

    def test_rrf_scoring_math(self):
        """Verify Reciprocal Rank Fusion score computation and mutual rank boost."""
        doc1 = self._make_dummy_result(1, "Doc 1")
        doc2 = self._make_dummy_result(2, "Doc 2")
        doc3 = self._make_dummy_result(3, "Doc 3")

        # Dense ranking: [Doc 1, Doc 2]
        dense_results = [doc1, doc2]
        # Sparse ranking: [Doc 2, Doc 3]
        sparse_results = [doc2, doc3]

        fused = self.retriever.fuse_rankings(dense_results, sparse_results, top_k=3)
        self.assertEqual(len(fused), 3)

        # Expected RRF scores:
        # Doc 1: dense_rank=1, sparse=None -> 1/(60+1) = 1/61 ~ 0.01639
        # Doc 2: dense_rank=2, sparse_rank=1 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.01613 + 0.01639 = 0.03252
        # Doc 3: dense=None, sparse_rank=2 -> 1/(60+2) = 1/62 ~ 0.01613
        # Therefore, Doc 2 must be ranked 1st due to mutual presence!
        top_doc = fused[0]
        self.assertEqual(top_doc.chunk_id, 2)
        self.assertEqual(top_doc.dense_rank, 2)
        self.assertEqual(top_doc.sparse_rank, 1)
        self.assertAlmostEqual(top_doc.rrf_score, (1/62) + (1/61), places=4)

        # Second doc must be Doc 1
        self.assertEqual(fused[1].chunk_id, 1)

    def test_end_to_end_hybrid_search(self):
        """Verify hybrid search returns fused results with metadata filters."""
        query = "Infosys Cobalt and enterprise AI suite Topaz"
        q_vec = embedding_service.embed_text(query)

        fused_results = self.retriever.search(
            query=query,
            query_vector=q_vec,
            top_k=3,
            ticker="INFY",
        )
        self.assertGreater(len(fused_results), 0)
        self.assertIsInstance(fused_results[0], HybridSearchResult)
        self.assertEqual(fused_results[0].company_ticker, "INFY")
        self.assertGreater(fused_results[0].rrf_score, 0.0)


if __name__ == "__main__":
    unittest.main()
