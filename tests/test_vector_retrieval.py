"""
Unit Tests for Vector Embeddings and Baseline Vector Search Engine
"""

from __future__ import annotations

import math
import unittest
from core.database import db
from retrieval.embeddings import EmbeddingService
from retrieval.search_engine import VectorSearchEngine
from retrieval.vector_store import VectorStore


class TestVectorRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        self.embedding_service = EmbeddingService(dimension=384)

    def test_embedding_dimension_and_unit_normalization(self):
        """Verify embeddings have exactly 384 dimensions and unit norm length = 1.0."""
        text = "Infosys revenue growth was 4.7% in FY24."
        vector = self.embedding_service.embed_text(text)

        self.assertEqual(len(vector), 384)
        norm = math.sqrt(sum(v * v for v in vector))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_deterministic_embedding_consistency(self):
        """Verify identical inputs generate identical vector representations."""
        text = "Operating margin stood at 24.63% for TCS in FY24."
        vec1 = self.embedding_service.embed_text(text)
        vec2 = self.embedding_service.embed_text(text)

        self.assertEqual(vec1, vec2)

    def test_semantic_similarity_ranking(self):
        """Verify semantically similar financial text has higher similarity than unrelated text."""
        query_vec = self.embedding_service.embed_text("cash flow from operations and capital expenditure")
        relevant_vec = self.embedding_service.embed_text("operating cash flow reached INR 25,655 crore after capex of 2,741 crore")
        unrelated_vec = self.embedding_service.embed_text("astronomical observations of distant galaxies in deep space")

        score_relevant = sum(q * r for q, r in zip(query_vec, relevant_vec))
        score_unrelated = sum(q * u for q, u in zip(query_vec, unrelated_vec))

        self.assertGreater(score_relevant, score_unrelated)

    def test_vector_search_engine_execution_and_filtering(self):
        """Verify vector search engine executes query and applies metadata filters."""
        query = "operating cash flow and free cash flow"
        response = VectorSearchEngine.search(query=query, top_k=3, ticker="INFY")

        self.assertIsNotNone(response)
        self.assertEqual(response.query, query)
        self.assertIn("diagnostics", response.to_dict())
        self.assertIn("latency_ms", response.diagnostics)
        self.assertIn("retrieval_method", response.diagnostics)

        # All results should have company_ticker = INFY if any results are returned
        for res in response.results:
            self.assertEqual(res.company_ticker, "INFY")
            self.assertGreater(res.score, -1.0)


if __name__ == "__main__":
    unittest.main()
