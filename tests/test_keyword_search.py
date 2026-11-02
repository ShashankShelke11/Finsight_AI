"""
Unit Tests for BM25 Lexical Keyword Search Engine
"""

from __future__ import annotations

import unittest
from core.database import db
from retrieval.keyword_search import BM25Scorer, KeywordSearchEngine


class TestKeywordSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        from scripts.seed_sample_data import seed_database
        seed_database()

    def test_bm25_tokenization_and_scoring(self):
        """Verify BM25 tokenization and scoring formula."""
        scorer = BM25Scorer(k1=1.5, b=0.75)
        corpus = [
            {"id": 1, "clean_text": "Infosys revenue growth was 4.7% with robust cloud traction.", "section_name": "Overview"},
            {"id": 2, "clean_text": "TCS reported all-time high annual order book with AI WisdomNext platform.", "section_name": "Strategic Highlights"},
            {"id": 3, "clean_text": "Capital expenditure was INR 2,741 crore with free cash flow of INR 22,914 crore.", "section_name": "Cash Flow"},
        ]

        # Test query 1
        scores_1 = scorer.compute_scores("AI WisdomNext order book", corpus)
        self.assertEqual(len(scores_1), 3)
        # Doc 2 must have the highest score
        self.assertEqual(scores_1[1][0]["id"], 2)
        self.assertGreater(scores_1[1][1], scores_1[0][1])

        # Test query 2: cash flow
        scores_2 = scorer.compute_scores("free cash flow capital expenditure", corpus)
        self.assertEqual(scores_2[2][0]["id"], 3)
        self.assertGreater(scores_2[2][1], scores_2[0][1])

    def test_keyword_search_database_execution(self):
        """Verify KeywordSearchEngine queries database chunks with metadata filtering."""
        query = "WisdomNext generative AI"
        results = KeywordSearchEngine.search(query=query, top_k=3, ticker="TCS")

        self.assertGreater(len(results), 0)
        top_res = results[0]
        self.assertEqual(top_res.company_ticker, "TCS")
        self.assertIn("WisdomNext", top_res.content)
        self.assertGreater(top_res.score, 0.0)

    def test_empty_query_handling(self):
        """Verify empty or non-matching query returns safe result without error."""
        results = KeywordSearchEngine.search(query="", top_k=3)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
