"""
End-to-End Integration Tests for Document Ingestion Pipeline and Vector Retrieval
"""

from __future__ import annotations

import unittest
from pathlib import Path
from core.config import settings
from core.database import db
from ingestion.pipeline import IngestionPipeline
from retrieval.search_engine import VectorSearchEngine


class TestIngestionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        cls.pipeline = IngestionPipeline()
        cls.infy_excerpt = settings.SAMPLE_DATA_DIR / "reports" / "infy_fy2024_excerpt.txt"
        cls.tcs_excerpt = settings.SAMPLE_DATA_DIR / "reports" / "tcs_fy2024_excerpt.txt"

    def test_01_ingest_sample_reports(self):
        """Verify end-to-end ingestion of real filing excerpts."""
        self.assertTrue(self.infy_excerpt.exists(), f"Missing file: {self.infy_excerpt}")
        self.assertTrue(self.tcs_excerpt.exists(), f"Missing file: {self.tcs_excerpt}")

        # Ingest Infosys
        infy_res = self.pipeline.ingest_file(
            filepath=self.infy_excerpt,
            company_ticker="INFY",
            fiscal_year=2024,
            reporting_period="FY2024",
            force_reindex=True,
        )
        self.assertEqual(infy_res.status, "success")
        self.assertGreater(infy_res.chunks_created, 0)
        self.assertEqual(infy_res.company_ticker, "INFY")

        # Ingest TCS
        tcs_res = self.pipeline.ingest_file(
            filepath=self.tcs_excerpt,
            company_ticker="TCS",
            fiscal_year=2024,
            reporting_period="FY2024",
            force_reindex=True,
        )
        self.assertEqual(tcs_res.status, "success")
        self.assertGreater(tcs_res.chunks_created, 0)
        self.assertEqual(tcs_res.company_ticker, "TCS")

    def test_02_duplicate_document_detection(self):
        """Verify duplicate ingestion attempts are caught by file hash."""
        dup_res = self.pipeline.ingest_file(
            filepath=self.infy_excerpt,
            company_ticker="INFY",
            force_reindex=False,
        )
        self.assertEqual(dup_res.status, "already_exists")
        self.assertIn("already been ingested", dup_res.details)

    def test_03_vector_retrieval_over_ingested_chunks(self):
        """Verify vector retrieval correctly matches queries against ingested report chunks."""
        # Query 1: Infosys Large Deals
        infy_query = "large deal total contract value TCV Cobalt Topaz"
        infy_search = VectorSearchEngine.search(query=infy_query, top_k=3)
        self.assertGreater(len(infy_search.results), 0)
        top_infy = infy_search.results[0]
        self.assertEqual(top_infy.company_ticker, "INFY")
        self.assertTrue(
            "Cobalt" in top_infy.content or "deal" in top_infy.content.lower() or "17.7" in top_infy.content
        )

        # Query 2: TCS AI WisdomNext
        tcs_query = "AI WisdomNext generative AI aggregation platform"
        tcs_search = VectorSearchEngine.search(query=tcs_query, top_k=3)
        self.assertGreater(len(tcs_search.results), 0)
        top_tcs = tcs_search.results[0]
        self.assertEqual(top_tcs.company_ticker, "TCS")
        self.assertIn("WisdomNext", top_tcs.content)


if __name__ == "__main__":
    unittest.main()
