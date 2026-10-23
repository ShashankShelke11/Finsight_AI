"""
Unit and Integration Tests for Database Layer and Sample Data Seeding
"""

from __future__ import annotations

import unittest
from pathlib import Path
from core.config import settings
from core.database import DatabaseManager
from scripts.seed_sample_data import seed_database


class TestDatabaseAndSeeding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an isolated test database
        cls.test_db_path = settings.BASE_DIR / "test_finsight.db"
        if cls.test_db_path.exists():
            cls.test_db_path.unlink()
        cls.test_db_manager = DatabaseManager(f"sqlite:///{cls.test_db_path}")

    @classmethod
    def tearDownClass(cls):
        if cls.test_db_path.exists():
            try:
                cls.test_db_path.unlink()
            except PermissionError:
                pass

    def test_01_init_db(self):
        """Verify database tables and indexes are created without error."""
        self.test_db_manager.init_db()
        tables = self.test_db_manager.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        table_names = {t["name"] for t in tables}
        expected = {"companies", "documents", "document_chunks", "financial_facts", "research_queries"}
        self.assertTrue(expected.issubset(table_names), f"Missing tables: {expected - table_names}")

    def test_02_seeding(self):
        """Verify data seeder populates companies, documents, chunks, and facts."""
        # Run seeder on the test database by temporarily overriding DATABASE_URL
        orig_url = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = f"sqlite:///{self.test_db_path}"
            from core.database import db
            db.database_url = settings.DATABASE_URL
            db.is_sqlite = True

            seed_database()

            # Check companies
            companies = db.fetch_all("SELECT ticker, name FROM companies ORDER BY ticker;")
            self.assertEqual(len(companies), 2)
            tickers = [c["ticker"] for c in companies]
            self.assertIn("INFY", tickers)
            self.assertIn("TCS", tickers)

            # Check documents
            docs = db.fetch_all("SELECT id, title, fiscal_year FROM documents;")
            self.assertGreaterEqual(len(docs), 6)

            # Check facts
            facts = db.fetch_all("SELECT * FROM financial_facts WHERE company_id = (SELECT id FROM companies WHERE ticker = 'INFY');")
            self.assertGreaterEqual(len(facts), 20)

            # Verify specific ground truth fact for Infosys FY2024 Revenue
            rev_row = db.fetch_one(
                """
                SELECT value, unit, currency, source_page FROM financial_facts
                WHERE metric_name = 'revenue' AND fiscal_year = 2024
                AND company_id = (SELECT id FROM companies WHERE ticker = 'INFY');
                """
            )
            self.assertIsNotNone(rev_row)
            self.assertEqual(float(rev_row["value"]), 153670.0)
            self.assertEqual(rev_row["unit"], "crore")
            self.assertEqual(rev_row["currency"], "INR")
            self.assertEqual(rev_row["source_page"], 160)

            # Check document chunks
            chunks = db.fetch_all("SELECT id, section_name, clean_text FROM document_chunks;")
            self.assertGreater(len(chunks), 0)

        finally:
            settings.DATABASE_URL = orig_url


if __name__ == "__main__":
    unittest.main()
