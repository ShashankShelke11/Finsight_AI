"""
Integration Tests for Financial Analytics Engine and Multi-Company Comparison
Tests database fact loading, multi-year scorecard generation, and comparative reporting between INFY and TCS.
"""

from __future__ import annotations

import unittest
from core.database import db
from finance.analytics import FinancialAnalyticsEngine
from finance.models import MetricType
from scripts.seed_sample_data import seed_database


class TestFinancialAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        seed_database()
        cls.engine = FinancialAnalyticsEngine()

    def test_get_company_facts_from_db(self):
        """Verify audited financial facts are retrieved from database with provenance citations."""
        facts_infy = self.engine.get_company_facts("INFY", [2022, 2023, 2024])

        self.assertIn(2022, facts_infy)
        self.assertIn(2023, facts_infy)
        self.assertIn(2024, facts_infy)

        # Verify FY24 Revenue fact
        rev_fact = facts_infy[2024].get(MetricType.REVENUE.value)
        self.assertIsNotNone(rev_fact)
        self.assertEqual(rev_fact.value, 153670.0)
        self.assertEqual(rev_fact.unit, "crore")
        self.assertEqual(rev_fact.currency, "INR")
        self.assertEqual(rev_fact.citation.page_number, 160)
        self.assertIn("Note 2.15", rev_fact.citation.source_reference)

    def test_compute_financial_scorecard(self):
        """Verify complete deterministic scorecard for Infosys across FY23 and FY24."""
        scorecard = self.engine.compute_financial_scorecard("INFY", [2023, 2024])

        self.assertIn(2023, scorecard)
        self.assertIn(2024, scorecard)

        fy24 = scorecard[2024]

        # YoY Growth
        self.assertAlmostEqual(fy24[MetricType.YOY_REVENUE_GROWTH.value].value, 4.7034, places=3)
        # Operating Margin
        self.assertAlmostEqual(fy24[MetricType.OPERATING_MARGIN.value].value, 20.6033, places=3)
        # Net Margin
        self.assertAlmostEqual(fy24[MetricType.NET_PROFIT_MARGIN.value].value, 17.0710, places=3)
        # Free Cash Flow
        self.assertEqual(fy24[MetricType.FREE_CASH_FLOW.value].value, 22914.0)
        # FCF Conversion
        self.assertAlmostEqual(fy24["fcf_conversion"].value, 87.3480, places=3)
        # Current Ratio
        self.assertAlmostEqual(fy24[MetricType.CURRENT_RATIO.value].value, 1.8087, places=3)
        # Debt to Equity
        self.assertAlmostEqual(fy24[MetricType.DEBT_TO_EQUITY.value].value, 0.0982, places=3)
        # ROE
        self.assertAlmostEqual(fy24[MetricType.RETURN_ON_EQUITY.value].value, 30.4560, places=3)
        # ROIC
        self.assertAlmostEqual(fy24[MetricType.RETURN_ON_INVESTED_CAPITAL.value].value, 25.1028, places=3)

    def test_compare_companies_infy_vs_tcs(self):
        """Verify multi-company comparison between Infosys and TCS over FY22-FY24."""
        comparison = self.engine.compare_companies(["INFY", "TCS"], [2022, 2023, 2024])

        self.assertEqual(comparison.companies, ["INFY", "TCS"])
        self.assertEqual(comparison.fiscal_years, [2022, 2023, 2024])

        # Verify TCS metrics
        tcs_metrics = comparison.metrics["TCS"]
        # TCS FY24 Operating Margin: 59324 / 240893 = 24.6267%
        tcs_op_margins = tcs_metrics[MetricType.OPERATING_MARGIN.value]
        self.assertEqual(len(tcs_op_margins), 3)
        self.assertAlmostEqual(tcs_op_margins[-1].value, 24.6267, places=3)

        # TCS FY24 Free Cash Flow: 44342 - 3460 = 40882 Cr
        tcs_fcf = tcs_metrics[MetricType.FREE_CASH_FLOW.value]
        self.assertEqual(tcs_fcf[-1].value, 40882.0)

        # Verify synthesis notes
        self.assertGreater(len(comparison.synthesis_notes), 0)
        # TCS should be identified as operating margin leader in FY24 (24.63% vs 20.60%)
        notes_text = " ".join(comparison.synthesis_notes)
        self.assertIn("TCS demonstrated higher operating profitability", notes_text)
        self.assertIn("TCS expanded faster in FY2024", notes_text)

        # Verify markdown table generation
        table_md = self.engine.format_comparison_markdown_table(comparison)
        self.assertIn("### Financial Comparison: INFY vs TCS (2022 - 2024)", table_md)
        self.assertIn("| YoY Revenue Growth (%) | **INFY** |", table_md)
        self.assertIn("| Operating Margin (%) | **TCS** |", table_md)
        self.assertIn("#### Key Analytical Takeaways:", table_md)


if __name__ == "__main__":
    unittest.main()
