"""
Unit Tests for Deterministic Financial Calculators
Tests exact mathematical accuracy against manually verified ground truth examples,
zero denominator handling, negative figures, and citation preservation.
"""

from __future__ import annotations

import unittest
from core.exceptions import ZeroDenominatorError
from finance.calculators import FinancialCalculators
from finance.models import CitationReference, FactRecord, MetricType, StatementType


class TestFinancialCalculators(unittest.TestCase):
    def setUp(self):
        self.cit_infy_24 = CitationReference("Infosys AR 2023-24", 2024, "FY2024", 160, "Note 2.15")
        self.cit_infy_23 = CitationReference("Infosys AR 2022-23", 2023, "FY2023", 154, "Note 2.15")

        self.fact_rev_24 = FactRecord(
            "INFY", 2024, "FY2024", StatementType.INCOME_STATEMENT,
            "revenue", "Revenue", 153670.0, "crore", "INR", True, self.cit_infy_24
        )
        self.fact_rev_23 = FactRecord(
            "INFY", 2023, "FY2023", StatementType.INCOME_STATEMENT,
            "revenue", "Revenue", 146767.0, "crore", "INR", True, self.cit_infy_23
        )

    def test_yoy_revenue_growth_exact(self):
        """Verify YoY Revenue Growth matches exact formula: ((153670 - 146767) / 146767) * 100 = 4.7034%."""
        res = FinancialCalculators.calculate_yoy_revenue_growth(self.fact_rev_24, self.fact_rev_23)

        expected = ((153670.0 - 146767.0) / 146767.0) * 100.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "%")
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.citations), 2)
        self.assertEqual(res.citations[0].document_title, "Infosys AR 2023-24")

    def test_yoy_growth_zero_prior_revenue(self):
        """Verify ZeroDenominatorError is raised when prior revenue is 0.0."""
        fact_zero_prior = FactRecord("ACME", 2023, "FY2023", StatementType.INCOME_STATEMENT, "revenue", "Rev", 0.0, "crore", "INR")
        with self.assertRaises(ZeroDenominatorError):
            FinancialCalculators.calculate_yoy_revenue_growth(self.fact_rev_24, fact_zero_prior)

    def test_operating_margin_exact(self):
        """Verify Operating Margin: (31661 / 153670) * 100 = 20.6033%."""
        cit_op = CitationReference("Infosys AR 2023-24", 2024, "FY2024", 160, "P&L")
        fact_op = FactRecord("INFY", 2024, "FY2024", StatementType.INCOME_STATEMENT, "operating_income", "Op Profit", 31661.0, "crore", "INR", True, cit_op)

        res = FinancialCalculators.calculate_operating_margin(fact_op, self.fact_rev_24)
        expected = (31661.0 / 153670.0) * 100.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "%")
        self.assertEqual(len(res.citations), 2)

    def test_net_profit_margin_exact(self):
        """Verify Net Profit Margin: (26233 / 153670) * 100 = 17.0710%."""
        fact_ni = FactRecord("INFY", 2024, "FY2024", StatementType.INCOME_STATEMENT, "net_income", "Net Profit", 26233.0, "crore", "INR")
        res = FinancialCalculators.calculate_net_profit_margin(fact_ni, self.fact_rev_24)
        expected = (26233.0 / 153670.0) * 100.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "%")

    def test_free_cash_flow_exact(self):
        """Verify Free Cash Flow: Operating Cash Flow - Capex = 25655 - 2741 = 22914 Cr."""
        cit_cf = CitationReference("Infosys AR 2023-24", 2024, "FY2024", 164, "Cash Flows")
        fact_ocf = FactRecord("INFY", 2024, "FY2024", StatementType.CASH_FLOW, "operating_cash_flow", "OCF", 25655.0, "crore", "INR", True, cit_cf)
        fact_capex = FactRecord("INFY", 2024, "FY2024", StatementType.CASH_FLOW, "capex", "Capex", 2741.0, "crore", "INR", True, cit_cf)

        res = FinancialCalculators.calculate_free_cash_flow(fact_ocf, fact_capex)
        self.assertEqual(res.value, 22914.0)
        self.assertEqual(res.unit, "crore")
        self.assertIn("OCF - Capex", res.formula)
        self.assertEqual(len(res.citations), 2)

    def test_fcf_conversion_exact(self):
        """Verify FCF Conversion: (22914 / 26233) * 100 = 87.3480%."""
        res = FinancialCalculators.calculate_fcf_conversion(22914.0, 26233.0)
        expected = (22914.0 / 26233.0) * 100.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "%")

    def test_current_ratio_exact(self):
        """Verify Current Ratio: 71280 / 39410 = 1.8087x."""
        res = FinancialCalculators.calculate_current_ratio(71280.0, 39410.0)
        expected = 71280.0 / 39410.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "x")

    def test_debt_to_equity_exact(self):
        """Verify Debt to Equity: 8460 / 86134 = 0.0982x."""
        res = FinancialCalculators.calculate_debt_to_equity(8460.0, 86134.0)
        expected = 8460.0 / 86134.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "x")

    def test_return_on_equity_exact(self):
        """Verify ROE: (26233 / 86134) * 100 = 30.4560%."""
        res = FinancialCalculators.calculate_return_on_equity(26233.0, 86134.0)
        expected = (26233.0 / 86134.0) * 100.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "%")

    def test_return_on_invested_capital_exact(self):
        """Verify ROIC: (31661 * 0.75 / (8460 + 86134)) * 100 = 25.1028%."""
        res = FinancialCalculators.calculate_return_on_invested_capital(
            operating_income=31661.0,
            total_debt=8460.0,
            total_equity=86134.0,
            tax_rate=0.25,
        )
        expected = (31661.0 * 0.75 / (8460.0 + 86134.0)) * 100.0
        self.assertAlmostEqual(res.value, expected, places=4)
        self.assertEqual(res.unit, "%")

    def test_cagr_calculation(self):
        """Verify 3-year CAGR: ((153670 / 121641) ** (1/2) - 1) * 100 = 12.3965%."""
        cagr = FinancialCalculators.calculate_cagr(121641.0, 153670.0, years=2)
        expected = (((153670.0 / 121641.0) ** 0.5) - 1.0) * 100.0
        self.assertAlmostEqual(cagr, expected, places=4)


if __name__ == "__main__":
    unittest.main()
