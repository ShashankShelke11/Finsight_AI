"""
Unit Tests for Financial Normalizer and Accounting Validator
"""

from __future__ import annotations

import unittest
from core.exceptions import InconsistentPeriodError, UnitMismatchError
from finance.models import CitationReference, FactRecord, StatementType
from finance.normalizer import UnitNormalizer
from finance.validator import FinancialValidator


class TestFinancialNormalizerAndValidator(unittest.TestCase):
    def test_unit_conversion(self):
        """Verify scale conversion between Crore, Million, Lakh, and Billion."""
        # 1 Crore = 10 Million
        self.assertAlmostEqual(UnitNormalizer.convert_amount(1.0, "crore", "million"), 10.0)
        # 100 Crore = 1 Billion (10^7 * 100 = 10^9)
        self.assertAlmostEqual(UnitNormalizer.convert_amount(100.0, "crore", "billion"), 1.0)
        # 1 Crore = 100 Lakhs
        self.assertAlmostEqual(UnitNormalizer.convert_amount(1.0, "crore", "lakh"), 100.0)

    def test_unit_alignment(self):
        """Verify aligning two values with different units."""
        v1, v2, common_unit = UnitNormalizer.align_units(5.0, "crore", 20.0, "million")
        self.assertEqual(common_unit, "crore")
        self.assertEqual(v1, 5.0)
        self.assertEqual(v2, 2.0)  # 20 million = 2 crore

    def test_fiscal_year_parsing(self):
        """Verify parsing of various fiscal year notations."""
        self.assertEqual(UnitNormalizer.normalize_fiscal_year("FY24"), 2024)
        self.assertEqual(UnitNormalizer.normalize_fiscal_year("FY2024"), 2024)
        self.assertEqual(UnitNormalizer.normalize_fiscal_year("2023-24"), 2024)
        self.assertEqual(UnitNormalizer.normalize_fiscal_year(2024), 2024)

    def test_accounting_period_validation(self):
        """Verify exception raised when comparing misaligned periods (e.g. Q1 vs FY)."""
        cit = CitationReference("Report", 2024, "FY2024", 1, "Ref")
        f_annual = FactRecord("INFY", 2024, "FY2024", StatementType.INCOME_STATEMENT, "revenue", "Rev", 153670.0, "crore", "INR", True, cit)
        f_quarter = FactRecord("INFY", 2024, "Q1_FY2024", StatementType.INCOME_STATEMENT, "revenue", "Rev", 38000.0, "crore", "INR", True, cit)

        with self.assertRaises(InconsistentPeriodError):
            FinancialValidator.validate_period_comparability(f_annual, f_quarter)

    def test_currency_mismatch_validation(self):
        """Verify exception raised when operating directly across different currencies."""
        cit = CitationReference("Report", 2024, "FY2024", 1, "Ref")
        f_inr = FactRecord("INFY", 2024, "FY2024", StatementType.INCOME_STATEMENT, "revenue", "Rev", 153670.0, "crore", "INR", True, cit)
        f_usd = FactRecord("ACME", 2024, "FY2024", StatementType.INCOME_STATEMENT, "revenue", "Rev", 2000.0, "million", "USD", True, cit)

        with self.assertRaises(UnitMismatchError):
            FinancialValidator.validate_currency_and_unit(f_inr, f_usd)

    def test_ratio_bounds_checking(self):
        """Verify detection of anomalous margins or solvency ratios."""
        self.assertIsNotNone(FinancialValidator.check_ratio_bounds("operating_margin", 95.0))
        self.assertIsNotNone(FinancialValidator.check_ratio_bounds("current_ratio", 0.75))
        self.assertIsNotNone(FinancialValidator.check_ratio_bounds("debt_to_equity", -1.5))
        self.assertIsNone(FinancialValidator.check_ratio_bounds("operating_margin", 24.5))


if __name__ == "__main__":
    unittest.main()
