"""
Unit Tests for Financial Document Text Cleaner
"""

from __future__ import annotations

import unittest
from ingestion.cleaner import TextCleaner


class TestTextCleaner(unittest.TestCase):
    def test_unicode_and_zero_width_removal(self):
        """Verify unicode normalization and zero-width spaces are stripped."""
        text_with_zero_width = "Infosys\u200B\u200C Revenue\uFEFF FY24"
        cleaned = TextCleaner.clean_text(text_with_zero_width)
        self.assertEqual(cleaned, "Infosys Revenue FY24")

    def test_quote_and_dash_standardization(self):
        """Verify curly quotes and em-dashes are standardized."""
        curly = '“Infosys” reported ‘record’ margin — operating at 20.6% – 21.0%'
        cleaned = TextCleaner.clean_text(curly)
        self.assertIn('"Infosys"', cleaned)
        self.assertIn("'record'", cleaned)
        self.assertIn(" - ", cleaned)

    def test_footer_and_page_number_removal(self):
        """Verify standalone page numbers and blank page notices are stripped."""
        text_with_footers = (
            "Management Discussion\n"
            "Page 14\n"
            "Operating margins improved by 50 bps.\n"
            "15 of 320\n"
            "Cash flow from operations reached INR 25,655 Cr.\n"
            "This page intentionally left blank\n"
        )
        cleaned = TextCleaner.clean_text(text_with_footers)
        self.assertNotIn("Page 14", cleaned)
        self.assertNotIn("15 of 320", cleaned)
        self.assertNotIn("This page intentionally left blank", cleaned)
        self.assertIn("Operating margins improved by 50 bps.", cleaned)
        self.assertIn("Cash flow from operations reached INR 25,655 Cr.", cleaned)

    def test_table_line_detection(self):
        """Verify detection of markdown table pipes and tabular numerical rows."""
        self.assertTrue(TextCleaner.is_likely_table_line("| Segment | FY23 | FY24 |"))
        self.assertTrue(TextCleaner.is_likely_table_line("Revenues INR 146,767 Cr  INR 153,670 Cr"))
        self.assertFalse(TextCleaner.is_likely_table_line("The company witnessed strong traction in cloud."))


if __name__ == "__main__":
    unittest.main()
