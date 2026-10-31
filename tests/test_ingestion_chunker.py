"""
Unit Tests for Structure-Aware Document Chunker
"""

from __future__ import annotations

import unittest
from ingestion.chunker import StructureAwareChunker


class TestStructureAwareChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = StructureAwareChunker(target_chunk_size=100, chunk_overlap=20)

    def test_section_splitting_and_headings(self):
        """Verify chunker detects markdown headings and associates section names."""
        doc_text = """
# Strategic Performance
Infosys delivered resilient financial performance amidst macroeconomic uncertainties.
Total revenue reached INR 153,670 crore.

## Capital Allocation Policy
The company returned 85% of free cash flow to shareholders via dividends and buybacks.
Free cash flow stood at INR 22,914 crore.

## Key Risk Factors
Macroeconomic slowdown and delayed discretionary spending could impact realizations.
"""
        chunks = self.chunker.chunk_document(doc_text)
        self.assertGreaterEqual(len(chunks), 3)

        section_names = [c.section_name for c in chunks]
        self.assertIn("Strategic Performance", section_names)
        self.assertIn("Capital Allocation Policy", section_names)
        self.assertIn("Key Risk Factors", section_names)

    def test_table_preservation(self):
        """Verify tables are preserved intact without tearing tabular structure."""
        table_text = """
### Financial Highlights Table
| Metric | FY23 | FY24 | YoY Growth |
|---|---|---|---|
| Revenue (Cr) | 146767 | 153670 | 4.7% |
| Operating Profit (Cr) | 30905 | 31661 | 2.4% |
| Net Profit (Cr) | 24095 | 26233 | 8.9% |
"""
        chunks = self.chunker.chunk_document(table_text)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_table)
        self.assertIn("Revenue (Cr)", chunks[0].content)
        self.assertIn("Net Profit (Cr)", chunks[0].content)

    def test_sliding_window_overlap(self):
        """Verify long paragraphs are split with overlap and appropriate token counts."""
        long_paragraph = "word " * 350
        chunks = self.chunker.chunk_document(long_paragraph)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertGreater(c.token_count, 0)
            self.assertLessEqual(c.token_count, 150)


if __name__ == "__main__":
    unittest.main()
