"""
Unit Tests for Multi-Format Document Loader (PDF, DOCX, TXT, CSV)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from core.exceptions import UnsupportedFileTypeError
from ingestion.loader import DocumentLoader


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_text_file(self):
        """Verify text/markdown files are loaded properly."""
        txt_path = self.dir_path / "report.txt"
        txt_path.write_text("Infosys delivered strong performance in FY24.", encoding="utf-8")

        content, page_map, total_pages = DocumentLoader.load(txt_path)
        self.assertIn("Infosys delivered strong performance", content)
        self.assertEqual(total_pages, 1)
        self.assertIn(1, page_map)

    def test_load_csv_file(self):
        """Verify CSV files are parsed and converted to markdown table format."""
        csv_path = self.dir_path / "metrics.csv"
        csv_path.write_text("Metric,FY23,FY24\nRevenue,146767,153670\nOperatingProfit,30905,31661\n", encoding="utf-8")

        content, page_map, total_pages = DocumentLoader.load(csv_path)
        self.assertIn("| Metric | FY23 | FY24 |", content)
        self.assertIn("| Revenue | 146767 | 153670 |", content)

    def test_load_docx_file(self):
        """Verify docx documents are loaded with paragraphs and tables."""
        import docx

        docx_path = self.dir_path / "statement.docx"
        doc = docx.Document()
        doc.add_heading("Management Overview", level=1)
        doc.add_paragraph("TCS achieved an all-time high annual order book.")
        table = doc.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Company"
        table.rows[0].cells[1].text = "Order Book"
        table.rows[1].cells[0].text = "TCS"
        table.rows[1].cells[1].text = "$42.7B"
        doc.save(str(docx_path))

        content, page_map, total_pages = DocumentLoader.load(docx_path)
        self.assertIn("TCS achieved an all-time high", content)
        self.assertIn("| Company | Order Book |", content)

    def test_load_pdf_file(self):
        """Verify PDF extraction using PyMuPDF (fitz)."""
        import pymupdf as fitz

        pdf_path = self.dir_path / "annual_report.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Infosys Topaz Enterprise AI Suite launch in FY24.")
        doc.save(str(pdf_path))
        doc.close()

        content, page_map, total_pages = DocumentLoader.load(pdf_path)
        self.assertIn("Infosys Topaz Enterprise AI Suite", content)
        self.assertEqual(total_pages, 1)

    def test_unsupported_extension(self):
        """Verify UnsupportedFileTypeError is raised for invalid file types."""
        exe_path = self.dir_path / "payload.exe"
        exe_path.write_bytes(b"MZfake")

        with self.assertRaises(UnsupportedFileTypeError):
            DocumentLoader.load(exe_path)


if __name__ == "__main__":
    unittest.main()
