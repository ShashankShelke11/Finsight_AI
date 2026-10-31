"""
Multi-Format Document Loader for FinSight AI
Extracts text, page numbers, and tables from PDF, DOCX, TXT, MD, and CSV files.
Uses PyMuPDF (fitz) for PDF extraction and python-docx for Word documents.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from core.exceptions import UnsupportedFileTypeError
from core.logging import logger


class DocumentLoader:
    """Extracts text and tabular content across multiple financial document formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".json", ".csv"}

    @classmethod
    def load(cls, filepath: Path | str) -> Tuple[str, Dict[int, str], int]:
        """
        Loads document and returns:
        - full_text: complete extracted text
        - page_map: mapping of page_number -> page_text
        - total_pages: total number of pages/sections
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Document file does not exist: {filepath}")

        ext = p.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported format '{ext}'. Supported: {sorted(cls.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            return cls._load_pdf(p)
        elif ext in (".docx", ".doc"):
            return cls._load_docx(p)
        elif ext == ".csv":
            return cls._load_csv(p)
        elif ext == ".json":
            return cls._load_json(p)
        else:
            return cls._load_text(p)

    @classmethod
    def _load_pdf(cls, filepath: Path) -> Tuple[str, Dict[int, str], int]:
        """Extracts text and tables from PDF using PyMuPDF (fitz)."""
        try:
            import pymupdf as fitz
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) is required to parse PDF documents. Install via `pip install pymupdf`")

        page_map: Dict[int, str] = {}
        full_text_parts: List[str] = []

        with fitz.open(str(filepath)) as doc:
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc[page_idx]

                # 1. Try to extract structured tables first if available in newer PyMuPDF
                table_texts: List[str] = []
                try:
                    tables = page.find_tables()
                    if tables and len(tables.tables) > 0:
                        for tbl in tables:
                            df = tbl.extract()
                            if df and len(df) > 1:
                                header = [str(c or "").strip() for c in df[0]]
                                md_table = [f"| {' | '.join(header)} |", f"| {' | '.join(['---'] * len(header))} |"]
                                for row in df[1:]:
                                    md_table.append(f"| {' | '.join(str(c or '').strip() for c in row)} |")
                                table_texts.append("\n".join(md_table))
                except Exception as e:
                    logger.debug(f"PyMuPDF table extraction fallback on page {page_num}: {e}")

                # 2. Extract standard text blocks
                raw_text = page.get_text("text").strip()

                page_content = raw_text
                if table_texts:
                    page_content = f"{raw_text}\n\n### Extracted Tables:\n" + "\n\n".join(table_texts)

                page_map[page_num] = page_content
                full_text_parts.append(f"<!-- Page {page_num} -->\n{page_content}")

        return "\n\n".join(full_text_parts), page_map, total_pages

    @classmethod
    def _load_docx(cls, filepath: Path) -> Tuple[str, Dict[int, str], int]:
        """Extracts text and tables from Word (.docx) documents."""
        try:
            import docx
        except ImportError:
            raise RuntimeError("python-docx is required to parse DOCX documents. Install via `pip install python-docx`")

        doc = docx.Document(str(filepath))
        parts: List[str] = []

        # Extract paragraphs
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                parts.append(txt)

        # Extract tables
        for tbl_idx, table in enumerate(doc.tables):
            table_rows: List[str] = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                table_rows.append(f"| {' | '.join(cells)} |")
            if len(table_rows) > 1:
                # Add markdown divider after header
                col_count = len(table.rows[0].cells)
                table_rows.insert(1, f"| {' | '.join(['---'] * col_count)} |")
                parts.append(f"\n### Table {tbl_idx + 1}:\n" + "\n".join(table_rows))

        full_text = "\n\n".join(parts)
        page_map = {1: full_text}
        return full_text, page_map, 1

    @classmethod
    def _load_text(cls, filepath: Path) -> Tuple[str, Dict[int, str], int]:
        """Extracts plain text or markdown files."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()

        page_map = {1: content}
        return content, page_map, 1

    @classmethod
    def _load_csv(cls, filepath: Path) -> Tuple[str, Dict[int, str], int]:
        """Loads CSV as markdown table for tabular analysis."""
        rows: List[List[str]] = []
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([cell.strip() for cell in row])

        if not rows:
            return "", {1: ""}, 1

        header = rows[0]
        md_lines = [f"| {' | '.join(header)} |", f"| {' | '.join(['---'] * len(header))} |"]
        for row in rows[1:]:
            md_lines.append(f"| {' | '.join(row)} |")

        content = "\n".join(md_lines)
        return content, {1: content}, 1

    @classmethod
    def _load_json(cls, filepath: Path) -> Tuple[str, Dict[int, str], int]:
        """Loads JSON formatted records."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        return content, {1: content}, 1
