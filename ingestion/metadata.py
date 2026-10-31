"""
Document Metadata Extraction and Validation for FinSight AI
Extracts and validates company identity, fiscal year, reporting periods,
document types, and computes SHA-256 integrity hashes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DocumentMetadata:
    """Provenance and classification metadata for an ingested financial document."""
    company_ticker: str
    title: str
    document_type: str  # annual_report, quarterly_filing, earnings_transcript, etc.
    fiscal_year: int
    reporting_period: str  # FY2024, Q1_FY2024, etc.
    publication_date: Optional[str] = None
    source_url: Optional[str] = None
    file_hash: str = ""
    file_path: Optional[str] = None
    total_pages: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_ticker": self.company_ticker,
            "title": self.title,
            "document_type": self.document_type,
            "fiscal_year": self.fiscal_year,
            "reporting_period": self.reporting_period,
            "publication_date": self.publication_date,
            "source_url": self.source_url,
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
        }


def compute_file_hash(filepath: Path | str) -> str:
    """Computes SHA-256 checksum of a file for idempotency and duplicate detection."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {filepath}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def infer_metadata_from_filename_or_text(
    filename: str,
    text_sample: str = "",
    default_ticker: str = "GENERIC",
) -> DocumentMetadata:
    """
    Infers document metadata from filename patterns or the opening text of a report.
    Handles typical patterns:
    - infy_fy2024_excerpt.txt -> INFY, FY2024, 2024
    - TCS_Annual_Report_2023.pdf -> TCS, FY2023, 2023
    """
    clean_name = Path(filename).stem.upper()

    # 1. Infer Ticker
    ticker = default_ticker
    known_tickers = ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "RELIANCE", "HDFCBANK", "ICICIBANK"]
    for t in known_tickers:
        if t in clean_name or (text_sample and t in text_sample[:500].upper()):
            ticker = t
            break

    # 2. Infer Fiscal Year (FY2024, FY24, 2024)
    fy_match = re.search(r"(?:FY\s*[-_]?\s*|20)(\d{2,4})", clean_name)
    fiscal_year = 2024
    if fy_match:
        val = int(fy_match.group(1))
        if val < 100:
            val += 2000
        fiscal_year = val
    elif text_sample:
        text_fy = re.search(r"(?:Fiscal\s+Year|FY\s*[-_]?\s*ended\s+March\s+31,?\s*|FY\s*)(\d{2,4})", text_sample, re.IGNORECASE)
        if text_fy:
            val = int(text_fy.group(1))
            if val < 100:
                val += 2000
            fiscal_year = val

    # 3. Infer Reporting Period
    reporting_period = f"FY{fiscal_year}"
    q_match = re.search(r"(Q[1-4])", clean_name)
    if q_match:
        reporting_period = f"{q_match.group(1)}_{reporting_period}"

    # 4. Infer Document Type
    doc_type = "annual_report"
    if "TRANSCRIPT" in clean_name or "CONCALL" in clean_name:
        doc_type = "earnings_transcript"
    elif "Q" in reporting_period:
        doc_type = "quarterly_filing"

    title = f"{ticker} {doc_type.replace('_', ' ').title()} {reporting_period}"

    return DocumentMetadata(
        company_ticker=ticker,
        title=title,
        document_type=doc_type,
        fiscal_year=fiscal_year,
        reporting_period=reporting_period,
        file_path=str(filename),
    )
