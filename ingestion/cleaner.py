"""
Financial Document Text Cleaner and Normalizer for FinSight AI
Cleans, normalizes, and sanitizes text extracted from financial filings,
preserving numerical figures, currencies, statement line items, and tables.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


class TextCleaner:
    """Normalizes raw text extracted from PDF, DOCX, or text files."""

    # Regex patterns for cleaning
    RE_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
    RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
    RE_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")
    RE_PAGE_NUM_FOOTER = re.compile(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
    RE_DISCLAIMER_NOISE = re.compile(
        r"^(?:this\s+page\s+(?:intentionally\s+)?left\s+blank|all\s+rights\s+reserved\.?|confidential)$",
        re.IGNORECASE,
    )

    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        """
        Cleans and normalizes raw text:
        1. Unicode normalization (NFKC).
        2. Strip null bytes and non-printable control chars while preserving newlines and tabs.
        3. Standardize quotes, dashes, and currency symbols.
        4. Remove zero-width spaces.
        5. Compress excessive whitespace while respecting line breaks and markdown formatting.
        """
        if not raw_text:
            return ""

        # 1. Unicode normalization
        text = unicodedata.normalize("NFKC", raw_text)

        # 2. Strip control characters (retain \n, \r, \t)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # 3. Remove zero-width spaces
        text = cls.RE_ZERO_WIDTH.sub("", text)

        # 4. Standardize quotes and dashes commonly found in investor reports
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        text = text.replace("—", " - ").replace("–", " - ")

        # 5. Process line by line to filter running footer/header noise
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()

            # Skip standalone page numbers or blank notices
            if cls.RE_PAGE_NUM_FOOTER.match(stripped) or cls.RE_DISCLAIMER_NOISE.match(stripped):
                continue

            # Normalize horizontal spaces inside the line
            norm_line = cls.RE_HORIZONTAL_WHITESPACE.sub(" ", line).strip()
            if norm_line:
                cleaned_lines.append(norm_line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                # Keep single paragraph breaks
                cleaned_lines.append("")

        # 6. Reassemble and compress redundant newlines
        reconstructed = "\n".join(cleaned_lines)
        return cls.RE_MULTIPLE_NEWLINES.sub("\n\n", reconstructed).strip()

    @classmethod
    def is_likely_table_line(cls, line: str) -> bool:
        """Determines if a line is part of a markdown table or structured numerical row."""
        stripped = line.strip()
        if "|" in stripped:
            return True
        # Check for multiple numbers or currency figures in line
        numbers = re.findall(r"(?:₹|\$|INR|USD)?\s*\d+(?:,\d+)*(?:\.\d+)?", stripped)
        return len(numbers) >= 2
