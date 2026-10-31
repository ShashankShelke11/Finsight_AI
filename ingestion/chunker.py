"""
Structure-Aware Financial Document Chunker for FinSight AI
Splits extracted document text into semantic chunks while preserving
section headers, financial tables, statement disclosures, and page references.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ingestion.cleaner import TextCleaner


@dataclass
class DocumentChunk:
    """Represents a discrete searchable chunk of a financial document."""
    chunk_index: int
    page_number: int
    section_name: str
    content: str
    clean_text: str
    is_table: bool = False
    table_data_json: Optional[str] = None
    token_count: int = 0
    document_id: Optional[int] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "content": self.content,
            "clean_text": self.clean_text,
            "is_table": self.is_table,
            "table_data_json": self.table_data_json,
            "token_count": self.token_count,
            "document_id": self.document_id,
            "metadata": self.metadata or {},
        }


class StructureAwareChunker:
    """
    Chunks financial text respecting document structure:
    - Splits on markdown headings (##, ###) and explicit financial section indicators.
    - Preserves tables and prevents tearing tabular rows.
    - Applies sliding window with overlap for long narrative sections.
    """

    HEADING_PATTERN = re.compile(
        r"^(?:#{1,4}\s+|[0-9]{1,2}\.\s+|[A-Z\s]{4,}:|Section\s+\d+:|Note\s+\d+:?)(.+)$",
        re.MULTILINE,
    )
    TABLE_BORDER_PATTERN = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

    def __init__(
        self,
        target_chunk_size: int = 400,  # approximate words/tokens
        chunk_overlap: int = 50,
        min_chunk_size: int = 40,
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def count_tokens(self, text: str) -> int:
        """Approximates token count based on whitespace and punctuation splitting."""
        if not text:
            return 0
        return len(text.split())

    def chunk_document(
        self,
        text: str,
        initial_page_number: int = 1,
        page_map: Optional[Dict[int, str]] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunks the document into structure-aware segments.
        If page_map is provided (page_num -> text), chunks per page while maintaining
        running section context. Otherwise chunks unified text using internal page markers.
        """
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # If a page_map is provided, process page by page
        if page_map:
            current_section = "Overview"
            for page_num, page_text in sorted(page_map.items()):
                cleaned_page = TextCleaner.clean_text(page_text)
                if not cleaned_page:
                    continue

                page_chunks = self._chunk_section_text(
                    cleaned_page,
                    default_page=page_num,
                    starting_chunk_idx=chunk_idx,
                    inherited_section=current_section,
                    doc_metadata=doc_metadata,
                )
                for c in page_chunks:
                    current_section = c.section_name
                    chunks.append(c)
                    chunk_idx += 1
            return chunks

        # Unified text chunking
        cleaned_text = TextCleaner.clean_text(text)
        return self._chunk_section_text(
            cleaned_text,
            default_page=initial_page_number,
            starting_chunk_idx=0,
            inherited_section="Overview",
            doc_metadata=doc_metadata,
        )

    def _chunk_section_text(
        self,
        text: str,
        default_page: int,
        starting_chunk_idx: int,
        inherited_section: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Splits text into chunks preserving sections, headers, and tables."""
        chunks: List[DocumentChunk] = []
        curr_idx = starting_chunk_idx

        # 1. Detect page numbers if embedded (e.g. "(Page 142)")
        page_matches = list(re.finditer(r"\(Page\s+(\d+)\)", text, re.IGNORECASE))

        # 2. Split on markdown headings or section breaks
        sections = re.split(r"\n(?=#{1,4}\s+)", text)
        current_section = inherited_section

        for section in sections:
            sec_text = section.strip()
            if not sec_text:
                continue

            # Extract heading if present
            heading_match = re.match(r"^#{1,4}\s+(.+?)(?:\n|$)", sec_text)
            if heading_match:
                current_section = heading_match.group(1).strip()
                # Remove header from body if desired, or keep for context
            
            # Determine page number for this section
            page_num = default_page
            for pm in page_matches:
                if pm.start() <= text.find(sec_text):
                    page_num = int(pm.group(1))

            # Check if section is a table
            is_table = self._is_table_block(sec_text)
            sec_token_count = self.count_tokens(sec_text)

            # If it's a table or within target chunk size, keep as single chunk
            if is_table or sec_token_count <= (self.target_chunk_size + self.chunk_overlap):
                chunk = DocumentChunk(
                    chunk_index=curr_idx,
                    page_number=page_num,
                    section_name=current_section,
                    content=sec_text,
                    clean_text=sec_text,
                    is_table=is_table,
                    table_data_json=json.dumps({"rows": [l for l in sec_text.splitlines() if "|" in l]}) if is_table else None,
                    token_count=sec_token_count,
                    metadata=doc_metadata or {},
                )
                chunks.append(chunk)
                curr_idx += 1
            else:
                # Long section: split by paragraphs or sliding window with overlap
                sub_chunks = self._sliding_window_split(
                    sec_text,
                    page_num=page_num,
                    section_name=current_section,
                    start_idx=curr_idx,
                    doc_metadata=doc_metadata,
                )
                chunks.extend(sub_chunks)
                curr_idx += len(sub_chunks)

        return chunks

    def _is_table_block(self, text: str) -> bool:
        """Determines if a block is primarily tabular data."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return False
        table_lines = sum(1 for l in lines if "|" in l or TextCleaner.is_likely_table_line(l))
        return (table_lines / len(lines)) >= 0.5 and len(lines) >= 2

    def _sliding_window_split(
        self,
        text: str,
        page_num: int,
        section_name: str,
        start_idx: int,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Splits long text blocks using sliding window word count with overlap."""
        words = text.split()
        chunks: List[DocumentChunk] = []
        curr_idx = start_idx
        start = 0

        while start < len(words):
            end = min(start + self.target_chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                DocumentChunk(
                    chunk_index=curr_idx,
                    page_number=page_num,
                    section_name=section_name,
                    content=chunk_text,
                    clean_text=chunk_text,
                    is_table=False,
                    token_count=len(chunk_words),
                    metadata=doc_metadata or {},
                )
            )
            curr_idx += 1
            if end >= len(words):
                break
            start += max(1, self.target_chunk_size - self.chunk_overlap)

        return chunks
