"""
End-to-End Document Ingestion Pipeline for FinSight AI
Coordinates file validation, text extraction, cleaning, structure-aware chunking,
embedding generation, duplicate detection, and persistent database storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.database import db
from core.exceptions import DocumentIngestionError, PromptInjectionDetected
from core.logging import logger
from core.security import check_prompt_injection, sanitize_text, validate_file_upload
from ingestion.chunker import DocumentChunk, StructureAwareChunker
from ingestion.cleaner import TextCleaner
from ingestion.loader import DocumentLoader
from ingestion.metadata import compute_file_hash, infer_metadata_from_filename_or_text
from retrieval.embeddings import embedding_service
from retrieval.vector_store import VectorStore


@dataclass
class IngestionResult:
    """Outcome of document ingestion."""
    document_id: int
    company_ticker: str
    title: str
    fiscal_year: int
    reporting_period: str
    total_pages: int
    chunks_created: int
    file_hash: str
    status: str
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "company_ticker": self.company_ticker,
            "title": self.title,
            "fiscal_year": self.fiscal_year,
            "reporting_period": self.reporting_period,
            "total_pages": self.total_pages,
            "chunks_created": self.chunks_created,
            "file_hash": self.file_hash,
            "status": self.status,
            "details": self.details,
        }


class IngestionPipeline:
    """Production ingestion workflow with validation, duplicate detection, and embedding."""

    def __init__(self, chunker: Optional[StructureAwareChunker] = None):
        self.chunker = chunker or StructureAwareChunker(target_chunk_size=400, chunk_overlap=50)

    def ingest_file(
        self,
        filepath: Path | str,
        company_ticker: Optional[str] = None,
        title: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        reporting_period: Optional[str] = None,
        document_type: Optional[str] = None,
        force_reindex: bool = False,
    ) -> IngestionResult:
        """
        Executes complete ingestion for a file:
        1. Validate size and extension.
        2. Compute SHA-256 file hash and detect duplicates.
        3. Extract text and tables (PDF, DOCX, TXT, etc.).
        4. Clean and normalize text.
        5. Structure-aware chunking.
        6. Compute embeddings.
        7. Save document and chunks to database.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"File does not exist: {filepath}")

        # 1. Validate file size and extension
        file_size = p.stat().st_size
        validate_file_upload(p.name, file_size)

        # 2. Compute SHA-256 hash
        file_hash = compute_file_hash(p)

        # Check for duplicates
        existing_doc = db.fetch_one(
            "SELECT id, title, total_pages FROM documents WHERE file_hash = ?",
            (file_hash,),
        )
        if existing_doc and not force_reindex:
            doc_id = existing_doc["id"]
            existing_chunks = db.fetch_all(
                "SELECT COUNT(*) as count FROM document_chunks WHERE document_id = ?",
                (doc_id,),
            )
            chunk_count = existing_chunks[0]["count"] if existing_chunks else 0
            logger.info(f"Duplicate document detected (file_hash={file_hash[:12]}...). Skipping ingestion.")
            return IngestionResult(
                document_id=doc_id,
                company_ticker=company_ticker or "UNKNOWN",
                title=existing_doc["title"],
                fiscal_year=fiscal_year or 0,
                reporting_period=reporting_period or "",
                total_pages=existing_doc["total_pages"],
                chunks_created=chunk_count,
                file_hash=file_hash,
                status="already_exists",
                details="Document with identical SHA-256 hash has already been ingested.",
            )

        # 3. Extract text and tables
        logger.info(f"Extracting content from: {p.name}")
        full_text, page_map, total_pages = DocumentLoader.load(p)

        # 4. Clean and sanitize text
        cleaned_text = TextCleaner.clean_text(full_text)
        if check_prompt_injection(cleaned_text):
            logger.warning(f"Adversarial prompt injection pattern flagged in document: {p.name}")

        # 5. Infer or assign metadata
        inferred = infer_metadata_from_filename_or_text(
            filename=p.name,
            text_sample=cleaned_text[:1000],
            default_ticker=company_ticker or "GENERIC",
        )
        final_ticker = (company_ticker or inferred.company_ticker).upper()
        final_title = title or inferred.title
        final_fy = fiscal_year or inferred.fiscal_year
        final_period = reporting_period or inferred.reporting_period
        final_type = document_type or inferred.document_type

        # Ensure company exists in database
        company_row = db.fetch_one("SELECT id FROM companies WHERE ticker = ?", (final_ticker,))
        if not company_row:
            with db.get_connection() as conn:
                if db.is_sqlite:
                    cursor = conn.execute(
                        """
                        INSERT INTO companies (ticker, name, exchange, reporting_currency)
                        VALUES (?, ?, 'NSE', 'INR') RETURNING id;
                        """,
                        (final_ticker, f"{final_ticker} Corp"),
                    )
                    company_id = cursor.fetchone()[0]
                else:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO companies (ticker, name, exchange, reporting_currency)
                        VALUES (%s, %s, 'NSE', 'INR') RETURNING id;
                        """,
                        (final_ticker, f"{final_ticker} Corp"),
                    )
                    company_id = cursor.fetchone()["id"]
        else:
            company_id = company_row["id"]

        # Insert or update document record
        with db.get_connection() as conn:
            if db.is_sqlite:
                cursor = conn.execute(
                    """
                    INSERT INTO documents (
                        company_id, title, document_type, fiscal_year, reporting_period,
                        file_hash, file_path, total_pages, ingestion_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed')
                    ON CONFLICT(file_hash) DO UPDATE SET
                        title=excluded.title,
                        total_pages=excluded.total_pages
                    RETURNING id;
                    """,
                    (
                        company_id,
                        final_title,
                        final_type,
                        final_fy,
                        final_period,
                        file_hash,
                        str(p),
                        total_pages,
                    ),
                )
                doc_id = cursor.fetchone()[0]
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO documents (
                        company_id, title, document_type, fiscal_year, reporting_period,
                        file_hash, file_path, total_pages, ingestion_status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'completed')
                    ON CONFLICT(file_hash) DO UPDATE SET
                        title=EXCLUDED.title,
                        total_pages=EXCLUDED.total_pages
                    RETURNING id;
                    """,
                    (
                        company_id,
                        final_title,
                        final_type,
                        final_fy,
                        final_period,
                        file_hash,
                        str(p),
                        total_pages,
                    ),
                )
                doc_id = cursor.fetchone()["id"]

        # If re-indexing, remove old chunks for this document
        if force_reindex and existing_doc:
            db.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))

        # 6. Structure-aware chunking
        doc_meta = {
            "company_ticker": final_ticker,
            "fiscal_year": final_fy,
            "reporting_period": final_period,
            "document_id": doc_id,
        }
        chunks = self.chunker.chunk_document(cleaned_text, page_map=page_map, doc_metadata=doc_meta)
        if not chunks:
            raise DocumentIngestionError(f"No valid chunks produced for document {p.name}")

        for c in chunks:
            c.document_id = doc_id

        # 7. Generate embeddings
        logger.info(f"Generating dense embeddings for {len(chunks)} chunks...")
        texts_to_embed = [c.clean_text for c in chunks]
        embeddings = embedding_service.embed_documents(texts_to_embed)

        # 8. Save chunks and embeddings
        saved_count = VectorStore.save_chunks(doc_id, chunks, embeddings)
        logger.info(f"Successfully ingested '{final_title}' ({saved_count} chunks indexed).")

        return IngestionResult(
            document_id=doc_id,
            company_ticker=final_ticker,
            title=final_title,
            fiscal_year=final_fy,
            reporting_period=final_period,
            total_pages=total_pages,
            chunks_created=saved_count,
            file_hash=file_hash,
            status="success",
            details="Ingestion and vector indexing completed successfully.",
        )


# Global ingestion pipeline instance
pipeline = IngestionPipeline()
