"""
Vector Store Interface for FinSight AI
Supports pgvector in PostgreSQL and high-performance in-memory cosine similarity over SQLite.
Enforces metadata filtering (company ticker, fiscal year, reporting period).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from core.database import db
from core.logging import logger

if TYPE_CHECKING:
    from ingestion.chunker import DocumentChunk


@dataclass
class SearchResult:
    """Retrieved document chunk with provenance metadata and similarity score."""
    chunk_id: int
    document_id: int
    company_ticker: str
    document_title: str
    fiscal_year: int
    reporting_period: str
    page_number: int
    section_name: str
    content: str
    clean_text: str
    score: float  # Cosine similarity (0.0 to 1.0)
    is_table: bool = False
    source_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "company_ticker": self.company_ticker,
            "document_title": self.document_title,
            "fiscal_year": self.fiscal_year,
            "reporting_period": self.reporting_period,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "content": self.content,
            "score": round(self.score, 4),
            "is_table": self.is_table,
            "source_reference": self.source_reference,
        }


class VectorStore:
    """Vector database access layer supporting both SQLite and PostgreSQL pgvector."""

    @classmethod
    def save_chunks(
        cls,
        document_id: int,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Saves document chunks along with their dense embedding vectors."""
        if not chunks:
            return 0

        saved_count = 0
        with db.get_connection() as conn:
            for chunk, emb in zip(chunks, embeddings):
                if db.is_sqlite:
                    emb_json = json.dumps(emb)
                    conn.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id, chunk_index, page_number, section_name,
                            content, clean_text, is_table, table_data_json,
                            token_count, embedding_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            document_id,
                            chunk.chunk_index,
                            chunk.page_number,
                            chunk.section_name,
                            chunk.content,
                            chunk.clean_text,
                            1 if chunk.is_table else 0,
                            chunk.table_data_json,
                            chunk.token_count,
                            emb_json,
                        ),
                    )
                else:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id, chunk_index, page_number, section_name,
                            content, clean_text, is_table, table_data_json,
                            token_count, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            document_id,
                            chunk.chunk_index,
                            chunk.page_number,
                            chunk.section_name,
                            chunk.content,
                            chunk.clean_text,
                            chunk.is_table,
                            chunk.table_data_json,
                            chunk.token_count,
                            emb,
                        ),
                    )
                saved_count += 1

        logger.info(f"Saved {saved_count} chunks with embeddings for document_id={document_id}")
        return saved_count

    @classmethod
    def search_similar(
        cls,
        query_vector: List[float],
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        reporting_period: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Retrieves top_k document chunks closest to query_vector using cosine similarity.
        Applies metadata filters when provided.
        """
        if db.is_sqlite:
            return cls._search_sqlite(query_vector, top_k, ticker, fiscal_year, reporting_period)
        else:
            return cls._search_postgres(query_vector, top_k, ticker, fiscal_year, reporting_period)

    @classmethod
    def _search_sqlite(
        cls,
        query_vector: List[float],
        top_k: int,
        ticker: Optional[str],
        fiscal_year: Optional[int],
        reporting_period: Optional[str],
    ) -> List[SearchResult]:
        """In-memory cosine similarity calculation over SQLite stored embeddings."""
        query_sql = """
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                d.fiscal_year,
                d.reporting_period,
                comp.ticker AS company_ticker,
                c.page_number,
                c.section_name,
                c.content,
                c.clean_text,
                c.is_table,
                c.embedding_json
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN companies comp ON d.company_id = comp.id
            WHERE 1=1
        """
        params: List[Any] = []
        if ticker:
            query_sql += " AND comp.ticker = ?"
            params.append(ticker.upper())
        if fiscal_year:
            query_sql += " AND d.fiscal_year = ?"
            params.append(fiscal_year)
        if reporting_period:
            query_sql += " AND d.reporting_period = ?"
            params.append(reporting_period)

        rows = db.fetch_all(query_sql, tuple(params))
        scored_results: List[SearchResult] = []

        for row in rows:
            emb_json = row.get("embedding_json")
            if not emb_json:
                continue
            emb = json.loads(emb_json)

            # Dot product of unit-normalized vectors equals cosine similarity
            score = sum(q * v for q, v in zip(query_vector, emb))

            ref = f"{row['company_ticker']} {row['reporting_period']}, Page {row['page_number']} ({row['section_name']})"
            scored_results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    company_ticker=row["company_ticker"],
                    document_title=row["document_title"],
                    fiscal_year=row["fiscal_year"],
                    reporting_period=row["reporting_period"],
                    page_number=row["page_number"],
                    section_name=row["section_name"] or "Overview",
                    content=row["content"],
                    clean_text=row["clean_text"],
                    score=score,
                    is_table=bool(row["is_table"]),
                    source_reference=ref,
                )
            )

        # Sort descending by score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    @classmethod
    def _search_postgres(
        cls,
        query_vector: List[float],
        top_k: int,
        ticker: Optional[str],
        fiscal_year: Optional[int],
        reporting_period: Optional[str],
    ) -> List[SearchResult]:
        """pgvector HNSW cosine distance search."""
        query_sql = """
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                d.fiscal_year,
                d.reporting_period,
                comp.ticker AS company_ticker,
                c.page_number,
                c.section_name,
                c.content,
                c.clean_text,
                c.is_table,
                1 - (c.embedding <=> %s::vector) AS score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN companies comp ON d.company_id = comp.id
            WHERE 1=1
        """
        params: List[Any] = [query_vector]
        if ticker:
            query_sql += " AND comp.ticker = %s"
            params.append(ticker.upper())
        if fiscal_year:
            query_sql += " AND d.fiscal_year = %s"
            params.append(fiscal_year)
        if reporting_period:
            query_sql += " AND d.reporting_period = %s"
            params.append(reporting_period)

        query_sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        params.extend([query_vector, top_k])

        rows = db.fetch_all(query_sql, tuple(params))
        results: List[SearchResult] = []
        for row in rows:
            ref = f"{row['company_ticker']} {row['reporting_period']}, Page {row['page_number']} ({row['section_name']})"
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    company_ticker=row["company_ticker"],
                    document_title=row["document_title"],
                    fiscal_year=row["fiscal_year"],
                    reporting_period=row["reporting_period"],
                    page_number=row["page_number"],
                    section_name=row["section_name"] or "Overview",
                    content=row["content"],
                    clean_text=row["clean_text"],
                    score=float(row["score"]),
                    is_table=bool(row["is_table"]),
                    source_reference=ref,
                )
            )
        return results
