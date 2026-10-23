"""
Universal Database Access Layer for FinSight AI
Supports SQLite (zero-dependency local development & testing)
and PostgreSQL + pgvector (production & Docker deployment).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional
from core.config import settings
from core.logging import logger


class DatabaseManager:
    """Manages database connections, table creation, and execution."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.is_sqlite = self.database_url.startswith("sqlite")

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Provides a managed database connection context."""
        if self.is_sqlite:
            # Extract file path from sqlite:///path/to/db
            raw_path = self.database_url.replace("sqlite:///", "")
            db_path = Path(raw_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for SQLite
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            # PostgreSQL production connection
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
            except ImportError:
                raise RuntimeError(
                    "psycopg2 is required for PostgreSQL. Please install psycopg2-binary or use SQLite."
                )

    def init_db(self) -> None:
        """Initializes database schema. Uses SQLite DDL if running SQLite, or PostgreSQL DDL."""
        logger.info(f"Initializing database schema on: {self.database_url}")
        if self.is_sqlite:
            self._init_sqlite()
        else:
            self._init_postgres()

    def _init_sqlite(self) -> None:
        """Creates tables in SQLite with equivalent structure to PostgreSQL schema."""
        sqlite_ddl = """
        -- 1. Companies Table
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            exchange TEXT NOT NULL,
            sector TEXT,
            industry TEXT,
            reporting_currency TEXT NOT NULL DEFAULT 'INR',
            country TEXT NOT NULL DEFAULT 'India',
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);

        -- 2. Documents Table
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            document_type TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            reporting_period TEXT NOT NULL,
            publication_date TEXT,
            source_url TEXT,
            file_hash TEXT UNIQUE NOT NULL,
            file_path TEXT,
            total_pages INTEGER DEFAULT 1,
            ingestion_status TEXT DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_docs_company_period ON documents(company_id, fiscal_year, reporting_period);
        CREATE INDEX IF NOT EXISTS idx_docs_file_hash ON documents(file_hash);

        -- 3. Document Chunks (Dense Vector + Clean Text)
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            section_name TEXT,
            content TEXT NOT NULL,
            clean_text TEXT NOT NULL,
            is_table INTEGER DEFAULT 0,
            table_data_json TEXT,
            token_count INTEGER DEFAULT 0,
            embedding_json TEXT, -- JSON array of floats for cosine similarity
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_page ON document_chunks(document_id, page_number);

        -- 4. Structured Financial Facts Table (Auditable ground truth)
        CREATE TABLE IF NOT EXISTS financial_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            fiscal_year INTEGER NOT NULL,
            reporting_period TEXT NOT NULL,
            statement_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_display_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            currency TEXT NOT NULL,
            is_consolidated INTEGER DEFAULT 1,
            source_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
            source_page INTEGER,
            source_reference TEXT,
            audit_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(company_id, fiscal_year, reporting_period, metric_name, is_consolidated)
        );

        CREATE INDEX IF NOT EXISTS idx_facts_lookup ON financial_facts(company_id, fiscal_year, metric_name);
        CREATE INDEX IF NOT EXISTS idx_facts_statement ON financial_facts(statement_type, metric_name);

        -- 5. Research Queries and Sessions Audit Trail
        CREATE TABLE IF NOT EXISTS research_queries (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_query TEXT NOT NULL,
            query_type TEXT,
            plan_json TEXT,
            synthesized_response TEXT,
            calculated_metrics_json TEXT,
            citations_json TEXT,
            retrieval_diagnostics_json TEXT,
            status TEXT DEFAULT 'completed',
            execution_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_queries_created_at ON research_queries(created_at DESC);
        """
        with self.get_connection() as conn:
            conn.executescript(sqlite_ddl)
        logger.info("SQLite schema initialized successfully.")

    def _init_postgres(self) -> None:
        """Executes the PostgreSQL init_db.sql schema."""
        init_sql_path = settings.BASE_DIR / "infra" / "init_db.sql"
        if not init_sql_path.exists():
            raise FileNotFoundError(f"Missing PostgreSQL schema file at: {init_sql_path}")
        with open(init_sql_path, "r", encoding="utf-8") as f:
            ddl = f.read()
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        logger.info("PostgreSQL schema initialized successfully.")

    def execute(self, sql: str, params: tuple | dict = ()) -> None:
        """Executes a SQL query with parameters."""
        with self.get_connection() as conn:
            conn.execute(sql, params)

    def fetch_all(self, sql: str, params: tuple | dict = ()) -> List[dict[str, Any]]:
        """Executes a query and returns all matching rows as dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            if self.is_sqlite:
                columns = [col[0] for col in cursor.description] if cursor.description else []
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                return cursor.fetchall()

    def fetch_one(self, sql: str, params: tuple | dict = ()) -> Optional[dict[str, Any]]:
        """Executes a query and returns a single row as a dictionary."""
        rows = self.fetch_all(sql, params)
        return rows[0] if rows else None


# Global database manager instance
db = DatabaseManager()
