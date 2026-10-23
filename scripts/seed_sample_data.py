"""
Database Seeding Script for FinSight AI
Populates the database with audited financial facts, company metadata,
documents, and sample report chunks for Infosys and TCS (FY22, FY23, FY24).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from core.database import db
from core.logging import logger
from core.security import sanitize_text


def seed_database() -> None:
    """Initializes tables and populates sample data."""
    logger.info("Initializing schema...")
    db.init_db()

    statements_path = settings.SAMPLE_DATA_DIR / "statements.json"
    if not statements_path.exists():
        raise FileNotFoundError(f"Sample statements file not found at: {statements_path}")

    with open(statements_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Seed Companies
    companies_map: dict[str, int] = {}
    with db.get_connection() as conn:
        for comp in data.get("companies", []):
            if db.is_sqlite:
                cursor = conn.execute(
                    """
                    INSERT INTO companies (ticker, name, exchange, sector, industry, reporting_currency, country, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        name=excluded.name,
                        exchange=excluded.exchange,
                        sector=excluded.sector,
                        industry=excluded.industry,
                        description=excluded.description
                    RETURNING id;
                    """,
                    (
                        comp["ticker"],
                        comp["name"],
                        comp["exchange"],
                        comp.get("sector"),
                        comp.get("industry"),
                        comp.get("reporting_currency", "INR"),
                        comp.get("country", "India"),
                        comp.get("description"),
                    ),
                )
                row = cursor.fetchone()
                companies_map[comp["ticker"]] = row[0]
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO companies (ticker, name, exchange, sector, industry, reporting_currency, country, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(ticker) DO UPDATE SET
                        name=EXCLUDED.name,
                        exchange=EXCLUDED.exchange,
                        sector=EXCLUDED.sector,
                        industry=EXCLUDED.industry,
                        description=EXCLUDED.description
                    RETURNING id;
                    """,
                    (
                        comp["ticker"],
                        comp["name"],
                        comp["exchange"],
                        comp.get("sector"),
                        comp.get("industry"),
                        comp.get("reporting_currency", "INR"),
                        comp.get("country", "India"),
                        comp.get("description"),
                    ),
                )
                row = cursor.fetchone()
                companies_map[comp["ticker"]] = row["id"]

    logger.info(f"Loaded {len(companies_map)} companies: {list(companies_map.keys())}")

    # 2. Seed Documents
    docs_map: dict[str, int] = {}  # title -> doc_id
    with db.get_connection() as conn:
        for doc in data.get("documents", []):
            ticker = doc["company_ticker"]
            company_id = companies_map[ticker]
            if db.is_sqlite:
                cursor = conn.execute(
                    """
                    INSERT INTO documents (company_id, title, document_type, fiscal_year, reporting_period, publication_date, source_url, file_hash, total_pages)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_hash) DO UPDATE SET
                        title=excluded.title,
                        total_pages=excluded.total_pages
                    RETURNING id;
                    """,
                    (
                        company_id,
                        doc["title"],
                        doc["document_type"],
                        doc["fiscal_year"],
                        doc["reporting_period"],
                        doc.get("publication_date"),
                        doc.get("source_url"),
                        doc["file_hash"],
                        doc.get("total_pages", 1),
                    ),
                )
                row = cursor.fetchone()
                docs_map[doc["title"]] = row[0]
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO documents (company_id, title, document_type, fiscal_year, reporting_period, publication_date, source_url, file_hash, total_pages)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(file_hash) DO UPDATE SET
                        title=EXCLUDED.title,
                        total_pages=EXCLUDED.total_pages
                    RETURNING id;
                    """,
                    (
                        company_id,
                        doc["title"],
                        doc["document_type"],
                        doc["fiscal_year"],
                        doc["reporting_period"],
                        doc.get("publication_date"),
                        doc.get("source_url"),
                        doc["file_hash"],
                        doc.get("total_pages", 1),
                    ),
                )
                row = cursor.fetchone()
                docs_map[doc["title"]] = row["id"]

    logger.info(f"Loaded {len(docs_map)} document references.")

    # 3. Seed Financial Facts
    facts_inserted = 0
    with db.get_connection() as conn:
        for fact in data.get("facts", []):
            company_id = companies_map[fact["company_ticker"]]
            doc_id = docs_map.get(fact.get("source_document_title"))
            raw_hash = f"{company_id}_{fact['fiscal_year']}_{fact['reporting_period']}_{fact['metric_name']}"
            audit_hash = hashlib.sha256(raw_hash.encode()).hexdigest()[:16]

            if db.is_sqlite:
                conn.execute(
                    """
                    INSERT INTO financial_facts (
                        company_id, fiscal_year, reporting_period, statement_type,
                        metric_name, metric_display_name, value, unit, currency,
                        is_consolidated, source_document_id, source_page, source_reference, audit_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(company_id, fiscal_year, reporting_period, metric_name, is_consolidated) DO UPDATE SET
                        value=excluded.value,
                        unit=excluded.unit,
                        currency=excluded.currency,
                        source_reference=excluded.source_reference,
                        source_page=excluded.source_page;
                    """,
                    (
                        company_id,
                        fact["fiscal_year"],
                        fact["reporting_period"],
                        fact["statement_type"],
                        fact["metric_name"],
                        fact["metric_display_name"],
                        fact["value"],
                        fact["unit"],
                        fact["currency"],
                        1 if fact.get("is_consolidated", True) else 0,
                        doc_id,
                        fact.get("source_page"),
                        fact.get("source_reference"),
                        audit_hash,
                    ),
                )
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO financial_facts (
                        company_id, fiscal_year, reporting_period, statement_type,
                        metric_name, metric_display_name, value, unit, currency,
                        is_consolidated, source_document_id, source_page, source_reference, audit_hash
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(company_id, fiscal_year, reporting_period, metric_name, is_consolidated) DO UPDATE SET
                        value=EXCLUDED.value,
                        unit=EXCLUDED.unit,
                        currency=EXCLUDED.currency,
                        source_reference=EXCLUDED.source_reference,
                        source_page=EXCLUDED.source_page;
                    """,
                    (
                        company_id,
                        fact["fiscal_year"],
                        fact["reporting_period"],
                        fact["statement_type"],
                        fact["metric_name"],
                        fact["metric_display_name"],
                        fact["value"],
                        fact["unit"],
                        fact["currency"],
                        fact.get("is_consolidated", True),
                        doc_id,
                        fact.get("source_page"),
                        fact.get("source_reference"),
                        audit_hash,
                    ),
                )
            facts_inserted += 1

    logger.info(f"Loaded {facts_inserted} audited financial facts across FY22-FY24.")

    # 4. Seed Report Excerpt Chunks
    reports_dir = settings.SAMPLE_DATA_DIR / "reports"
    chunks_seeded = 0
    report_files = [
        ("INFY", "Infosys Integrated Annual Report 2023-24", reports_dir / "infy_fy2024_excerpt.txt"),
        ("TCS", "TCS Annual Report 2023-24", reports_dir / "tcs_fy2024_excerpt.txt"),
    ]

    for ticker, doc_title, filepath in report_files:
        if not filepath.exists():
            continue
        doc_id = docs_map.get(doc_title)
        if not doc_id:
            continue

        with open(filepath, "r", encoding="utf-8") as rf:
            content = rf.read()

        # Split into logical sections by Markdown headings
        sections = re.split(r"\n(?=###?\s+)", content)
        with db.get_connection() as conn:
            for idx, sec in enumerate(sections):
                sec = sec.strip()
                if not sec:
                    continue
                # Determine section name and page
                header_match = re.match(r"^###?\s+([^\n]+)", sec)
                section_name = header_match.group(1).strip() if header_match else "Overview"
                page_match = re.search(r"\(Page\s+(\d+)\)", section_name)
                page_number = int(page_match.group(1)) if page_match else 1
                clean_sec = sanitize_text(sec)

                # Store embedding placeholder or dummy normalized vector (384 dims)
                embedding_json = json.dumps([0.0] * settings.EMBEDDING_DIMENSION)

                if db.is_sqlite:
                    conn.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id, chunk_index, page_number, section_name,
                            content, clean_text, is_table, token_count, embedding_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            doc_id,
                            idx,
                            page_number,
                            section_name,
                            sec,
                            clean_sec,
                            0,
                            len(clean_sec.split()),
                            embedding_json,
                        ),
                    )
                else:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id, chunk_index, page_number, section_name,
                            content, clean_text, is_table, token_count
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            doc_id,
                            idx,
                            page_number,
                            section_name,
                            sec,
                            clean_sec,
                            False,
                            len(clean_sec.split()),
                        ),
                    )
                chunks_seeded += 1

    logger.info(f"Loaded {chunks_seeded} structured report chunks.")
    logger.info("Database seeding completed successfully!")


if __name__ == "__main__":
    seed_database()
