-- ==============================================================================
-- FinSight AI: PostgreSQL + pgvector Database Initialization Schema
-- ==============================================================================

-- Enable pgvector extension for dense embedding storage and HNSW indexing
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID extension if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Companies Table
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    sector VARCHAR(128),
    industry VARCHAR(128),
    reporting_currency VARCHAR(8) NOT NULL DEFAULT 'INR',
    country VARCHAR(64) NOT NULL DEFAULT 'India',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);

-- 2. Documents Table (Annual Reports, Statements, Transcripts)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    document_type VARCHAR(64) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    reporting_period VARCHAR(32) NOT NULL,
    publication_date DATE,
    source_url TEXT,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_path TEXT,
    total_pages INTEGER DEFAULT 1,
    ingestion_status VARCHAR(32) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_docs_company_period ON documents(company_id, fiscal_year, reporting_period);
CREATE INDEX IF NOT EXISTS idx_docs_file_hash ON documents(file_hash);

-- 3. Document Chunks (Hybrid Retrieval with Dense Vectors + Full-Text TSVector)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    section_name VARCHAR(128),
    content TEXT NOT NULL,
    clean_text TEXT NOT NULL,
    is_table BOOLEAN DEFAULT FALSE,
    table_data_json JSONB,
    token_count INTEGER DEFAULT 0,
    embedding vector(384),
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', clean_text)) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON document_chunks(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN(tsv_content);

-- HNSW Vector Index for Cosine Similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Structured Financial Facts Table (Ground-Truth Auditable Metrics)
CREATE TABLE IF NOT EXISTS financial_facts (
    id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    fiscal_year INTEGER NOT NULL,
    reporting_period VARCHAR(32) NOT NULL,
    statement_type VARCHAR(64) NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    metric_display_name VARCHAR(255) NOT NULL,
    value NUMERIC(20, 4) NOT NULL,
    unit VARCHAR(32) NOT NULL,
    currency VARCHAR(8) NOT NULL,
    is_consolidated BOOLEAN DEFAULT TRUE,
    source_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    source_page INTEGER,
    source_reference TEXT,
    audit_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_company_period_metric UNIQUE(company_id, fiscal_year, reporting_period, metric_name, is_consolidated)
);

CREATE INDEX IF NOT EXISTS idx_facts_lookup ON financial_facts(company_id, fiscal_year, metric_name);
CREATE INDEX IF NOT EXISTS idx_facts_statement ON financial_facts(statement_type, metric_name);

-- 5. Research Sessions and Query Audit Trail
CREATE TABLE IF NOT EXISTS research_queries (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64),
    user_query TEXT NOT NULL,
    query_type VARCHAR(64),
    plan_json JSONB,
    synthesized_response TEXT,
    calculated_metrics_json JSONB,
    citations_json JSONB,
    retrieval_diagnostics_json JSONB,
    status VARCHAR(32) DEFAULT 'completed',
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queries_session ON research_queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_created_at ON research_queries(created_at DESC);
