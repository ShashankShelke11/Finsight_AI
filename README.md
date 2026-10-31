# FinSight AI — Agentic Financial Research & Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Agentic RAG](https://img.shields.io/badge/Architecture-Agentic%20RAG-orange.svg)](https://langchain-ai.github.io/langgraph/)

FinSight AI is a production-oriented, evidence-grounded financial intelligence and research platform. It analyzes public-company annual reports, quarterly filings, earnings transcripts, and historical statements to generate verifiable, citation-backed equity research reports.

The platform guarantees **deterministic financial calculations**, eliminating LLM arithmetic hallucinations and enforcing strict source citation validation.

---

## 1. System Architecture

```mermaid
flowchart TD
    User([Financial Analyst / User]) --> UI[Streamlit & Plotly Dashboard]
    UI --> API[FastAPI Backend /api/v1]
    
    subgraph Core Engine [FinSight AI Platform]
        API --> Guardrails[Security & Prompt Injection Guardrails]
        Guardrails --> LangGraph[LangGraph Agentic Orchestrator]
        
        subgraph Workflow Nodes
            Planner[1. Query Planner] --> Retriever[2. Evidence Retriever]
            Retriever --> Verifier[4. Evidence Verifier]
            Verifier -- Insufficient Evidence --> Refiner[5. Query Refinement (Max 2)]
            Refiner --> Retriever
            Verifier -- Grounded Evidence --> CalcTool[3. Financial Calculation Tool]
            CalcTool --> Synthesizer[6. Research Synthesizer]
            Synthesizer --> CitValidator[7. Citation Validator]
            CitValidator --> FinalResp[8. Final Verified Response]
        end
        
        CalcTool --> DeterministicEngine[Deterministic Python Financial Analytics]
        Retriever --> HybridSearch[Hybrid Retrieval Engine]
        
        subgraph Storage Layer
            HybridSearch --> VectorDB[(pgvector Dense Vectors)]
            HybridSearch --> FTS[(PostgreSQL Full-Text Search / FTS)]
            HybridSearch --> Reranker[Cross-Encoder Reranker]
            CalcTool --> FactsDB[(Structured Audited Facts Store)]
        end
    end
    
    FinalResp --> API
    API --> UI
```

---

## 2. Repository Layout

```
finsight-ai/
├── api/
│   ├── app.py                   # FastAPI application factory & routes
│   ├── dependencies.py          # Database & auth dependency injection
│   └── routers/
│       ├── health.py            # Health and readiness probes
│       ├── analytics.py         # Deterministic financial math endpoints
│       ├── research.py          # Agentic research & streaming query endpoints
│       └── documents.py         # Document ingestion & management
├── ingestion/
│   ├── loader.py                # PyMuPDF & docx document extractors
│   ├── cleaner.py               # Text normalization and whitespace sanitizer
│   ├── chunker.py               # Structure-aware and table-preserving chunking
│   └── metadata.py              # Filing metadata and reporting period parser
├── retrieval/
│   ├── hybrid.py                # Reciprocal Rank Fusion (RRF) hybrid search
│   ├── reranker.py              # Cross-encoder semantic reranking
│   ├── vector_store.py          # pgvector & local vector store adapter
│   └── search_engine.py         # Search engine coordinator with metadata filters
├── finance/
│   ├── calculators.py          # Deterministic financial ratio & growth functions
│   ├── models.py                # Financial statement schemas & Fact types
│   ├── normalizer.py            # Currency, units, and period alignment
│   └── validator.py             # Accounting comparability checks
├── agents/
│   ├── state.py                 # ResearchState TypedDict schema
│   ├── graph.py                 # LangGraph orchestration graph
│   └── nodes/                   # Specialized LangGraph nodes (Planner, Calc, etc.)
├── core/
│   ├── config.py                # Settings & environment configuration
│   ├── database.py              # Universal DB manager (SQLite & PostgreSQL)
│   ├── exceptions.py            # Domain-specific exceptions
│   ├── logging.py               # Structured JSON & console logger
│   └── security.py              # Prompt-injection sanitizer & input validation
├── data/
│   └── sample/
│       ├── statements.json      # Audited 3-year financial facts for INFY & TCS
│       └── reports/             # Sample filing excerpts with commentary
├── infra/
│   ├── init_db.sql              # Production PostgreSQL + pgvector DDL
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   └── seed_sample_data.py      # Database population CLI script
├── tests/
├── tests/
│   ├── test_database.py           # Schema and seeding unit tests
│   ├── test_security.py           # Security & guardrail unit tests
│   ├── test_ingestion_cleaner.py  # Text cleaning & normalization unit tests
│   ├── test_ingestion_chunker.py  # Structure-aware & table chunking unit tests
│   ├── test_ingestion_loader.py   # Multi-format document loader unit tests (PDF, DOCX, CSV)
│   ├── test_vector_retrieval.py   # Embedding & cosine similarity unit tests
│   └── test_ingestion_pipeline.py # End-to-end ingestion & vector retrieval integration tests
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Environment Setup & Quickstart

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- (Optional) Docker & Docker Compose for PostgreSQL + pgvector

### Local Setup (Zero-Dependency SQLite Mode)

```bash
# 1. Clone the repository
git clone https://github.com/ShashankShelke11/FinSight_AI.git
cd FinSight_AI

# 2. Set up virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\Activate.ps1

# 3. Create .env configuration
cp .env.example .env

# 4. Initialize and Seed the Database with Sample Data
python scripts/seed_sample_data.py

# 5. Run the Test Suite (25 Tests)
python -m unittest discover -s tests -p "test_*.py"
```

### Document Ingestion & Vector Retrieval Quickstart

```python
from ingestion.pipeline import pipeline
from retrieval.search_engine import search_engine

# 1. Ingest an annual report (PDF, DOCX, TXT, CSV)
result = pipeline.ingest_file(
    filepath="data/sample/reports/infy_fy2024_excerpt.txt",
    company_ticker="INFY",
    fiscal_year=2024,
    reporting_period="FY2024",
)
print(f"Ingested {result.title}: {result.chunks_created} chunks indexed.")

# 2. Execute Baseline Vector Retrieval
response = search_engine.search(
    query="large deal total contract value TCV Cobalt Topaz",
    top_k=3,
    ticker="INFY",
)
for res in response.results:
    print(f"[{res.score:.4f}] {res.source_reference}:\n{res.content[:200]}...\n")
```

---

## 4. Key Financial & Technical Capabilities

- **Zero-Hallucination Math**: All ratios (Operating Margin, Net Margin, Current Ratio, Debt-to-Equity, FCF, ROE, ROIC) and YoY growth rates are evaluated deterministically in Python.
- **Traceable Provenance**: Every financial fact and retrieved chunk links to an exact filing name, fiscal year, reporting period, section, and page number.
- **Structure-Aware Chunking**: Intelligently identifies headings (`##`), disclosure notes, and preserves financial tables without tearing rows.
- **Multi-Format Extraction**: Parses PDFs (via `pymupdf`), Word docs (via `python-docx`), CSVs, and plain text with automatic duplicate detection via SHA-256 hashes.
- **Dense Vector Search**: Generates 384-dimensional unit-normalized embeddings with cosine similarity querying and metadata filtering.
- **Adversarial Hardening**: Untrusted annual report content is isolated from LLM system prompts; injection payloads are detected and neutralized.

---

## 5. Implementation Roadmap

- [x] **Phase 1: Foundation & Architecture** (Database schema, seeder, security guardrails, models)
- [x] **Phase 2: Document Ingestion & Baseline Vector RAG** (PyMuPDF/DOCX extraction, cleaner, structure-aware chunker, dense vector store, retrieval engine, 25 tests passing)
- [ ] **Phase 3: Hybrid Retrieval & Cross-Encoder Reranking** (Dense + FTS keyword search + Reciprocal Rank Fusion + Cross-Encoder reranker)
- [ ] **Phase 4: Deterministic Financial Calculation Engine** (Python financial analytics engine with division-by-zero guards)
- [ ] **Phase 5: LangGraph Agent Orchestration & Citation Verification** (Query planner, retriever, calculator, verifier, refiner, synthesizer, citation validator)
- [ ] **Phase 6: Golden Benchmark Evaluation Suite & Observability** (Recall@5, MRR, calculation accuracy, citation precision benchmarks)
- [ ] **Phase 7: FastAPI Endpoints & Streamlit Dashboard** (Interactive Plotly visualizations and report export)
- [ ] **Phase 8: Docker, CI/CD, Documentation & Portfolio Highlights** (Containerization, GitHub Actions, portfolio showcase)
