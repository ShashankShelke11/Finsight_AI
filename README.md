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
│   ├── test_database.py         # Schema and seeding unit tests
│   └── test_security.py         # Security & guardrail unit tests
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
git clone https://github.com/your-username/finsight-ai.git
cd finsight-ai

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

# 5. Run the Test Suite
python -m unittest discover -s tests -p "test_*.py"
```

### Production Setup (PostgreSQL + pgvector with Docker)

```bash
# 1. Start PostgreSQL with pgvector
docker compose up -d postgres

# 2. Update .env:
# DATABASE_URL=postgresql://finsight_user:finsight_secret@localhost:5432/finsight_db

# 3. Seed PostgreSQL database
python scripts/seed_sample_data.py
```

---

## 4. Key Financial Capabilities

- **Zero-Hallucination Math**: All ratios (Operating Margin, Net Margin, Current Ratio, Debt-to-Equity, FCF, ROE, ROIC) and YoY growth rates are evaluated deterministically in Python.
- **Traceable Provenance**: Every financial fact links to an exact filing name, fiscal year, reporting period, and page number.
- **Adversarial Hardening**: Untrusted annual report content is isolated from LLM system prompts; injection payloads are detected and neutralized.
- **Hybrid Retrieval**: Dense semantic similarity search combined with sparse keyword search via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
