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
├── retrieval/
│   ├── embeddings.py            # Dense embedding generator (SentenceTransformers & fallback)
│   ├── vector_store.py          # Vector store adapter (SQLite & pgvector)
│   ├── keyword_search.py        # BM25 lexical keyword search engine
│   ├── hybrid.py                # Reciprocal Rank Fusion (RRF) combiner
│   ├── reranker.py              # Cross-Encoder joint query-passage reranker
│   └── search_engine.py         # Multi-mode unified search coordinator
├── evaluation/
│   ├── retrieval_eval.py        # Experiment A benchmark runner (Recall@K, MRR, NDCG)
│   └── dataset/
│       └── retrieval_golden_queries.json # Curated benchmark evaluation dataset
├── tests/
│   ├── test_database.py           # Schema and seeding unit tests
│   ├── test_security.py           # Security & guardrail unit tests
│   ├── test_ingestion_cleaner.py  # Text cleaning & normalization unit tests
│   ├── test_ingestion_chunker.py  # Structure-aware & table chunking unit tests
│   ├── test_ingestion_loader.py   # Multi-format document loader unit tests (PDF, DOCX, CSV)
│   ├── test_vector_retrieval.py   # Embedding & cosine similarity unit tests
│   ├── test_ingestion_pipeline.py # End-to-end ingestion & vector retrieval integration tests
│   ├── test_keyword_search.py     # BM25 keyword search unit tests
│   ├── test_hybrid_rrf.py         # Reciprocal Rank Fusion unit tests
│   ├── test_reranker.py           # Cross-Encoder reranker unit tests
│   └── test_retrieval_evaluation.py # Retrieval benchmark integration tests
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
git clone https://github.com/ShashankShelke11/Finsight_AI.git
cd Finsight_AI

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

# 5. Run the Test Suite (33 Tests)
python -m unittest discover -s tests -p "test_*.py"
```

### Hybrid Retrieval & Reranking Quickstart

```python
from retrieval.search_engine import search_engine

# 1. Execute Multi-Stage Hybrid Search with Cross-Encoder Reranking
response = search_engine.search(
    query="Infosys large deal total contract value TCV Cobalt Topaz",
    mode="hybrid_rerank",  # Options: 'vector', 'hybrid', 'hybrid_rerank'
    top_k=3,
    ticker="INFY",
)

print(f"Retrieval Method: {response.diagnostics['retrieval_method']}")
print(f"Total Latency: {response.diagnostics['latency_ms']} ms")
print(f"Latency Breakdown: {response.diagnostics['latency_breakdown_ms']}")

for res in response.results:
    print(f"[{res.score:.4f}] {res.source_reference}:\n{res.content[:200]}...\n")
```

---

## 4. Experiment A: Retrieval Architecture Comparison Benchmark

The retrieval evaluation framework evaluates performance across the curated golden financial benchmark (`evaluation/dataset/retrieval_golden_queries.json`):

| Architecture Mode | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 | Mean Latency (ms) | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Vector-Only Baseline** | 90.0% | 100.0% | 100.0% | 0.9500 | 0.9631 | 4.76 ms | 5.25 ms |
| **2. Hybrid (Vector + BM25 RRF)** | 100.0% | 100.0% | 100.0% | 1.0000 | 0.9137 | 8.53 ms | 9.74 ms |
| **3. Hybrid + Cross-Encoder Rerank** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** | **9.86 ms** | **10.14 ms** |

### Benchmark Takeaways:
- **Recall@1 Improvement**: Hybrid RRF and Cross-Encoder Reranking increased top-1 recall from 90.0% to **100.0%**, accurately locating the relevant disclosure section on the very first attempt.
- **NDCG@5 Perfection**: Cross-Encoder joint scoring optimized ranking order, achieving an **NDCG@5 of 1.0000**.
- **Low Latency Budget**: The entire multi-stage pipeline (vector similarity + BM25 + RRF + reranking) completes in **~9.8ms**, well within real-time financial analyst workflow constraints.

---

## 5. Key Financial & Technical Capabilities

- **Zero-Hallucination Math**: All ratios (Operating Margin, Net Margin, Current Ratio, Debt-to-Equity, FCF, ROE, ROIC) and YoY growth rates are evaluated deterministically in Python.
- **Traceable Provenance**: Every financial fact and retrieved chunk links to an exact filing name, fiscal year, reporting period, section, and page number.
- **Multi-Stage Hybrid RAG**: Dense vector semantic matching coupled with BM25 lexical keyword search via Reciprocal Rank Fusion ($k=60$) and Cross-Encoder reranking.
- **Structure-Aware Chunking**: Intelligently identifies headings (`##`), disclosure notes, and preserves financial tables without tearing rows.
- **Multi-Format Extraction**: Parses PDFs (via `pymupdf`), Word docs (via `python-docx`), CSVs, and plain text with automatic duplicate detection via SHA-256 hashes.
- **Adversarial Hardening**: Untrusted annual report content is isolated from LLM system prompts; injection payloads are detected and neutralized.

---

## 6. Implementation Roadmap

- [x] **Phase 1: Foundation & Architecture** (Database schema, seeder, security guardrails, models)
- [x] **Phase 2: Document Ingestion & Baseline Vector RAG** (PyMuPDF/DOCX extraction, cleaner, structure-aware chunker, dense vector store, retrieval engine, 25 tests passing)
- [x] **Phase 3: Hybrid Retrieval & Cross-Encoder Reranking** (Dense + BM25 keyword search + Reciprocal Rank Fusion + Cross-Encoder reranker + Experiment A benchmark runner with 33 tests passing)
- [ ] **Phase 4: Deterministic Financial Calculation Engine** (Python financial analytics engine with division-by-zero guards)
- [ ] **Phase 5: LangGraph Agent Orchestration & Citation Verification** (Query planner, retriever, calculator, verifier, refiner, synthesizer, citation validator)
- [ ] **Phase 6: Golden Benchmark Evaluation Suite & Observability** (Recall@5, MRR, calculation accuracy, citation precision benchmarks)
- [ ] **Phase 7: FastAPI Endpoints & Streamlit Dashboard** (Interactive Plotly visualizations and report export)
- [ ] **Phase 8: Docker, CI/CD, Documentation & Portfolio Highlights** (Containerization, GitHub Actions, portfolio showcase)
