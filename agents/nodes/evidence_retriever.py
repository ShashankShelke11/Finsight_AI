"""
Evidence Retriever Node for FinSight AI
Executes multi-query hybrid search (BM25 + Dense RRF + Cross-Encoder Rerank)
and retrieves audited financial statements from the database.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set
from agents.state import FinancialResearchState
from core.logging import get_logger
from finance.analytics import FinancialAnalyticsEngine
from finance.models import FactRecord
from retrieval.search_engine import SearchResult, search_engine

logger = get_logger("agents.evidence_retriever")


class EvidenceRetrieverNode:
    """Retrieves both unstructured report passages and structured audited facts."""

    @classmethod
    def execute(cls, state: FinancialResearchState) -> Dict[str, Any]:
        """LangGraph node handler."""
        start_time = time.perf_counter()

        target_companies = state.get("target_companies", ["INFY", "TCS"])
        target_years = state.get("target_years", [2022, 2023, 2024])
        sub_queries = state.get("sub_queries", [state.get("sanitized_query", "")])

        logger.info(
            f"Executing Evidence Retriever: {len(sub_queries)} sub-queries, "
            f"companies={target_companies}, years={target_years}"
        )

        # 1. Unstructured Passage Retrieval (Hybrid + Reranking)
        all_chunks: List[SearchResult] = list(state.get("retrieved_chunks", []))
        seen_chunk_ids: Set[int] = {c.chunk_id for c in all_chunks}
        retrieval_latencies: List[float] = []

        for query_str in sub_queries:
            # Query without filter first, or filter by company if targeted
            ticker_hint = None
            for t in target_companies:
                if t in query_str.upper() or ("INFOSYS" in query_str.upper() and t == "INFY"):
                    ticker_hint = t
                    break

            try:
                search_resp = search_engine.search(
                    query=query_str,
                    mode="hybrid_rerank",
                    top_k=3,
                    ticker=ticker_hint,
                )
                retrieval_latencies.append(search_resp.diagnostics.get("latency_ms", 0.0))

                for res in search_resp.results:
                    if res.chunk_id not in seen_chunk_ids:
                        all_chunks.append(res)
                        seen_chunk_ids.add(res.chunk_id)
            except Exception as e:
                logger.warning(f"Retrieval error for sub-query '{query_str}': {e}")

        # 2. Structured Financial Facts Retrieval
        all_facts: List[FactRecord] = list(state.get("retrieved_facts", []))
        seen_fact_keys: Set[str] = {
            f"{f.company_ticker}_{f.fiscal_year}_{f.metric_name}" for f in all_facts
        }

        for ticker in target_companies:
            facts_by_year = FinancialAnalyticsEngine.get_company_facts(ticker, target_years)
            for fy, metrics_dict in facts_by_year.items():
                for metric_name, fact_record in metrics_dict.items():
                    key = f"{fact_record.company_ticker}_{fact_record.fiscal_year}_{fact_record.metric_name}"
                    if key not in seen_fact_keys:
                        all_facts.append(fact_record)
                        seen_fact_keys.add(key)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.info(
            f"Evidence Retriever completed in {elapsed_ms}ms: "
            f"{len(all_chunks)} chunks, {len(all_facts)} audited facts."
        )

        diagnostics = {
            "total_elapsed_ms": elapsed_ms,
            "chunks_retrieved": len(all_chunks),
            "facts_retrieved": len(all_facts),
            "sub_queries_executed": len(sub_queries),
            "mean_search_latency_ms": round(sum(retrieval_latencies) / len(retrieval_latencies), 2) if retrieval_latencies else 0.0,
        }

        execution_log = {
            "node": "evidence_retriever",
            "chunks_count": len(all_chunks),
            "facts_count": len(all_facts),
            "elapsed_ms": elapsed_ms,
        }

        trace = list(state.get("execution_trace", []))
        trace.append(execution_log)

        return {
            "retrieved_chunks": all_chunks,
            "retrieved_facts": all_facts,
            "retrieval_diagnostics": diagnostics,
            "execution_trace": trace,
        }
