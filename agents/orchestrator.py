"""
High-Level Agentic Research Orchestrator for FinSight AI
Coordinates query execution via LangGraph, audit logging, and structured response delivery.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from agents.state import FinancialResearchState
from agents.workflow import financial_research_app
from core.database import db
from core.logging import get_logger

logger = get_logger("agents.orchestrator")


class FinancialResearchOrchestrator:
    """Entry point for executing agentic financial research workflows."""

    @classmethod
    def run(
        cls,
        query: str,
        session_id: Optional[str] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end financial research query through the LangGraph workflow.
        Returns full structured response including synthesized report, metrics, citations, and trace.
        """
        start_time = time.perf_counter()
        query_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())

        logger.info(f"Starting financial research workflow [Query ID: {query_id}] for query: '{query[:80]}...'")

        initial_state: FinancialResearchState = {
            "raw_query": query,
            "sanitized_query": query,
            "is_safe": True,
            "security_flag": None,
            "retry_count": 0,
            "max_retries": max_retries,
            "execution_trace": [],
            "retrieved_chunks": [],
            "retrieved_facts": [],
            "calculated_metrics": [],
            "verification_issues": [],
            "errors": [],
        }

        try:
            # Execute LangGraph state machine
            final_state = financial_research_app.invoke(initial_state)
            status = "completed" if final_state.get("is_safe", True) else "security_blocked"
        except Exception as e:
            logger.error(f"Workflow execution failure: {e}", exc_info=True)
            status = "failed"
            final_state = {
                **initial_state,
                "final_markdown_report": f"### Research Workflow Failed\n\nAn unexpected error occurred during research execution: {e}",
                "executive_summary": "Error during execution.",
                "errors": [str(e)],
            }

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"Financial research completed in {elapsed_ms}ms with status '{status}'.")

        # Serialize metrics and citations for database audit logging
        calculated_metrics = final_state.get("calculated_metrics", [])
        serialized_metrics = [
            m.model_dump() if hasattr(m, "model_dump") else str(m)
            for m in calculated_metrics
        ]

        citations = final_state.get("validated_citations", [])
        serialized_citations = [
            c.model_dump() if hasattr(c, "model_dump") else str(c)
            for c in citations
        ]

        plan_info = {
            "query_type": final_state.get("query_type"),
            "target_companies": final_state.get("target_companies"),
            "target_years": final_state.get("target_years"),
            "target_metrics": final_state.get("target_metrics"),
            "sub_queries": final_state.get("sub_queries"),
        }

        # Persist audit record in research_queries table
        try:
            sql = """
                INSERT INTO research_queries (
                    id, session_id, user_query, query_type, plan_json,
                    synthesized_response, calculated_metrics_json, citations_json,
                    retrieval_diagnostics_json, status, execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                query_id,
                session_id,
                query,
                final_state.get("query_type", "unknown"),
                json.dumps(plan_info),
                final_state.get("final_markdown_report", ""),
                json.dumps(serialized_metrics),
                json.dumps(serialized_citations),
                json.dumps(final_state.get("retrieval_diagnostics", {})),
                status,
                elapsed_ms,
            )
            db.execute(sql, params)
        except Exception as e:
            logger.warning(f"Could not persist research query audit record: {e}")

        return {
            "query_id": query_id,
            "session_id": session_id,
            "status": status,
            "execution_time_ms": elapsed_ms,
            "query": query,
            "query_type": final_state.get("query_type", "factual_lookup"),
            "target_companies": final_state.get("target_companies", []),
            "target_years": final_state.get("target_years", []),
            "executive_summary": final_state.get("executive_summary", ""),
            "markdown_report": final_state.get("final_markdown_report", ""),
            "comparison_table": final_state.get("comparison_table_md"),
            "calculated_metrics": serialized_metrics,
            "citations": serialized_citations,
            "source_documents": final_state.get("source_documents", []),
            "methodology_and_caveats": final_state.get("methodology_and_caveats", []),
            "execution_trace": final_state.get("execution_trace", []),
            "errors": final_state.get("errors", []),
        }


# Global singleton instance
orchestrator = FinancialResearchOrchestrator()
