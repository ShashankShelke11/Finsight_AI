"""
Query Refiner Node for FinSight AI
Bounded iterative retrieval refinement when evidence verification detects missing items.
"""

from __future__ import annotations

from typing import Any, Dict, List
from agents.state import FinancialResearchState
from core.logging import get_logger

logger = get_logger("agents.query_refiner")


class QueryRefinerNode:
    """Reformulates queries to target missing financial disclosures or passages."""

    @classmethod
    def execute(cls, state: FinancialResearchState) -> Dict[str, Any]:
        """LangGraph node execution handler."""
        current_retry = state.get("retry_count", 0) + 1
        missing = state.get("missing_elements", [])
        companies = state.get("target_companies", ["INFY", "TCS"])
        years = state.get("target_years", [2024])

        logger.info(
            f"Executing Query Refiner: Attempt {current_retry}, missing elements: {missing}"
        )

        refined_queries: List[str] = []

        # Formulate specific targeted fallback queries based on what's missing
        for elem in missing:
            if "INFY" in elem or "facts_INFY" in elem:
                refined_queries.append("Infosys consolidated financial statements revenue profit cash flow FY2024")
                refined_queries.append("Infosys Limited annual report audited balance sheet income statement")
            elif "TCS" in elem or "facts_TCS" in elem:
                refined_queries.append("Tata Consultancy Services consolidated financial statements revenue profit FY2024")
                refined_queries.append("TCS annual report audited cash flows balance sheet")
            elif "qualitative" in elem:
                for c in companies:
                    c_name = "Infosys" if c == "INFY" else "Tata Consultancy Services"
                    refined_queries.append(f"{c_name} strategic outlook AI generative cloud risk factors")
            elif "operating_margin" in elem:
                refined_queries.append("Operating income operating margin EBIT employee benefit expenses")
            elif "free_cash_flow" in elem:
                refined_queries.append("Operating cash flow capital expenditure capex purchase of property plant equipment")

        if not refined_queries:
            # Broadened general financial query
            refined_queries.append(f"Financial statements income statement balance sheet cash flows {companies} {years}")

        execution_log = {
            "node": "query_refiner",
            "retry_count": current_retry,
            "new_sub_queries": refined_queries,
        }

        trace = list(state.get("execution_trace", []))
        trace.append(execution_log)

        return {
            "sub_queries": refined_queries,
            "retry_count": current_retry,
            "execution_trace": trace,
        }
