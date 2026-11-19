"""
Financial Analysis Tool Node for FinSight AI
Executes deterministic financial calculations using Python functions from Phase 4.
LLMs are NEVER asked to perform arithmetic.
"""

from __future__ import annotations

from typing import Any, Dict, List
from agents.state import FinancialResearchState
from core.logging import get_logger
from finance.analytics import FinancialAnalyticsEngine
from finance.models import CalculatedMetricResult

logger = get_logger("agents.financial_tool")


class FinancialAnalysisToolNode:
    """Deterministic calculation execution node in the LangGraph workflow."""

    @classmethod
    def execute(cls, state: FinancialResearchState) -> Dict[str, Any]:
        """LangGraph node execution handler."""
        target_companies = state.get("target_companies", ["INFY", "TCS"])
        target_years = state.get("target_years", [2022, 2023, 2024])
        query_type = state.get("query_type", "company_comparison")

        logger.info(
            f"Executing Financial Analysis Tool for {target_companies} across {target_years} (type: {query_type})"
        )

        calculated_metrics: List[CalculatedMetricResult] = []
        scorecards: Dict[str, Any] = {}
        comparison_table_md: str | None = None

        # 1. Generate multi-year scorecards for each company
        for ticker in target_companies:
            try:
                card = FinancialAnalyticsEngine.compute_financial_scorecard(ticker, target_years)
                scorecards[ticker] = card
                # Flatten calculated metrics from scorecard
                for fy, metrics_dict in card.items():
                    for metric_key, res in metrics_dict.items():
                        if isinstance(res, CalculatedMetricResult):
                            calculated_metrics.append(res)
            except Exception as e:
                logger.warning(f"Error calculating scorecard for {ticker}: {e}")

        # 2. If multi-company comparison or comparison requested, generate side-by-side comparative table
        if len(target_companies) > 1 or query_type == "company_comparison":
            try:
                comp_summary = FinancialAnalyticsEngine.compare_companies(
                    tickers=target_companies,
                    fiscal_years=target_years,
                )
                comparison_table_md = FinancialAnalyticsEngine.format_comparison_markdown_table(comp_summary)
            except Exception as e:
                logger.warning(f"Error generating comparison summary: {e}")

        execution_log = {
            "node": "financial_analysis_tool",
            "companies_analyzed": target_companies,
            "calculated_metrics_count": len(calculated_metrics),
            "scorecards_generated": list(scorecards.keys()),
            "comparison_table_generated": comparison_table_md is not None,
        }

        trace = list(state.get("execution_trace", []))
        trace.append(execution_log)

        return {
            "calculated_metrics": calculated_metrics,
            "company_scorecards": scorecards,
            "comparison_table_md": comparison_table_md,
            "execution_trace": trace,
        }
