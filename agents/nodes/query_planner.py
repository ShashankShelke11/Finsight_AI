"""
Query Planner Node for FinSight AI
Classifies query intent, extracts company tickers, fiscal years, target metrics,
and generates focused retrieval sub-queries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set
from agents.state import FinancialResearchState
from core.logging import get_logger
from core.security import guardrails
from finance.normalizer import UnitNormalizer

logger = get_logger("agents.query_planner")

COMPANY_SYNONYMS: Dict[str, str] = {
    "INFOSYS": "INFY",
    "INFY": "INFY",
    "TATA CONSULTANCY SERVICES": "TCS",
    "TCS": "TCS",
    "TATA CONSULTANCY": "TCS",
}

METRIC_KEYWORDS: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "turnover", "topline", "top line"],
    "operating_margin": ["operating margin", "operating profit", "ebit", "operating income"],
    "net_profit_margin": ["net margin", "net profit margin", "pat margin", "net income margin", "profitability"],
    "free_cash_flow": ["free cash flow", "fcf", "cash generation"],
    "operating_cash_flow": ["operating cash flow", "cash flow from operations", "ocf"],
    "capex": ["capex", "capital expenditure", "capital additions"],
    "debt_to_equity": ["debt to equity", "debt/equity", "d/e", "leverage", "solvency"],
    "current_ratio": ["current ratio", "liquidity ratio", "working capital ratio"],
    "return_on_equity": ["roe", "return on equity"],
    "return_on_invested_capital": ["roic", "return on invested capital"],
    "risks": ["risk", "risks", "uncertainty", "headwinds", "challenges", "regulatory", "cybersecurity"],
}


class QueryPlannerNode:
    """Decomposes complex financial queries into structured plans for retrieval and calculation."""

    @classmethod
    def execute(cls, state: FinancialResearchState) -> Dict[str, Any]:
        """LangGraph node execution handler."""
        raw_query = state.get("raw_query", "")
        logger.info(f"Executing Query Planner for query: '{raw_query[:80]}...'")

        # 1. Security & Guardrails check
        sanitized_query, sec_result = guardrails.sanitize_user_prompt(raw_query)
        if not sec_result.is_safe:
            logger.warning(f"Security guardrails triggered: {sec_result.flagged_reasons}")
            return {
                "sanitized_query": sanitized_query,
                "is_safe": False,
                "security_flag": sec_result.risk_level.value,
                "final_markdown_report": f"### Security Policy Violation\n\nYour request was flagged by the security guardrails: {', '.join(sec_result.flagged_reasons)}. For system safety, this query cannot be processed.",
                "executive_summary": "Security Policy Violation: query blocked by guardrails.",
                "errors": [f"Query failed security inspection: {sec_result.flagged_reasons}"],
            }

        # 2. Extract Companies
        target_companies = cls._extract_companies(sanitized_query)
        if not target_companies:
            # Default to primary tracked universe if unstated
            target_companies = ["INFY", "TCS"]

        # 3. Extract Fiscal Years
        target_years = cls._extract_years(sanitized_query)
        if not target_years:
            # Default to latest 3 reported financial years
            target_years = [2022, 2023, 2024]

        # 4. Extract Metrics & Topics
        target_metrics = cls._extract_metrics(sanitized_query)

        # 5. Classify Query Type
        query_type = cls._classify_intent(sanitized_query, target_companies, target_metrics)

        # 6. Generate Targeted Retrieval Sub-Queries
        sub_queries = cls._generate_sub_queries(sanitized_query, target_companies, target_years, target_metrics, query_type)

        execution_log = {
            "node": "query_planner",
            "query_type": query_type,
            "target_companies": target_companies,
            "target_years": target_years,
            "target_metrics": target_metrics,
            "sub_queries_count": len(sub_queries),
        }

        trace = list(state.get("execution_trace", []))
        trace.append(execution_log)

        return {
            "sanitized_query": sanitized_query,
            "is_safe": True,
            "security_flag": None,
            "query_type": query_type,
            "target_companies": target_companies,
            "target_years": target_years,
            "target_metrics": target_metrics,
            "sub_queries": sub_queries,
            "explanation_needed": True,
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", 2),
            "execution_trace": trace,
        }

    @classmethod
    def _extract_companies(cls, query: str) -> List[str]:
        found: List[str] = []
        q_upper = query.upper()
        # Word boundary matching
        for name, ticker in COMPANY_SYNONYMS.items():
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, q_upper):
                if ticker not in found:
                    found.append(ticker)
        return found

    @classmethod
    def _extract_years(cls, query: str) -> List[int]:
        q_lower = query.lower()
        years_found: Set[int] = set()

        # Check for phrases like "last three reported financial years" / "last 3 years"
        if re.search(r"last\s+(?:three|3)\s+(?:reported\s+)?(?:financial\s+|fiscal\s+)?years", q_lower):
            return [2022, 2023, 2024]
        if re.search(r"last\s+(?:two|2)\s+(?:reported\s+)?(?:financial\s+|fiscal\s+)?years", q_lower):
            return [2023, 2024]

        # Explicit FY matches: FY24, FY 2023, 2023-24
        matches = re.findall(r"(?:FY\s*|FY-)?(?:20)?(\d{2})", query, re.IGNORECASE)
        for m in matches:
            try:
                y = UnitNormalizer.normalize_fiscal_year(f"FY{m}")
                if 2015 <= y <= 2030:
                    years_found.add(y)
            except Exception:
                pass

        # Standalone 4-digit years: 2022, 2023, 2024
        matches_4 = re.findall(r"\b(20\d{2})\b", query)
        for m in matches_4:
            y = int(m)
            if 2015 <= y <= 2030:
                years_found.add(y)

        return sorted(list(years_found)) if years_found else []

    @classmethod
    def _extract_metrics(cls, query: str) -> List[str]:
        q_lower = query.lower()
        extracted: List[str] = []
        for metric, keywords in METRIC_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    if metric not in extracted:
                        extracted.append(metric)
                    break
        return extracted

    @classmethod
    def _classify_intent(cls, query: str, companies: List[str], metrics: List[str]) -> str:
        q_lower = query.lower()
        if len(companies) > 1 or any(w in q_lower for w in ["compare", "comparison", "vs", "versus"]):
            return "company_comparison"
        if any(w in q_lower for w in ["trend", "cagr", "historical", "over time", "multi-year"]):
            return "trend_analysis"
        if "risks" in metrics or any(w in q_lower for w in ["risk", "threat", "regulatory", "headwind"]):
            return "risk_analysis"
        if any(m in metrics for m in ["operating_margin", "net_profit_margin", "free_cash_flow", "current_ratio", "debt_to_equity", "return_on_equity", "return_on_invested_capital"]):
            return "financial_calculation"
        return "factual_lookup"

    @classmethod
    def _generate_sub_queries(
        cls,
        raw_query: str,
        companies: List[str],
        years: List[int],
        metrics: List[str],
        query_type: str,
    ) -> List[str]:
        sub_queries: List[str] = []

        # Always include the core user query as base
        sub_queries.append(raw_query)

        # Company-specific detailed retrieval queries
        for ticker in companies:
            comp_name = "Infosys" if ticker == "INFY" else "Tata Consultancy Services TCS"
            # Financial metrics query
            sub_queries.append(f"{comp_name} revenue operating margin net profit free cash flow FY2024")
            # Strategy and qualitative operational commentary
            sub_queries.append(f"{comp_name} strategic growth drivers AI cloud total contract value deal pipeline")
            # Risk disclosures
            if "risks" in metrics or query_type in ("risk_analysis", "company_comparison"):
                sub_queries.append(f"{comp_name} major risk factors regulatory compliance macroeconomic client spending")

        return sub_queries
