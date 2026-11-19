"""
Agentic Financial Research State Definition for LangGraph
Defines the typed workflow state passed between all agent nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from finance.models import CalculatedMetricResult, CitationReference, FactRecord
from retrieval.search_engine import SearchResult


class FinancialResearchState(TypedDict, total=False):
    """
    Typed state container for FinSight AI's LangGraph research workflow.
    Every node in the controlled graph reads from and writes to this structure.
    """

    # --- 1. User Input & Security ---
    raw_query: str
    sanitized_query: str
    is_safe: bool
    security_flag: Optional[str]

    # --- 2. Query Planning & Decomposition ---
    query_type: str  # 'factual_lookup', 'financial_calculation', 'company_comparison', 'trend_analysis', 'risk_analysis'
    target_companies: List[str]  # e.g. ['INFY', 'TCS']
    target_years: List[int]  # e.g. [2022, 2023, 2024]
    target_metrics: List[str]  # e.g. ['revenue', 'operating_margin', 'free_cash_flow']
    sub_queries: List[str]  # targeted queries for retrieval
    explanation_needed: bool

    # --- 3. Retrieved Evidence ---
    retrieved_chunks: List[SearchResult]
    retrieved_facts: List[FactRecord]
    retrieval_diagnostics: Dict[str, Any]

    # --- 4. Deterministic Financial Computations ---
    calculated_metrics: List[CalculatedMetricResult]
    company_scorecards: Dict[str, Any]
    comparison_table_md: Optional[str]

    # --- 5. Verification & Controlled Refinement ---
    verification_passed: bool
    verification_issues: List[str]
    missing_elements: List[str]
    retry_count: int
    max_retries: int

    # --- 6. Synthesis & Citation Validation ---
    draft_synthesis: Optional[str]
    validated_citations: List[CitationReference]
    citation_accuracy_score: float
    unsupported_claims: List[str]

    # --- 7. Final Response Output ---
    final_markdown_report: str
    executive_summary: str
    methodology_and_caveats: List[str]
    source_documents: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]
    errors: List[str]
