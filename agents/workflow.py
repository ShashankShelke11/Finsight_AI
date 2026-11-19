"""
LangGraph Controlled Research Workflow for FinSight AI
Assembles the multi-node agentic graph with query planning, hybrid evidence retrieval,
deterministic financial analytics, evidence verification, bounded query refinement,
evidence-grounded synthesis, and citation validation.
"""

from __future__ import annotations

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from agents.nodes import (
    CitationValidatorNode,
    EvidenceRetrieverNode,
    EvidenceVerifierNode,
    FinancialAnalysisToolNode,
    QueryPlannerNode,
    QueryRefinerNode,
    ResearchSynthesizerNode,
)
from agents.state import FinancialResearchState
from core.logging import get_logger

logger = get_logger("agents.workflow")


def security_router(state: FinancialResearchState) -> str:
    """Routes to citation_validator / early exit if prompt injection or safety risk is flagged."""
    if not state.get("is_safe", True):
        logger.warning("Routing to early exit due to security violation.")
        return "citation_validator"
    return "evidence_retriever"


def verification_router(state: FinancialResearchState) -> str:
    """
    Evaluates evidence verification outcome:
    - If verification passed -> proceed to synthesis.
    - If failed AND retry budget available -> route to query refiner (bounded <= max_retries).
    - If failed AND retry budget exhausted -> proceed to synthesis with transparent caveats.
    """
    passed = state.get("verification_passed", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not passed and retry_count < max_retries:
        logger.info(f"Verification failed; routing to query_refiner (retry {retry_count + 1}/{max_retries})")
        return "query_refiner"

    logger.info("Verification complete or retry budget reached; routing to synthesizer.")
    return "synthesizer"


def build_financial_research_graph() -> StateGraph:
    """Constructs and compiles the production LangGraph state graph."""
    graph = StateGraph(FinancialResearchState)

    # 1. Register logical nodes
    graph.add_node("query_planner", QueryPlannerNode.execute)
    graph.add_node("evidence_retriever", EvidenceRetrieverNode.execute)
    graph.add_node("financial_tool", FinancialAnalysisToolNode.execute)
    graph.add_node("evidence_verifier", EvidenceVerifierNode.execute)
    graph.add_node("query_refiner", QueryRefinerNode.execute)
    graph.add_node("synthesizer", ResearchSynthesizerNode.execute)
    graph.add_node("citation_validator", CitationValidatorNode.execute)

    # 2. Add edges & conditional routing
    graph.add_edge(START, "query_planner")

    graph.add_conditional_edges(
        "query_planner",
        security_router,
        {
            "evidence_retriever": "evidence_retriever",
            "citation_validator": "citation_validator",
        },
    )

    graph.add_edge("evidence_retriever", "financial_tool")
    graph.add_edge("financial_tool", "evidence_verifier")

    graph.add_conditional_edges(
        "evidence_verifier",
        verification_router,
        {
            "query_refiner": "query_refiner",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("query_refiner", "evidence_retriever")
    graph.add_edge("synthesizer", "citation_validator")
    graph.add_edge("citation_validator", END)

    return graph


# Compiled workflow application singleton
financial_research_app = build_financial_research_graph().compile()
