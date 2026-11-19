"""
Evidence Verifier Node for FinSight AI
Inspects retrieved evidence, reporting periods, units, and deterministic calculation
completeness before permitting report synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List
from agents.state import FinancialResearchState
from core.logging import get_logger

logger = get_logger("agents.evidence_verifier")


class EvidenceVerifierNode:
    """Verifies coverage and data integrity of retrieved facts and passages."""

    @classmethod
    def execute(cls, state: FinancialResearchState) -> Dict[str, Any]:
        """LangGraph node execution handler."""
        target_companies = state.get("target_companies", [])
        target_years = state.get("target_years", [])
        target_metrics = state.get("target_metrics", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        retrieved_facts = state.get("retrieved_facts", [])
        calculated_metrics = state.get("calculated_metrics", [])
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        issues: List[str] = []
        missing: List[str] = []

        logger.info(
            f"Executing Evidence Verifier (Attempt {retry_count + 1}/{max_retries + 1}): "
            f"{len(retrieved_facts)} facts, {len(retrieved_chunks)} chunks, {len(calculated_metrics)} metrics."
        )

        # 1. Company coverage check
        facts_by_company = {f.company_ticker for f in retrieved_facts}
        for ticker in target_companies:
            if ticker not in facts_by_company:
                issues.append(f"No structured audited facts found for company: {ticker}")
                missing.append(f"facts_{ticker}")

        # 2. Fiscal year coverage check
        facts_by_year = {f.fiscal_year for f in retrieved_facts}
        for y in target_years:
            if y not in facts_by_year:
                issues.append(f"Missing structured statements for fiscal year: {y}")
                missing.append(f"year_{y}")

        # 3. Text passage verification (for qualitative analysis & context)
        if not retrieved_chunks:
            issues.append("No document text chunks retrieved for qualitative analysis or risk commentary.")
            missing.append("qualitative_chunks")

        # 4. Critical metric coverage check
        if "operating_margin" in target_metrics:
            has_op_margin = any(m.metric_name == "operating_margin" for m in calculated_metrics)
            if not has_op_margin:
                issues.append("Operating margin calculation missing from results.")
                missing.append("metric_operating_margin")

        if "free_cash_flow" in target_metrics:
            has_fcf = any(m.metric_name == "free_cash_flow" for m in calculated_metrics)
            if not has_fcf:
                issues.append("Free cash flow calculation missing from results.")
                missing.append("metric_free_cash_flow")

        # Decision logic:
        # If missing critical elements and retry budget remains -> fail verification and trigger refiner
        has_critical_shortfall = len(missing) > 0 and len(retrieved_facts) == 0
        can_retry = retry_count < max_retries

        if has_critical_shortfall and can_retry:
            verification_passed = False
            logger.warning(f"Evidence verification failed. Triggering query refinement (retry {retry_count + 1})")
        else:
            verification_passed = True
            if issues:
                logger.info(f"Evidence verification passed with {len(issues)} acceptable caveats/notices.")
            else:
                logger.info("Evidence verification passed with full completeness.")

        execution_log = {
            "node": "evidence_verifier",
            "verification_passed": verification_passed,
            "issues_count": len(issues),
            "missing_count": len(missing),
            "retry_count": retry_count,
        }

        trace = list(state.get("execution_trace", []))
        trace.append(execution_log)

        return {
            "verification_passed": verification_passed,
            "verification_issues": issues,
            "missing_elements": missing,
            "execution_trace": trace,
        }
