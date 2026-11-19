"""
Accounting Comparability and Financial Sanity Validator for FinSight AI
Validates accounting policy consistency, currency comparability, consolidated vs standalone distinctions,
and numerical plausibility bounds.
"""

from __future__ import annotations

from typing import List, Optional
from core.exceptions import InconsistentPeriodError, UnitMismatchError
from finance.models import FactRecord


class FinancialValidator:
    """Enforces rigorous accounting consistency rules across metrics and periods."""

    @classmethod
    def validate_period_comparability(cls, fact_current: FactRecord, fact_prior: FactRecord) -> None:
        """Ensures two facts represent comparable reporting intervals (e.g. FY vs FY, not FY vs Quarter)."""
        curr_p = fact_current.reporting_period.upper()
        prior_p = fact_prior.reporting_period.upper()

        curr_is_q = "Q" in curr_p
        prior_is_q = "Q" in prior_p

        if curr_is_q != prior_is_q:
            raise InconsistentPeriodError(
                f"Cannot compare quarterly reporting period '{curr_p}' with annual reporting period '{prior_p}'."
            )

        if fact_current.fiscal_year <= fact_prior.fiscal_year and not curr_is_q:
            raise InconsistentPeriodError(
                f"Current fiscal year ({fact_current.fiscal_year}) must be after prior fiscal year ({fact_prior.fiscal_year})."
            )

    @classmethod
    def validate_currency_and_unit(cls, fact1: FactRecord, fact2: FactRecord) -> None:
        """Validates that two facts share compatible currencies or explicitly flags currency translation requirements."""
        if fact1.currency.upper() != fact2.currency.upper():
            raise UnitMismatchError(
                f"Currency mismatch: {fact1.company_ticker} reports in {fact1.currency}, "
                f"while {fact2.company_ticker} reports in {fact2.currency}. Direct arithmetic without FX translation is prohibited."
            )

    @classmethod
    def validate_consolidation_status(cls, fact1: FactRecord, fact2: FactRecord) -> Optional[str]:
        """Warns if attempting to compare consolidated financial figures with standalone figures."""
        if fact1.is_consolidated != fact2.is_consolidated:
            status1 = "Consolidated" if fact1.is_consolidated else "Standalone"
            status2 = "Consolidated" if fact2.is_consolidated else "Standalone"
            return (
                f"Comparability Warning: Comparing {status1} statements of {fact1.company_ticker} "
                f"with {status2} statements of {fact2.company_ticker}. Intersubsidiary eliminations may bias ratios."
            )
        return None

    @classmethod
    def check_ratio_bounds(cls, metric_name: str, value: float) -> Optional[str]:
        """Flags anomalous or accounting edge-case values for calculated metrics."""
        warnings: List[str] = []

        if metric_name in ("operating_margin", "net_profit_margin", "gross_margin"):
            if value > 90.0:
                warnings.append(f"Unusually high margin ({value:.2f}%). Verify if non-operating items or exceptional gains were included.")
            elif value < -50.0:
                warnings.append(f"Severe negative margin ({value:.2f}%). Reflects deep operating losses in the reporting period.")

        elif metric_name == "current_ratio":
            if value < 0.0:
                warnings.append("Current ratio cannot be negative; indicates negative liabilities or assets entry anomaly.")
            elif value < 1.0:
                warnings.append(f"Current ratio ({value:.2f}x) is below 1.0. Indicates working capital deficit and short-term liquidity pressure.")
            elif value > 10.0:
                warnings.append(f"Unusually high current ratio ({value:.2f}x). May indicate excess idle cash or unutilized working capital.")

        elif metric_name == "debt_to_equity":
            if value < 0.0:
                warnings.append("Negative debt-to-equity ratio indicates negative shareholders' equity (accumulated losses exceed share capital).")
            elif value > 5.0:
                warnings.append(f"High financial leverage ({value:.2f}x debt-to-equity). Significant reliance on debt financing.")

        elif metric_name == "yoy_revenue_growth":
            if value < -50.0:
                warnings.append(f"Steep revenue decline of {value:.2f}%. Check for business divestments or discontinued operations.")
            elif value > 100.0:
                warnings.append(f"Extraordinary revenue growth of {value:.2f}%. Check for major mergers, acquisitions, or base-period anomalies.")

        return " | ".join(warnings) if warnings else None
