"""
Domain Models and Schemas for Financial Statements, Facts, and Derived Ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(str, Enum):
    # Primary Statement Line Items (Audited Ground Truth)
    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    NET_INCOME = "net_income"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPEX = "capital_expenditures"
    TOTAL_DEBT = "total_debt"
    TOTAL_EQUITY = "total_equity"
    CURRENT_ASSETS = "current_assets"
    CURRENT_LIABILITIES = "current_liabilities"
    DILUTED_EPS = "diluted_eps"

    # Derived Ratios and Growth Metrics (Deterministic Calculations)
    YOY_REVENUE_GROWTH = "yoy_revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    NET_PROFIT_MARGIN = "net_profit_margin"
    FREE_CASH_FLOW = "free_cash_flow"
    CURRENT_RATIO = "current_ratio"
    DEBT_TO_EQUITY = "debt_to_equity"
    RETURN_ON_EQUITY = "return_on_equity"
    RETURN_ON_INVESTED_CAPITAL = "return_on_invested_capital"


class StatementType(str, Enum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    DERIVED_RATIOS = "derived_ratios"


@dataclass
class CitationReference:
    """Traceable evidence citation pointing to original filing and page number."""
    document_title: str
    fiscal_year: int
    reporting_period: str
    page_number: int
    source_reference: str
    url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_title": self.document_title,
            "fiscal_year": self.fiscal_year,
            "reporting_period": self.reporting_period,
            "page_number": self.page_number,
            "source_reference": self.source_reference,
            "url": self.url,
        }


@dataclass
class FactRecord:
    """An individual audited financial figure with full provenance metadata."""
    company_ticker: str
    fiscal_year: int
    reporting_period: str
    statement_type: StatementType
    metric_name: str
    metric_display_name: str
    value: float
    unit: str  # e.g. "crore", "million", "USD", "INR"
    currency: str  # "INR", "USD", etc.
    is_consolidated: bool = True
    citation: Optional[CitationReference] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_ticker": self.company_ticker,
            "fiscal_year": self.fiscal_year,
            "reporting_period": self.reporting_period,
            "statement_type": self.statement_type.value if isinstance(self.statement_type, StatementType) else str(self.statement_type),
            "metric_name": self.metric_name,
            "metric_display_name": self.metric_display_name,
            "value": self.value,
            "unit": self.unit,
            "currency": self.currency,
            "is_consolidated": self.is_consolidated,
            "citation": self.citation.to_dict() if self.citation else None,
        }


@dataclass
class CalculatedMetricResult:
    """Result of a deterministic Python calculation with explicit formula and inputs."""
    metric_name: str
    display_name: str
    formula: str
    inputs: Dict[str, float]
    value: Optional[float]
    unit: str  # "%", "x", "INR Crore", etc.
    citations: List[CitationReference] = field(default_factory=list)
    is_valid: bool = True
    caveats: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "display_name": self.display_name,
            "formula": self.formula,
            "inputs": self.inputs,
            "value": self.value,
            "unit": self.unit,
            "citations": [c.to_dict() for c in self.citations],
            "is_valid": self.is_valid,
            "caveats": self.caveats,
        }


@dataclass
class CompanyComparisonSummary:
    """Multi-company or multi-year financial comparison container."""
    companies: List[str]
    fiscal_years: List[int]
    metrics: Dict[str, Dict[str, List[CalculatedMetricResult]]]  # company -> metric -> yearly results
    synthesis_notes: List[str] = field(default_factory=list)
