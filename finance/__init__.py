"""
FinSight AI Financial Analytics Package
Deterministic calculators, fact models, unit normalizers, accounting validators,
and financial analytics comparison engine.
"""

from finance.analytics import FinancialAnalyticsEngine, analytics_engine
from finance.calculators import FinancialCalculators
from finance.models import (
    CalculatedMetricResult,
    CitationReference,
    CompanyComparisonSummary,
    FactRecord,
    MetricType,
    StatementType,
)
from finance.normalizer import UnitNormalizer
from finance.validator import FinancialValidator

__all__ = [
    "MetricType",
    "StatementType",
    "CitationReference",
    "FactRecord",
    "CalculatedMetricResult",
    "CompanyComparisonSummary",
    "FinancialCalculators",
    "FinancialAnalyticsEngine",
    "analytics_engine",
    "UnitNormalizer",
    "FinancialValidator",
]
