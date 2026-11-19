"""
Deterministic Financial Calculation Engine for FinSight AI
Implements tested, zero-hallucination financial formulas in Python.
All calculations preserve explicit formulas, input parameters, citations, and accounting caveats.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from core.exceptions import FinancialCalculationError, ZeroDenominatorError
from finance.models import CalculatedMetricResult, CitationReference, FactRecord, MetricType
from finance.normalizer import UnitNormalizer
from finance.validator import FinancialValidator


def _extract_citations(*facts: Optional[FactRecord]) -> List[CitationReference]:
    """Collects citations from input FactRecords."""
    citations: List[CitationReference] = []
    for f in facts:
        if f and f.citation:
            citations.append(f.citation)
    return citations


class FinancialCalculators:
    """Core deterministic calculation engine for financial ratios and multi-year metrics."""

    @classmethod
    def calculate_yoy_revenue_growth(
        cls,
        current_rev: FactRecord | float,
        prior_rev: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Year-over-Year Revenue Growth (%):
        Formula: ((Current Revenue - Prior Revenue) / Prior Revenue) * 100
        """
        citations = _extract_citations(
            current_rev if isinstance(current_rev, FactRecord) else None,
            prior_rev if isinstance(prior_rev, FactRecord) else None,
        )

        val_curr = current_rev.value if isinstance(current_rev, FactRecord) else float(current_rev)
        val_prior = prior_rev.value if isinstance(prior_rev, FactRecord) else float(prior_rev)

        unit_curr = current_rev.unit if isinstance(current_rev, FactRecord) else unit
        unit_prior = prior_rev.unit if isinstance(prior_rev, FactRecord) else unit

        # Align units if different
        val_curr, val_prior, common_unit = UnitNormalizer.align_units(val_curr, unit_curr, val_prior, unit_prior)

        caveats_list: List[str] = []
        if isinstance(current_rev, FactRecord) and isinstance(prior_rev, FactRecord):
            FinancialValidator.validate_period_comparability(current_rev, prior_rev)
            FinancialValidator.validate_currency_and_unit(current_rev, prior_rev)
            cons_warn = FinancialValidator.validate_consolidation_status(current_rev, prior_rev)
            if cons_warn:
                caveats_list.append(cons_warn)

        if val_prior == 0.0:
            raise ZeroDenominatorError("Prior period revenue is zero; YoY growth rate is mathematically undefined.")

        if val_prior < 0.0:
            caveats_list.append("Prior period revenue was negative; percentage growth may not convey economic intuition.")

        growth_pct = ((val_curr - val_prior) / abs(val_prior)) * 100.0

        bound_warn = FinancialValidator.check_ratio_bounds("yoy_revenue_growth", growth_pct)
        if bound_warn:
            caveats_list.append(bound_warn)

        formula = "((Current Revenue - Prior Revenue) / |Prior Revenue|) * 100"
        return CalculatedMetricResult(
            metric_name=MetricType.YOY_REVENUE_GROWTH.value,
            display_name="YoY Revenue Growth",
            formula=formula,
            inputs={"current_revenue": val_curr, "prior_revenue": val_prior},
            value=round(growth_pct, 4),
            unit="%",
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list) if caveats_list else None,
        )

    @classmethod
    def calculate_operating_margin(
        cls,
        operating_income: FactRecord | float,
        revenue: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Operating Profit Margin (%):
        Formula: (Operating Income / Revenue) * 100
        """
        citations = _extract_citations(
            operating_income if isinstance(operating_income, FactRecord) else None,
            revenue if isinstance(revenue, FactRecord) else None,
        )

        op_val = operating_income.value if isinstance(operating_income, FactRecord) else float(operating_income)
        rev_val = revenue.value if isinstance(revenue, FactRecord) else float(revenue)

        op_unit = operating_income.unit if isinstance(operating_income, FactRecord) else unit
        rev_unit = revenue.unit if isinstance(revenue, FactRecord) else unit

        op_val, rev_val, _ = UnitNormalizer.align_units(op_val, op_unit, rev_val, rev_unit)

        caveats_list: List[str] = []
        if isinstance(operating_income, FactRecord) and isinstance(revenue, FactRecord):
            FinancialValidator.validate_currency_and_unit(operating_income, revenue)
            cons_warn = FinancialValidator.validate_consolidation_status(operating_income, revenue)
            if cons_warn:
                caveats_list.append(cons_warn)

        if rev_val == 0.0:
            raise ZeroDenominatorError("Revenue is zero; operating margin cannot be calculated.")

        margin_pct = (op_val / rev_val) * 100.0

        bound_warn = FinancialValidator.check_ratio_bounds("operating_margin", margin_pct)
        if bound_warn:
            caveats_list.append(bound_warn)

        formula = "(Operating Income / Revenue) * 100"
        return CalculatedMetricResult(
            metric_name=MetricType.OPERATING_MARGIN.value,
            display_name="Operating Margin",
            formula=formula,
            inputs={"operating_income": op_val, "revenue": rev_val},
            value=round(margin_pct, 4),
            unit="%",
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list) if caveats_list else None,
        )

    @classmethod
    def calculate_net_profit_margin(
        cls,
        net_income: FactRecord | float,
        revenue: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Net Profit Margin (%):
        Formula: (Net Income / Revenue) * 100
        """
        citations = _extract_citations(
            net_income if isinstance(net_income, FactRecord) else None,
            revenue if isinstance(revenue, FactRecord) else None,
        )

        ni_val = net_income.value if isinstance(net_income, FactRecord) else float(net_income)
        rev_val = revenue.value if isinstance(revenue, FactRecord) else float(revenue)

        ni_unit = net_income.unit if isinstance(net_income, FactRecord) else unit
        rev_unit = revenue.unit if isinstance(revenue, FactRecord) else unit

        ni_val, rev_val, _ = UnitNormalizer.align_units(ni_val, ni_unit, rev_val, rev_unit)

        caveats_list: List[str] = []
        if isinstance(net_income, FactRecord) and isinstance(revenue, FactRecord):
            FinancialValidator.validate_currency_and_unit(net_income, revenue)

        if rev_val == 0.0:
            raise ZeroDenominatorError("Revenue is zero; net profit margin cannot be computed.")

        margin_pct = (ni_val / rev_val) * 100.0

        bound_warn = FinancialValidator.check_ratio_bounds("net_profit_margin", margin_pct)
        if bound_warn:
            caveats_list.append(bound_warn)

        formula = "(Net Income / Revenue) * 100"
        return CalculatedMetricResult(
            metric_name=MetricType.NET_PROFIT_MARGIN.value,
            display_name="Net Profit Margin",
            formula=formula,
            inputs={"net_income": ni_val, "revenue": rev_val},
            value=round(margin_pct, 4),
            unit="%",
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list) if caveats_list else None,
        )

    @classmethod
    def calculate_free_cash_flow(
        cls,
        operating_cash_flow: FactRecord | float,
        capex: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Free Cash Flow (FCF):
        Definition: Operating Cash Flow - Capital Expenditures (Capex)
        Applied consistently across all corporate analyses.
        """
        citations = _extract_citations(
            operating_cash_flow if isinstance(operating_cash_flow, FactRecord) else None,
            capex if isinstance(capex, FactRecord) else None,
        )

        ocf_val = operating_cash_flow.value if isinstance(operating_cash_flow, FactRecord) else float(operating_cash_flow)
        capex_val = capex.value if isinstance(capex, FactRecord) else float(capex)

        ocf_unit = operating_cash_flow.unit if isinstance(operating_cash_flow, FactRecord) else unit
        capex_unit = capex.unit if isinstance(capex, FactRecord) else unit

        ocf_val, capex_val, common_unit = UnitNormalizer.align_units(ocf_val, ocf_unit, capex_val, capex_unit)

        # Capex is treated as an outflow. If entered as positive, subtract it.
        # If entered as negative in cash flow statements, add it.
        effective_capex = abs(capex_val)
        fcf = ocf_val - effective_capex

        caveats_list: List[str] = [
            "Definition applied: FCF = Operating Cash Flow - Capital Expenditures (Capex)."
        ]
        if fcf < 0.0:
            caveats_list.append("Negative Free Cash Flow: Capital investments exceeded operating cash generation.")

        formula = "Operating Cash Flow - Capex (OCF - Capex)"
        return CalculatedMetricResult(
            metric_name=MetricType.FREE_CASH_FLOW.value,
            display_name="Free Cash Flow",
            formula=formula,
            inputs={"operating_cash_flow": ocf_val, "capital_expenditures": effective_capex},
            value=round(fcf, 4),
            unit=common_unit,
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list),
        )

    @classmethod
    def calculate_fcf_conversion(
        cls,
        free_cash_flow: FactRecord | float,
        net_income: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Free Cash Flow Conversion (% of Net Income):
        Formula: (Free Cash Flow / Net Income) * 100
        Reflects cash flow quality and earnings conversion efficiency.
        """
        citations = _extract_citations(
            free_cash_flow if isinstance(free_cash_flow, FactRecord) else None,
            net_income if isinstance(net_income, FactRecord) else None,
        )

        fcf_val = free_cash_flow.value if isinstance(free_cash_flow, FactRecord) else float(free_cash_flow)
        ni_val = net_income.value if isinstance(net_income, FactRecord) else float(net_income)

        fcf_unit = free_cash_flow.unit if isinstance(free_cash_flow, FactRecord) else unit
        ni_unit = net_income.unit if isinstance(net_income, FactRecord) else unit

        fcf_val, ni_val, _ = UnitNormalizer.align_units(fcf_val, fcf_unit, ni_val, ni_unit)

        if ni_val == 0.0:
            raise ZeroDenominatorError("Net income is zero; FCF conversion percentage is undefined.")

        conv_pct = (fcf_val / ni_val) * 100.0

        caveats_list: List[str] = []
        if conv_pct > 100.0:
            caveats_list.append("FCF conversion exceeds 100%: Cash flow benefited from working capital release or non-cash depreciation.")
        elif conv_pct < 0.0:
            caveats_list.append("Negative cash flow conversion: Working capital absorption or negative operating cash flow.")

        formula = "(Free Cash Flow / Net Income) * 100"
        return CalculatedMetricResult(
            metric_name="fcf_conversion",
            display_name="FCF to Net Income Conversion",
            formula=formula,
            inputs={"free_cash_flow": fcf_val, "net_income": ni_val},
            value=round(conv_pct, 4),
            unit="%",
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list) if caveats_list else None,
        )

    @classmethod
    def calculate_current_ratio(
        cls,
        current_assets: FactRecord | float,
        current_liabilities: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Current Ratio (x):
        Formula: Current Assets / Current Liabilities
        """
        citations = _extract_citations(
            current_assets if isinstance(current_assets, FactRecord) else None,
            current_liabilities if isinstance(current_liabilities, FactRecord) else None,
        )

        ca_val = current_assets.value if isinstance(current_assets, FactRecord) else float(current_assets)
        cl_val = current_liabilities.value if isinstance(current_liabilities, FactRecord) else float(current_liabilities)

        ca_unit = current_assets.unit if isinstance(current_assets, FactRecord) else unit
        cl_unit = current_liabilities.unit if isinstance(current_liabilities, FactRecord) else unit

        ca_val, cl_val, _ = UnitNormalizer.align_units(ca_val, ca_unit, cl_val, cl_unit)

        if cl_val == 0.0:
            raise ZeroDenominatorError("Current liabilities are zero; current ratio cannot be computed.")

        ratio = ca_val / cl_val

        caveats = FinancialValidator.check_ratio_bounds("current_ratio", ratio)
        formula = "Current Assets / Current Liabilities"
        return CalculatedMetricResult(
            metric_name=MetricType.CURRENT_RATIO.value,
            display_name="Current Ratio",
            formula=formula,
            inputs={"current_assets": ca_val, "current_liabilities": cl_val},
            value=round(ratio, 4),
            unit="x",
            citations=citations,
            is_valid=True,
            caveats=caveats,
        )

    @classmethod
    def calculate_debt_to_equity(
        cls,
        total_debt: FactRecord | float,
        total_equity: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Debt-to-Equity Ratio (x):
        Formula: Total Debt / Total Shareholders' Equity
        """
        citations = _extract_citations(
            total_debt if isinstance(total_debt, FactRecord) else None,
            total_equity if isinstance(total_equity, FactRecord) else None,
        )

        debt_val = total_debt.value if isinstance(total_debt, FactRecord) else float(total_debt)
        eq_val = total_equity.value if isinstance(total_equity, FactRecord) else float(total_equity)

        d_unit = total_debt.unit if isinstance(total_debt, FactRecord) else unit
        eq_unit = total_equity.unit if isinstance(total_equity, FactRecord) else unit

        debt_val, eq_val, _ = UnitNormalizer.align_units(debt_val, d_unit, eq_val, eq_unit)

        if eq_val == 0.0:
            raise ZeroDenominatorError("Shareholders' equity is zero; debt-to-equity ratio is undefined.")

        ratio = debt_val / eq_val

        caveats_list: List[str] = []
        if debt_val == 0.0:
            caveats_list.append("Zero debt balance: company operates with zero financial debt leverage.")
        bound_warn = FinancialValidator.check_ratio_bounds("debt_to_equity", ratio)
        if bound_warn:
            caveats_list.append(bound_warn)

        formula = "Total Debt / Total Equity"
        return CalculatedMetricResult(
            metric_name=MetricType.DEBT_TO_EQUITY.value,
            display_name="Debt to Equity Ratio",
            formula=formula,
            inputs={"total_debt": debt_val, "total_equity": eq_val},
            value=round(ratio, 4),
            unit="x",
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list) if caveats_list else None,
        )

    @classmethod
    def calculate_return_on_equity(
        cls,
        net_income: FactRecord | float,
        total_equity: FactRecord | float,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Return on Equity (ROE) (%):
        Formula: (Net Income / Total Shareholders' Equity) * 100
        """
        citations = _extract_citations(
            net_income if isinstance(net_income, FactRecord) else None,
            total_equity if isinstance(total_equity, FactRecord) else None,
        )

        ni_val = net_income.value if isinstance(net_income, FactRecord) else float(net_income)
        eq_val = total_equity.value if isinstance(total_equity, FactRecord) else float(total_equity)

        ni_unit = net_income.unit if isinstance(net_income, FactRecord) else unit
        eq_unit = total_equity.unit if isinstance(total_equity, FactRecord) else unit

        ni_val, eq_val, _ = UnitNormalizer.align_units(ni_val, ni_unit, eq_val, eq_unit)

        if eq_val == 0.0:
            raise ZeroDenominatorError("Total equity is zero; ROE cannot be computed.")

        roe_pct = (ni_val / eq_val) * 100.0

        caveats_list: List[str] = []
        if eq_val < 0.0:
            caveats_list.append("Negative equity produces mathematically positive ROE if net income is negative; economic interpretation requires caution.")

        formula = "(Net Income / Total Equity) * 100"
        return CalculatedMetricResult(
            metric_name=MetricType.RETURN_ON_EQUITY.value,
            display_name="Return on Equity (ROE)",
            formula=formula,
            inputs={"net_income": ni_val, "total_equity": eq_val},
            value=round(roe_pct, 4),
            unit="%",
            citations=citations,
            is_valid=True,
            caveats=" | ".join(caveats_list) if caveats_list else None,
        )

    @classmethod
    def calculate_return_on_invested_capital(
        cls,
        operating_income: FactRecord | float,
        total_debt: FactRecord | float,
        total_equity: FactRecord | float,
        tax_rate: float = 0.25,
        unit: str = "crore",
    ) -> CalculatedMetricResult:
        """
        Calculates Return on Invested Capital (ROIC) (%):
        Formula: (Operating Income * (1 - Tax Rate) / (Total Debt + Total Equity)) * 100
        NOPAT = Operating Income * (1 - Tax Rate)
        Invested Capital = Total Debt + Total Equity
        """
        citations = _extract_citations(
            operating_income if isinstance(operating_income, FactRecord) else None,
            total_debt if isinstance(total_debt, FactRecord) else None,
            total_equity if isinstance(total_equity, FactRecord) else None,
        )

        op_val = operating_income.value if isinstance(operating_income, FactRecord) else float(operating_income)
        debt_val = total_debt.value if isinstance(total_debt, FactRecord) else float(total_debt)
        eq_val = total_equity.value if isinstance(total_equity, FactRecord) else float(total_equity)

        op_unit = operating_income.unit if isinstance(operating_income, FactRecord) else unit
        d_unit = total_debt.unit if isinstance(total_debt, FactRecord) else unit
        eq_unit = total_equity.unit if isinstance(total_equity, FactRecord) else unit

        # Align all to unit
        op_val = UnitNormalizer.convert_amount(op_val, op_unit, unit)
        debt_val = UnitNormalizer.convert_amount(debt_val, d_unit, unit)
        eq_val = UnitNormalizer.convert_amount(eq_val, eq_unit, unit)

        invested_capital = debt_val + eq_val
        if invested_capital == 0.0:
            raise ZeroDenominatorError("Invested capital (Total Debt + Total Equity) is zero; ROIC is undefined.")

        nopat = op_val * (1.0 - tax_rate)
        roic_pct = (nopat / invested_capital) * 100.0

        formula = f"(Operating Income * (1 - {tax_rate}) / (Total Debt + Total Equity)) * 100"
        return CalculatedMetricResult(
            metric_name=MetricType.RETURN_ON_INVESTED_CAPITAL.value,
            display_name="Return on Invested Capital (ROIC)",
            formula=formula,
            inputs={
                "operating_income": op_val,
                "total_debt": debt_val,
                "total_equity": eq_val,
                "effective_tax_rate": tax_rate,
                "invested_capital": invested_capital,
                "nopat": nopat,
            },
            value=round(roic_pct, 4),
            unit="%",
            citations=citations,
            is_valid=True,
            caveats=f"Assumed effective tax rate of {tax_rate*100:.1f}%. Invested Capital defined as Total Debt + Total Equity.",
        )

    @classmethod
    def calculate_cagr(
        cls,
        beginning_val: float,
        ending_val: float,
        years: int,
    ) -> float:
        """
        Calculates Compound Annual Growth Rate (%):
        Formula: ((Ending Value / Beginning Value) ** (1 / years) - 1) * 100
        """
        if years <= 0:
            raise ValueError("Number of years for CAGR must be strictly positive.")
        if beginning_val <= 0.0 or ending_val <= 0.0:
            raise FinancialCalculationError("CAGR calculation requires strictly positive beginning and ending values.")

        cagr = ((ending_val / beginning_val) ** (1.0 / years) - 1.0) * 100.0
        return round(cagr, 4)
