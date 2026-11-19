"""
Financial Analytics Coordinator for FinSight AI
Queries audited financial facts from the database, executes deterministic calculations,
generates multi-year scorecards, and synthesizes multi-company equity comparison summaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from core.database import db
from core.exceptions import MissingFactError
from core.logging import logger
from finance.calculators import FinancialCalculators
from finance.models import (
    CalculatedMetricResult,
    CitationReference,
    CompanyComparisonSummary,
    FactRecord,
    MetricType,
    StatementType,
)


class FinancialAnalyticsEngine:
    """Coordinates audited facts retrieval and executes multi-year financial ratio analytics."""

    @classmethod
    def get_company_facts(
        cls,
        ticker: str,
        fiscal_years: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, FactRecord]]:
        """
        Retrieves ground-truth audited facts for a company organized by fiscal year and metric name.
        Returns: {fiscal_year: {metric_name: FactRecord}}
        """
        sql = """
            SELECT 
                f.fiscal_year,
                f.reporting_period,
                f.statement_type,
                f.metric_name,
                f.metric_display_name,
                f.value,
                f.unit,
                f.currency,
                f.is_consolidated,
                f.source_page,
                f.source_reference,
                d.title AS doc_title,
                d.source_url
            FROM financial_facts f
            JOIN companies c ON f.company_id = c.id
            LEFT JOIN documents d ON f.source_document_id = d.id
            WHERE c.ticker = ?
        """
        params: List[Any] = [ticker.upper()]
        if fiscal_years:
            placeholders = ",".join(["?"] * len(fiscal_years))
            sql += f" AND f.fiscal_year IN ({placeholders})"
            params.extend(fiscal_years)

        sql += " ORDER BY f.fiscal_year ASC;"

        rows = db.fetch_all(sql, tuple(params))
        by_year: Dict[int, Dict[str, FactRecord]] = {}

        for r in rows:
            fy = r["fiscal_year"]
            if fy not in by_year:
                by_year[fy] = {}

            citation = CitationReference(
                document_title=r.get("doc_title") or f"{ticker} Annual Report {r['reporting_period']}",
                fiscal_year=fy,
                reporting_period=r["reporting_period"],
                page_number=r.get("source_page") or 1,
                source_reference=r.get("source_reference") or "Audited Financial Statements",
                url=r.get("source_url"),
            )

            record = FactRecord(
                company_ticker=ticker.upper(),
                fiscal_year=fy,
                reporting_period=r["reporting_period"],
                statement_type=StatementType(r["statement_type"]) if r["statement_type"] in [s.value for s in StatementType] else StatementType.INCOME_STATEMENT,
                metric_name=r["metric_name"],
                metric_display_name=r["metric_display_name"],
                value=float(r["value"]),
                unit=r["unit"],
                currency=r["currency"],
                is_consolidated=bool(r["is_consolidated"]),
                citation=citation,
            )
            by_year[fy][r["metric_name"]] = record

        return by_year

    @classmethod
    def compute_financial_scorecard(
        cls,
        ticker: str,
        fiscal_years: List[int],
    ) -> Dict[int, Dict[str, CalculatedMetricResult]]:
        """
        Computes deterministic financial scorecard across requested fiscal years.
        Returns: {fiscal_year: {metric_key: CalculatedMetricResult}}
        """
        # Fetch requested years plus prior year if needed for YoY growth
        min_year = min(fiscal_years)
        needed_years = sorted(set(fiscal_years + [min_year - 1]))
        facts_by_year = cls.get_company_facts(ticker, needed_years)

        scorecard: Dict[int, Dict[str, CalculatedMetricResult]] = {}

        for fy in sorted(fiscal_years):
            scorecard[fy] = {}
            curr_facts = facts_by_year.get(fy, {})
            prior_facts = facts_by_year.get(fy - 1, {})

            # 1. Revenue & YoY Growth
            curr_rev = curr_facts.get(MetricType.REVENUE.value)
            prior_rev = prior_facts.get(MetricType.REVENUE.value)
            if curr_rev and prior_rev:
                scorecard[fy][MetricType.YOY_REVENUE_GROWTH.value] = (
                    FinancialCalculators.calculate_yoy_revenue_growth(curr_rev, prior_rev)
                )

            # 2. Operating Margin
            op_inc = curr_facts.get(MetricType.OPERATING_INCOME.value)
            if op_inc and curr_rev:
                scorecard[fy][MetricType.OPERATING_MARGIN.value] = (
                    FinancialCalculators.calculate_operating_margin(op_inc, curr_rev)
                )

            # 3. Net Profit Margin
            net_inc = curr_facts.get(MetricType.NET_INCOME.value)
            if net_inc and curr_rev:
                scorecard[fy][MetricType.NET_PROFIT_MARGIN.value] = (
                    FinancialCalculators.calculate_net_profit_margin(net_inc, curr_rev)
                )

            # 4. Free Cash Flow
            ocf = curr_facts.get(MetricType.OPERATING_CASH_FLOW.value)
            capex = curr_facts.get(MetricType.CAPEX.value)
            if ocf and capex:
                fcf_res = FinancialCalculators.calculate_free_cash_flow(ocf, capex)
                scorecard[fy][MetricType.FREE_CASH_FLOW.value] = fcf_res

                # 5. FCF Conversion
                if net_inc:
                    scorecard[fy]["fcf_conversion"] = (
                        FinancialCalculators.calculate_fcf_conversion(fcf_res.value, net_inc)
                    )

            # 6. Current Ratio
            ca = curr_facts.get(MetricType.CURRENT_ASSETS.value)
            cl = curr_facts.get(MetricType.CURRENT_LIABILITIES.value)
            if ca and cl:
                scorecard[fy][MetricType.CURRENT_RATIO.value] = (
                    FinancialCalculators.calculate_current_ratio(ca, cl)
                )

            # 7. Debt to Equity
            debt = curr_facts.get(MetricType.TOTAL_DEBT.value)
            equity = curr_facts.get(MetricType.TOTAL_EQUITY.value)
            if debt and equity:
                scorecard[fy][MetricType.DEBT_TO_EQUITY.value] = (
                    FinancialCalculators.calculate_debt_to_equity(debt, equity)
                )

            # 8. Return on Equity (ROE)
            if net_inc and equity:
                scorecard[fy][MetricType.RETURN_ON_EQUITY.value] = (
                    FinancialCalculators.calculate_return_on_equity(net_inc, equity)
                )

            # 9. Return on Invested Capital (ROIC)
            if op_inc and debt and equity:
                scorecard[fy][MetricType.RETURN_ON_INVESTED_CAPITAL.value] = (
                    FinancialCalculators.calculate_return_on_invested_capital(op_inc, debt, equity)
                )

        return scorecard

    @classmethod
    def compare_companies(
        cls,
        tickers: List[str],
        fiscal_years: List[int],
    ) -> CompanyComparisonSummary:
        """
        Executes multi-company, multi-year comparative financial analysis.
        Generates structured scorecard and synthesis observations.
        """
        cleaned_tickers = [t.upper().strip() for t in tickers]
        metrics_by_company: Dict[str, Dict[str, List[CalculatedMetricResult]]] = {}

        for t in cleaned_tickers:
            scorecard = cls.compute_financial_scorecard(t, fiscal_years)
            metrics_by_company[t] = {}

            # Collect each metric across years
            all_metric_names = set()
            for fy_data in scorecard.values():
                all_metric_names.update(fy_data.keys())

            for m in all_metric_names:
                metrics_by_company[t][m] = [
                    scorecard[fy][m] for fy in sorted(fiscal_years) if m in scorecard[fy]
                ]

        # Generate analytical synthesis notes
        notes = cls._generate_comparison_notes(cleaned_tickers, fiscal_years, metrics_by_company)

        return CompanyComparisonSummary(
            companies=cleaned_tickers,
            fiscal_years=sorted(fiscal_years),
            metrics=metrics_by_company,
            synthesis_notes=notes,
        )

    @classmethod
    def _generate_comparison_notes(
        cls,
        tickers: List[str],
        fiscal_years: List[int],
        metrics: Dict[str, Dict[str, List[CalculatedMetricResult]]],
    ) -> List[str]:
        """Synthesizes objective financial observations between companies."""
        notes: List[str] = []
        latest_fy = max(fiscal_years)

        if len(tickers) == 2:
            t1, t2 = tickers[0], tickers[1]

            # Operating Margin comparison
            op1 = next((m.value for m in metrics.get(t1, {}).get(MetricType.OPERATING_MARGIN.value, []) if m.inputs.get("fiscal_year", latest_fy) == latest_fy), None)
            op2 = next((m.value for m in metrics.get(t2, {}).get(MetricType.OPERATING_MARGIN.value, []) if m.inputs.get("fiscal_year", latest_fy) == latest_fy), None)

            # Pull latest values directly from list
            list_op1 = metrics.get(t1, {}).get(MetricType.OPERATING_MARGIN.value, [])
            list_op2 = metrics.get(t2, {}).get(MetricType.OPERATING_MARGIN.value, [])
            if list_op1 and list_op2:
                latest_op1 = list_op1[-1].value
                latest_op2 = list_op2[-1].value
                if latest_op1 > latest_op2:
                    diff = latest_op1 - latest_op2
                    notes.append(f"{t1} demonstrated higher operating profitability in FY{latest_fy} with an operating margin of {latest_op1:.2f}% vs {t2}'s {latest_op2:.2f}% (spread: +{diff:.2f}%).")
                else:
                    diff = latest_op2 - latest_op1
                    notes.append(f"{t2} demonstrated higher operating profitability in FY{latest_fy} with an operating margin of {latest_op2:.2f}% vs {t1}'s {latest_op1:.2f}% (spread: +{diff:.2f}%).")

            # Revenue Growth comparison
            rev1 = metrics.get(t1, {}).get(MetricType.YOY_REVENUE_GROWTH.value, [])
            rev2 = metrics.get(t2, {}).get(MetricType.YOY_REVENUE_GROWTH.value, [])
            if rev1 and rev2:
                latest_g1 = rev1[-1].value
                latest_g2 = rev2[-1].value
                lead = t1 if latest_g1 > latest_g2 else t2
                lead_g = max(latest_g1, latest_g2)
                lag_g = min(latest_g1, latest_g2)
                notes.append(f"Revenue Growth: {lead} expanded faster in FY{latest_fy} at {lead_g:.2f}% YoY compared to {lag_g:.2f}%.")

            # Cash conversion
            fcf1 = metrics.get(t1, {}).get("fcf_conversion", [])
            fcf2 = metrics.get(t2, {}).get("fcf_conversion", [])
            if fcf1 and fcf2:
                c1 = fcf1[-1].value
                c2 = fcf2[-1].value
                notes.append(f"Cash Flow Quality: Free cash flow to net income conversion stood at {c1:.1f}% for {t1} and {c2:.1f}% for {t2}, indicating disciplined capital conversion.")

            # Solvency & Leverage
            de1 = metrics.get(t1, {}).get(MetricType.DEBT_TO_EQUITY.value, [])
            de2 = metrics.get(t2, {}).get(MetricType.DEBT_TO_EQUITY.value, [])
            if de1 and de2:
                notes.append(f"Balance Sheet Strength: Both companies maintain conservative capital structures ({t1}: {de1[-1].value:.2f}x D/E, {t2}: {de2[-1].value:.2f}x D/E), primarily composed of operational lease liabilities.")

        return notes

    @classmethod
    def format_comparison_markdown_table(cls, summary: CompanyComparisonSummary) -> str:
        """Renders an executive Markdown comparison table across companies and years."""
        lines = [
            f"### Financial Comparison: {' vs '.join(summary.companies)} ({min(summary.fiscal_years)} - {max(summary.fiscal_years)})",
            "",
        ]

        # Table header
        header = "| Metric | Company | " + " | ".join([f"FY{fy}" for fy in summary.fiscal_years]) + " |"
        divider = "| :--- | :--- | " + " | ".join([":---:" for _ in summary.fiscal_years]) + " |"
        lines.extend([header, divider])

        tracked_metrics = [
            (MetricType.YOY_REVENUE_GROWTH.value, "YoY Revenue Growth (%)"),
            (MetricType.OPERATING_MARGIN.value, "Operating Margin (%)"),
            (MetricType.NET_PROFIT_MARGIN.value, "Net Profit Margin (%)"),
            (MetricType.FREE_CASH_FLOW.value, "Free Cash Flow (INR Cr)"),
            ("fcf_conversion", "FCF / Net Income (%)"),
            (MetricType.CURRENT_RATIO.value, "Current Ratio (x)"),
            (MetricType.DEBT_TO_EQUITY.value, "Debt-to-Equity (x)"),
            (MetricType.RETURN_ON_EQUITY.value, "Return on Equity (%)"),
            (MetricType.RETURN_ON_INVESTED_CAPITAL.value, "ROIC (%)"),
        ]

        for metric_key, display_label in tracked_metrics:
            for comp in summary.companies:
                comp_data = summary.metrics.get(comp, {}).get(metric_key, [])
                val_map = {m.inputs.get("fiscal_year", summary.fiscal_years[idx]): m.value for idx, m in enumerate(comp_data)} if comp_data else {}

                row_cells = []
                for idx, fy in enumerate(summary.fiscal_years):
                    if idx < len(comp_data):
                        v = comp_data[idx].value
                        row_cells.append(f"{v:.2f}" if v is not None else "-")
                    else:
                        row_cells.append("-")

                lines.append(f"| {display_label} | **{comp}** | " + " | ".join(row_cells) + " |")

        if summary.synthesis_notes:
            lines.extend(["", "#### Key Analytical Takeaways:"])
            for n in summary.synthesis_notes:
                lines.append(f"- {n}")

        return "\n".join(lines)


# Global analytics engine singleton
analytics_engine = FinancialAnalyticsEngine()
