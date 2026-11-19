"""
Currency, Unit, and Period Normalization for FinSight AI
Ensures consistent financial arithmetic across varying reporting standards,
currencies, units (Crores, Millions, Billions), and reporting fiscal intervals.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple
from core.exceptions import UnitMismatchError


class UnitNormalizer:
    """Standardizes financial units and scales to prevent dimensional errors."""

    # Unit multipliers relative to base unit (1.0)
    UNIT_MULTIPLIERS: Dict[str, float] = {
        "raw": 1.0,
        "single": 1.0,
        "unit": 1.0,
        "inr": 1.0,
        "usd": 1.0,
        "thousand": 1_000.0,
        "k": 1_000.0,
        "lakh": 100_000.0,
        "million": 1_000_000.0,
        "mn": 1_000_000.0,
        "m": 1_000_000.0,
        "crore": 10_000_000.0,
        "cr": 10_000_000.0,
        "billion": 1_000_000_000.0,
        "bn": 1_000_000_000.0,
        "b": 1_000_000_000.0,
        "trillion": 1_000_000_000_000.0,
        "percentage": 1.0,
        "%": 1.0,
        "ratio": 1.0,
        "x": 1.0,
    }

    # Reference FX rates (for comparative displays only, labeled as estimates)
    REFERENCE_FX_TO_INR: Dict[str, float] = {
        "INR": 1.0,
        "USD": 83.5,
        "EUR": 90.2,
        "GBP": 106.0,
    }

    @classmethod
    def normalize_unit_string(cls, unit_str: str) -> str:
        """Cleans and standardizes unit strings."""
        if not unit_str:
            return "raw"
        cleaned = unit_str.lower().strip().replace(".", "")
        if cleaned in ("cr", "crores", "crore"):
            return "crore"
        if cleaned in ("mn", "million", "millions"):
            return "million"
        if cleaned in ("bn", "billion", "billions"):
            return "billion"
        if cleaned in ("k", "thousand", "thousands"):
            return "thousand"
        if cleaned in ("lakh", "lakhs", "lac", "lacs"):
            return "lakh"
        if cleaned in ("%", "pct", "percent", "percentage"):
            return "%"
        if cleaned in ("x", "ratio", "times"):
            return "x"
        return cleaned

    @classmethod
    def convert_amount(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Converts an amount from one scale unit to another (e.g. million to crore)."""
        u_from = cls.normalize_unit_string(from_unit)
        u_to = cls.normalize_unit_string(to_unit)

        if u_from == u_to:
            return value

        mult_from = cls.UNIT_MULTIPLIERS.get(u_from)
        mult_to = cls.UNIT_MULTIPLIERS.get(u_to)

        if mult_from is None or mult_to is None:
            raise UnitMismatchError(
                f"Cannot convert between unmapped units: '{from_unit}' -> '{to_unit}'"
            )

        # Convert to base, then to target
        base_value = value * mult_from
        return base_value / mult_to

    @classmethod
    def align_units(
        cls,
        val1: float,
        unit1: str,
        val2: float,
        unit2: str,
        target_unit: Optional[str] = None,
    ) -> Tuple[float, float, str]:
        """Aligns two financial figures to the same unit for arithmetic operations."""
        u1 = cls.normalize_unit_string(unit1)
        u2 = cls.normalize_unit_string(unit2)

        if u1 == u2:
            return val1, val2, u1

        chosen_unit = target_unit or u1
        norm_val1 = cls.convert_amount(val1, u1, chosen_unit)
        norm_val2 = cls.convert_amount(val2, u2, chosen_unit)
        return norm_val1, norm_val2, chosen_unit

    @classmethod
    def normalize_fiscal_year(cls, period_str: str | int) -> int:
        """Parses fiscal year representations (e.g., 'FY24', 'FY2024', '2023-24', 2024) to integer 2024."""
        if isinstance(period_str, int):
            return period_str
        s = str(period_str).strip().upper()

        # Check for range format like '2023-24' or '2023-2024' (ending year is standard FY)
        m_range = re.search(r"(\d{4})\s*[-/]\s*(\d{2,4})", s)
        if m_range:
            end_yr = m_range.group(2)
            if len(end_yr) == 2:
                century = int(m_range.group(1)[:2]) * 100
                return century + int(end_yr)
            return int(end_yr)

        # Check for 'FY24' or 'FY 2024'
        m_fy = re.search(r"FY\s*(\d{2,4})", s)
        if m_fy:
            yr = m_fy.group(1)
            if len(yr) == 2:
                return 2000 + int(yr)
            return int(yr)

        # Look for standalone 4-digit year (e.g. 2024)
        m_year = re.search(r"\b(19\d{2}|20\d{2})\b", s)
        if m_year:
            return int(m_year.group(1))

        raise ValueError(f"Could not parse fiscal year from: '{period_str}'")
