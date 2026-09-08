"""Display helpers: Steam's review vocabulary and number formatting."""

from __future__ import annotations

from .constants import SENTIMENT_COLORS


def review_label(pct: float) -> tuple[str, str]:
    """Map a predicted positive percentage onto Steam's review summary label.

    The store also gates its extreme labels ("Overwhelmingly", "Very") on review
    volume, but review *count* is not one of the trained targets -- only
    `positive_review_percentage` is. So the bands below are applied on the
    percentage alone, and the volume-gated wording is not claimed.

    Returns (label, hex_colour).
    """
    if pct >= 95:
        label, tone = "Overwhelmingly Positive", "positive"
    elif pct >= 80:
        label, tone = "Very Positive", "positive"
    elif pct >= 70:
        label, tone = "Mostly Positive", "positive"
    elif pct >= 40:
        label, tone = "Mixed", "mixed"
    elif pct >= 20:
        label, tone = "Mostly Negative", "negative"
    else:
        label, tone = "Overwhelmingly Negative", "negative"

    return label, SENTIMENT_COLORS[tone]


def compact_number(value: float) -> str:
    """1_234_567 -> '1.2M'. Used on the metric tiles."""
    value = float(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            scaled = value / threshold
            precision = 0 if abs(scaled) >= 100 else 1
            return f"{scaled:,.{precision}f}{suffix}"
    return f"{value:,.0f}"


def money(value: float) -> str:
    return f"${compact_number(value)}"


def price_label(price: float) -> str:
    return "Free to Play" if price <= 0 else f"${price:,.2f}"
