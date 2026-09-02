"""Display helpers: Steam's review vocabulary and number formatting."""

from __future__ import annotations

from .constants import SENTIMENT_COLORS


def review_summary(positive_ratio: float, total_reviews: int) -> tuple[str, str]:
    """Map a positive ratio + review count onto Steam's review summary label.

    Reproduces the store's published thresholds: the extreme labels
    ("Overwhelmingly", "Very") need a minimum review volume, otherwise the
    plain label is used.

    Returns (label, hex_colour).
    """
    pct = positive_ratio * 100 if positive_ratio <= 1 else positive_ratio

    if total_reviews < 10:
        return "Need more reviews", SENTIMENT_COLORS["mixed"]

    if pct >= 95 and total_reviews >= 500:
        label, tone = "Overwhelmingly Positive", "positive"
    elif pct >= 80 and total_reviews >= 50:
        label, tone = "Very Positive", "positive"
    elif pct >= 80:
        label, tone = "Positive", "positive"
    elif pct >= 70:
        label, tone = "Mostly Positive", "positive"
    elif pct >= 40:
        label, tone = "Mixed", "mixed"
    elif pct >= 20:
        label, tone = "Mostly Negative", "negative"
    elif total_reviews >= 500:
        label, tone = "Overwhelmingly Negative", "negative"
    elif total_reviews >= 50:
        label, tone = "Very Negative", "negative"
    else:
        label, tone = "Negative", "negative"

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
