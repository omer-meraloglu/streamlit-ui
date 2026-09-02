"""Result tiles, review badge and driver chart.

Tiles map to the target columns of `datascientiststeamproject.kaggle.Steam`.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from demo import GameSpec, PreviewNumbers
from utils.constants import STEAM
from utils.formatting import compact_number, money, price_label, review_summary


def _tile(
    label: str, value: str, sub: str = "", tone: str = "", compact: bool = False
) -> str:
    tone_class = f" is-{tone}" if tone else ""
    value_class = " compact" if compact else ""
    return (
        f'<div class="metric-tile{tone_class}">'
        f'<div class="label">{label}</div>'
        f'<div class="value{value_class}">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f"</div>"
    )


def render_empty_state() -> None:
    st.markdown(
        '<div class="steam-card" style="text-align:center;padding:46px 20px">'
        '<div style="font-size:15px;color:#c7d5e0">'
        "Configure a store page on the left, then hit "
        "<b>Predict performance</b>.</div>"
        '<div style="font-size:13px;color:#8f98a0;margin-top:8px">'
        "Sample values only — estimated_owners, positive / negative, peak_ccu, "
        "recommendations and metacritic_score.</div></div>",
        unsafe_allow_html=True,
    )


def render_results(
    numbers: PreviewNumbers, spec: GameSpec, settings: dict
) -> None:
    label, color = review_summary(numbers.positive_ratio, numbers.total_reviews)
    pct = numbers.positive_ratio * 100
    effective_price = 0.0 if spec.price_status != "Paid" else spec.price

    # ---- headline: the game as a store capsule ---------------------------
    tags_html = "".join(
        f'<span class="capsule">{t}</span>' for t in (spec.genres + spec.tags)[:8]
    )
    byline = spec.developers or "Unknown developer"
    if spec.publishers and spec.publishers != spec.developers:
        byline += f" &nbsp;&middot;&nbsp; {spec.publishers}"

    st.markdown(
        f'<div class="steam-card">'
        f'<div style="font-size:22px;color:#fff">{spec.name}</div>'
        f'<div style="font-size:13px;color:#8f98a0;margin:4px 0 10px">'
        f"{price_label(effective_price)} &nbsp;&middot;&nbsp; "
        f"{spec.release_date:%d %b %Y} &nbsp;&middot;&nbsp; "
        f'{", ".join(spec.platforms)} &nbsp;&middot;&nbsp; {byline}</div>'
        f"{tags_html}</div>",
        unsafe_allow_html=True,
    )

    # ---- metric tiles ------------------------------------------------------
    owners_sub = (
        "bucket · owners" if settings["show_intervals"] else "estimated_owners"
    )

    # 3 per row x 2 rows, so the six targets fit without scrolling.
    top = st.columns(3)
    with top[0]:
        st.markdown(
            _tile(
                "estimated_owners",
                numbers.estimated_owners,
                owners_sub,
                compact=True,
            ),
            unsafe_allow_html=True,
        )
    with top[1]:
        st.markdown(
            _tile(
                "revenue",
                money(numbers.revenue),
                "owners × price",
                tone="green",
            ),
            unsafe_allow_html=True,
        )
    with top[2]:
        st.markdown(
            _tile("review score", f"{pct:.0f}%", label, tone="gold"),
            unsafe_allow_html=True,
        )

    bottom = st.columns(3)
    with bottom[0]:
        st.markdown(
            _tile("peak_ccu", compact_number(numbers.peak_ccu), "at launch"),
            unsafe_allow_html=True,
        )
    with bottom[1]:
        st.markdown(
            _tile(
                "metacritic_score",
                str(numbers.metacritic_score),
                f"user_score {numbers.user_score}",
            ),
            unsafe_allow_html=True,
        )
    with bottom[2]:
        st.markdown(
            _tile(
                "recommendations",
                compact_number(numbers.recommendations),
                "store recs",
            ),
            unsafe_allow_html=True,
        )

    # ---- review summary, styled like the store ----------------------------
    st.markdown(
        f'<div class="steam-card">'
        f'<div class="card-title">Expected review summary</div>'
        f'<div class="sentiment-row">'
        f'<span class="sentiment-badge" style="color:{color}">{label}</span>'
        f'<span class="sentiment-meta">{numbers.positive:,} positive / '
        f"{numbers.negative:,} negative &nbsp;&middot;&nbsp; "
        f"{numbers.total_reviews:,} total</span>"
        f"</div>"
        f'<div class="score-bar"><div style="width:{pct:.1f}%;'
        f'background:{color}"></div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- drivers -----------------------------------------------------------
    if settings["show_drivers"]:
        st.markdown(
            '<div class="card-title" style="margin-top:8px">Drivers</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(_driver_chart(numbers.drivers))
        st.caption("Sample weights for layout purposes.")


def _driver_chart(drivers: dict[str, float]) -> alt.Chart:
    frame = pd.DataFrame(
        {"feature": list(drivers), "impact": list(drivers.values())}
    )
    frame["direction"] = frame["impact"].apply(
        lambda v: "Increases" if v >= 0 else "Decreases"
    )

    return (
        alt.Chart(frame)
        # No fixed mark height: at this chart height a hard 16px collapses the
        # seven bands onto one row. Let the band scale size the bars.
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("impact:Q", title="Impact on estimated_owners"),
            # labelLimit: column names like supported_languages get an ellipsis
            # at Altair's 180px default once the legend eats the width.
            y=alt.Y(
                "feature:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=200),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Increases", "Decreases"],
                    range=[STEAM["blue"], STEAM["rust"]],
                ),
                legend=alt.Legend(title=None, orient="right"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Column"),
                alt.Tooltip("impact:Q", title="Impact", format="+.2f"),
            ],
        )
        .properties(height=175, width="container")
    )
