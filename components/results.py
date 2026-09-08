"""Result tiles, review badge and driver chart.

Every value shown is a model output or is derived from one:

  estimated_owners   model_owners  (classifier -> the bucket's own range)
  review score       model_review  (regressor, positive_review_percentage)
  suggested price    model_price   (regressor on log1p(price), inverted)
  confidence         model_owners  predict_proba on the chosen bucket
  revenue            derived: the owners range x the price you entered
  drivers            SHAP values from explainer_owners
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from demo import GameSpec
from utils.constants import STEAM
from utils.formatting import compact_number, money, price_label, review_label
from utils.model_backend import Prediction


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
        "Scored by the trained models — estimated_owners, "
        "positive_review_percentage and price.</div></div>",
        unsafe_allow_html=True,
    )


def render_results(
    prediction: Prediction, spec: GameSpec, settings: dict
) -> None:
    label, color = review_label(prediction.review_pct)
    pct = prediction.review_pct

    # ---- headline capsule: price, platforms and the picked chips ----------
    tags_html = "".join(
        f'<span class="capsule">{t}</span>' for t in (spec.genres + spec.tags)[:8]
    )
    st.markdown(
        f'<div class="steam-card">'
        f'<div style="font-size:24px;color:#fff">'
        f"{price_label(prediction.entered_price)}"
        f'<span style="font-size:14px;color:#8f98a0"> &nbsp;&middot;&nbsp; '
        f'{", ".join(spec.platforms)}</span></div>'
        f'<div style="margin-top:9px">{tags_html}</div></div>',
        unsafe_allow_html=True,
    )

    # ---- metric tiles ------------------------------------------------------
    owners_value = (
        f"{compact_number(prediction.owners_low)}–"
        f"{compact_number(prediction.owners_high)}"
    )
    # The wider span is only worth showing when it adds something: with 99% of
    # the mass on one bucket it is identical to the bucket itself.
    wider = (
        prediction.spread_low < prediction.owners_low
        or prediction.spread_high > prediction.owners_high
    )
    if settings["show_intervals"] and wider:
        owners_sub = (
            f"80% of mass: {compact_number(prediction.spread_low)}–"
            f"{compact_number(prediction.spread_high)}"
        )
    else:
        owners_sub = "predicted owner range"

    # 3 per row x 2 rows, so the six values fit without scrolling.
    top = st.columns(3)
    with top[0]:
        st.markdown(
            _tile("estimated_owners", owners_value, owners_sub, compact=True),
            unsafe_allow_html=True,
        )
    with top[1]:
        st.markdown(
            _tile(
                "revenue",
                f"{money(prediction.revenue_low)}–{money(prediction.revenue_high)}",
                "owner range × your price",
                tone="green",
                compact=True,
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
            _tile(
                "suggested price",
                price_label(prediction.suggested_price),
                "what model_price would charge",
            ),
            unsafe_allow_html=True,
        )
    with bottom[1]:
        gap = prediction.price_gap
        st.markdown(
            _tile(
                "price gap",
                f"{gap:+,.2f}",
                "above suggested" if gap >= 0 else "below suggested",
                tone="green" if abs(gap) < 2 else "",
            ),
            unsafe_allow_html=True,
        )
    with bottom[2]:
        st.markdown(
            _tile(
                "confidence",
                f"{prediction.owners_confidence:.0%}",
                "on the owners bucket",
            ),
            unsafe_allow_html=True,
        )

    # ---- review summary, styled like the store ----------------------------
    # Review *counts* are not a trained target, so the badge reports the
    # predicted percentage rather than a positive / negative split.
    st.markdown(
        f'<div class="steam-card">'
        f'<div class="card-title">Expected review summary</div>'
        f'<div class="sentiment-row">'
        f'<span class="sentiment-badge" style="color:{color}">{label}</span>'
        f'<span class="sentiment-meta">{pct:.1f}% positive &nbsp;&middot;&nbsp; '
        f"predicted by model_review</span>"
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
        if prediction.drivers:
            st.altair_chart(_driver_chart(prediction.drivers))
            st.caption(
                f"SHAP values from explainer_owners ({prediction.backend}) for the "
                "predicted bucket — log-odds, not owners."
            )
        else:
            st.caption("No SHAP explainer available for this model set.")


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
        # bands onto one row. Let the band scale size the bars.
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("impact:Q", title="SHAP impact on predicted owners bucket"),
            # labelLimit: names like "Steam Trading Cards (category)" get an
            # ellipsis at Altair's 180px default once the legend eats the width.
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
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("impact:Q", title="SHAP", format="+.3f"),
            ],
        )
        .properties(height=175, width="container")
    )
