"""Result tiles, review badge and driver chart.

Every value shown is a model output or is derived from one:

  estimated_owners   owners model  (classifier -> the bucket's own range)
  review score       review model  (regressor, positive_review_percentage)
  suggested price    price model   (regressor on log1p(price), inverted)
  confidence         owners model  predict_proba on the chosen bucket
  revenue            derived: the owners range x the price you entered
  drivers            SHAP values for the owners model

The error-margins card reports the miss measured on the held-out set when
training, shipped inside the bundles -- see `Prediction.price_interval` and
`review_interval`. Model sets that carry no margins simply do not get the card.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from demo import GameSpec
from utils.constants import STEAM
from utils.formatting import (
    money,
    price_label,
    range_label,
    review_label,
)
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
    price_source = (
        "the model's suggestion" if prediction.price_is_suggested else "your price"
    )

    # ---- headline capsule: price, platforms and the picked chips ----------
    tags_html = "".join(
        f'<span class="capsule">{t}</span>' for t in (spec.genres + spec.tags)[:8]
    )
    st.markdown(
        f'<div class="steam-card">'
        f'<div style="font-size:24px;color:#fff">'
        f"{price_label(prediction.entered_price)}"
        f'<span style="font-size:14px;color:#8f98a0"> &nbsp;&middot;&nbsp; '
        f"{price_source} &nbsp;&middot;&nbsp; "
        f'{", ".join(spec.platforms)}</span></div>'
        f'<div style="margin-top:9px">{tags_html}</div></div>',
        unsafe_allow_html=True,
    )

    # ---- metric tiles ------------------------------------------------------
    owners_value = range_label(prediction.owners_low, prediction.owners_high)
    # The wider span is only worth showing when it adds something: with 99% of
    # the mass on one bucket it is identical to the bucket itself.
    wider = (
        prediction.spread_low < prediction.owners_low
        or prediction.spread_high > prediction.owners_high
    )
    if settings["show_intervals"] and wider:
        owners_sub = (
            "80% of mass: "
            f"{range_label(prediction.spread_low, prediction.spread_high)}"
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
                range_label(prediction.revenue_low, prediction.revenue_high, money),
                f"owner range × {price_source}",
                tone="green",
                compact=True,
            ),
            unsafe_allow_html=True,
        )
    with top[2]:
        review_sub = label
        if settings["show_intervals"] and prediction.review_margin:
            low, high = prediction.review_interval
            review_sub = f"{label} · typically {low:.0f}–{high:.0f}%"
        st.markdown(
            _tile("review score", f"{pct:.0f}%", review_sub, tone="gold"),
            unsafe_allow_html=True,
        )

    bottom = st.columns(3)
    with bottom[0]:
        price_sub = "what the price model would charge"
        if settings["show_intervals"] and prediction.price_margin:
            price_sub = f"typically ±${prediction.price_margin:,.2f}"
        st.markdown(
            _tile(
                "suggested price",
                price_label(prediction.suggested_price),
                price_sub,
            ),
            unsafe_allow_html=True,
        )
    with bottom[1]:
        gap = prediction.price_gap
        if prediction.price_is_suggested:
            # Nothing was typed in, so there is no gap to report -- the entered
            # price IS the suggestion and the difference is always zero.
            gap_tile = _tile("price gap", "—", "no price entered")
        else:
            gap_tile = _tile(
                "price gap",
                f"{gap:+,.2f}",
                "above suggested" if gap >= 0 else "below suggested",
                tone="green" if abs(gap) < 2 else "",
            )
        st.markdown(gap_tile, unsafe_allow_html=True)
    with bottom[2]:
        conf_sub = "on the owners bucket"
        average = prediction.owners_mean_confidence
        if average:
            conf_sub = f"on the owners bucket · {average:.0%} average"
        st.markdown(
            _tile(
                "confidence",
                f"{prediction.owners_confidence:.0%}",
                conf_sub,
            ),
            unsafe_allow_html=True,
        )

    if settings["show_intervals"] and prediction.has_margins:
        _render_intervals(prediction)

    # ---- review summary, styled like the store ----------------------------
    # Review *counts* are not a trained target, so the badge reports the
    # predicted percentage rather than a positive / negative split.
    st.markdown(
        f'<div class="steam-card">'
        f'<div class="card-title">Expected review summary</div>'
        f'<div class="sentiment-row">'
        f'<span class="sentiment-badge" style="color:{color}">{label}</span>'
        f'<span class="sentiment-meta">{pct:.1f}% positive &nbsp;&middot;&nbsp; '
        f"predicted by the review model</span>"
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
                f"SHAP values for the owners model ({prediction.backend}) on the "
                "predicted bucket — log-odds, not owners. release_year and "
                "app_id are left out: neither is a choice you make."
            )
        else:
            st.caption("No SHAP explainer available for this model set.")


def _interval_row(label: str, value: str, basis: str) -> str:
    return (
        f'<div class="interval-row">'
        f'<span class="interval-label">{label}</span>'
        f'<span class="interval-value">{value}</span>'
        f'<span class="interval-basis">{basis}</span>'
        f"</div>"
    )


def _render_intervals(prediction: Prediction) -> None:
    """The three predictions, each with the error margin it was trained with.

    The two regressors ship `mae` / `std` / `confidence_95` measured on the
    held-out set. The band shown is the point estimate ± MAE -- the average
    miss -- with the far wider 95% band (1.96σ of the same residuals) reported
    beside it rather than as the headline. The owners model is a classifier, so
    there is no residual to take a σ of: its band is the set of buckets holding
    the middle 80% of the predicted probability.
    """
    rows = []

    owners = prediction.margins.get("owners", {})
    if owners:
        rows.append(_interval_row(
            "estimated_owners",
            range_label(prediction.spread_low, prediction.spread_high),
            f"{prediction.owners_confidence:.0%} on the picked bucket · "
            f"{owners.get('mean_confidence', 0):.0%} average when trained",
        ))

    review = prediction.margins.get("review", {})
    if review:
        low, high = prediction.review_interval
        rows.append(_interval_row(
            "review score",
            f"{low:.1f}–{high:.1f}%",
            f"±{prediction.review_margin:.1f} pts typical · "
            f"95% band ±{prediction.review_band95:.1f}",
        ))

    price = prediction.margins.get("price", {})
    if price:
        low, high = prediction.price_interval
        rows.append(_interval_row(
            "suggested price",
            # not price_label(): a lower edge clipped to 0 is the bottom of an
            # interval, not a free-to-play release
            f"${low:,.2f}–${high:,.2f}",
            f"±${prediction.price_margin:,.2f} typical · "
            f"95% band ±${prediction.price_band95:,.2f}",
        ))

    if not rows:
        return

    st.markdown(
        '<div class="steam-card">'
        '<div class="card-title">Error margins</div>'
        + "".join(rows)
        + '<div class="interval-note">The band is the average miss (MAE) on the '
          f"held-out set when {prediction.backend} was trained, not an error "
          "computed for this configuration. The 95% band is wider because a "
          "few predictions miss badly.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


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
        # ~28px a band: at 12 drivers anything tighter makes Altair drop every
        # other axis label.
        .properties(height=340, width="container")
    )
