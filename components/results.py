"""Result tiles, review badge and driver chart.

Every value shown is a model output or is derived from one:

  estimated_owners   owners model  (a bucket, or the band an index rounds to)
  review score       review model  (a percentage, or a predicted band)
  suggested price    price model   (regressor on log1p(price), inverted)
  confidence         owners model  predict_proba on the chosen bucket
  revenue            derived: the owners range x the price you entered
  drivers            SHAP values, for whichever of the three is picked

The error-margins card reports the miss measured on the held-out set when
training, shipped inside the bundles -- see `Prediction.price_interval` and
`review_interval`. Model sets that carry no margins simply do not get the card.

A model set predicts owners and reviews in one of two shapes (see
`utils.model_backend`), and both are rendered here: a classifier's bucket plus
its probability, or an ordinal band index plus the two bands it falls between.
The percentage badge and the 0-100 bar only appear for a set that predicts a
percentage; an ordinal set gets its band and the position of that band on the
scale instead.
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

# Which model the drivers panel explains. SHAP is computed for all three on
# every prediction (~23ms together), so switching is just a redraw.
#
# The unit differs per model and has to be said out loud: the owners model is a
# classifier, so its values are log-odds; the price model was fitted on
# log1p(price), so its values are not dollars.
DRIVER_MODELS = {
    "estimated owners": "owners",
    "review score": "review",
    "suggested price": "price",
}
# Axis title and unit per target. Both depend on the shape the loaded set
# predicts in, not just on which model is being explained: a classifier's SHAP
# values are log-odds, an ordinal regressor's are band index.
def _driver_axis(target: str, prediction: Prediction) -> str:
    if target == "owners":
        if prediction.owners_confidence is None:
            return "SHAP impact on the owners band index"
        return "SHAP impact on the predicted owners bucket"
    if target == "review":
        if prediction.review_pct is None:
            return "SHAP impact on the review band index"
        return "SHAP impact on review score"
    return "SHAP impact on log1p(price)"


def _driver_unit(target: str, prediction: Prediction) -> str:
    if target == "owners":
        if prediction.owners_confidence is None:
            return "band index, not a number of owners"
        return "log-odds on the predicted bucket, not owners"
    if target == "review":
        if prediction.review_pct is None:
            return "band index, not a percentage"
        return "percentage points of positive reviews"
    return "on log1p(price), not dollars"


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
    ordinal_review = prediction.review_pct is None
    if ordinal_review:
        bands = prediction.review_bands
        label = prediction.review_band or "—"
        # Position on the band scale, so the bar and the colour still mean
        # something without a percentage behind them.
        pct = 100.0 * prediction.review_index / max(len(bands) - 1, 1)
        color = review_label(pct)[1]
    else:
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
    if not settings["show_intervals"] or not wider:
        owners_sub = "predicted owner range"
    elif prediction.owners_confidence is None:
        owners_sub = (
            "falls between "
            f"{range_label(prediction.spread_low, prediction.spread_high)}"
        )
    else:
        owners_sub = (
            "80% of mass: "
            f"{range_label(prediction.spread_low, prediction.spread_high)}"
        )

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
        if ordinal_review:
            top_band = len(prediction.review_bands) - 1
            st.markdown(
                _tile(
                    "review score", label,
                    f"band {prediction.review_index:.1f} of {top_band}",
                    tone="gold", compact=True,
                ),
                unsafe_allow_html=True,
            )
        else:
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
        if prediction.owners_confidence is None:
            # A regressor over band indexes has no per-bucket probability;
            # the index itself is the closest thing to report.
            top_band = len(prediction.review_bands or [0] * 5) - 1
            st.markdown(
                _tile(
                    "owners band",
                    f"{prediction.owners_index:.2f}",
                    f"ordinal index, 0–{top_band}",
                ),
                unsafe_allow_html=True,
            )
        else:
            conf_sub = "on the owners bucket"
            average = prediction.owners_mean_confidence
            if average:
                conf_sub = f"on the owners bucket · {average:.0%} average"
            st.markdown(
                _tile("confidence", f"{prediction.owners_confidence:.0%}", conf_sub),
                unsafe_allow_html=True,
            )

    if settings["show_intervals"] and prediction.has_margins:
        _render_intervals(prediction)

    # ---- review summary, styled like the store ----------------------------
    # Review *counts* are not a trained target, so the badge reports the
    # predicted percentage rather than a positive / negative split.
    if ordinal_review:
        meta = (
            f"band {prediction.review_index:.2f} of "
            f"{len(prediction.review_bands) - 1} &nbsp;&middot;&nbsp; "
            "predicted by the review model"
        )
    else:
        meta = (
            f"{pct:.1f}% positive &nbsp;&middot;&nbsp; "
            "predicted by the review model"
        )
    st.markdown(
        f'<div class="steam-card">'
        f'<div class="card-title">Expected review summary</div>'
        f'<div class="sentiment-row">'
        f'<span class="sentiment-badge" style="color:{color}">{label}</span>'
        f'<span class="sentiment-meta">{meta}</span>'
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
        # The label is collapsed: the card title above already says Drivers,
        # and the three options name themselves.
        choice = st.segmented_control(
            "drivers for",
            list(DRIVER_MODELS),
            default="estimated owners",
            key="drivers_model",
            label_visibility="collapsed",
        )
        target = DRIVER_MODELS.get(choice or "estimated owners", "owners")
        drivers = prediction.drivers.get(target, {})

        if drivers:
            st.altair_chart(
                _driver_chart(drivers, _driver_axis(target, prediction))
            )
            st.caption(
                f"SHAP values for the {target} model ({prediction.backend}) — "
                f"{_driver_unit(target, prediction)}. release_year and app_id "
                "are left out: neither is a choice you make."
            )
        else:
            st.caption(
                f"No SHAP explainer available for the {target} model in this "
                "model set."
            )


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

    What each row can honestly say depends on what its bundle measured:

    * a price or percentage regressor ships `mae` / `confidence_95`, so its
      row is the estimate ± MAE -- the average miss -- with the far wider 95%
      band (1.96σ of the same residuals) beside it rather than as the headline.
    * a classifier ships no residual to take a σ of, so its row is the buckets
      holding the middle 80% of the predicted probability.
    * an ordinal band model ships neither. Its row is the two bands the
      prediction falls between, which comes from the prediction itself, and
      the spread beside it is how widely that model's predictions were spread
      when trained -- context, not an error, and said as much.
    """
    rows, notes = [], []

    owners = prediction.margins.get("owners", {})
    if owners and prediction.owners_confidence is None:
        rows.append(_interval_row(
            "estimated_owners",
            range_label(prediction.spread_low, prediction.spread_high),
            f"index {prediction.owners_index:.2f}, so between these two bands · "
            f"predictions spread ±{prediction.owners_spread:.2f} bands",
        ))
    elif owners:
        rows.append(_interval_row(
            "estimated_owners",
            range_label(prediction.spread_low, prediction.spread_high),
            f"{prediction.owners_confidence:.0%} on the picked bucket · "
            f"{owners.get('mean_confidence', 0):.0%} average when trained",
        ))

    review = prediction.margins.get("review", {})
    if review and prediction.review_pct is None:
        bands = prediction.review_bands
        index = prediction.review_index
        low = bands[max(int(index), 0)]
        high = bands[min(int(index) + 1, len(bands) - 1)]
        rows.append(_interval_row(
            "review score",
            low if low == high else f"{low} – {high}",
            f"index {index:.2f}, so between these two bands · "
            f"predictions spread ±{prediction.review_spread:.2f} bands",
        ))
    elif review:
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

    # Only explain the kinds of row actually on screen -- which ones appear
    # depends on the shapes this model set predicts in.
    if prediction.price_margin or prediction.review_margin:
        notes.append(
            "A miss (MAE) was measured on the held-out set when "
            f"{prediction.backend} was trained, not computed for this "
            "configuration; the 95% band is wider because a few predictions "
            "miss badly."
        )
    if prediction.owners_confidence is None or prediction.review_pct is None:
        notes.append(
            "A row quoting bands has no error to report — that model predicts "
            "an index, and the range is the two bands the index falls between."
        )

    st.markdown(
        '<div class="steam-card">'
        '<div class="card-title">Error margins</div>'
        + "".join(rows)
        + f'<div class="interval-note">{" ".join(notes)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _driver_chart(drivers: dict[str, float], axis_title: str) -> alt.Chart:
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
            x=alt.X("impact:Q", title=axis_title),
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
