"""Top nav bar and sidebar."""

from __future__ import annotations

import streamlit as st

APP_NAME = "STEAM<span>CAST</span>"
TAGLINE = "Sales &amp; review forecasting for Steam releases"


def render_header() -> None:
    st.markdown(
        f"""
        <div class="steam-nav">
          <div class="brand">{APP_NAME}<small>{TAGLINE}</small></div>
          <div class="nav-links"><span class="active">Predict</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(backends: dict) -> dict:
    """Model picker + display toggles. Returns what the result panel reads."""
    with st.sidebar:
        st.markdown("## Model")
        names = list(backends)
        backend = st.radio(
            "trained model set", names, index=0,
            help="Both are trained by Steam-Price-Popularity-Predictor: "
                 "saved_models (LightGBM) and saved_models_xgb (XGBoost).",
        )
        # Two roots can both hold models; show the path so it is obvious
        # which one is live.
        st.caption(f"`{backends[backend]}`")

        st.divider()
        st.markdown("## Display")
        show_intervals = st.toggle("Show ranges", value=True)
        show_drivers = st.toggle("Show drivers", value=True)

        st.divider()
        st.markdown("## About")
        st.caption(
            "Predictions come from the three models trained by "
            "`2_train_models.py`: owners (classifier), review percentage and "
            "price (regressors). Revenue is derived, not predicted."
        )

        return {
            "backend": backend,
            "show_intervals": show_intervals,
            "show_drivers": show_drivers,
        }
