"""SteamCast -- Steam-themed predictor front-end.

Scores a prospective store page with the models trained by
Steam-Price-Popularity-Predictor. Run with:  streamlit run app.py

Point STEAMCAST_MODEL_ROOT at that project's ml_models/ directory if it is not
a sibling of this checkout.
"""

from __future__ import annotations

import streamlit as st

from components.header import render_header, render_settings
from components.inputs import render_feature_form
from components.results import render_empty_state, render_results
from utils.model_backend import (
    available_backends,
    load_bundle,
    predict,
    search_roots,
)
from utils.theme import inject_theme, register_chart_theme

st.set_page_config(
    page_title="SteamCast — Sales & Review Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()
register_chart_theme()

backends = available_backends()
if not backends:
    render_header()
    searched = "\n".join(f"- `{r}`" for r in search_roots())
    st.error(
        "No trained models found. Searched, in order:\n\n"
        f"{searched}\n\n"
        "Run `2_train_models.py` in Steam-Price-Popularity-Predictor, upload the "
        "`bundle_*.pkl` files into this repo's `models/bundles` or "
        "`models/bundles_xgb` directory, or set `STEAMCAST_MODEL_ROOT`."
    )
    st.stop()

render_header()

form_col, result_col = st.columns([1, 1.15], gap="large")

# The settings card sits under the form but has to be *read* first: the picked
# model set decides which genres/tags the form may offer. Claiming both slots
# up front fixes the on-screen order, then each is filled in dependency order.
with form_col:
    form_slot = st.container()
    settings_slot = st.container()

with settings_slot:
    settings = render_settings(backends)

bundle = load_bundle(settings["backend"])

with form_slot:
    spec = render_feature_form(bundle)
    if spec is not None:
        st.session_state["spec"] = spec

with result_col:
    # Re-scored every run, so switching model set updates the panel without
    # needing the form resubmitted.
    if "spec" in st.session_state:
        render_results(
            predict(st.session_state["spec"], bundle),
            st.session_state["spec"],
            settings,
        )
    else:
        render_empty_state()
