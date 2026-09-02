"""SteamCast -- Steam-themed page mockup.

Visual template only. The numbers are sample values from demo.py, not
predictions. Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from components.header import render_header, render_sidebar
from components.inputs import render_feature_form
from components.results import render_empty_state, render_results
from demo import preview_numbers
from utils.theme import inject_theme, register_chart_theme

st.set_page_config(
    page_title="SteamCast — Sales & Review Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
register_chart_theme()

settings = render_sidebar()
render_header()

form_col, result_col = st.columns([1, 1.15], gap="large")

with form_col:
    spec = render_feature_form()
    if spec is not None:
        st.session_state["spec"] = spec
        st.session_state["numbers"] = preview_numbers(spec)

with result_col:
    if "numbers" in st.session_state:
        render_results(
            st.session_state["numbers"], st.session_state["spec"], settings
        )
    else:
        render_empty_state()
