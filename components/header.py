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


def render_sidebar() -> dict:
    """Display toggles. Returns what the result panel reads."""
    with st.sidebar:
        st.markdown("## Display")
        show_intervals = st.toggle("Show ranges", value=True)
        show_drivers = st.toggle("Show drivers", value=True)

        st.divider()
        st.markdown("## About")
        st.caption(
            "Visual template. The numbers on screen are sample values, not "
            "predictions."
        )

        return {"show_intervals": show_intervals, "show_drivers": show_drivers}
