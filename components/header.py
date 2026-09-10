"""Top nav bar and the settings row.

There is no sidebar: the model picker and the display toggles sit in a card
under the input form, so the whole app is one page.
"""

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


def render_settings(backends: dict) -> dict:
    """Model picker + display toggles. Returns what the result panel reads.

    Rendered into a container the caller placed under the form, but read
    before the form is drawn -- the picked model set decides which vocabulary
    the form can offer, and Streamlit keeps a container's position regardless
    of when it is filled.
    """
    with st.container(border=True):
        st.markdown('<div class="card-title">Model &amp; display</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns([1.3, 1])
        with col1:
            backend = st.radio(
                "trained model set", list(backends), index=0,
                horizontal=True, label_visibility="collapsed",
                help="Both sets are trained by "
                     "Steam-Price-Popularity-Predictor on the same data.",
            )
        with col2:
            show_intervals = st.toggle("Show error margins", value=True)
            show_drivers = st.toggle("Show drivers", value=True)

        # Several roots can hold models; show the path so it is obvious which
        # one is live. Written out rather than st.caption()'d: the caption
        # element overruns the card's bottom padding and ends up sitting on
        # the border, and it carries no class of its own to correct that.
        st.markdown(
            f'<div class="model-path"><code>{backends[backend]}</code></div>',
            unsafe_allow_html=True,
        )

    return {
        "backend": backend,
        "show_intervals": show_intervals,
        "show_drivers": show_drivers,
    }
