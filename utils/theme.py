"""Injects the Steam stylesheet and builds Steam-coloured Altair charts."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st

from .constants import STEAM

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "styles.css"


@st.cache_data(show_spinner=False)
def _read_css(path: str, mtime: float) -> str:
    """`mtime` busts the cache when the stylesheet is edited during development."""
    return Path(path).read_text(encoding="utf-8")


def inject_theme() -> None:
    """Call once, right after `st.set_page_config`."""
    css = _read_css(str(CSS_PATH), CSS_PATH.stat().st_mtime)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def steam_chart_theme() -> dict:
    """Altair theme so charts sit on the dark background without a white box."""
    return {
        "config": {
            "background": "transparent",
            "font": "Arial, Helvetica, sans-serif",
            "axis": {
                "labelColor": STEAM["text_muted"],
                "titleColor": STEAM["text_muted"],
                "gridColor": "rgba(102, 192, 244, 0.10)",
                "domainColor": "rgba(102, 192, 244, 0.20)",
                "tickColor": "rgba(102, 192, 244, 0.20)",
                "labelFontSize": 11,
                "titleFontSize": 11,
            },
            "legend": {
                "labelColor": STEAM["text_muted"],
                "titleColor": STEAM["text_muted"],
            },
            "view": {"stroke": "transparent"},
        }
    }


def register_chart_theme(name: str = "steam") -> None:
    """Register + enable the Altair theme, tolerating both Altair 4 and 5 APIs."""
    try:  # Altair >= 5.5 wants a callable registered through the plugin registry
        alt.theme.register(name, enable=True)(steam_chart_theme)
    except AttributeError:  # Altair 4 / early 5
        alt.themes.register(name, steam_chart_theme)
        alt.themes.enable(name)
