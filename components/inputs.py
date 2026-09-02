"""The game-configuration form.

Fields mirror the pre-launch columns of
`datascientiststeamproject.kaggle.Steam`. Post-launch columns (estimated_owners,
positive, negative, peak_ccu, recommendations, metacritic_score, user_score,
playtimes) are targets and deliberately absent.

Laid out to fit one screen without scrolling: fields are packed into rows and
the vertical rhythm is tightened in assets/styles.css.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from demo import GameSpec
from utils.constants import PLATFORMS, PRICE_STATUS, VOCAB

DEFAULT_GENRES = ["Action", "Indie"]
DEFAULT_CATEGORIES = ["Single-player", "Steam Achievements"]
DEFAULT_TAGS = ["Pixel Graphics", "Story Rich"]


def _section_title(text: str) -> None:
    """Small uppercase card heading, matching the store's panel headers."""
    st.markdown(f'<div class="card-title">{text}</div>', unsafe_allow_html=True)


def render_feature_form() -> GameSpec | None:
    """Draw the form. Returns the spec on submit, otherwise None.

    Note: bordered `st.container`s are used for the card look -- raw <div>
    wrappers cannot contain Streamlit widgets, since each widget renders into
    its own DOM container.
    """
    with st.form("game_spec", border=False):
        with st.container(border=True):
            _section_title("Store page")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                name = st.text_input("name", value="Neon Drifter")
            with col2:
                price = st.number_input(
                    "price", min_value=0.0, max_value=200.0, value=19.99, step=1.0
                )
            with col3:
                price_status = st.selectbox("price_status", PRICE_STATUS, index=0)

            col4, col5, col6 = st.columns(3)
            with col4:
                release_date = st.date_input("release_date", value=date(2026, 11, 12))
            with col5:
                developers = st.text_input("developers", value="Drift Collective")
            with col6:
                publishers = st.text_input("publishers", value="Drift Collective")

        # One card, paired rows: six multiselects stacked full-width overflow a
        # 768px-tall screen by ~100px.
        with st.container(border=True):
            _section_title("Content & reach")
            col7, col8 = st.columns(2)
            with col7:
                genres = st.multiselect(
                    "genres", VOCAB["genres"], default=DEFAULT_GENRES
                )
            with col8:
                tags = st.multiselect("tags", VOCAB["tags"], default=DEFAULT_TAGS)

            col9, col10 = st.columns([1.4, 1])
            with col9:
                categories = st.multiselect(
                    "categories", VOCAB["categories"], default=DEFAULT_CATEGORIES
                )
            with col10:
                platforms = st.multiselect(
                    "windows / mac / linux", PLATFORMS, default=["Windows"]
                )

            col11, col12 = st.columns(2)
            with col11:
                supported_languages = st.multiselect(
                    "supported_languages", VOCAB["languages"], default=["English"]
                )
            with col12:
                full_audio_languages = st.multiselect(
                    "full_audio_languages", VOCAB["languages"], default=[]
                )

            # number_input rather than slider: sliders cost ~30px more height.
            col13, col14, col15, col16 = st.columns(4)
            with col13:
                achievements = st.number_input(
                    "achievements", min_value=0, max_value=1000, value=24
                )
            with col14:
                dlc_count = st.number_input(
                    "dlc_count", min_value=0, max_value=100, value=0
                )
            with col15:
                n_screenshots = st.number_input(
                    "screenshots", min_value=0, max_value=30, value=8
                )
            with col16:
                n_movies = st.number_input(
                    "movies", min_value=0, max_value=10, value=2
                )
            has_website = st.checkbox("website", value=True)

        submitted = st.form_submit_button("Predict performance", type="primary")

    if not submitted:
        return None

    return GameSpec(
        name=name or "Untitled Game",
        release_date=release_date,
        price=float(price),
        price_status=price_status,
        developers=developers,
        publishers=publishers,
        genres=genres,
        categories=categories,
        tags=tags,
        achievements=int(achievements),
        dlc_count=int(dlc_count),
        supported_languages=supported_languages or ["English"],
        full_audio_languages=full_audio_languages,
        platforms=platforms or ["Windows"],
        n_screenshots=int(n_screenshots),
        n_movies=int(n_movies),
        has_website=has_website,
    )
