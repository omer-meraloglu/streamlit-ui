"""The game-configuration form.

Every field here is a real model input. The genre / category / tag options are
read off the trained models themselves (`bundle.genres` etc.), so the form can
only offer one-hot columns the models were actually fitted on -- picking
something the model never saw is not possible.

Post-launch columns (estimated_owners_avg, positive_review_percentage) are
targets and deliberately absent.

Laid out to fit one screen without scrolling: fields are packed into rows and
the vertical rhythm is tightened in assets/styles.css.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from demo import GameSpec
from utils.constants import PLATFORMS, PRICE_STATUS

DEFAULT_GENRES = ["Action", "Indie"]
DEFAULT_CATEGORIES = ["Single-player", "Steam Achievements"]
DEFAULT_TAGS = ["Pixel Graphics", "Story Rich"]


def _section_title(text: str) -> None:
    """Small uppercase card heading, matching the store's panel headers."""
    st.markdown(f'<div class="card-title">{text}</div>', unsafe_allow_html=True)


def _defaults(wanted: list[str], vocab: dict[str, str]) -> list[str]:
    """Drop defaults the loaded model does not know -- Streamlit errors on those."""
    return [w for w in wanted if w in vocab]


def render_feature_form(bundle) -> GameSpec | None:
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
                    "genres", list(bundle.genres),
                    default=_defaults(DEFAULT_GENRES, bundle.genres),
                )
            with col8:
                tags = st.multiselect(
                    "tags", list(bundle.tags),
                    default=_defaults(DEFAULT_TAGS, bundle.tags),
                )

            col9, col10 = st.columns([1.4, 1])
            with col9:
                categories = st.multiselect(
                    "categories", list(bundle.categories),
                    default=_defaults(DEFAULT_CATEGORIES, bundle.categories),
                )
            with col10:
                platforms = st.multiselect(
                    "windows / mac / linux", PLATFORMS, default=["Windows"]
                )

            # number_input rather than slider: sliders cost ~30px more height.
            col11, col12, col13, col14 = st.columns(4)
            with col11:
                achievements = st.number_input(
                    "achievements", min_value=0, max_value=1000, value=24
                )
            with col12:
                dlc_count = st.number_input(
                    "dlc_count", min_value=0, max_value=100, value=0
                )
            with col13:
                n_screenshots = st.number_input(
                    "screenshots", min_value=0, max_value=30, value=8
                )
            with col14:
                n_movies = st.number_input(
                    "movies", min_value=0, max_value=10, value=2
                )

            # has_website / has_support_url / has_support_email are three
            # separate BOOLEAN feature columns.
            col15, col16, col17 = st.columns(3)
            with col15:
                has_website = st.checkbox("website", value=True)
            with col16:
                has_support_url = st.checkbox("support_url", value=True)
            with col17:
                has_support_email = st.checkbox("support_email", value=True)

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
        genre_columns=[bundle.genres[g] for g in genres],
        category_columns=[bundle.categories[c] for c in categories],
        tag_columns=[bundle.tags[t] for t in tags],
        achievements=int(achievements),
        dlc_count=int(dlc_count),
        platforms=platforms or ["Windows"],
        n_screenshots=int(n_screenshots),
        n_movies=int(n_movies),
        has_website=has_website,
        has_support_url=has_support_url,
        has_support_email=has_support_email,
    )
