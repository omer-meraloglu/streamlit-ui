"""The input schema the form fills in and the models score.

This used to hold mock numbers. Predictions now come from the trained models in
`Steam-Price-Popularity-Predictor` -- see `utils/model_backend.py`. What is left
here is `GameSpec`: the pre-launch fields, in the shape the feature builder
expects.

`genres` / `categories` / `tags` / `languages` carry display labels for the
store capsule; `*_columns` carry the matching one-hot column names the models
were trained on. The form resolves one to the other via the vocabulary read off
the models.

`name`, `developers` and `publishers` used to live here. None were model
features -- they only decorated the result capsule -- so the form no longer
asks for them. `release_date` stays because `release_year` IS a feature, but it
is no longer editable either: it defaults to the current year.

`price` is optional. Left as None it means "no price decided yet": the price
model runs first and its own suggestion is what the owners and review models
are scored against. `price_status` is gone -- the form only ever offered
"Paid", and no model in either set was fitted on `is_free` anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class GameSpec:
    """Pre-launch columns -- the model inputs."""

    # Not user-editable: feeds the `release_year` feature.
    release_date: date = field(default_factory=date.today)
    # None = no price entered; the price model's own suggestion is used instead.
    price: float | None = None

    # Display labels (shown on the capsule)
    genres: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Store-page localisation: the final models were fitted on `lang_*` columns.
    languages: list[str] = field(default_factory=lambda: ["English"])

    # Resolved one-hot column names (fed to the models)
    genre_columns: list[str] = field(default_factory=list)
    category_columns: list[str] = field(default_factory=list)
    tag_columns: list[str] = field(default_factory=list)
    language_columns: list[str] = field(default_factory=list)

    achievements: int = 0
    dlc_count: int = 0
    platforms: list[str] = field(default_factory=lambda: ["Windows"])
    n_screenshots: int = 8
    n_movies: int = 2
    has_website: bool = True
    has_support_url: bool = False
    has_support_email: bool = False

    # The table stores platforms as three BOOLEAN columns.
    @property
    def windows(self) -> bool:
        return "Windows" in self.platforms

    @property
    def mac(self) -> bool:
        return "Mac" in self.platforms

    @property
    def linux(self) -> bool:
        return "Linux" in self.platforms
