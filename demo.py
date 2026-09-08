"""The input schema the form fills in and the models score.

This used to hold mock numbers. Predictions now come from the trained models in
`Steam-Price-Popularity-Predictor` -- see `utils/model_backend.py`. What is left
here is `GameSpec`: the pre-launch fields, in the shape the feature builder
expects.

`genres` / `categories` / `tags` carry display labels for the store capsule;
`*_columns` carry the matching one-hot column names the models were trained on.
The form resolves one to the other via the vocabulary read off the models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class GameSpec:
    """Pre-launch columns -- the model inputs."""

    name: str = "Neon Drifter"
    release_date: date = field(default_factory=date.today)
    price: float = 19.99
    price_status: str = "Paid"
    developers: str = ""
    publishers: str = ""

    # Display labels (shown on the capsule)
    genres: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Resolved one-hot column names (fed to the models)
    genre_columns: list[str] = field(default_factory=list)
    category_columns: list[str] = field(default_factory=list)
    tag_columns: list[str] = field(default_factory=list)

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
