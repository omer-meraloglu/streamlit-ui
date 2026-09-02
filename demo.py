"""Sample numbers so the page has something to show.

This is NOT a model -- it is mock data for the mockup. Values are seeded from
the form input so the page reacts when you change something, and stays stable
when you don't.

Field names mirror the BigQuery table `datascientiststeamproject.kaggle.Steam`,
so the visuals line up with the real columns:

  inputs  (known before launch) -> GameSpec
  targets (what you predict)    -> PreviewNumbers
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

# `estimated_owners` is a STRING bucket in the source table, not a number.
OWNER_BUCKETS = [
    (0, 20_000), (20_000, 50_000), (50_000, 100_000), (100_000, 200_000),
    (200_000, 500_000), (500_000, 1_000_000), (1_000_000, 2_000_000),
    (2_000_000, 5_000_000), (5_000_000, 10_000_000), (10_000_000, 20_000_000),
    (20_000_000, 50_000_000), (50_000_000, 100_000_000),
]


@dataclass
class GameSpec:
    """Pre-launch columns -- the model inputs."""

    name: str = "Neon Drifter"
    release_date: date = field(default_factory=date.today)
    price: float = 19.99
    price_status: str = "Paid"
    developers: str = ""
    publishers: str = ""
    genres: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    achievements: int = 0
    dlc_count: int = 0
    supported_languages: list[str] = field(default_factory=lambda: ["English"])
    full_audio_languages: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=lambda: ["Windows"])
    n_screenshots: int = 8
    n_movies: int = 2
    has_website: bool = True

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


@dataclass
class PreviewNumbers:
    """Post-launch columns -- the prediction targets."""

    estimated_owners: str      # STRING bucket, e.g. "20,000 - 50,000"
    owners_mid: float          # numeric mid-point, for the derived revenue
    positive: int
    negative: int
    peak_ccu: int
    recommendations: int
    metacritic_score: int
    user_score: int
    revenue: float             # derived: owners x price, not a table column
    drivers: dict[str, float]

    @property
    def total_reviews(self) -> int:
        return self.positive + self.negative

    @property
    def positive_ratio(self) -> float:
        total = self.total_reviews
        return self.positive / total if total else 0.0


def _owner_bucket(owners: float) -> str:
    for low, high in OWNER_BUCKETS:
        if owners < high:
            return f"{low:,} - {high:,}"
    return f"{OWNER_BUCKETS[-1][1]:,}+"


def preview_numbers(spec: GameSpec) -> PreviewNumbers:
    rng = np.random.default_rng(
        int(hashlib.sha256(
            json.dumps(asdict(spec), sort_keys=True, default=str).encode()
        ).hexdigest()[:8], 16)
    )

    price = 0.0 if spec.price_status != "Paid" else max(spec.price, 0.0)

    scale = 9.6
    scale += 0.9 if price == 0 else -0.30 * np.log1p(price / 12.0)
    scale += 0.16 * np.log1p(len(spec.supported_languages))
    scale += 0.12 * np.log1p(len(spec.tags))
    scale += 0.15 * (len(spec.platforms) - 1)
    scale += 0.20 * len(spec.genres)
    scale += 0.18 * np.log1p(spec.dlc_count)
    scale += 0.10 * np.log1p(spec.achievements)
    scale += rng.normal(0, 0.15)

    owners = float(np.exp(scale))
    ratio = float(np.clip(0.80 + rng.normal(0, 0.06), 0.05, 0.99))
    total_reviews = int(max(owners * 0.021, 5))
    positive = int(total_reviews * ratio)

    return PreviewNumbers(
        estimated_owners=_owner_bucket(owners),
        owners_mid=owners,
        positive=positive,
        negative=total_reviews - positive,
        peak_ccu=int(max(owners * 0.012, 1)),
        recommendations=int(max(owners * 0.008, 0)),
        metacritic_score=int(np.clip(55 + ratio * 40 + rng.normal(0, 4), 0, 100)),
        user_score=int(np.clip(ratio * 100 + rng.normal(0, 3), 0, 100)),
        revenue=owners * price * 0.72,
        drivers={
            "price": -0.30 * np.log1p(price / 12.0),
            "genres": 0.20 * len(spec.genres),
            "tags": 0.12 * np.log1p(len(spec.tags)),
            "supported_languages": 0.16 * np.log1p(len(spec.supported_languages)),
            "platforms": 0.15 * (len(spec.platforms) - 1),
            "dlc_count": 0.18 * np.log1p(spec.dlc_count),
            "achievements": 0.10 * np.log1p(spec.achievements),
        },
    )
