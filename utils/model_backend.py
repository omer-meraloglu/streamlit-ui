"""Loads the trained models from Steam-Price-Popularity-Predictor and runs them.

Three models were trained by that project's `2_train_models.py`:

  model_price   LGBM/XGB Regressor  -> log1p(price)                 (no `price` feature)
  model_owners  LGBM/XGB Classifier -> estimated_owners_avg bucket  (uses `price`)
  model_review  LGBM/XGB Regressor  -> positive_review_percentage   (uses `price`)

Each was fitted on its *own* selected feature subset (80 / 81 / 82 columns), so
every model gets a frame built to its own `feature_names`, in its own order --
XGBoost raises on a mismatch and LightGBM would silently misalign.

Set STEAMCAST_MODEL_ROOT to point somewhere other than the sibling checkout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Where the .pkl files live
# --------------------------------------------------------------------------
DEFAULT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "Steam-Price-Popularity-Predictor"
    / "ml_models"
)

BACKENDS = {"LightGBM": "saved_models", "XGBoost": "saved_models_xgb"}
TARGETS = ("price", "owners", "review")


def model_root() -> Path:
    return Path(os.environ.get("STEAMCAST_MODEL_ROOT", DEFAULT_ROOT))


def available_backends() -> dict[str, Path]:
    """Backends whose directory holds a complete set of models."""
    root = model_root()
    found = {}
    for label, folder in BACKENDS.items():
        d = root / folder
        needed = [d / f"model_{t}.pkl" for t in TARGETS]
        needed.append(d / "label_encoder_owners.pkl")
        if all(p.exists() for p in needed):
            found[label] = d
    return found


# --------------------------------------------------------------------------
# Column name -> human label, so the form offers exactly what the models know
# --------------------------------------------------------------------------
_SLUG = {
    "co_op": "Co-op", "online_co_op": "Online Co-op", "online_pvp": "Online PvP",
    "sci_fi": "Sci-Fi", "souls_like": "Souls-like", "roguelite": "Rogue-lite",
    "shared_split_screen": "Shared/Split Screen", "in_app_purchases": "In-App Purchases",
    "vr_only": "VR Only", "vr_supported": "VR Supported",
    "single_player": "Single-player", "multi_player": "Multi-player",
    "post_apocalyptic": "Post-apocalyptic", "turn_based": "Turn-Based",
    "hand_drawn": "Hand-drawn", "free_to_play": "Free To Play",
    "hack_and_slash": "Hack and Slash", "massively_multiplayer": "Massively Multiplayer",
}
_WORD = {
    "rpg": "RPG", "mmo": "MMO", "pvp": "PvP", "fps": "FPS", "vr": "VR",
    "2d": "2D", "3d": "3D", "dlc": "DLC", "and": "and",
}


def humanize(slug: str) -> str:
    if slug in _SLUG:
        return _SLUG[slug]
    return " ".join(_WORD.get(w, w.capitalize()) for w in slug.split("_"))


def _vocab(features: list[str], prefix: str) -> dict[str, str]:
    """{'Action': 'genre_action', ...} for one one-hot family, label-sorted."""
    pairs = [(humanize(c[len(prefix):]), c) for c in features if c.startswith(prefix)]
    return dict(sorted(pairs))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Bundle:
    """Everything needed to score one GameSpec."""

    label: str
    directory: Path
    models: dict
    features: dict[str, list[str]]
    owner_classes: np.ndarray      # owner mid-points, ascending
    class_order: np.ndarray        # encoder index for each entry above
    explainers: dict
    genres: dict[str, str]
    categories: dict[str, str]
    tags: dict[str, str]

    @property
    def all_features(self) -> set[str]:
        return set().union(*self.features.values())


def _feature_names(model, directory: Path, target: str) -> list[str]:
    """LightGBM exposes feature_name_, XGBoost feature_names_in_."""
    for attr in ("feature_name_", "feature_names_in_"):
        names = getattr(model, attr, None)
        if names is not None and len(names):
            return list(names)
    booster = getattr(model, "get_booster", None)
    if booster is not None:
        names = booster().feature_names
        if names:
            return list(names)
    # Last resort: the test frame saved alongside the model.
    X, _ = joblib.load(directory / f"test_data_{target}.pkl")
    return list(X.columns)


@lru_cache(maxsize=4)
def load_bundle(label: str) -> Bundle:
    directory = available_backends()[label]

    models, features, explainers = {}, {}, {}
    for target in TARGETS:
        models[target] = joblib.load(directory / f"model_{target}.pkl")
        features[target] = _feature_names(models[target], directory, target)
        path = directory / f"explainer_{target}.pkl"
        if path.exists():
            try:
                explainers[target] = joblib.load(path)
            except Exception:      # a stale explainer must not block predicting
                pass

    encoder = joblib.load(directory / "label_encoder_owners.pkl")
    # Classes are numeric owner mid-points stored as strings ('10000', '150000'),
    # so LabelEncoder ordered them lexically. Sort numerically for display.
    values = np.array([float(c) for c in encoder.classes_])
    order = np.argsort(values)

    union = sorted(set().union(*features.values()))
    return Bundle(
        label=label,
        directory=directory,
        models=models,
        features=features,
        owner_classes=values[order],
        class_order=order,
        explainers=explainers,
        genres=_vocab(union, "genre_"),
        categories=_vocab(union, "cat_"),
        tags=_vocab(union, "tag_"),
    )


# --------------------------------------------------------------------------
# GameSpec -> model frame
# --------------------------------------------------------------------------
def build_frame(spec, features: list[str], price: float | None) -> pd.DataFrame:
    """One row, columns exactly `features` in order. Unlisted one-hots stay 0."""
    values: dict[str, float] = {
        "release_year": float(spec.release_date.year),
        "achievements_count": float(spec.achievements),
        "dlc_count": float(spec.dlc_count),
        "screenshot_count": float(spec.n_screenshots),
        "movie_count": float(spec.n_movies),
        "has_website": float(spec.has_website),
        "has_support_url": float(getattr(spec, "has_support_url", False)),
        "has_support_email": float(getattr(spec, "has_support_email", False)),
        "supports_windows": float(spec.windows),
        "supports_mac": float(spec.mac),
        "supports_linux": float(spec.linux),
        "is_free": float(spec.price_status != "Paid" or spec.price <= 0),
    }
    if price is not None:
        values["price"] = float(price)

    for column in spec.genre_columns + spec.category_columns + spec.tag_columns:
        values[column] = 1.0

    row = [values.get(name, 0.0) for name in features]
    return pd.DataFrame([row], columns=features).astype("float64")


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    """Real model output. Every field below is predicted or derived from one."""

    owners: float                  # predicted bucket mid-point
    owners_confidence: float       # probability mass on that bucket
    owners_low: float              # 10th percentile of the class distribution
    owners_high: float             # 90th percentile
    review_pct: float              # positive_review_percentage, 0-100
    suggested_price: float         # what the price model would charge
    entered_price: float           # what the user typed (drives owners/review)
    revenue: float                 # derived: owners x entered price
    drivers: dict[str, float]      # SHAP values for the owners model
    backend: str

    @property
    def price_gap(self) -> float:
        return self.entered_price - self.suggested_price


def _shap_for_owners(bundle: Bundle, frame: pd.DataFrame, class_index: int,
                     top_n: int = 8) -> dict[str, float]:
    """SHAP contributions for the predicted owner bucket, largest |value| first."""
    explainer = bundle.explainers.get("owners")
    if explainer is None:
        return {}
    try:
        values = explainer.shap_values(frame)
    except Exception:
        return {}

    array = np.array(values)
    if array.ndim == 3:                      # (rows, features, classes) or (classes, rows, features)
        if array.shape[0] == len(frame):
            row = array[0, :, class_index]
        else:
            row = array[class_index, 0, :]
    elif array.ndim == 2:
        row = array[0]
    else:
        return {}

    pairs = sorted(
        zip(frame.columns, row), key=lambda kv: abs(kv[1]), reverse=True
    )[:top_n]
    return {driver_label(k): float(v) for k, v in pairs}


def driver_label(column: str) -> str:
    """'tag_story_rich' -> 'Story Rich (tag)', 'release_year' -> 'Release Year'."""
    for prefix, family in (("genre_", "genre"), ("cat_", "category"), ("tag_", "tag")):
        if column.startswith(prefix):
            return f"{humanize(column[len(prefix):])} ({family})"
    return humanize(column)


def predict(spec, bundle: Bundle) -> Prediction:
    price = 0.0 if spec.price_status != "Paid" else max(float(spec.price), 0.0)

    # price model predicts log1p(price) and never sees `price` as an input
    price_frame = build_frame(spec, bundle.features["price"], price=None)
    suggested = float(np.expm1(bundle.models["price"].predict(price_frame)[0]))
    suggested = max(suggested, 0.0)

    owners_frame = build_frame(spec, bundle.features["owners"], price=price)
    owners_model = bundle.models["owners"]
    proba = owners_model.predict_proba(owners_frame)[0][bundle.class_order]
    best = int(np.argmax(proba))

    cumulative = np.cumsum(proba)
    low = float(bundle.owner_classes[int(np.searchsorted(cumulative, 0.10))])
    high = float(bundle.owner_classes[
        min(int(np.searchsorted(cumulative, 0.90)), len(bundle.owner_classes) - 1)
    ])

    review_frame = build_frame(spec, bundle.features["review"], price=price)
    review_pct = float(np.clip(bundle.models["review"].predict(review_frame)[0], 0, 100))

    owners = float(bundle.owner_classes[best])
    return Prediction(
        owners=owners,
        owners_confidence=float(proba[best]),
        owners_low=low,
        owners_high=high,
        review_pct=review_pct,
        suggested_price=suggested,
        entered_price=price,
        revenue=owners * price,
        drivers=_shap_for_owners(bundle, owners_frame, int(bundle.class_order[best])),
        backend=bundle.label,
    )
