"""Loads the trained models from Steam-Price-Popularity-Predictor and runs them.

Three models are trained by that project, one per target:

  price   Regressor  -> log1p(price)                 (no `price` feature)
  owners  Classifier -> estimated owners bucket      (uses `price`)
  review  Regressor  -> positive_review_percentage   (uses `price`)

Each set ships as three bundles. `bundle_<target>.pkl` is a dict carrying the
fitted estimator, its `selected_features`, the `target_transform` applied while
training and the `error_margins` measured on the held-out set -- so a
prediction can be reported with its error margin instead of as a bare number.

The two sets do not predict owners and reviews the same way, so each target is
read in whichever of three modes its own bundle describes (see `_mode`):

  value    a number in its own unit -- price, in dollars once the transform is
           undone. Error margins are `mae` / `confidence_95`.
  classes  a classifier over named buckets. `predict_proba` gives the bucket
           and the confidence; the names come from `classes` / `label_encoder`,
           or from `classes_` when the model was fitted on them directly.
  ordinal  a regressor over a band *index* -- 0.0 is the bottom band, and the
           value lands between bands as often as on one. Recognised by
           `mean_continuous_pred` in the margins, which is where a bundle in
           this mode reports the spread of its predictions instead of an error.

The band names are not in an ordinal bundle, so BAND_NAMES below supplies
them. Owners reuses the buckets its own classifier was trained on; the review
names are the assumption, and the one place to correct if the training script
banded reviews differently.

Each model was fitted on its *own* selected feature subset, so every model gets
a frame built to its own feature list, in its own order -- XGBoost raises on a
mismatch and LightGBM would silently misalign.

Set STEAMCAST_MODEL_ROOT to point somewhere other than the sibling checkout.
"""

from __future__ import annotations

import math
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
APP_DIR = Path(__file__).resolve().parent.parent

# Bundles copied into this repo. Searched FIRST: these are the ones that get
# deployed, and the ones that get replaced by hand when a new training run is
# handed over -- a stale sibling checkout silently shadowing them is how the
# app ends up scoring with last week's models while looking up to date.
LOCAL_ROOT = APP_DIR / "models"

# The training project, checked out next to this one. Only reached for a set
# this repo does not carry, so a fresh run there still shows up on its own.
SIBLING_ROOT = APP_DIR.parent / "Steam-Price-Popularity-Predictor" / "ml_models"

TARGETS = ("price", "owners", "review")

# The two model sets, and the folder names each is written to. Two names per
# set because the training project and this repo disagree: `2_train_models.py`
# writes into saved_models/ and saved_models_xgb_final/, while an upload into
# this repo lands in bundles/ and bundles_xgb/. First match wins.
BACKENDS = {
    "LightGBM": ("bundles", "saved_models"),
    "XGBoost": ("bundles_xgb", "saved_models_xgb_final"),
}

# `app_id` is one of the selected features, but a game that has not shipped has
# no id yet. Steam hands them out in ascending order, so a release configured
# today would draw one near the top of the range -- feeding that is closer to
# the truth than 0, which would sit the prediction next to the 2003 catalogue.
DEFAULT_APP_ID = 3_000_000.0

# Names for the bands an `ordinal` target indexes, lowest first. An ordinal
# bundle ships the index and nothing to decode it with, so these are supplied
# here.
#
# `owners` repeats the buckets the classifier version of the same model was
# trained on, and the numbers below are what the app reports as an owner range.
# `review` is the assumed banding -- the model output is verifiably a 0-based
# index (its predictions floor at ~0 and the two targets behave identically),
# but nothing in the bundle names the bands. Correct this list if the training
# script split reviews another way; nothing else needs to change.
BAND_NAMES = {
    "owners": ["0-20k", "20k-50k", "50k-200k", "200k-1M", "1M+"],
    "review": [
        "Mostly Negative", "Mixed", "Mostly Positive",
        "Very Positive", "Overwhelmingly Positive",
    ],
}

def search_roots() -> list[Path]:
    """Directories to search, highest priority first."""
    roots = []
    override = os.environ.get("STEAMCAST_MODEL_ROOT")
    if override:
        roots.append(Path(override))
    roots.append(LOCAL_ROOT)
    roots.append(SIBLING_ROOT)
    return roots


def _has_bundles(directory: Path) -> bool:
    """Current layout: one self-describing bundle per target."""
    return directory.is_dir() and all(
        (directory / f"bundle_{t}.pkl").exists() for t in TARGETS
    )


def available_backends() -> dict[str, Path]:
    """Backends holding a complete set of models, nearest root winning.

    Each root is checked for every folder name a set can go by. A label found
    in an earlier root is never overwritten by a later one, so what this repo
    carries takes precedence over the sibling checkout.
    """
    found: dict[str, Path] = {}
    for root in search_roots():
        for label, folders in BACKENDS.items():
            if label in found:
                continue
            for folder in folders:
                if _has_bundles(root / folder):
                    found[label] = root / folder
                    break

    return {label: found[label] for label in BACKENDS if label in found}


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
    transforms: dict[str, str | None]   # target -> transform applied in training
    margins: dict[str, dict]            # target -> held-out error stats ({} if none)
    modes: dict[str, str]               # target -> "value" | "classes" | "ordinal"
    owner_labels: list[str]             # bucket names, smallest first
    owner_bounds: list                  # (low, high) owners, aligned with the above
    review_bands: list[str]             # review band names, worst first ([] unless ordinal)
    class_order: np.ndarray             # model class index for each entry above
    explainers: dict
    genres: dict[str, str]
    categories: dict[str, str]
    tags: dict[str, str]
    languages: dict[str, str]

    @property
    def all_features(self) -> set[str]:
        return set().union(*self.features.values())


_SUFFIXES = {"k": 1e3, "m": 1e6, "b": 1e9}


def _owner_edge(text: str) -> float:
    """'20k' -> 20000.0."""
    text = text.strip().lower().replace(",", "")
    if text and text[-1] in _SUFFIXES:
        return float(text[:-1]) * _SUFFIXES[text[-1]]
    return float(text)


def _owner_range(label: str) -> tuple[float, float]:
    """'20k-50k' -> (20000, 50000); the open-ended '1M+' -> (1000000, inf)."""
    label = label.strip()
    if label.endswith("+"):
        return _owner_edge(label[:-1]), math.inf
    low, _, high = label.partition("-")
    return _owner_edge(low), _owner_edge(high)


@lru_cache(maxsize=4)
def load_bundle(label: str) -> Bundle:
    return _load_bundled(label, available_backends()[label])


def _explainer_width(explainer) -> int | None:
    """How many columns an explainer's own tree expects, when it will say."""
    try:
        return int(explainer.model.original_model.num_feature())
    except Exception:
        return None


def _explainers(directory: Path, features: dict[str, list[str]]) -> dict:
    """Any explainer_<target>.pkl sitting next to the models, if it still fits.

    Retraining rewrites the models but leaves the old explainer pickles in
    place, and one fitted on a different feature set does not fail until it is
    asked for values -- taking the whole drivers panel down with it. Checking
    the width here means a stale pickle is simply ignored and the explainer
    gets rebuilt from the model instead.
    """
    found = {}
    for target in TARGETS:
        path = directory / f"explainer_{target}.pkl"
        if not path.exists():
            continue
        try:
            explainer = joblib.load(path)
        except Exception:          # a stale explainer must not block predicting
            continue
        width = _explainer_width(explainer)
        if width is not None and width != len(features[target]):
            continue
        found[target] = explainer
    return found


def _assemble(label, directory, models, features, transforms, margins, modes,
              owner_labels, owner_bounds, review_bands, class_order,
              explainers) -> Bundle:
    union = sorted(set().union(*features.values()))
    return Bundle(
        label=label,
        directory=directory,
        models=models,
        features=features,
        transforms=transforms,
        margins=margins,
        modes=modes,
        owner_labels=owner_labels,
        owner_bounds=owner_bounds,
        review_bands=review_bands,
        class_order=class_order,
        explainers=explainers,
        genres=_vocab(union, "genre_"),
        categories=_vocab(union, "cat_"),
        tags=_vocab(union, "tag_"),
        languages=_vocab(union, "lang_"),
    )


def _mode(target: str, saved: dict, model) -> str:
    """Which of the three target shapes this bundle holds -- see the module docstring."""
    if "mean_continuous_pred" in (saved.get("error_margins") or {}):
        return "ordinal"
    if hasattr(model, "predict_proba") and getattr(model, "classes_", None) is not None:
        return "classes"
    return "value"


def _owner_names(saved: dict, model) -> list[str]:
    """Bucket names in the model's own class order.

    LightGBM was fitted on the names themselves, so `classes_` already holds
    them. XGBoost needs integer classes, so its bundle was fitted on encoded
    labels and ships the names alongside -- as `classes`, or on the
    `label_encoder` it used. Either way index i names class i.
    """
    names = saved.get("classes")
    if names is None and saved.get("label_encoder") is not None:
        names = list(saved["label_encoder"].classes_)
    if names is None:
        names = list(model.classes_)
    return [str(n) for n in names]


def _load_bundled(label: str, directory: Path) -> Bundle:
    """One self-describing pickle per target."""
    models, features, transforms, margins, modes = {}, {}, {}, {}, {}
    owners_saved = None
    for target in TARGETS:
        saved = joblib.load(directory / f"bundle_{target}.pkl")
        models[target] = saved["model"]
        features[target] = list(saved["selected_features"])
        transforms[target] = saved.get("target_transform")
        margins[target] = dict(saved.get("error_margins") or {})
        modes[target] = _mode(target, saved, saved["model"])
        if target == "owners":
            owners_saved = saved

    if modes["owners"] == "ordinal":
        # An ordinal bundle indexes the bands and ships no names for them, so
        # they are already in size order and each class is its own index.
        names = list(BAND_NAMES["owners"])
        order = np.arange(len(names))
    else:
        # Whichever way the names arrive, they are in the encoder's lexical
        # order -- '1M+' lands second. Sort by lower edge to display by size.
        names = _owner_names(owners_saved, models["owners"])
        order = np.argsort([low for low, _ in (_owner_range(n) for n in names)])
        names = [names[i] for i in order]

    return _assemble(
        label, directory, models, features, transforms, margins, modes,
        owner_labels=names,
        owner_bounds=[_owner_range(n) for n in names],
        review_bands=list(BAND_NAMES["review"]) if modes["review"] == "ordinal" else [],
        class_order=order,
        explainers=_explainers(directory, features),
    )


# --------------------------------------------------------------------------
# GameSpec -> model frame
# --------------------------------------------------------------------------
def build_frame(spec, features: list[str], price: float | None) -> pd.DataFrame:
    """One row, columns exactly `features` in order. Unlisted one-hots stay 0."""
    values: dict[str, float] = {
        "app_id": DEFAULT_APP_ID,
        "release_year": float(spec.release_date.year),
        "achievements_count": float(spec.achievements),
        "dlc_count": float(spec.dlc_count),
        "screenshot_count": float(spec.n_screenshots),
        "movie_count": float(spec.n_movies),
        "total_media_count": float(spec.n_screenshots + spec.n_movies),
        "has_website": float(spec.has_website),
        "has_support_url": float(getattr(spec, "has_support_url", False)),
        "has_support_email": float(getattr(spec, "has_support_email", False)),
        "supports_windows": float(spec.windows),
        "supports_mac": float(spec.mac),
        "supports_linux": float(spec.linux),
    }
    if price is not None:
        values["is_free"] = float(price <= 0)
    if price is not None:
        values["price"] = float(price)

    one_hots = (
        spec.genre_columns
        + spec.category_columns
        + spec.tag_columns
        + getattr(spec, "language_columns", [])
    )
    for column in one_hots:
        values[column] = 1.0

    row = [values.get(name, 0.0) for name in features]
    return pd.DataFrame([row], columns=features).astype("float64")


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    """Real model output. Every field below is predicted or derived from one."""

    owners_low: float              # predicted bucket's lower edge
    owners_high: float             # predicted bucket's upper edge
    owners_confidence: float | None    # probability on that bucket (classifier only)
    owners_index: float | None         # raw ordinal band index (regressor only)
    spread_low: float              # lower edge spanning the middle 80% of mass
    spread_high: float             # upper edge of the same span
    # Exactly one of these carries the review prediction, depending on the
    # bundle's mode: a percentage, or a band index with the band it rounds to.
    review_pct: float | None       # positive_review_percentage, 0-100
    review_index: float | None     # raw ordinal band index, 0.0 = bottom band
    review_band: str | None        # the band `review_index` rounds to
    review_bands: list[str]        # every band name, worst first
    suggested_price: float         # what the price model would charge
    entered_price: float           # the price the other two models were given
    price_is_suggested: bool       # True when no price was typed in
    revenue_low: float             # derived: owners range x entered price
    revenue_high: float
    drivers: dict[str, dict[str, float]]   # target -> SHAP values for that model
    margins: dict[str, dict]       # held-out error stats, straight off the bundle
    backend: str

    @property
    def price_gap(self) -> float:
        return self.entered_price - self.suggested_price

    # ---- error margins ---------------------------------------------------
    # Two numbers were measured on the held-out set, both half-widths in the
    # target's own unit -- dollars for price, percentage points for review:
    #
    #   mae            the average miss. This is what gets shown, because it is
    #                  what "the model is off by about this much" means.
    #   confidence_95  1.96 x the standard deviation of the same residuals. A
    #                  genuine 95% band, but several times wider than the
    #                  typical miss, because a handful of predictions are very
    #                  wrong and blow up the standard deviation. Kept as
    #                  context rather than as the headline.
    #
    # Nothing was saved for the older model sets, hence the 0.0 default and
    # `has_margins`, which hides the panel instead of printing a made-up +/- 0.
    @property
    def price_margin(self) -> float:
        return float(self.margins.get("price", {}).get("mae", 0.0))

    @property
    def review_margin(self) -> float:
        return float(self.margins.get("review", {}).get("mae", 0.0))

    @property
    def price_band95(self) -> float:
        return float(self.margins.get("price", {}).get("confidence_95", 0.0))

    @property
    def review_band95(self) -> float:
        return float(self.margins.get("review", {}).get("confidence_95", 0.0))

    # An `ordinal` bundle carries no error at all -- `std_continuous_pred` is
    # how widely the model's predictions were spread over the held-out set, in
    # band units. Useful context, but it is not a miss and is not labelled as
    # one; the band range shown comes from the prediction itself instead.
    @property
    def owners_spread(self) -> float:
        return float(self.margins.get("owners", {}).get("std_continuous_pred", 0.0))

    @property
    def review_spread(self) -> float:
        return float(self.margins.get("review", {}).get("std_continuous_pred", 0.0))

    @property
    def owners_mean_confidence(self) -> float:
        """Average winning-class probability over the held-out set."""
        return float(self.margins.get("owners", {}).get("mean_confidence", 0.0))

    @property
    def has_margins(self) -> bool:
        return any(self.margins.get(t) for t in ("price", "owners", "review"))

    @property
    def price_interval(self) -> tuple[float, float]:
        """Typical-miss band around the suggested price, floored at free."""
        return (
            max(self.suggested_price - self.price_margin, 0.0),
            self.suggested_price + self.price_margin,
        )

    @property
    def review_interval(self) -> tuple[float, float]:
        """Typical-miss band around the review score, clipped to 0-100."""
        return (
            max(self.review_pct - self.review_margin, 0.0),
            min(self.review_pct + self.review_margin, 100.0),
        )


def _explainer(bundle: Bundle, target: str):
    """One target's explainer, rebuilt from the model when no pickle shipped.

    explainer_owners.pkl is 26MB -- larger than every other artefact combined --
    so a deployment can ship the models alone and have the explainers
    reconstructed here. shap is imported lazily, and building one costs 0.4s
    (1.8s for owners), but the Bundle is cached so that is paid once per model
    set rather than once per prediction.
    """
    if target not in bundle.explainers:
        try:
            import shap

            bundle.explainers[target] = shap.TreeExplainer(bundle.models[target])
        except Exception:
            return None
    return bundle.explainers[target]


# Fed a constant (app_id) or pinned to the current year (release_year), so
# neither says anything about the configuration being scored -- and release_year
# dominates the ranking, pushing out the features that are actually a choice.
UNACTIONABLE = {"app_id", "release_year"}


def _shap_drivers(bundle: Bundle, target: str, frame: pd.DataFrame,
                  class_index: int | None = None,
                  top_n: int = 12) -> dict[str, float]:
    """SHAP contributions for one model, largest |value| first.

    The regressors give one value per feature. The owners classifier gives one
    per feature *per class*, so `class_index` picks the predicted bucket's
    slice -- SHAP has shipped both axis orders, hence the shape check.
    """
    explainer = _explainer(bundle, target)
    if explainer is None:
        return {}
    try:
        values = explainer.shap_values(frame)
    except Exception:
        return {}

    array = np.array(values)
    if array.ndim == 3:                      # (rows, features, classes) or (classes, rows, features)
        if class_index is None:
            return {}
        if array.shape[0] == len(frame):
            row = array[0, :, class_index]
        else:
            row = array[class_index, 0, :]
    elif array.ndim == 2:
        row = array[0]
    else:
        return {}

    pairs = sorted(
        ((c, v) for c, v in zip(frame.columns, row) if c not in UNACTIONABLE),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_n]
    return {driver_label(k): float(v) for k, v in pairs}


def driver_label(column: str) -> str:
    """'tag_story_rich' -> 'Story Rich (tag)', 'release_year' -> 'Release Year'."""
    for prefix, family in (
        ("genre_", "genre"), ("cat_", "category"), ("tag_", "tag"),
        ("lang_", "language"),
    ):
        if column.startswith(prefix):
            return f"{humanize(column[len(prefix):])} ({family})"
    return humanize(column)


def _band(value: float, count: int) -> tuple[int, int, int]:
    """An ordinal prediction as (nearest band, band below, band above).

    A value of 1.4 is the model saying "band 1, leaning 2" -- the pair either
    side of it is the honest range, and it is derived from the prediction
    itself rather than from an error statistic the bundle does not carry.
    """
    value = float(np.clip(value, 0.0, count - 1))
    return (
        int(round(value)),
        int(np.floor(value)),
        int(np.ceil(value)),
    )


def _untransform(value: float, transform: str | None) -> float:
    """Undo the transform the target was trained on."""
    if transform == "log1p":
        return float(np.expm1(value))
    return float(value)


def predict(spec, bundle: Bundle) -> Prediction:
    # The price model never sees `price` as an input, so it can run first --
    # which is what makes leaving the field empty work: with no price typed in,
    # the owners and review models are handed the price this model suggests.
    price_frame = build_frame(spec, bundle.features["price"], price=None)
    suggested = _untransform(
        bundle.models["price"].predict(price_frame)[0], bundle.transforms["price"]
    )
    suggested = max(suggested, 0.0)

    entered = spec.price
    price = suggested if entered is None else max(float(entered), 0.0)

    owners_frame = build_frame(spec, bundle.features["owners"], price=price)
    owners_model = bundle.models["owners"]
    last = len(bundle.owner_bounds) - 1

    if bundle.modes["owners"] == "ordinal":
        # A band index. There is no per-class probability to rank, so the SHAP
        # call below takes the regressor's single output rather than a slice,
        # and the range shown is the pair of bands the value falls between.
        owners_index = float(owners_model.predict(owners_frame)[0])
        best, lo_idx, hi_idx = _band(owners_index, last + 1)
        proba = None
        shap_class = None
    else:
        owners_index = None
        proba = owners_model.predict_proba(owners_frame)[0][bundle.class_order]
        best = int(np.argmax(proba))
        cumulative = np.cumsum(proba)
        lo_idx = min(int(np.searchsorted(cumulative, 0.10)), last)
        hi_idx = min(int(np.searchsorted(cumulative, 0.90)), last)
        shap_class = int(bundle.class_order[best])

    review_frame = build_frame(spec, bundle.features["review"], price=price)
    review_raw = _untransform(
        bundle.models["review"].predict(review_frame)[0],
        bundle.transforms["review"],
    )
    if bundle.modes["review"] == "ordinal":
        review_pct = None
        review_index = float(review_raw)
        review_band = bundle.review_bands[
            _band(review_index, len(bundle.review_bands))[0]
        ]
    else:
        review_pct = float(np.clip(review_raw, 0, 100))
        review_index = None
        review_band = None

    owners_low, owners_high = bundle.owner_bounds[best]
    return Prediction(
        owners_low=owners_low,
        owners_high=owners_high,
        owners_confidence=None if proba is None else float(proba[best]),
        owners_index=owners_index,
        spread_low=bundle.owner_bounds[lo_idx][0],
        spread_high=bundle.owner_bounds[hi_idx][1],
        review_pct=review_pct,
        review_index=review_index,
        review_band=review_band,
        review_bands=list(bundle.review_bands),
        suggested_price=suggested,
        entered_price=price,
        price_is_suggested=entered is None,
        revenue_low=owners_low * price,
        # The top bucket is open-ended, so its upper edge is infinite; a free
        # release would turn that into a NaN rather than a revenue of nothing.
        revenue_high=owners_high * price if price > 0 else 0.0,
        drivers={
            "owners": _shap_drivers(bundle, "owners", owners_frame, shap_class),
            "review": _shap_drivers(bundle, "review", review_frame),
            "price": _shap_drivers(bundle, "price", price_frame),
        },
        margins=bundle.margins,
        backend=bundle.label,
    )
