"""Loads the trained models from Steam-Price-Popularity-Predictor and runs them.

Three models are trained by that project, one per target:

  price   Regressor  -> log1p(price)                 (no `price` feature)
  owners  Classifier -> estimated owners bucket      (uses `price`)
  review  Regressor  -> positive_review_percentage   (uses `price`)

Two on-disk layouts are supported:

  bundles/    the current one. `bundle_<target>.pkl` is a dict carrying the
              fitted estimator, its `selected_features`, the `target_transform`
              applied while training and the `error_margins` measured on the
              held-out set -- so a prediction can be reported with a confidence
              interval instead of as a bare number.
  saved_models[_xgb]/   the older one: a bare estimator per target plus
              `label_encoder_owners.pkl`, and no error margins.

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

# The training project, checked out next to this one -- the local-dev case, and
# still what is searched first so a fresh training run keeps winning.
SIBLING_ROOT = APP_DIR.parent / "Steam-Price-Popularity-Predictor" / "ml_models"

# Fallback: .pkl files uploaded by hand into this repo, which is how the
# deployed app gets its models -- there is no sibling checkout in the cloud.
LOCAL_ROOT = APP_DIR / "models"

# estimated_owners_avg holds the MID-POINT of a Steam owner range ("0 - 20,000"
# is stored as 10000). These edges reproduce the ranges, so a prediction can be
# reported as the interval it actually is rather than a spuriously exact number.
OWNER_BUCKET_EDGES = [
    0, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000,
    5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000,
]

TARGETS = ("price", "owners", "review")

# The current format: bundle_<target>.pkl, self-describing, error margins
# included. Listed first so it is what the app opens on.
BUNDLE_FOLDER = "bundles"
BUNDLE_LABEL = "LightGBM (final)"

# The earlier format, kept selectable so old runs stay reproducible. Keyed by
# folder name: the training script has since started dropping bundles into
# saved_models/ too, and a directory has to be labelled by the format it really
# holds -- otherwise the bundles get served under the "legacy" name.
LEGACY_LABELS = {
    "saved_models": "LightGBM (legacy)",
    "saved_models_xgb": "XGBoost (legacy)",
}

# `app_id` is one of the selected features, but a game that has not shipped has
# no id yet. Steam hands them out in ascending order, so a release configured
# today would draw one near the top of the range -- feeding that is closer to
# the truth than 0, which would sit the prediction next to the 2003 catalogue.
DEFAULT_APP_ID = 3_000_000.0

# Label used when the .pkl files sit directly in a root rather than in a
# saved_models/ subfolder -- the shape a manual upload usually takes. Named
# neutrally because a flat drop does not say which library trained it.
FLAT_LABEL = "Uploaded"


def search_roots() -> list[Path]:
    """Directories to search, highest priority first."""
    roots = []
    override = os.environ.get("STEAMCAST_MODEL_ROOT")
    if override:
        roots.append(Path(override))
    roots.append(SIBLING_ROOT)
    roots.append(LOCAL_ROOT)
    return roots


def _has_bundles(directory: Path) -> bool:
    """Current layout: one self-describing bundle per target."""
    return directory.is_dir() and all(
        (directory / f"bundle_{t}.pkl").exists() for t in TARGETS
    )


def _has_estimators(directory: Path) -> bool:
    """Older layout: three bare estimators plus the owners encoder."""
    if not directory.is_dir():
        return False
    needed = [directory / f"model_{t}.pkl" for t in TARGETS]
    needed.append(directory / "label_encoder_owners.pkl")
    return all(p.exists() for p in needed)


def _is_complete(directory: Path) -> bool:
    """A directory usable as a backend, in either layout."""
    return _has_bundles(directory) or _has_estimators(directory)


def _label_for(directory: Path) -> str | None:
    """What this directory would be called, or None if it holds no model set.

    Bundles win over bare estimators in the same folder: they are the newer
    export, and they are the only one carrying error margins.
    """
    if _has_bundles(directory):
        return BUNDLE_LABEL
    if _has_estimators(directory):
        return LEGACY_LABELS.get(directory.name, FLAT_LABEL)
    return None


def available_backends() -> dict[str, Path]:
    """Backends holding a complete set of models, nearest root winning.

    Each root is checked for bundles/, then saved_models/ and
    saved_models_xgb/, then for a flat drop of .pkl files in the root itself. A
    label found in an earlier root is never overwritten by a later one, so the
    sibling checkout takes precedence over anything uploaded into this repo.
    """
    found: dict[str, Path] = {}
    for root in search_roots():
        candidates = [root / BUNDLE_FOLDER]
        candidates += [root / folder for folder in LEGACY_LABELS]
        candidates.append(root)
        for directory in candidates:
            label = _label_for(directory)
            if label is not None and label not in found:
                found[label] = directory

    order = [BUNDLE_LABEL] + list(LEGACY_LABELS.values()) + [FLAT_LABEL]
    return {label: found[label] for label in order if label in found}


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
    owner_labels: list[str]             # bucket names, smallest first
    owner_bounds: list                  # (low, high) owners, aligned with the above
    class_order: np.ndarray             # model class index for each entry above
    explainers: dict
    genres: dict[str, str]
    categories: dict[str, str]
    tags: dict[str, str]
    languages: dict[str, str]

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


def _owner_bounds(mids: np.ndarray) -> list[tuple[float, float]]:
    """Mid-points -> (low, high) edges, verified against OWNER_BUCKET_EDGES.

    Falls back to a degenerate range if the trained classes ever stop lining up
    with the edges, so a schema change degrades to the old point estimate
    instead of inventing an interval.
    """
    if len(mids) == len(OWNER_BUCKET_EDGES) - 1 and all(
        abs((OWNER_BUCKET_EDGES[i] + OWNER_BUCKET_EDGES[i + 1]) / 2 - m) < 1
        for i, m in enumerate(mids)
    ):
        return [
            (float(OWNER_BUCKET_EDGES[i]), float(OWNER_BUCKET_EDGES[i + 1]))
            for i in range(len(mids))
        ]
    return [(float(m), float(m)) for m in mids]


@lru_cache(maxsize=4)
def load_bundle(label: str) -> Bundle:
    directory = available_backends()[label]
    if _has_bundles(directory):
        return _load_bundled(label, directory)
    return _load_estimators(label, directory)


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


def _assemble(label, directory, models, features, transforms, margins,
              owner_labels, owner_bounds, class_order, explainers) -> Bundle:
    union = sorted(set().union(*features.values()))
    return Bundle(
        label=label,
        directory=directory,
        models=models,
        features=features,
        transforms=transforms,
        margins=margins,
        owner_labels=owner_labels,
        owner_bounds=owner_bounds,
        class_order=class_order,
        explainers=explainers,
        genres=_vocab(union, "genre_"),
        categories=_vocab(union, "cat_"),
        tags=_vocab(union, "tag_"),
        languages=_vocab(union, "lang_"),
    )


def _load_bundled(label: str, directory: Path) -> Bundle:
    """Current layout: each pickle is a dict describing its own model."""
    models, features, transforms, margins = {}, {}, {}, {}
    for target in TARGETS:
        saved = joblib.load(directory / f"bundle_{target}.pkl")
        models[target] = saved["model"]
        features[target] = list(saved["selected_features"])
        transforms[target] = saved.get("target_transform")
        margins[target] = dict(saved.get("error_margins") or {})

    # The classifier is fitted on the bucket names themselves ('20k-50k'), so
    # `classes_` is in lexical order -- '1M+' lands second. Sort by lower edge.
    names = [str(c) for c in models["owners"].classes_]
    bounds = [_owner_range(n) for n in names]
    order = np.argsort([low for low, _ in bounds])

    return _assemble(
        label, directory, models, features, transforms, margins,
        owner_labels=[names[i] for i in order],
        owner_bounds=[bounds[i] for i in order],
        class_order=order,
        explainers=_explainers(directory, features),
    )


def _load_estimators(label: str, directory: Path) -> Bundle:
    """Older layout: bare estimators, a separate encoder, no error margins."""
    models, features = {}, {}
    for target in TARGETS:
        models[target] = joblib.load(directory / f"model_{target}.pkl")
        features[target] = _feature_names(models[target], directory, target)

    encoder = joblib.load(directory / "label_encoder_owners.pkl")
    # Classes are numeric owner mid-points stored as strings ('10000', '150000'),
    # so LabelEncoder ordered them lexically. Sort numerically for display.
    values = np.array([float(c) for c in encoder.classes_])
    order = np.argsort(values)
    bounds = _owner_bounds(values[order])

    return _assemble(
        label, directory, models, features,
        # These models were trained with log1p(price) and untransformed targets
        # elsewhere; nothing was saved alongside them to say so.
        transforms={"price": "log1p", "owners": None, "review": None},
        margins={t: {} for t in TARGETS},
        owner_labels=[f"{low:,.0f}-{high:,.0f}" for low, high in bounds],
        owner_bounds=bounds,
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
    owners_confidence: float       # probability mass on that bucket
    spread_low: float              # lower edge spanning the middle 80% of mass
    spread_high: float             # upper edge of the same span
    review_pct: float              # positive_review_percentage, 0-100
    suggested_price: float         # what the price model would charge
    entered_price: float           # the price the other two models were given
    price_is_suggested: bool       # True when no price was typed in
    revenue_low: float             # derived: owners range x entered price
    revenue_high: float
    drivers: dict[str, float]      # SHAP values for the owners model
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


def _owners_explainer(bundle: Bundle):
    """The owners explainer, rebuilt from the model when no pickle shipped.

    explainer_owners.pkl is 26MB -- larger than every other artefact combined --
    so a deployment can ship the models alone and have the explainer
    reconstructed here. shap is imported lazily: it costs ~1s and is not needed
    when the drivers panel is switched off.
    """
    if "owners" not in bundle.explainers:
        try:
            import shap

            bundle.explainers["owners"] = shap.TreeExplainer(bundle.models["owners"])
        except Exception:
            return None
    return bundle.explainers["owners"]


# Fed a constant (app_id) or pinned to the current year (release_year), so
# neither says anything about the configuration being scored -- and release_year
# dominates the ranking, pushing out the features that are actually a choice.
UNACTIONABLE = {"app_id", "release_year"}


def _shap_for_owners(bundle: Bundle, frame: pd.DataFrame, class_index: int,
                     top_n: int = 12) -> dict[str, float]:
    """SHAP contributions for the predicted owner bucket, largest |value| first."""
    explainer = _owners_explainer(bundle)
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
    proba = owners_model.predict_proba(owners_frame)[0][bundle.class_order]
    best = int(np.argmax(proba))

    cumulative = np.cumsum(proba)
    last = len(bundle.owner_bounds) - 1
    lo_idx = min(int(np.searchsorted(cumulative, 0.10)), last)
    hi_idx = min(int(np.searchsorted(cumulative, 0.90)), last)

    review_frame = build_frame(spec, bundle.features["review"], price=price)
    review_pct = float(np.clip(
        _untransform(
            bundle.models["review"].predict(review_frame)[0],
            bundle.transforms["review"],
        ),
        0, 100,
    ))

    owners_low, owners_high = bundle.owner_bounds[best]
    return Prediction(
        owners_low=owners_low,
        owners_high=owners_high,
        owners_confidence=float(proba[best]),
        spread_low=bundle.owner_bounds[lo_idx][0],
        spread_high=bundle.owner_bounds[hi_idx][1],
        review_pct=review_pct,
        suggested_price=suggested,
        entered_price=price,
        price_is_suggested=entered is None,
        revenue_low=owners_low * price,
        # The top bucket is open-ended, so its upper edge is infinite; a free
        # release would turn that into a NaN rather than a revenue of nothing.
        revenue_high=owners_high * price if price > 0 else 0.0,
        drivers=_shap_for_owners(bundle, owners_frame, int(bundle.class_order[best])),
        margins=bundle.margins,
        backend=bundle.label,
    )
