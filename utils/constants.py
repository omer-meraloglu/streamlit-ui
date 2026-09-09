"""Palette and the fixed option lists.

The genre / category / tag vocabularies used to live here as hand-written lists.
They now come from the trained models (`utils.model_backend.Bundle`), because a
label the models were never fitted on cannot be scored -- keeping a second,
drifting copy here is how the form ends up offering options the models ignore.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Steam palette (sampled from the store front-end)
# --------------------------------------------------------------------------
STEAM = {
    "bg_dark": "#171a21",      # top nav / footer
    "bg": "#1b2838",           # page background
    "panel": "#16202d",        # card background
    "panel_alt": "#2a475e",    # raised panel / sidebar
    "blue": "#66c0f4",         # the signature Steam light blue
    "blue_dim": "#417a9b",
    "text": "#c7d5e0",
    "text_muted": "#8f98a0",
    "green": "#a4d007",        # "Play"/install button
    "gold": "#b9a074",         # "Mixed" reviews
    "rust": "#a34c25",         # "Negative" reviews
}

# Review-summary colours, matching how the store labels each bucket.
SENTIMENT_COLORS = {
    "positive": STEAM["blue"],
    "mixed": STEAM["gold"],
    "negative": STEAM["rust"],
}

# Three BOOLEAN feature columns: supports_windows / mac / linux.
PLATFORMS = ["Windows", "Mac", "Linux"]
