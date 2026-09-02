"""Palette and the option lists the form offers."""

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

# --------------------------------------------------------------------------
# Dropdown options
# --------------------------------------------------------------------------
GENRES = [
    "Action", "Adventure", "Casual", "Indie", "Massively Multiplayer",
    "RPG", "Racing", "Simulation", "Sports", "Strategy",
    "Early Access", "Free To Play", "Violent", "Gore", "Nudity",
]

CATEGORIES = [
    "Single-player", "Multi-player", "PvP", "Online PvP", "Co-op",
    "Online Co-op", "Shared/Split Screen", "Cross-Platform Multiplayer",
    "Steam Achievements", "Steam Cloud", "Steam Workshop",
    "Steam Trading Cards", "Steam Leaderboards", "Full controller support",
    "Partial Controller Support", "Remote Play Together", "VR Supported",
    "Includes level editor", "In-App Purchases", "Captions available",
]

TAGS = [
    "2D", "3D", "Action Roguelike", "Anime", "Atmospheric", "Automation",
    "Base Building", "Bullet Hell", "Card Game", "City Builder", "Colony Sim",
    "Comedy", "Crafting", "Cute", "Dark Fantasy", "Deckbuilding",
    "Difficult", "Exploration", "Farming Sim", "First-Person", "Souls-like",
    "Great Soundtrack", "Hack and Slash", "Horror", "Immersive Sim",
    "Metroidvania", "Multiplayer", "Narrative", "Open World", "Pixel Graphics",
    "Platformer", "Point & Click", "Procedural Generation", "Psychological Horror",
    "Puzzle", "Relaxing", "Replay Value", "Rogue-lite", "Sandbox", "Sci-fi",
    "Shooter", "Simulation", "Stealth", "Story Rich", "Survival",
    "Tactical", "Third Person", "Turn-Based", "Visual Novel", "Zombies",
]

LANGUAGES = [
    "English", "Simplified Chinese", "Traditional Chinese", "Japanese",
    "Korean", "Russian", "German", "French", "Spanish - Spain",
    "Spanish - Latin America", "Portuguese - Brazil", "Italian", "Polish",
    "Turkish", "Dutch", "Danish", "Finnish", "Norwegian", "Swedish",
    "Czech", "Hungarian", "Ukrainian", "Thai", "Vietnamese", "Arabic",
]

PLATFORMS = ["Windows", "Mac", "Linux"]

# `price_status` is a STRING column in the source table.
PRICE_STATUS = ["Paid", "Free", "Free To Play"]

VOCAB = {
    "genres": GENRES,
    "categories": CATEGORIES,
    "tags": TAGS,
    "languages": LANGUAGES,
}
