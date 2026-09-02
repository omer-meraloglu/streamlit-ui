# SteamCast — Steam-themed page mockup

Visual template for the game sales & review prediction UI. **No dataset, no
model, no BigQuery connection** — the numbers on screen are sample values from
`demo.py` so the layout has something to render.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Schema

Field names follow the BigQuery table
`datascientiststeamproject.kaggle.Steam`.

**Inputs** (`GameSpec`) — columns known before launch: `name`, `release_date`,
`price`, `price_status`, `developers`, `publishers`, `genres`, `categories`,
`tags`, `achievements`, `dlc_count`, `supported_languages`,
`full_audio_languages`, `windows` / `mac` / `linux`, `screenshots`, `movies`,
`website`.

**Targets** (`PreviewNumbers`) — what the models will predict:
`estimated_owners`, `positive`, `negative`, `peak_ccu`, `recommendations`,
`metacritic_score`, `user_score`. These are kept out of the form so a target
cannot be typed in as a feature.

Two details carried over from the real table:

- `estimated_owners` is a **STRING** bucket (`"20,000 - 50,000"`), not a
  number. The tile renders the bucket; `owners_mid` holds a numeric midpoint
  used only for the derived revenue figure.
- Reviews are two **INTEGER** columns, `positive` and `negative`. The review
  score and the Steam summary label are computed from them, not stored.

`revenue` is the one displayed figure with no matching column — it is derived
as owners × price and labelled as such.

## One-screen layout

The page is built to fit without scrolling, verified down to **1280×720**.
That constraint drives several choices — change them together or it will
start scrolling again:

- Fields are packed into rows (`st.columns`) rather than stacked; the six
  multiselects sit two-per-row in a single card.
- `number_input` instead of `st.slider` — sliders cost ~30px each.
- The vertical rhythm is tightened in `styles.css` (gaps, label size, control
  height). The multiselect "Clear all" button is hidden there too, so chips
  get its width; each chip keeps its own remove button.
- The drivers chart has no fixed bar height — a hard value collapses the
  bands onto one row at this chart size.

```
app.py                  the page
demo.py                 sample numbers (not a model)
components/header.py    top nav bar, sidebar
components/inputs.py    the form
components/results.py   tiles, review badge, chart
utils/constants.py      palette + dropdown options
utils/formatting.py     Steam review labels, number formatting
utils/theme.py          CSS injection, Altair theme
assets/styles.css       the Steam skin
.streamlit/config.toml  base dark theme
```
# streamlit-ui
