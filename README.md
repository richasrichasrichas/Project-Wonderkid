# Project Wonderkid

Scrapes player statistics from FBref and SofaScore, optionally cleans
noisy text fields with a Hugging Face NER model, and exports the result
to CSV via pandas.

## Structure

| File | Purpose |
|---|---|
| `config.py` | Base URLs, headers, timeouts, rate-limit delay |
| `fbref_scraper.py` | Fetches FBref pages and parses tables hidden inside HTML comments |
| `sofascore_scraper.py` | Queries SofaScore's internal JSON API |
| `text_cleaner.py` | `basic_clean` (regex, no deps) + `HFTextCleaner` (HF NER, lazy-loaded) |
| `pipeline.py` | Orchestrates scrape -> clean -> `to_csv(...)` |

## Install

```bash
pip install -r requirements.txt
# transformers + torch are only required if you use HFTextCleaner
```

## Usage

```python
from sports_scraper.pipeline import export_fbref_standard_stats, export_sofascore_player_stats

# FBref: Premier League 2023-2024 standard stats, regex-cleaned names
df = export_fbref_standard_stats(
    comp_id=9, season="2023-2024", comp_slug="Premier-League",
    output_path="fbref_standard_stats.csv",
)

# Same, but route player names through the HF NER cleaner instead
df = export_fbref_standard_stats(
    comp_id=9, season="2023-2024", comp_slug="Premier-League",
    output_path="fbref_standard_stats_hf.csv",
    use_hf_cleaning=True,
)

# SofaScore: season stats for a list of player IDs
df = export_sofascore_player_stats(
    player_ids=[12345, 67890],
    tournament_id=17, season_id=52186,
    output_path="sofascore_player_stats.csv",
)
```

## Important notes

- **FBref actively rate-limits scraping.** `config.REQUEST_DELAY_SECONDS`
  defaults to 3s between requests — don't lower it aggressively, you'll
  get temporarily blocked.
- **The SofaScore endpoints are unofficial** (no public API docs). Get
  `player_id` / `tournament_id` / `season_id` by inspecting the Network
  tab on sofascore.com — these can change without notice.
- **This code was validated offline** against synthetic HTML/JSON
  fixtures that mirror FBref's and SofaScore's real structure (comment-
  hidden tables, two-level headers, repeated header rows). It has *not*
  been run against the live sites in this environment, since the sandbox
  network is restricted to a small allowlist of domains (pypi, github,
  etc.) that doesn't include fbref.com or sofascore.com. Test against
  the real URLs in your own environment before relying on it.
- Respect each site's Terms of Service and `robots.txt` for your use case.