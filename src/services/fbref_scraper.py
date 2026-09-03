"""
fbref_scraper.py

Scrapes player/team statistics tables from FBref.

FBref hides most secondary tables inside HTML comments (a common
anti-scraping pattern), so a plain `pandas.read_html(url)` call misses
them. This module fetches the raw page, pulls the target table out of
its comment wrapper, and parses it with pandas.
"""

from __future__ import annotations
import re
import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from . import config


class FBrefError(Exception):
    """Raised when a page or table cannot be fetched/parsed from FBref."""


def _fetch_html(url: str) -> str:
    """Fetches a page's raw HTML with retries and a polite delay."""
    last_error: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=config.DEFAULT_HEADERS, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            time.sleep(config.REQUEST_DELAY_SECONDS)
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)
    raise FBrefError(f"Failed to fetch {url} after {config.MAX_RETRIES} attempts: {last_error}")


def extract_table_html(page_html: str, table_id: str) -> str:
    """
    Returns the raw HTML of a <table id="table_id"> element, whether it
    sits directly in the page or is wrapped inside an HTML comment
    (FBref's usual pattern for secondary tables).
    """
    soup = BeautifulSoup(page_html, "lxml")

    table = soup.find("table", id=table_id)
    if table is not None:
        return str(table)

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if table_id not in comment:
            continue
        comment_soup = BeautifulSoup(comment, "lxml")
        table = comment_soup.find("table", id=table_id)
        if table is not None:
            return str(table)

    raise FBrefError(f"Table id '{table_id}' not found in page (checked comments too).")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FBref stat tables commonly use a two-level header (e.g. "Playing Time"
    -> "MP"). Flattens that into single column names like "Playing Time_MP",
    and drops the "Unnamed" prefix noise pandas generates for single-level
    headers spanning a MultiIndex.
    """
    if isinstance(df.columns, pd.MultiIndex):
        flat_columns = []
        for top, bottom in df.columns:
            top = "" if str(top).startswith("Unnamed") else str(top).strip()
            bottom = str(bottom).strip()
            flat_columns.append(f"{top}_{bottom}" if top else bottom)
        df = df.copy()
        df.columns = flat_columns
    return df


def _drop_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long FBref tables repeat the header row every N rows as a visual aid.
    Those show up as data rows where the first column's value equals its
    own column name (or a literal "Rk"/"Player" divider) — drop them.
    """
    if df.empty:
        return df
    first_col = df.columns[0]
    mask = df[first_col].astype(str) != str(first_col)
    mask &= df[first_col].astype(str) != "Rk"
    return df[mask].reset_index(drop=True)


def parse_table(page_html: str, table_id: str) -> pd.DataFrame:
    """Extracts and parses a single FBref table into a clean DataFrame."""
    table_html = extract_table_html(page_html, table_id)
    # Wrapped in StringIO: passing a raw HTML string directly can trip
    # pandas/lxml's "is this a file path or URL?" heuristic on short inputs.
    df = pd.read_html(StringIO(table_html))[0]
    df = _flatten_columns(df)
    df = _drop_repeated_header_rows(df)
    return df


def scrape_table(url: str, table_id: str) -> pd.DataFrame:
    """Fetches `url` and parses the table identified by `table_id`."""
    html = _fetch_html(url)
    return parse_table(html, table_id)


def scrape_standard_stats(comp_id: int, season: str, comp_slug: str,
                           table_id: str = "stats_standard") -> pd.DataFrame:
    """
    Scrapes the "Standard Stats" table for a given competition/season.

    Example: comp_id=9, season="2023-2024", comp_slug="Premier-League"
      -> https://fbref.com/en/comps/9/2023-2024/stats/2023-2024-Premier-League-Stats
    """
    url = f"{config.FBREF_BASE_URL}/en/comps/{comp_id}/{season}/stats/{season}-{comp_slug}-Stats"
    return scrape_table(url, table_id)


def scrape_player_scouting_report(player_id: str, player_slug: str,
                                   table_id: str = "scout_full_AS Roma") -> pd.DataFrame:
    """
    Scrapes a player's percentile scouting report table.
    Note: `table_id` includes the comparison group FBref chose for that
    player (e.g. "scout_full_Big 5 Leagues, 2023-24") — inspect the page
    to confirm the exact id before calling this.
    """
    url = f"{config.FBREF_BASE_URL}/en/players/{player_id}/scout/365_m1/{player_slug}-Scouting-Report"
    return scrape_table(url, table_id)