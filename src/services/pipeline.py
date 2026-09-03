"""
pipeline.py

High-level entry points that tie the scrapers and the text cleaner
together and export the result to CSV via pandas.
"""

from __future__ import annotations
from typing import List, Optional

import pandas as pd

from . import fbref_scraper, sofascore_scraper
from .text_cleaner import HFTextCleaner, basic_clean


def export_fbref_standard_stats(
    comp_id: int,
    season: str,
    comp_slug: str,
    output_path: str,
    name_column: str = "Player",
    use_hf_cleaning: bool = False,
) -> pd.DataFrame:
    """
    Scrapes the FBref "Standard Stats" table for a competition/season,
    cleans the player-name column, and writes the result to `output_path`.

    use_hf_cleaning=False (default) uses the fast regex-based cleaner.
    use_hf_cleaning=True routes the name column through HFTextCleaner
    (requires transformers/torch to be installed).
    """
    df = fbref_scraper.scrape_standard_stats(comp_id, season, comp_slug)

    if name_column in df.columns:
        if use_hf_cleaning:
            cleaner = HFTextCleaner()
            df = cleaner.clean_dataframe_column(df, name_column)
        else:
            df[name_column] = df[name_column].map(basic_clean)

    df.to_csv(output_path, index=False)
    return df


def export_sofascore_player_stats(
    player_ids: List[int],
    tournament_id: int,
    season_id: int,
    output_path: str,
) -> pd.DataFrame:
    """
    Fetches season statistics for a list of SofaScore player IDs and
    writes the combined result to `output_path`.
    """
    df = sofascore_scraper.get_players_season_statistics(player_ids, tournament_id, season_id)
    df.to_csv(output_path, index=False)
    return df


def merge_fbref_sofascore(
    fbref_df: pd.DataFrame,
    sofascore_df: pd.DataFrame,
    fbref_name_col: str = "Player",
    sofascore_name_col: str = "name",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merges an FBref DataFrame with a SofaScore DataFrame on player name.

    Player names rarely match exactly between sources (accents, suffixes,
    nickname vs. full name), so this does a normalized-string join —
    lowercased and stripped of accents — rather than a plain equality
    join. For production use, a fuzzy-matching step (e.g. rapidfuzz)
    is recommended for names that don't match exactly after normalization.
    """
    import re
    import unicodedata

    def _normalize(name: str) -> str:
        if not isinstance(name, str):
            return ""
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"\s+", " ", name.strip().lower())
        return name

    fbref_df = fbref_df.copy()
    sofascore_df = sofascore_df.copy()
    fbref_df["_merge_key"] = fbref_df[fbref_name_col].map(_normalize)
    sofascore_df["_merge_key"] = sofascore_df[sofascore_name_col].map(_normalize)

    merged = pd.merge(
        fbref_df, sofascore_df, on="_merge_key", how="inner", suffixes=("_fbref", "_sofascore")
    ).drop(columns=["_merge_key"])

    if output_path:
        merged.to_csv(output_path, index=False)
    return merged