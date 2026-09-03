"""
text_cleaner.py

Cleans noisy text extracted from scraped pages (footnote markers, extra
whitespace, embedded metadata). Provides a fast regex-based cleaner for
the common cases, plus an optional Hugging Face NER-based cleaner for
messier text where a plain regex isn't reliable enough — e.g. raw text
blocks where a player's name is embedded next to team/nationality
metadata with no clean delimiter.

`transformers` (and a backend like `torch`) is only imported lazily,
inside HFTextCleaner, so importing this module — or using `basic_clean`
— never requires those heavier dependencies to be installed.
"""

from __future__ import annotations
import re
from typing import List, Optional

import pandas as pd

_FOOTNOTE_SUFFIX = re.compile(r"[\*\d]+$")
_WHITESPACE = re.compile(r"\s+")


def basic_clean(text: Optional[str]) -> Optional[str]:
    """
    Fast, dependency-free cleanup for the common FBref/SofaScore noise:
    trailing footnote markers ("Player Name2", "Player Name*"), duplicate
    whitespace, and leading/trailing spaces.
    """
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = _FOOTNOTE_SUFFIX.sub("", text).strip()
    text = _WHITESPACE.sub(" ", text)
    return text


class HFTextCleaner:
    """
    Wraps a Hugging Face NER pipeline to pull a clean player name out of
    noisy raw text (e.g. a scraped block like "Rodrigo 'Rodri' Hernández
    Cascante Spain Manchester City" with no clear delimiters).

    The pipeline is lazy-loaded on first use, so constructing this class
    is cheap and importing the module doesn't require transformers/torch.

    Requires: pip install transformers torch
    """

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = model_name
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "HFTextCleaner requires the 'transformers' package "
                    "(and a backend like 'torch'). Install with: "
                    "pip install transformers torch"
                ) from exc
            self._pipeline = pipeline(
                "ner", model=self.model_name, aggregation_strategy="simple"
            )
        return self._pipeline

    def extract_person_name(self, raw_text: Optional[str]) -> Optional[str]:
        """
        Runs NER on `raw_text` and returns the longest entity tagged as a
        person. Falls back to `basic_clean(raw_text)` when no person
        entity is found, or when the input isn't a usable string.
        """
        text = basic_clean(raw_text)
        if not text:
            return text

        nlp = self._ensure_pipeline()
        entities = nlp(text)
        person_spans = [e["word"] for e in entities if e.get("entity_group") == "PER"]

        if not person_spans:
            return text
        return max(person_spans, key=len)

    def clean_column(self, values: List[Optional[str]]) -> List[Optional[str]]:
        """Applies `extract_person_name` across a list of raw text values."""
        return [self.extract_person_name(v) for v in values]

    def clean_dataframe_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Returns a copy of `df` with `column` cleaned via the HF pipeline."""
        df = df.copy()
        df[column] = self.clean_column(df[column].tolist())
        return df