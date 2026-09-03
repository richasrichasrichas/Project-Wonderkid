"""
sports_scraper

A small module to scrape player statistics from FBref and SofaScore,
clean noisy text fields with a Hugging Face NER model, and export the
results to CSV via pandas.
"""

__all__ = [
    "config",
    "fbref_scraper",
    "sofascore_scraper",
    "text_cleaner",
    "pipeline",
]