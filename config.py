"""
Shared configuration for the scraper module: base URLs, request headers,
timeouts and rate limiting.

FBref actively rate-limits/blocks aggressive scraping. Keep REQUEST_DELAY_SECONDS
reasonably high (their own guidance is roughly one request every 3 seconds)
to avoid getting temporarily blocked.
"""

FBREF_BASE_URL = "https://fbref.com"
SOFASCORE_API_BASE = "https://api.sofascore.com/api/v1"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS = 3.0   # be polite, especially with FBref
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0