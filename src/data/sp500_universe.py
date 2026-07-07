"""
sp500_universe.py — S&P 500 constituent list, scraped from Wikipedia and cached locally.

Why Wikipedia? It's free, always up to date, and has a clean table at
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies — no API key needed.

The CSV cache means we only hit Wikipedia once (or when you explicitly refresh),
which is polite to Wikipedia and fast for repeated runs.
"""

import requests
import pandas as pd
from config import UNIVERSE_FILE


def fetch_sp500_tickers() -> pd.DataFrame:
    """Scrape current S&P 500 constituents from Wikipedia.

    Returns a DataFrame with columns: ticker, company, sector, sub_industry.
    Dots in tickers are replaced with hyphens (e.g. BRK.B → BRK-B) because
    Alpaca's API uses hyphens as the separator for share classes.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia returns 403 for requests without a User-Agent header.
    # Sending a descriptive UA is also polite — it lets Wikipedia know who's asking.
    headers = {"User-Agent": "trading-research/1.0 (educational project; python-requests)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(response.text, flavor="lxml")
    # The first table on that page is always the constituent list
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["ticker", "company", "sector", "sub_industry"]
    # Alpaca uses hyphens, not dots (e.g. BRK.B → BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def load_sp500_tickers(refresh: bool = False) -> pd.DataFrame:
    """Return S&P 500 DataFrame, scraping Wikipedia if cache is missing or stale.

    Args:
        refresh: If True, re-scrape Wikipedia even if the CSV exists.

    Returns:
        DataFrame with columns: ticker, company, sector, sub_industry.
    """
    if not UNIVERSE_FILE.exists() or refresh:
        # Make sure the data/ directory exists before writing
        UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_sp500_tickers()
        df.to_csv(UNIVERSE_FILE, index=False)
    return pd.read_csv(UNIVERSE_FILE)


# Tickers that Alpaca's API rejects even after the dot→hyphen conversion.
# BRK/B and BF/B have slash-based share-class separators that neither format resolves.
_ALPACA_UNSUPPORTED: set[str] = {"BRK-B", "BF-B"}


def get_symbols(refresh: bool = False) -> list[str]:
    """Return list of S&P 500 ticker symbols (Alpaca-compatible).

    This is the main entry point for other modules that just need the ticker list.
    """
    all_tickers = load_sp500_tickers(refresh)["ticker"].tolist()
    return [t for t in all_tickers if t not in _ALPACA_UNSUPPORTED]


def get_sector_map(refresh: bool = False) -> dict[str, str]:
    """Return {ticker: sector} mapping for sector-cap enforcement.

    Used by the risk engine to check that no single GICS sector exceeds
    the MAX_SECTOR_PCT limit defined in config.py.
    """
    df = load_sp500_tickers(refresh)
    return dict(zip(df["ticker"], df["sector"]))
