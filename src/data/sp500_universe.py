"""
sp500_universe.py — S&P 500 constituent list, scraped from Wikipedia and cached locally.

Why Wikipedia? It's free, always up to date, and has a clean table at
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies — no API key needed.

The CSV cache means we only hit Wikipedia once (or when you explicitly refresh),
which is polite to Wikipedia and fast for repeated runs.

SURVIVORSHIP BIAS WARNING
--------------------------
Wikipedia's table lists the *current* S&P 500 members. Every price/news/
earnings/insider fetch that starts from get_symbols() uses this same list —
including when pulling several years of *historical* data. That means a
company that was in the index in, say, 2021 but has since been removed
(acquired, bankrupt, delisted, demoted) never appears anywhere in our
dataset, while every ticker we do have is, by definition, one that survived
to today. Backtests built on this universe will look better than a real
point-in-time strategy would have performed, because the failures have been
quietly erased from the sample. This is a known, *unfixed* limitation — see
get_symbols() for the runtime warning, and CLAUDE.md for the project-wide
rule this violates.
"""

import logging

import requests
import pandas as pd
from config import UNIVERSE_FILE

log = logging.getLogger(__name__)

# Set to True after the first survivorship-bias warning has been logged, so
# we nag once per process (e.g. once per backtest run) instead of spamming
# the log every time get_symbols() is called in a loop.
_survivorship_warned = False


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

    Warning: this is the *current* constituent list applied across all of
    history (see the module docstring's SURVIVORSHIP BIAS WARNING). We log
    a one-time warning the first time this is called in a process so it's
    impossible to run a backtest without seeing the caveat at least once.
    """
    global _survivorship_warned
    if not _survivorship_warned:
        log.warning(
            "sp500_universe.get_symbols() returns CURRENT S&P 500 members. "
            "Point-in-time membership masking (src/data/sp500_history.py) now "
            "corrects backfill bias in the feature matrix and backtest, but "
            "price HISTORY for delisted/acquired ex-members is often missing "
            "from free data feeds, so a residual survivorship tilt remains. "
            "Read backtest numbers with this in mind."
        )
        _survivorship_warned = True

    all_tickers = load_sp500_tickers(refresh)["ticker"].tolist()
    return [t for t in all_tickers if t not in _ALPACA_UNSUPPORTED]


def get_sector_map(refresh: bool = False) -> dict[str, str]:
    """Return {ticker: sector} mapping for sector-cap enforcement.

    Used by the risk engine to check that no single GICS sector exceeds
    the MAX_SECTOR_PCT limit defined in config.py.
    """
    df = load_sp500_tickers(refresh)
    return dict(zip(df["ticker"], df["sector"]))
