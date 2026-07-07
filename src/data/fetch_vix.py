"""
fetch_vix.py — Download and cache daily VIX closes from Yahoo Finance.

The VIX (CBOE Volatility Index) measures the market's expectation of 30-day
volatility. Think of it as the market's "fear gauge": VIX > 30 means elevated
fear, and our risk engine uses that threshold to halve position sizes.

We use yfinance (Yahoo Finance) rather than Alpaca because Alpaca doesn't carry
index data like ^VIX — only equity tickers. yfinance is free and works well for
daily index data.

Data is stored as Parquet (a compressed columnar format) rather than CSV because:
- It preserves the datetime index dtype without any parsing overhead.
- It's ~5x smaller than CSV for numeric data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from config import VIX_FILE, DATA_DIR


def fetch_vix(start: str = "2019-01-01") -> pd.DataFrame:
    """Download VIX daily closes from Yahoo Finance.

    Args:
        start: Start date string in YYYY-MM-DD format.

    Returns:
        DataFrame with a 'vix' column and a DatetimeIndex named 'date'.
    """
    raw = yf.download("^VIX", start=start, progress=False, auto_adjust=True)

    # Newer versions of yfinance may return a MultiIndex with the ticker name
    # as the top level (e.g. ('Close', '^VIX')). Squeezing/dropping that level
    # keeps things compatible across versions.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Close"]].copy()
    df.columns = ["vix"]
    df.index = pd.to_datetime(df.index)
    # Rename index from 'Date' (yfinance default) to lowercase 'date' for
    # consistency with the rest of our parquet files
    df.index.name = "date"
    return df


def update_vix() -> None:
    """Append latest VIX data to parquet, creating the file if it doesn't exist.

    On subsequent calls it only downloads the days since the last stored date,
    so it's fast after the first run (incremental update pattern).
    """
    DATA_DIR.mkdir(exist_ok=True)

    if VIX_FILE.exists():
        existing = pd.read_parquet(VIX_FILE)
        last = existing.index.max()
        # Start from the day after our last stored date
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        # If we're already up to date, nothing to do (avoids yfinance "start > end" error)
        if start > datetime.now().strftime("%Y-%m-%d"):
            return
        new_data = fetch_vix(start=start)
        combined = pd.concat([existing, new_data])
    else:
        combined = fetch_vix()

    # Remove any duplicate dates (keep the latest values) and sort chronologically
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(VIX_FILE)


def load_vix() -> pd.DataFrame:
    """Load VIX from parquet.

    Call update_vix() first if you want to ensure the data is current.

    Returns:
        DataFrame with 'vix' column and DatetimeIndex named 'date'.
    """
    return pd.read_parquet(VIX_FILE)
