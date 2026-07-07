"""Intraday bar fetcher: 30-minute bars from Alpaca for a single trading day.

Saves one parquet per (ticker, date) into:
    data/intraday/<TICKER>/YYYY-MM-DD.parquet

Run from the project root:
    python -m src.data.fetch_intraday
"""

import pandas as pd
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # TimeFrameUnit needed for custom intervals
from alpaca.data.enums import Adjustment, DataFeed

import config

# 30-minute bar interval — TimeFrame(30, TimeFrameUnit.Minute) is the correct
# way to specify a custom interval in alpaca-py; TimeFrame.Minute(30) doesn't exist.
_30MIN = TimeFrame(30, TimeFrameUnit.Minute)

# Canonical column set we promise to callers
_EXPECTED_COLS = ["open", "high", "low", "close", "volume", "vwap"]


def _make_client() -> StockHistoricalDataClient:
    """Mockable factory — swap this out in tests so we never hit Alpaca."""
    return StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


def fetch_intraday(client: StockHistoricalDataClient, ticker: str, date: str) -> pd.DataFrame:
    """Fetch 30-minute bars for *ticker* on *date* (YYYY-MM-DD string).

    Returns a DataFrame with columns [open, high, low, close, volume, vwap]
    and a DatetimeIndex in US/Eastern time.  Returns an *empty* DataFrame
    (same columns, no rows) if Alpaca returns nothing or raises.

    How the date window works
    -------------------------
    Alpaca's bar endpoint is UTC-based. To capture all bars on a US trading
    day we request [date 00:00 UTC, date+1 00:00 UTC).  The SDK converts the
    resulting timestamps to US/Eastern so pre-market / after-hours bars are
    also included (we can filter later if needed).
    """
    try:
        start_utc = f"{date}T00:00:00Z"
        next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end_utc = f"{next_day}T00:00:00Z"

        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=_30MIN,
            start=start_utc,
            end=end_utc,
            adjustment=Adjustment.ALL,  # consistent with daily bars
            feed=DataFeed.IEX,          # free tier; SIP requires paid subscription
        )

        bars = client.get_stock_bars(req)
        df = bars.df

        if df.empty:
            return _empty_df()

        # The SDK returns a MultiIndex (symbol, timestamp). Drop the symbol
        # level so the result is indexed by timestamp only — same pattern as
        # fetch_prices.fetch_symbol.
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        df.index.name = "timestamp"

        # Convert to US/Eastern so bar times are human-readable for US markets.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("US/Eastern")

        # Keep only the standard OHLCV + VWAP columns (drop trade_count etc.)
        present = [c for c in _EXPECTED_COLS if c in df.columns]
        df = df[present]
        return df

    except Exception as exc:
        # Never crash the caller — a missing/bad ticker just returns empty.
        print(f"  [warn] fetch_intraday({ticker}, {date}): {exc}")
        return _empty_df()


def _empty_df() -> pd.DataFrame:
    """Empty DataFrame with the correct schema."""
    return pd.DataFrame(columns=_EXPECTED_COLS)


def update_intraday(
    symbols: list[str] | None = None,
    target_date: str | None = None,
) -> None:
    """Fetch and persist 30-minute bars for every symbol on *target_date*.

    Parameters
    ----------
    symbols:
        Tickers to fetch. Defaults to config.FALLBACK_SYMBOLS (or the full
        universe loaded from config.UNIVERSE_FILE if you wire that up).
    target_date:
        Date string YYYY-MM-DD. Defaults to today.

    File layout:
        data/intraday/<TICKER>/<YYYY-MM-DD>.parquet

    Idempotent: if the parquet already exists the ticker is skipped — run it
    twice in a day and you won't burn API quota on reruns.
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    if symbols is None:
        symbols = config.FALLBACK_SYMBOLS

    client = _make_client()

    for ticker in symbols:
        ticker_dir = config.INTRADAY_DIR / ticker
        out_path = ticker_dir / f"{target_date}.parquet"

        if out_path.exists():
            print(f"  {ticker} {target_date}: already saved, skipping.")
            continue

        df = fetch_intraday(client, ticker, target_date)
        if df.empty:
            print(f"  {ticker} {target_date}: no data returned.")
            continue

        ticker_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        print(f"  {ticker} {target_date}: {len(df)} bars -> {out_path}")


def load_intraday(ticker: str, date: str) -> pd.DataFrame:
    """Load saved 30-minute bars from parquet.

    Parameters
    ----------
    ticker: e.g. "AAPL"
    date:   e.g. "2024-01-15"

    Raises FileNotFoundError if the parquet doesn't exist yet (call
    update_intraday first, or catch the error in the caller).
    """
    path = config.INTRADAY_DIR / ticker / f"{date}.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    update_intraday()
