"""Daily bar fetcher for up to 500 symbols.

Writes one parquet file per symbol into PRICES_DIR (data/prices/).
On subsequent runs it only fetches the bars that are missing — no duplicates.

Run from the project root:
    python -m src.data.fetch_prices
"""

import pandas as pd
from datetime import timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

from config import PRICES_DIR, START_DATE, END_DATE, ALPACA_API_KEY, ALPACA_SECRET_KEY


def _make_client() -> StockHistoricalDataClient:
    """Build the Alpaca data client from config credentials."""
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_symbol(
    client: StockHistoricalDataClient,
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Fetch adjusted daily bars for one symbol.

    The Alpaca SDK returns a DataFrame with a MultiIndex (symbol, timestamp).
    We drop the symbol level so the result is indexed by timestamp only —
    simpler to work with downstream.

    Returns:
        - An *empty* DataFrame when Alpaca legitimately has nothing new
          (e.g. we're already caught up to today).
        - None when the fetch itself failed (network error, API outage, bad
          symbol). This is deliberately distinguishable from "empty" — an
          empty DataFrame used to double as "failed", which meant an outage
          looked identical to "already up to date" and update_prices() would
          silently stop updating without telling anyone.
        Never raises — a bad symbol never aborts the whole batch.
    """
    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,  # split + dividend adjusted
            feed=DataFeed.IEX,          # free tier; SIP requires paid subscription
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return df
        # bars.df is multi-indexed (symbol, timestamp) — drop the symbol level
        # so callers just see a plain timestamp index.
        df = df.droplevel(0)
        df.index.name = "timestamp"
        return df
    except Exception as exc:
        print(f"  [warn] {ticker}: {exc}")
        return None


def update_prices(symbols: list[str] | None = None) -> None:
    """Incrementally update parquet files for every symbol in the list.

    Logic (think of it like a git pull):
    - If no file exists yet, fetch from START_DATE.
    - If the file exists, look at the last date stored and fetch from the
      next calendar day onwards — that way we never re-download old bars.
    - After fetching, concatenate old + new, deduplicate (keep the newest
      version of any duplicated row), sort, and write back to parquet.

    Fetch failures (network error, API outage) are tracked separately from
    "legitimately no new data" and summarised in a warning line at the end,
    so an outage is visible instead of silently looking like nothing changed.
    """
    from src.data.sp500_universe import get_symbols

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    client = _make_client()

    if symbols is None:
        symbols = get_symbols()

    failed_tickers: list[str] = []

    for ticker in symbols:
        path = PRICES_DIR / f"{ticker}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            # Resume from the day after the last stored bar
            last_date = existing.index.max()
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = None
            start = START_DATE

        new_data = fetch_symbol(client, ticker, start, END_DATE)
        if new_data is None:
            # The fetch itself failed — remember it so we can warn below,
            # rather than silently treating an outage as "up to date".
            failed_tickers.append(ticker)
            continue
        if new_data.empty:
            # Nothing new to add — file (if it existed) is already up to date
            continue

        if existing is not None:
            combined = pd.concat([existing, new_data])
        else:
            combined = new_data

        # Guard against any accidental overlap (e.g. mock returning stale dates).
        # keep="last" means the freshly fetched version wins on a tie.
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.to_parquet(path)
        print(f"{ticker}: {len(combined)} rows")

    if failed_tickers:
        preview = ", ".join(failed_tickers[:5])
        more = f" (+{len(failed_tickers) - 5} more)" if len(failed_tickers) > 5 else ""
        print(f"[warn] {len(failed_tickers)} ticker(s) failed to fetch: {preview}{more}")


def load_prices(ticker: str) -> pd.DataFrame:
    """Load the full daily price history for one symbol from parquet."""
    return pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")


if __name__ == "__main__":
    update_prices()
