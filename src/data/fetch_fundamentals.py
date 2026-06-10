"""
fetch_fundamentals.py — Pull gross profitability and P/B ratio via yfinance.

FMP's v3 key-metrics endpoint was retired for new accounts after August 2025.
yfinance (Yahoo Finance) is free and already a project dependency.

Metrics fetched from yf.Ticker(ticker).info per ticker:
  priceToBook   → pb_ratio
  grossMargins  → gross_profitability
    (grossMargins = gross profit / revenue — a close proxy for Novy-Marx
     gross profitability. The exact Novy-Marx definition uses total assets
     as the denominator, but that requires a separate balance-sheet API call
     per ticker which would make the 500-ticker update ~10× slower.)

45-day lag rule: enforced in fundamental_features.py, not here. This module
just stores today's snapshot with today's date. The feature function picks
the most recent snapshot that predates the query date by 45+ days.
"""

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

log = logging.getLogger(__name__)

_SLEEP_BETWEEN = 0.05  # seconds between tickers — polite to Yahoo Finance


def update_fundamentals(symbols: list[str] | None = None) -> None:
    """Fetch P/B ratio and gross margins for each symbol and save to parquet.

    Skips tickers where Yahoo Finance returns no data (common for recent IPOs).
    Progress is logged every 50 tickers.

    Args:
        symbols: List of ticker strings. Defaults to S&P 500 universe.
    """
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    today = date.today().isoformat()
    records = []

    for i, ticker in enumerate(symbols):
        try:
            info = yf.Ticker(ticker).info
            pb = info.get("priceToBook")
            gm = info.get("grossMargins")
            records.append({
                "ticker":              ticker,
                "date":                today,
                "gross_profitability": float(gm) if gm is not None else None,
                "pb_ratio":            float(pb) if pb is not None else None,
            })
        except Exception as exc:
            log.warning("fundamentals: skipping %s — %s", ticker, exc)

        if i > 0 and i % 50 == 0:
            log.info("fundamentals: %d/%d tickers fetched", i, len(symbols))

        time.sleep(_SLEEP_BETWEEN)

    if not records:
        log.warning("fundamentals: no data fetched — Yahoo Finance may be unavailable.")
        return

    new_df = pd.DataFrame(records)

    out_path: Path = config.FUNDAMENTALS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined.to_parquet(out_path, index=False)
    log.info("fundamentals: saved %d rows to %s", len(combined), out_path)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)-8s — %(message)s")
    from src.data.sp500_universe import get_symbols
    update_fundamentals(get_symbols())
