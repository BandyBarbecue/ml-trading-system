"""
fetch_insider.py — Per-ticker insider transactions via yfinance.

Uses Yahoo Finance data (via yfinance) instead of raw EDGAR API calls.
yfinance provides clean, pre-parsed insider transaction data that is
fast and reliable for our 500-ticker universe.

Transaction codes (our internal schema):
  P = Open-market purchase (bullish signal)
  S = Open-market sale   (bearish signal)
  Others (awards, exercises, gifts) are excluded — not informative signals.

Executive trades (CEO/CFO/President) get a 2× weight because they carry
more information content than director trades.

yfinance column mapping (from holders.py in the yfinance source):
  df["Text"]       → transaction description ("Purchase", "Sale", etc.)
  df["Position"]   → insider role (CEO, CFO, Director, ...)
  df["Start Date"] → transaction date (Timestamp)
  df["Shares"]     → number of shares transacted
  df["Value"]      → total transaction value in USD
"""

import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

log = logging.getLogger(__name__)

EXECUTIVE_TITLES = {"ceo", "cfo", "president", "chief executive", "chief financial"}


# ---------------------------------------------------------------------------
# "Skip if already fetched today" guard
# ---------------------------------------------------------------------------

def _last_fetch_path() -> Path:
    return config.DATA_DIR / "insider_last_fetch.json"


def _already_fetched_today() -> bool:
    path = _last_fetch_path()
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text())
        return state.get("date") == date.today().isoformat()
    except (json.JSONDecodeError, OSError):
        return False


def _mark_fetched_today() -> None:
    path = _last_fetch_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": date.today().isoformat()}))


# ---------------------------------------------------------------------------
# Per-ticker fetch
# ---------------------------------------------------------------------------

def _fetch_ticker_insider(sym: str) -> list[dict]:
    """Fetch and parse insider transactions for one ticker via yfinance.

    yfinance returns all available historical insider transactions for the ticker.
    We filter for open-market purchases (P) and sales (S) — grants, exercises,
    and gifts carry no informational signal about management conviction.
    """
    try:
        df = yf.Ticker(sym).insider_transactions
    except Exception as exc:
        log.debug("yfinance insider failed for %s: %s", sym, exc)
        return []

    if df is None or df.empty:
        return []

    rows: list[dict] = []
    for _, row in df.iterrows():
        # "Text" (from transactionText) describes the transaction:
        # "Purchase", "Sale", "Option Exercise", "Gift", etc.
        text = str(row.get("Text", "")).lower()
        if "purchase" in text:
            txn_type = "P"
        elif "sale" in text:
            txn_type = "S"
        else:
            continue  # grant, exercise, gift — skip

        # "Position" (from filerRelation) is the insider's role at the company.
        position = str(row.get("Position", "")).lower()
        is_executive = any(t in position for t in EXECUTIVE_TITLES)
        role_weight = 2.0 if is_executive else 1.0
        insider_type = position if position else "unknown"

        txn_date = row.get("Start Date")
        if pd.isnull(txn_date):
            continue
        try:
            txn_date_str = pd.Timestamp(txn_date).strftime("%Y-%m-%d")
        except Exception:
            continue

        try:
            shares    = float(row.get("Shares", 0) or 0)
            value_usd = float(row.get("Value",  0) or 0)
        except (ValueError, TypeError):
            shares, value_usd = 0.0, 0.0

        rows.append({
            "ticker":           sym,
            "date":             txn_date_str,
            "insider_type":     insider_type,
            "transaction_type": txn_type,
            "shares":           shares    * role_weight,
            "value_usd":        value_usd * role_weight,
        })

    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def update_insider(
    symbols: list[str] | None = None,
) -> None:
    """Fetch insider transactions for all S&P 500 tickers via yfinance.

    Skips if already run today. Merges new data with the existing parquet
    and de-duplicates on (ticker, date, insider_type, transaction_type, shares).
    """
    if _already_fetched_today():
        log.info("insider: already fetched today — skipping.")
        return

    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    rows: list[dict] = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        rows.extend(_fetch_ticker_insider(sym))

        if (i + 1) % 50 == 0:
            log.info("insider: %d/%d tickers fetched (%d rows so far)", i + 1, n, len(rows))

        time.sleep(0.1)  # polite pacing to avoid Yahoo Finance throttling

    _mark_fetched_today()

    out_path: Path = config.INSIDER_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()

    if not rows:
        log.info("insider: no P/S transactions found for any ticker.")
        return

    new_df = pd.DataFrame(rows)
    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        .drop_duplicates(
            subset=["ticker", "date", "insider_type", "transaction_type", "shares"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined.to_parquet(out_path, index=False)
    log.info("insider: wrote %d rows to %s", len(combined), out_path)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
    )
    update_insider()
