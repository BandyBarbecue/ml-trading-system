"""intraday_runner.py — 30-minute bar polling loop for intraday execution.

Runs from market open (09:30 ET) to INTRADAY_CLOSE_TIME (15:50 ET) each
trading day.  Every 30 minutes it:
  1. Gets current portfolio positions from Alpaca
  2. Fetches the latest 30-min bars for each held ticker
  3. Computes VWAP entry signals — adjusts position size if "half" or
     fires full entry
  4. Checks Claude NLP news signals for event-driven trades
     (confidence > 0.85, magnitude "large")
  5. Submits any triggered orders to Alpaca paper API
  6. Sleeps until the next 30-minute bar

Analogy: think of this as a security guard making rounds every 30 minutes.
On each round they check whether anything needs action (entry, resize, exit)
and act immediately, then go back to sleep until the next scheduled check.

Paper trading only — no live capital, ever.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Eastern timezone object — reused throughout so we don't rebuild it each call.
_ET = ZoneInfo("America/New_York")

# Market open time components (hour, minute).
_MARKET_OPEN_HOUR   = 9
_MARKET_OPEN_MINUTE = 30


def _now_et() -> datetime:
    """Return the current datetime in US/Eastern time.

    Extracted into its own function so tests can monkeypatch it without
    having to mock the entire datetime class (which breaks constructors).
    """
    return datetime.now(tz=_ET)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _is_market_hours(now: datetime) -> bool:
    """True if *now* is between 09:30 and 15:50 US/Eastern (inclusive start).

    *now* must already be timezone-aware.  If it is naive (no tzinfo), we
    assume it is already in ET.

    Think of this as the "are the doors open?" check before each patrol round.
    """
    # Parse the close time from config ("15:50" → hour=15, minute=50).
    close_h, close_m = map(int, config.INTRADAY_CLOSE_TIME.split(":"))

    # Work entirely in (hour, minute) to avoid date arithmetic.
    hm = (now.hour, now.minute)
    return (
        (_MARKET_OPEN_HOUR, _MARKET_OPEN_MINUTE) <= hm < (close_h, close_m)
    )


def _next_bar_time(now: datetime) -> datetime:
    """Return the next 30-minute bar boundary in US/Eastern time.

    Examples:
        10:47 → 11:00
        11:00 → 11:30   (if now is exactly on a boundary, advance to next)
        11:30 → 12:00

    The logic works by looking at the current minute offset within the hour:
      - If on the boundary (minute == 0 or 30), advance by 30 minutes.
      - Otherwise, round up to the next multiple-of-30 minute.

    *now* is treated as already being in the correct timezone; no conversion
    is applied here.
    """
    minutes = now.minute

    if minutes == 0:
        # Exactly on the hour → next boundary is :30 of the same hour.
        return now.replace(minute=30, second=0, microsecond=0)

    # next_minutes is the smallest multiple of 30 that is > current minutes.
    next_minutes = 30 * ((minutes // 30) + 1)

    if next_minutes >= 60:
        # e.g. minutes=45 → next_minutes=60 → wrap to next hour :00
        return now.replace(
            hour=now.hour + 1, minute=0, second=0, microsecond=0
        )

    return now.replace(minute=next_minutes, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Single-bar processor
# ---------------------------------------------------------------------------

def run_bar(
    bar_idx: int,
    held_tickers: list[str],
    news_signals: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """Process one 30-minute bar across all held tickers.

    For each ticker currently held in the portfolio:
      1. Fetch intraday 30-min bars for today.
      2. Get VWAP entry signal (full / half / eod).
      3. Translate the signal into a target weight:
           "full" → equal-weight  (1 / N positions)
           "half" → half weight   (0.5 / N positions)
           "eod"  → close out     (weight = 0)
      4. Compute the weight delta vs. current equal-weight holding and
         queue a rebalance order if the delta is non-zero.
      5. Independently check for Claude NLP event signals on the ticker;
         if one qualifies (confidence ≥ 0.85, magnitude "large"), add an
         event trade on top.

    Parameters
    ----------
    bar_idx:       Zero-based bar number (0 = market open bar).
    held_tickers:  List of ticker symbols currently held in the portfolio.
    news_signals:  List of pre-extracted NLP signal dicts (from extract_signals).
    dry_run:       If True, skip all order submission and return [].

    Returns
    -------
    List of order result dicts from submit_rebalance (empty on dry_run or
    if no orders were triggered).
    """
    # Lazy imports keep the module mockable in tests — patches on the module
    # names below will intercept calls before any Alpaca traffic occurs.
    from src.data.fetch_intraday import fetch_intraday, _make_client as _make_data_client
    from src.execution.intraday_signal import get_entry_signal, get_event_signal
    from src.execution.paper_trader import (
        _make_trading_client,
        get_current_positions,
        submit_rebalance,
        get_account_value,
    )

    if not held_tickers:
        logger.debug("run_bar %d: no held tickers, nothing to do.", bar_idx)
        return []

    today = _now_et().strftime("%Y-%m-%d")
    n     = len(held_tickers)

    # Equal-weight baseline: every held ticker targets 1/N of portfolio value.
    equal_weight = 1.0 / n

    # ------------------------------------------------------------------
    # Build target weights from VWAP entry signals
    # ------------------------------------------------------------------
    data_client = _make_data_client()

    target_weights: dict[str, float] = {}

    for ticker in held_tickers:
        bars = fetch_intraday(data_client, ticker, today)
        signal = get_entry_signal(bars, bar_idx)

        if signal == "full":
            target_weights[ticker] = equal_weight
        elif signal == "half":
            target_weights[ticker] = 0.5 / n
        elif signal == "eod":
            target_weights[ticker] = 0.0
        else:
            # "skip" or unknown — treat as full for now
            target_weights[ticker] = equal_weight

    # ------------------------------------------------------------------
    # Determine weight deltas vs. current positions
    # ------------------------------------------------------------------
    # We need the trading client to fetch current positions and submit orders.
    trading_client = _make_trading_client()
    current_weights = get_current_positions(trading_client)
    account_value   = get_account_value(trading_client)

    deltas: dict[str, float] = {}
    for ticker, target in target_weights.items():
        current = float(current_weights.get(ticker, 0.0))
        delta   = target - current
        if abs(delta) > 1e-6:
            deltas[ticker] = delta

    # ------------------------------------------------------------------
    # Event-driven trades (independent of VWAP signal)
    # ------------------------------------------------------------------
    for ticker in held_tickers:
        event = get_event_signal(news_signals, ticker)
        if event is not None:
            # direction "long" → positive weight delta; "short" → negative
            direction = event.get("direction", "long")
            size      = event.get("size", config.EVENT_POSITION_MAX)
            signed    = size if direction == "long" else -size
            # Accumulate — event trade is additive on top of any VWAP delta.
            deltas[ticker] = deltas.get(ticker, 0.0) + signed

    if dry_run or not deltas:
        if dry_run:
            logger.debug("run_bar %d: dry_run=True, skipping order submission.", bar_idx)
        else:
            logger.debug("run_bar %d: no deltas to trade.", bar_idx)
        return []

    # ------------------------------------------------------------------
    # Submit orders
    # ------------------------------------------------------------------
    delta_series = pd.Series(deltas)
    orders = submit_rebalance(trading_client, delta_series, account_value)

    logger.info(
        "run_bar %d: submitted %d order(s) for tickers: %s",
        bar_idx, len(orders), list(deltas.keys()),
    )
    return orders


# ---------------------------------------------------------------------------
# Main intraday loop
# ---------------------------------------------------------------------------

def run_intraday(dry_run: bool = False, date: str | None = None) -> None:
    """Main intraday loop.  Runs from market open to INTRADAY_CLOSE_TIME.

    On each 30-minute bar:
      1. Get held tickers from Alpaca (current positions).
      2. Load today's news signals from data/news/YYYY-MM-DD.parquet.
      3. Extract NLP signals with extract_signals().
      4. Call run_bar() with the current bar index and signals.
      5. Sleep until the next 30-minute boundary.
      6. Stop when time >= config.INTRADAY_CLOSE_TIME ("15:50").

    Parameters
    ----------
    dry_run: If True, all steps run but no orders are submitted to Alpaca.
    date:    Override for "today" (YYYY-MM-DD string).  Default: actual today.
             Useful when replaying a historical session in tests.
    """
    from src.execution.paper_trader import _make_trading_client, get_current_positions
    from src.features.nlp_features  import extract_signals

    logger.info(
        "=== Intraday runner starting (dry_run=%s, date=%s) ===",
        dry_run, date or "today",
    )

    # Determine the trading date string used for file loading.
    # run_intraday uses this only for news loading; actual "now" always comes
    # from datetime.now() so market-hour checks are accurate.
    today_str = date or _now_et().strftime("%Y-%m-%d")

    bar_idx = 0

    while _is_market_hours(_now_et()):
        now = _now_et()
        logger.info("--- Bar %d at %s ---", bar_idx, now.strftime("%H:%M:%S ET"))

        # Step 1: current holdings
        held_tickers: list[str] = []
        try:
            trading_client  = _make_trading_client()
            current_weights = get_current_positions(trading_client)
            held_tickers    = list(current_weights.index)
        except Exception as exc:
            logger.warning("Could not fetch positions: %s — using empty list.", exc)

        # Step 2: load today's raw headlines parquet
        news_df = pd.DataFrame()
        try:
            news_path = config.NEWS_DIR / f"{today_str}.parquet"
            news_df   = pd.read_parquet(news_path)
        except Exception as exc:
            logger.debug(
                "Could not load news parquet for %s: %s — proceeding with no news.",
                today_str, exc,
            )

        # Step 3: extract NLP signals from raw headlines
        # extract_signals expects a DataFrame with a "headline" column.
        # If news_df is empty or missing the column, we pass an empty list.
        news_signals: list[dict] = []
        if not news_df.empty and "headline" in news_df.columns:
            try:
                news_signals = extract_signals(news_df)
            except Exception as exc:
                logger.warning("extract_signals failed: %s — no NLP signals.", exc)

        # Step 4: process this bar
        try:
            run_bar(
                bar_idx      = bar_idx,
                held_tickers = held_tickers,
                news_signals = news_signals,
                dry_run      = dry_run,
            )
        except Exception as exc:
            logger.error("run_bar %d failed: %s", bar_idx, exc)

        bar_idx += 1

        # Step 5: sleep until the next 30-minute boundary
        now        = _now_et()
        next_bar   = _next_bar_time(now)
        sleep_secs = max(0.0, (next_bar - now).total_seconds())

        logger.info(
            "Sleeping %.0f s until next bar at %s.",
            sleep_secs, next_bar.strftime("%H:%M ET"),
        )
        time.sleep(sleep_secs)

    logger.info(
        "=== Intraday runner finished after %d bar(s). ===", bar_idx
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    log_dir = config.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"intraday_{datetime.now().strftime('%Y-%m-%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="Run the intraday 30-minute bar loop.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps but skip actual order submission.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override today's date for news loading (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    run_intraday(dry_run=args.dry_run, date=args.date)
