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

import json
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

# daily_run.py's Step 5 (portfolio construction) only reruns on Mondays, so a
# target_weights.json file up to about a week old is still this week's live
# rebalance. Anything older than this means a rebalance was probably missed
# (e.g. daily_run.py has been failing) — trusting HRP sizing that stale is
# riskier than just falling back to equal-weight, so we don't.
_TARGET_WEIGHTS_MAX_AGE_DAYS = 7

# The Task Scheduler trigger fires a few minutes before the 09:30 ET bell so
# the runner is ready for the opening bar. But if the trigger misfires hours
# early (e.g. the PC wakes at 3 AM), sleeping until open would hold run.lock
# all night — so we only wait if the open is at most this many seconds away.
_MAX_PREOPEN_WAIT_SECS = 2 * 60 * 60

# If run.lock is held at startup, wait and retry instead of giving up. This
# happens when the PC was asleep at both trigger times and Task Scheduler's
# StartWhenAvailable fires the daily run and this runner at the same moment
# (2026-07-10: daily grabbed the lock 1 second earlier and the intraday layer
# never ran all day). The daily run normally finishes in ~6 minutes, longer
# on retrain days — 45 minutes covers that with room to spare while still
# giving up on a genuinely wedged lock before the session is worthless.
_LOCK_RETRY_SECS    = 60
_LOCK_WAIT_MAX_SECS = 45 * 60


def _now_et() -> datetime:
    """Return the current datetime in US/Eastern time.

    Extracted into its own function so tests can monkeypatch it without
    having to mock the entire datetime class (which breaks constructors).
    """
    return datetime.now(tz=_ET)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _seconds_until_open(now: datetime) -> float:
    """Seconds from *now* until today's 09:30 ET open.

    Positive → the bell hasn't rung yet; zero or negative → already open
    (or past). *now* must be an ET datetime (see _now_et).
    """
    open_dt = now.replace(
        hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MINUTE,
        second=0, microsecond=0,
    )
    return (open_dt - now).total_seconds()


def _effective_close_hm(now: datetime) -> tuple[int, int]:
    """Return (hour, minute) for today's "stop trading" time.

    Normally this is 10 minutes before the actual NYSE close (16:00 ET ->
    15:50), which is exactly what config.INTRADAY_CLOSE_TIME hardcodes. On
    an early-close day (e.g. the day after Thanksgiving, actual close
    13:00 ET) it becomes 10 minutes before THAT bell (12:50) instead —
    otherwise the runner would keep polling for hours after the market has
    already shut.

    The calendar lookup is wrapped in try/except because this runs on every
    polling loop iteration: a calendar bug or bad install must never crash
    the intraday runner mid-session — it just falls back to the static
    config value, same as before this gate existed.
    """
    try:
        from src.data.market_calendar import market_close_et
        close = market_close_et(now.date())
        if close is not None:
            # "10 minutes before" via minute arithmetic — avoids pulling in
            # timedelta just for this one subtraction.
            total_minutes = close.hour * 60 + close.minute - 10
            return divmod(total_minutes, 60)
    except Exception as exc:
        logger.warning(
            "market_close_et lookup failed (%s) — falling back to "
            "config.INTRADAY_CLOSE_TIME.", exc,
        )

    return tuple(map(int, config.INTRADAY_CLOSE_TIME.split(":")))


def _is_market_hours(now: datetime) -> bool:
    """True if *now* is between 09:30 and the effective close US/Eastern
    (inclusive start, exclusive end — see _effective_close_hm).

    *now* must already be timezone-aware.  If it is naive (no tzinfo), we
    assume it is already in ET.

    Think of this as the "are the doors open?" check before each patrol round.
    """
    close_h, close_m = _effective_close_hm(now)

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


def _is_final_bar(now: datetime) -> bool:
    """True if the NEXT 30-minute boundary is at or past today's effective
    close time — i.e. this is the last bar that will be processed today.

    Bars are 30-min aligned but the effective close (15:50 normally, 12:50 on
    an early-close day — see _effective_close_hm) usually isn't, so we can't
    just compare `now` to the close time directly: the loop's outer
    `_is_market_hours` check already guarantees `now < close_time`, meaning
    "now == close time" never happens. Instead we look ahead to the next
    scheduled bar boundary — if that boundary would be at/after close, this
    bar is the final one, and this is where the "eod" signal should actually
    liquidate (see run_bar).
    """
    close_h, close_m = _effective_close_hm(now)
    next_bar = _next_bar_time(now)
    return (next_bar.hour, next_bar.minute) >= (close_h, close_m)


# ---------------------------------------------------------------------------
# HRP target weights (persisted by daily_run.py Step 5)
# ---------------------------------------------------------------------------

def _load_target_weights(now: datetime | None = None) -> dict[str, float]:
    """Load Step 5's HRP-sized target weights from data/target_weights.json.

    Without this, run_bar previously sized every "full" entry as a naive
    1/N equal weight, throwing away all the HRP risk-parity sizing daily_run
    computed on Monday. Falls back to an empty dict — meaning callers use
    per-ticker equal-weight instead — if the file is missing, corrupt, or
    older than _TARGET_WEIGHTS_MAX_AGE_DAYS.

    Returns
    -------
    dict[ticker, weight] — empty dict on any failure (logged as a warning).
    """
    path = config.DATA_DIR / "target_weights.json"
    if not path.exists():
        logger.warning(
            "target_weights.json not found — falling back to equal-weight sizing."
        )
        return {}

    try:
        payload = json.loads(path.read_text())
        file_date = datetime.strptime(payload["date"], "%Y-%m-%d").date()
        weights = {k: float(v) for k, v in payload.get("weights", {}).items()}
    except Exception as exc:
        logger.warning(
            "target_weights.json unreadable (%s) — falling back to equal-weight sizing.",
            exc,
        )
        return {}

    today_date = (now or _now_et()).date()
    age_days = (today_date - file_date).days
    if age_days > _TARGET_WEIGHTS_MAX_AGE_DAYS:
        logger.warning(
            "target_weights.json is %d day(s) old (> %d) — falling back to "
            "equal-weight sizing.", age_days, _TARGET_WEIGHTS_MAX_AGE_DAYS,
        )
        return {}

    return weights


# ---------------------------------------------------------------------------
# Single-bar processor
# ---------------------------------------------------------------------------

def run_bar(
    bar_idx: int,
    held_tickers: list[str],
    news_signals: list[dict],
    dry_run: bool = False,
    is_final_bar: bool = False,
) -> list[dict]:
    """Process one 30-minute bar across all held tickers.

    For each ticker currently held in the portfolio:
      1. Fetch intraday 30-min bars for today.
      2. Get VWAP entry signal (full / half / eod).
      3. Translate the signal into a target weight, scaled toward the HRP
         weight daily_run.py's Step 5 computed for this ticker (falling back
         to equal-weight 1/N per ticker if target_weights.json is missing or
         stale — see _load_target_weights):
           "full" → the full HRP target weight
           "half" → half of the HRP target weight
           "eod"  → on the FINAL bar of the day: close out (weight = 0).
                    On any earlier bar: hold — no trade this bar at all
                    (see is_final_bar below for why).
      4. Compute the weight delta vs. current holding and queue a rebalance
         order if the delta is non-zero.
      5. Independently check for Claude NLP event signals on the ticker;
         if one qualifies (confidence ≥ 0.85, magnitude "large"), add an
         event trade on top.

    Parameters
    ----------
    bar_idx:       Zero-based bar number (0 = market open bar).
    held_tickers:  List of ticker symbols currently held in the portfolio.
    news_signals:  List of pre-extracted NLP signal dicts (from extract_signals).
    dry_run:       If True, skip all order submission and return [].
    is_final_bar:  True only for the last 30-min bar of the trading day (see
                   _is_final_bar). Controls whether "eod" actually liquidates
                   (final bar) or just means "hold, don't trade yet" (every
                   other bar) — mapping "eod" to an immediate weight-0 target
                   on EVERY bar caused a sell/rebuy whipsaw every 30 minutes.

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

    # Equal-weight fallback — used per-ticker only when target_weights.json
    # doesn't have an entry for that ticker (missing/stale file, or a ticker
    # that isn't part of the current HRP portfolio yet).
    equal_weight = 1.0 / n
    hrp_targets  = _load_target_weights()

    # ------------------------------------------------------------------
    # Build target weights from VWAP entry signals
    # ------------------------------------------------------------------
    data_client = _make_data_client()

    target_weights: dict[str, float] = {}

    for ticker in held_tickers:
        bars = fetch_intraday(data_client, ticker, today)
        signal = get_entry_signal(bars, bar_idx)
        hrp_target = hrp_targets.get(ticker, equal_weight)

        if signal == "full":
            target_weights[ticker] = hrp_target
        elif signal == "half":
            target_weights[ticker] = 0.5 * hrp_target
        elif signal == "eod":
            if is_final_bar:
                # The actual end of the trading day — liquidate.
                target_weights[ticker] = 0.0
            else:
                # Not the final bar: leave the position unchanged. Skipping
                # this ticker entirely (not adding it to target_weights)
                # means no delta is computed for it below, so no order is
                # submitted this bar — the whole point is to avoid selling
                # now only to rebuy 30 minutes later.
                continue
        else:
            # "skip" or unknown — treat as full for now
            target_weights[ticker] = hrp_target

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

    Before doing anything else, this checks two restart-safety guards:
      - risk_state.is_halted(today): if the kill switch fired earlier today
        (in this run or a crashed prior run), refuse to trade again today.
      - paper_trader.acquire_run_lock(): refuse to start if another run
        (e.g. an overlapping Task Scheduler fire) already holds the lock.

    Parameters
    ----------
    dry_run: If True, all steps run but no orders are submitted to Alpaca.
    date:    Override for "today" (YYYY-MM-DD string).  Default: actual today.
             Useful when replaying a historical session in tests.
    """
    from src.execution.paper_trader import (
        _make_trading_client,
        get_current_positions,
        acquire_run_lock,
        release_run_lock,
    )
    from src.execution.risk_state  import is_halted
    from src.features.nlp_features import extract_signals

    logger.info(
        "=== Intraday runner starting (dry_run=%s, date=%s) ===",
        dry_run, date or "today",
    )

    # Determine the trading date string used for file loading.
    # run_intraday uses this only for news loading; actual "now" always comes
    # from datetime.now() so market-hour checks are accurate.
    today_str = date or _now_et().strftime("%Y-%m-%d")

    # --- Restart-safe halt check -----------------------------------------
    if is_halted(today_str):
        logger.error(
            "Trading halted for %s (kill switch fired earlier today) — "
            "refusing to run the intraday loop.", today_str,
        )
        return

    # --- NYSE calendar gate ------------------------------------------------
    # Weekends and holidays have no bars to poll. Checked before the run
    # lock so a Task Scheduler misfire on a holiday exits without even
    # touching the lock file or the network.
    from src.data.market_calendar import is_trading_day
    if not is_trading_day(today_str):
        logger.info("Market closed today (weekend/holiday) — nothing to do.")
        return

    # --- Run lock ----------------------------------------------------------
    # Wait-and-retry rather than exit: a held lock at startup almost always
    # means the daily run is mid-flight (catch-up fires launch both tasks at
    # the same instant), and it will release the lock in a few minutes.
    lock_deadline = time.monotonic() + _LOCK_WAIT_MAX_SECS
    while not acquire_run_lock():
        if time.monotonic() >= lock_deadline:
            logger.error(
                "run.lock still held after waiting %.0f minutes — giving up "
                "on today's intraday session.", _LOCK_WAIT_MAX_SECS / 60,
            )
            return
        logger.info(
            "run.lock is held (daily run probably in progress) — retrying "
            "in %d s.", _LOCK_RETRY_SECS,
        )
        time.sleep(_LOCK_RETRY_SECS)

    try:
        # --- Wait for the opening bell (#21) --------------------------------
        # The scheduled task starts a few minutes before 09:30 ET. Without
        # this wait, the while-loop's market-hours check was False on the
        # very first pass and the runner exited silently with 0 bars — the
        # intraday layer never ran at all. Waiting here (inside the run
        # lock) also means an overlapping scheduler fire during the wait
        # bounces off the lock instead of starting a duplicate runner.
        wait_secs = _seconds_until_open(_now_et())
        if 0 < wait_secs <= _MAX_PREOPEN_WAIT_SECS:
            logger.info(
                "Market opens in %.0f s — sleeping until the 09:30 ET bell.",
                wait_secs,
            )
            time.sleep(wait_secs)

        bar_idx = 0

        while _is_market_hours(_now_et()):
            now = _now_et()
            final_bar = _is_final_bar(now)
            logger.info(
                "--- Bar %d at %s%s ---",
                bar_idx, now.strftime("%H:%M:%S ET"),
                " (final bar)" if final_bar else "",
            )

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
                    is_final_bar = final_bar,
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
    finally:
        release_run_lock()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    log_dir = config.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"intraday_{_now_et().strftime('%Y-%m-%d')}.log"

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
