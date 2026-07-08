"""Earnings fetcher: pulls EPS actuals vs estimates from Yahoo Finance (yfinance).

Why yfinance and not FMP?
    This module originally used FMP's /earnings-surprises endpoint. That
    endpoint was retired for accounts created after 2025-08-31 (it returns
    403 "Legacy Endpoint"), and FMP's free-tier replacement
    (stable/earnings-calendar) only serves a truncated ~3-month window with
    historical ranges paywalled — useless for SUE, which needs up to 8 past
    quarterly reports per ticker. Yahoo Finance provides ~6 years of EPS
    estimates AND actuals per ticker for free, with exact announcement
    timestamps. Those timestamps are a genuine upgrade: they let us derive
    the BMO/AMC report-time flag (FMP returned no time field at all), which
    activates the day-0 earnings signal in earnings_features.compute_sue().

All symbols are stored in a single file: EARNINGS_FILE (data/earnings/earnings.parquet).
Earnings arrive quarterly, so on subsequent runs we skip tickers whose most
recent stored report is younger than STALE_AFTER_DAYS — no point asking Yahoo
for a new quarterly number 3 weeks after the last one.

Run from the project root:
    python -m src.data.fetch_earnings
"""

import time

import pandas as pd
import yfinance as yf

# Import the *module* so tests can monkeypatch config.EARNINGS_FILE and our
# code sees the patched value at call time.  Using `from config import X`
# would bind a local copy at import time, which the monkeypatch can't affect.
import config

# How many past report dates to request per ticker. Yahoo interprets this
# loosely (a request for 12 often returns ~25 rows) — 28 comfortably covers
# the ~8 quarters SUE needs plus several years of backtest history.
HISTORY_LIMIT = 28

# Quarterly reports are ~91 days apart. If the newest stored report for a
# ticker is younger than this, the next report can't exist yet — skip the
# ticker entirely and save a Yahoo request. 75 gives a safety margin for
# companies that report early.
STALE_AFTER_DAYS = 75

# If this many tickers fail IN A ROW, assume a systemic problem (Yahoo
# throttling/ban, network down) and abort instead of hammering through the
# remaining hundreds of symbols. Isolated per-ticker failures (delisted,
# no coverage) just skip that ticker and keep going.
MAX_CONSECUTIVE_FAILURES = 10


def _classify_report_time(ts: pd.Timestamp) -> str:
    """Derive "bmo" / "amc" / "unknown" from an announcement timestamp.

    Why this matters: a Before-Market-Open (BMO) report is public ~7 hours
    before that day's 16:00 ET close, so it is legitimate information for
    day-T features.  An After-Market-Close (AMC) report is NOT public until
    ~30 minutes after the close, so using it on day T would be look-ahead
    bias.  When we can't tell, we store "unknown" and the feature code
    treats it like AMC (the conservative choice).

    Yahoo timestamps are US/Eastern (e.g. 06:00 = pre-market call, 16:30 =
    after the close). Rules:
      - exactly midnight → Yahoo had no time info, only a date → "unknown"
      - before 09:30 ET  → "bmo"
      - 16:00 ET or later → "amc"
      - during market hours (rare data quirk, e.g. 13:00) → "unknown"
    """
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    if ts.hour == 0 and ts.minute == 0:
        return "unknown"
    if (ts.hour, ts.minute) < (9, 30):
        return "bmo"
    if ts.hour >= 16:
        return "amc"
    return "unknown"


def _fetch_symbol_earnings(ticker: str) -> pd.DataFrame:
    """Fetch the earnings-date history for one ticker from Yahoo.

    Returns yfinance's frame: index = tz-aware "Earnings Date", columns
    include "EPS Estimate" and "Reported EPS" (NaN for future dates).

    Yahoo spells share classes with dashes ("BRK-B") where our universe file
    uses dots ("BRK.B") — translate for the request only; the parquet keeps
    our spelling so joins against the price data still work.
    """
    yahoo_symbol = ticker.replace(".", "-")
    frame = yf.Ticker(yahoo_symbol).get_earnings_dates(limit=HISTORY_LIMIT)
    return frame if frame is not None else pd.DataFrame()


def _rows_from_earnings_frame(ticker: str, frame: pd.DataFrame) -> list[dict]:
    """Convert one ticker's yfinance earnings frame into parquet-schema rows.

    Drops rows that are unusable:
      - future/unreported quarters (Reported EPS is NaN)
      - missing estimates (can't compute a surprise without one)
      - zero estimates (division-by-zero guard in the surprise calculation)
    """
    rows: list[dict] = []
    if frame.empty:
        return rows

    for ts, row in frame.iterrows():
        actual = row.get("Reported EPS")
        estimated = row.get("EPS Estimate")
        if pd.isna(actual) or pd.isna(estimated) or estimated == 0:
            continue

        rows.append({
            "ticker": ticker,
            # Store the calendar DATE only (tz-naive midnight): the feature
            # pipeline's daily index is tz-naive, and the intraday timing
            # information is captured separately in report_time.
            "date": pd.Timestamp(ts.date()),
            "actual_eps": float(actual),
            "estimated_eps": float(estimated),
            # surprise_pct = (actual − estimate) / |estimate|
            # Positive → beat, negative → miss.  Dividing by the absolute value
            # handles negative estimates correctly (e.g. a loss smaller than
            # expected is still a "beat").
            "surprise_pct": (float(actual) - float(estimated)) / abs(float(estimated)),
            # When the report was released relative to the trading session.
            # BMO reports may be used on day T; AMC/unknown only from T+1.
            "report_time": _classify_report_time(ts),
        })
    return rows


def update_earnings(symbols: list[str] | None = None) -> None:
    """Fetch earnings surprise data for all symbols and append to the parquet file.

    Args:
        symbols: Tickers to fetch.  Defaults to the S&P 500 universe from
                 ``sp500_universe.get_symbols()``.

    Skips tickers whose most recent stored report is younger than
    STALE_AFTER_DAYS (their next quarterly report can't exist yet).
    The first ever run therefore backfills all ~500 tickers (slow, minutes);
    daily runs only touch the handful of tickers due to report.

    Output:
        config.EARNINGS_FILE — columns: ticker, date, actual_eps,
        estimated_eps, surprise_pct, report_time ("bmo"/"amc"/"unknown").
    """
    from src.data.sp500_universe import get_symbols

    # Read config.EARNINGS_FILE at call time (not import time) so that
    # monkeypatch.setattr(config, "EARNINGS_FILE", ...) takes effect in tests.
    earnings_file = config.EARNINGS_FILE
    earnings_file.parent.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = get_symbols()

    # Load what we already have so we can skip tickers that can't have news.
    if earnings_file.exists():
        existing = pd.read_parquet(earnings_file)
        latest_stored = existing.groupby("ticker")["date"].max()
    else:
        existing = pd.DataFrame()
        latest_stored = pd.Series(dtype="datetime64[ns]")

    freshness_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=STALE_AFTER_DAYS)

    rows: list[dict] = []
    consecutive_failures = 0
    fetched = skipped = 0

    for ticker in symbols:
        if ticker in latest_stored.index and latest_stored[ticker] > freshness_cutoff:
            # Newest stored report is recent — the next quarterly one isn't
            # due yet, so a Yahoo request would return nothing new.
            skipped += 1
            continue

        try:
            frame = _fetch_symbol_earnings(ticker)
            consecutive_failures = 0
        except Exception as exc:
            # One ticker failing (delisted, no Yahoo coverage) shouldn't kill
            # the whole update — but a long unbroken failure streak means
            # Yahoo is refusing us and continuing would only make it worse.
            consecutive_failures += 1
            err_type = type(exc).__name__
            print(f"[warn] Earnings fetch failed for {ticker}: {err_type}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[warn] {MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                    "aborting earnings update (likely rate-limited)."
                )
                break
            continue

        rows.extend(_rows_from_earnings_frame(ticker, frame))
        fetched += 1
        # Be a polite Yahoo client: a short pause between requests keeps a
        # full 500-ticker backfill from looking like a scraping burst.
        time.sleep(0.1)

    if not rows:
        # Nothing new fetched — leave the file as-is (or don't create it)
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Legacy rows written before report_time existed come out of the concat
    # as NaN in that column — backfill them to "unknown" so the parquet is
    # always a clean string column with only three possible values.
    combined["report_time"] = combined["report_time"].fillna("unknown")

    # Deduplicate on (ticker, date) — keep the most recent version of any row
    # (a re-fetch may carry a corrected EPS figure).
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined.to_parquet(earnings_file, index=False)
    # ASCII only: this print runs under Task Scheduler's cp1252 console, where
    # a "→" raises UnicodeEncodeError AFTER the parquet is already saved —
    # making a successful run exit 1.
    print(
        f"Earnings: {len(combined)} records for {combined['ticker'].nunique()} tickers "
        f"({fetched} fetched, {skipped} fresh-skipped) -> {earnings_file.name}"
    )


def load_earnings() -> pd.DataFrame:
    """Load the full earnings history from parquet.

    Returns an empty DataFrame if the file doesn't exist yet (avoids callers
    having to check existence themselves).
    """
    earnings_file = config.EARNINGS_FILE
    return pd.read_parquet(earnings_file) if earnings_file.exists() else pd.DataFrame()


if __name__ == "__main__":
    update_earnings()
