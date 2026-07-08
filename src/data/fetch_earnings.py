"""Earnings fetcher: pulls EPS actuals vs estimates from FMP and writes parquet.

All symbols are stored in a single file: EARNINGS_FILE (data/earnings/earnings.parquet).
On subsequent runs, tickers that are already in the file are skipped so we don't
burn API calls re-fetching data we already have.

Run from the project root:
    python -m src.data.fetch_earnings
"""

import requests
import pandas as pd

# Import the *module* so tests can monkeypatch config.EARNINGS_FILE and our
# code sees the patched value at call time.  Using `from config import X`
# would bind a local copy at import time, which the monkeypatch can't affect.
import config


def _parse_report_time(raw: object) -> str:
    """Normalise a report-time field into "bmo" / "amc" / "unknown".

    Why this matters: a Before-Market-Open (BMO) report is public ~7 hours
    before that day's 16:00 ET close, so it is legitimate information for
    day-T features.  An After-Market-Close (AMC) report is NOT public until
    ~30 minutes after the close, so using it on day T would be look-ahead
    bias.  When we can't tell, we store "unknown" and the feature code
    treats it like AMC (the conservative choice).

    Note: FMP's /earnings-surprises endpoint currently returns no time field
    at all, so in practice this yields "unknown" for every row — but we parse
    defensively in case the endpoint (or a future data source) provides one.
    """
    if not isinstance(raw, str):
        return "unknown"
    value = raw.strip().lower()
    # FMP's earnings-calendar uses "bmo"/"amc"; accept only those two exactly
    # and treat anything else ("--", "", "dmh", ...) as unknown.
    if value in ("bmo", "amc"):
        return value
    return "unknown"


def _fetch_symbol_earnings(ticker: str) -> list[dict]:
    """Fetch EPS surprise history for one ticker from FMP.

    FMP endpoint: /earnings-surprises/{ticker}
    Returns a list of dicts like:
        {"symbol": "AAPL", "date": "2024-02-01", "eps": 2.18, "epsEstimated": 2.10}
    """
    url = f"{config.FMP_BASE_URL}/earnings-surprises/{ticker}?apikey={config.FMP_API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def update_earnings(symbols: list[str] | None = None) -> None:
    """Fetch earnings surprise data for all symbols and append to the parquet file.

    Args:
        symbols: Tickers to fetch.  Defaults to the S&P 500 universe from
                 ``sp500_universe.get_symbols()``.

    Skips any ticker already present in the existing file (incremental update).
    Drops rows where ``eps`` or ``epsEstimated`` is None or zero (avoids
    division-by-zero in the surprise calculation).

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

    # Load whatever we already have so we can skip tickers already stored.
    if earnings_file.exists():
        existing = pd.read_parquet(earnings_file)
        fetched_tickers = set(existing["ticker"].unique())
    else:
        existing = pd.DataFrame()
        fetched_tickers = set()

    if not config.FMP_API_KEY:
        print("[info] FMP_API_KEY not set — skipping earnings update.")
        return

    rows = []
    for ticker in symbols:
        if ticker in fetched_tickers:
            # Already have full history for this ticker — skip to save API quota
            continue

        try:
            data = _fetch_symbol_earnings(ticker)
        except Exception as exc:
            # 403 = no paid FMP subscription; other errors = network/rate-limit.
            # Log without the URL (it contains the API key) and abort immediately.
            err_type = type(exc).__name__
            print(f"[warn] Earnings fetch failed for {ticker}: {err_type} — aborting earnings update.")
            return
        for item in data:
            actual = item.get("eps")
            estimated = item.get("epsEstimated")

            # Guard: drop rows with missing values or a zero estimate (can't
            # compute a meaningful surprise percentage if the estimate is 0).
            if actual is None or estimated is None or estimated == 0:
                continue

            # surprise_pct = (actual − estimate) / |estimate|
            # Positive → beat, negative → miss.  Dividing by the absolute value
            # handles negative estimates correctly (e.g. a loss smaller than
            # expected is still a "beat").
            rows.append({
                "ticker": ticker,
                "date": pd.Timestamp(item["date"]),
                "actual_eps": float(actual),
                "estimated_eps": float(estimated),
                "surprise_pct": (float(actual) - float(estimated)) / abs(float(estimated)),
                # When the report was released relative to the trading session.
                # BMO reports may be used on day T; AMC/unknown only from T+1.
                "report_time": _parse_report_time(item.get("time")),
            })

    if not rows:
        # Nothing new fetched — leave the file as-is (or don't create it)
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Legacy rows written before report_time existed come out of the concat
    # as NaN in that column — backfill them to "unknown" so the parquet is
    # always a clean string column with only three possible values.
    combined["report_time"] = combined["report_time"].fillna("unknown")

    # Deduplicate on (ticker, date) — keep the most recent version of any row.
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined.to_parquet(earnings_file, index=False)
    print(
        f"Earnings: {len(combined)} records "
        f"for {combined['ticker'].nunique()} tickers → {earnings_file.name}"
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
