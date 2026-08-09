"""News fetcher: pulls headlines from the Alpaca News API and writes daily parquet files.

Each file is saved to NEWS_DIR / YYYY-MM-DD.parquet with one row per
(ticker, headline) pair — a single article mentioning two tickers becomes
two rows, so you can filter by ticker downstream without any explode step.

Run from the project root:
    python -m src.data.fetch_news
"""

import logging

import pandas as pd
from datetime import date, datetime, timedelta, timezone

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

# Import the module-level values we need.  NEWS_DIR is also exposed at the top
# level so tests can monkeypatch it: `monkeypatch.setattr(fetch_news, "NEWS_DIR", ...)`
from config import NEWS_DIR, ALPACA_API_KEY, ALPACA_SECRET_KEY

# logging (not print) so news fetches show up in data/logs/daily_run_*.log —
# the 2026-07-10 post-mortem found the news pipeline had been silently stale
# for days precisely because its output never reached the log files.
logger = logging.getLogger(__name__)


def _make_client() -> NewsClient:
    """Build the Alpaca NewsClient from config credentials.

    Kept as a separate function so tests can patch it with
    `patch.object(fetch_news, "_make_client")`.
    """
    return NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _get_news_items(client: NewsClient, symbols: list[str], start: str, end: str) -> list:
    """Single page of Alpaca news items for a symbol list and date range."""
    req = NewsRequest(
        symbols=",".join(symbols),
        start=start,
        end=end,
        limit=1000,
    )
    response = client.get_news(req)
    if hasattr(response, "data") and isinstance(response.data, dict):
        return response.data.get("news", [])
    if hasattr(response, "news") and isinstance(response.news, list):
        return response.news
    return list(response)


def update_news(
    symbols: list[str] | None = None,
    lookback_days: int = 7,
    target_date: str | None = None,
) -> None:
    """Fetch recent headlines and write one parquet file per calendar date.

    Fetches the last *lookback_days* calendar days in a single API call (so
    weekend articles published on Saturday/Sunday are not missed on Monday),
    then splits results by publication date and saves each date separately.
    Dates that already have a parquet file are skipped.

    Args:
        symbols:      List of ticker symbols.  Defaults to S&P 500 universe.
        lookback_days: How many calendar days to look back (default 7).
        target_date:  ISO date string (YYYY-MM-DD).  When provided, fetch the
                      window ending on this date instead of yesterday.

    Output:
        NEWS_DIR / YYYY-MM-DD.parquet — columns: ticker, headline,
        published_at, source.  One file per date that has at least one article.
    """
    # Lazy import avoids a circular dependency in tests that mock this module.
    from src.data.sp500_universe import get_symbols

    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = get_symbols()

    # The API's `end` bound is a TIMESTAMP: a bare date string means midnight
    # at the START of that day, so passing "yesterday" actually excluded all
    # of yesterday's articles and the newest file was always two days old
    # (the 2026-07-10 "news is structurally stale" bug). To INCLUDE a day's
    # articles the bound must be past the end of that day.
    if target_date is not None:
        end_date   = (date.fromisoformat(target_date) + timedelta(days=1)).isoformat()
        start_date = (date.fromisoformat(target_date) - timedelta(days=lookback_days)).isoformat()
    else:
        end_date   = datetime.now(timezone.utc).isoformat()   # right now — includes today so far
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    client = _make_client()
    news_items = _get_news_items(client, symbols, start_date, end_date)

    if not news_items:
        return

    # Fan out items into (ticker, date) rows then group by date for saving.
    rows = []
    for item in news_items:
        # Alpaca's created_at is UTC. Bucketing on the raw UTC calendar date
        # would push anything published after ~20:00 ET (market close + a
        # few hours) into the *next* trading day's file — e.g. an article at
        # 2024-01-03 01:00 UTC is really 2024-01-02 20:00 ET and belongs with
        # Jan 2's news. So we convert to US/Eastern BEFORE taking the date,
        # same pattern as fetch_intraday.py's tz handling.
        ts = pd.Timestamp(item.created_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")  # defensive: assume UTC if tz-naive input slips through
        pub_date = ts.tz_convert("US/Eastern").normalize().tz_localize(None)
        date_str = pub_date.strftime("%Y-%m-%d")
        # summary may be absent on older SDK versions or stub objects in tests.
        # getattr returns None/MagicMock when missing, so we guard with isinstance.
        raw_summary = getattr(item, "summary", None)
        summary = raw_summary if isinstance(raw_summary, str) else ""
        for ticker in item.symbols:
            rows.append({
                "ticker":       ticker,
                "headline":     item.headline,
                "published_at": item.created_at,
                "source":       item.source,
                "summary":      summary,
                "date":         date_str,
            })

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"])

    for day, group in df.groupby("date"):
        out_path = NEWS_DIR / f"{day}.parquet"
        out_df = group.drop(columns=["date"])
        if out_path.exists():
            # Now that we fetch through "right now", a file written earlier
            # today (or yesterday intraday) only holds PART of that day's
            # articles. Merge instead of skip-or-overwrite: union the old and
            # new rows and drop duplicates, so re-fetching can only ever ADD
            # articles, never lose ones a previous run already saved (the
            # fetch is capped at 1000 items, so a wide window can return
            # fewer old articles than we already have on disk).
            existing = pd.read_parquet(out_path)
            merged = pd.concat([existing, out_df], ignore_index=True)
            # keep="last" prefers the freshly fetched copy of a duplicate row
            # (it may carry a backfilled summary the old row lacked).
            merged = merged.drop_duplicates(
                subset=["ticker", "headline", "published_at"], keep="last"
            )
            out_df = merged
        out_df.to_parquet(out_path, index=False)
        logger.info("News %s: %d headlines -> %s", day, len(out_df), out_path.name)


def load_recent_news(days: int = 3, as_of: date | None = None) -> pd.DataFrame:
    """Load the last *days* daily news parquet files into one DataFrame.

    Useful for building a short feature window without scanning the whole
    news directory.

    Args:
        days:  How many calendar days back (from the anchor) to look.
        as_of: Anchor date for "recent". Defaults to None, which means real
               -world "today" — the original behaviour, still what the
               intraday runner and daily_run want (they always want news as
               of right now). Pass an explicit historical date to load the
               window ending on THAT date instead. This is what lets a
               5-year historical feature matrix load news relative to the
               date it's currently assembling, instead of always relative
               to whatever day the script happens to be run on (the bug:
               without this, a build run today only ever sees news from the
               last few days near today, no matter which historical date
               each row is for).
    """
    anchor = as_of if as_of is not None else date.today()
    frames = []
    # i=0 is the anchor day ITSELF, not the day before. Excluding the anchor
    # (the old `i + 1`) meant "today's news" never included today: the 8:30 AM
    # run couldn't see overnight headlines and the intraday event layer read a
    # file that was never part of the window. Look-ahead safety for historical
    # builds is handled downstream by _aggregate_scores' 16:00-ET close cutoff,
    # not by hiding the anchor day's file.
    for i in range(days):
        d = (anchor - timedelta(days=i)).isoformat()
        path = NEWS_DIR / f"{d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    update_news()
