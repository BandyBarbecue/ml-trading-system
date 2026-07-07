"""News fetcher: pulls headlines from the Alpaca News API and writes daily parquet files.

Each file is saved to NEWS_DIR / YYYY-MM-DD.parquet with one row per
(ticker, headline) pair — a single article mentioning two tickers becomes
two rows, so you can filter by ticker downstream without any explode step.

Run from the project root:
    python -m src.data.fetch_news
"""

import pandas as pd
from datetime import date, timedelta

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

# Import the module-level values we need.  NEWS_DIR is also exposed at the top
# level so tests can monkeypatch it: `monkeypatch.setattr(fetch_news, "NEWS_DIR", ...)`
from config import NEWS_DIR, ALPACA_API_KEY, ALPACA_SECRET_KEY


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

    if target_date is not None:
        end_date   = target_date
        start_date = (date.fromisoformat(target_date) - timedelta(days=lookback_days)).isoformat()
    else:
        end_date   = (date.today() - timedelta(days=1)).isoformat()   # yesterday
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    client = _make_client()
    news_items = _get_news_items(client, symbols, start_date, end_date)

    if not news_items:
        return

    # Fan out items into (ticker, date) rows then group by date for saving.
    rows = []
    for item in news_items:
        pub_date = pd.Timestamp(item.created_at).normalize()
        if pub_date.tzinfo is not None:
            pub_date = pub_date.tz_localize(None)
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
        if out_path.exists():
            continue  # already saved from a previous run — don't overwrite
        out_df = group.drop(columns=["date"])
        out_df.to_parquet(out_path, index=False)
        print(f"News {day}: {len(out_df)} headlines -> {out_path.name}")


def load_recent_news(days: int = 3) -> pd.DataFrame:
    """Load the last *days* daily news parquet files into one DataFrame.

    Useful for building a short feature window without scanning the whole
    news directory.
    """
    frames = []
    for i in range(days):
        d = (date.today() - timedelta(days=i + 1)).isoformat()
        path = NEWS_DIR / f"{d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    update_news()
