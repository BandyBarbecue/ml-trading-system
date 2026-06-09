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


def update_news(
    symbols: list[str] | None = None,
    target_date: str | None = None,
) -> None:
    """Fetch all headlines published on *target_date* for the given symbols.

    Args:
        symbols:     List of ticker symbols to filter by.  Defaults to the
                     full S&P 500 universe from ``sp500_universe.get_symbols()``.
        target_date: ISO date string (YYYY-MM-DD).  Defaults to yesterday,
                     because today's trading day is not yet complete.

    Output:
        NEWS_DIR / <target_date>.parquet — columns: ticker, headline,
        published_at, source.  File is only created when at least one
        headline is found.
    """
    # Lazy import avoids a circular dependency in tests that mock this module.
    from src.data.sp500_universe import get_symbols

    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = get_symbols()

    if target_date is None:
        # Yesterday: today's session isn't finished yet
        target_date = (date.today() - timedelta(days=1)).isoformat()

    client = _make_client()

    # NewsRequest.symbols expects a comma-separated string, not a list.
    # (Checked against the installed alpaca-py source: `symbols: Optional[str]`.)
    # limit=1000 is the per-page maximum; the SDK handles pagination automatically.
    req = NewsRequest(
        symbols=",".join(symbols),
        start=target_date,
        end=target_date,
        limit=1000,
    )
    news_items = client.get_news(req)

    # Each news item can reference multiple tickers in item.symbols.
    # We fan those out so every row belongs to exactly one ticker — easier
    # to join against price data later.
    rows = []
    for item in news_items:
        for ticker in item.symbols:
            rows.append({
                "ticker": ticker,
                "headline": item.headline,
                "published_at": item.created_at,
                "source": item.source,
            })

    if not rows:
        # Nothing to write — don't create a zero-byte / empty parquet file
        return

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"])

    out_path = NEWS_DIR / f"{target_date}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"News {target_date}: {len(df)} headlines → {out_path.name}")


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
