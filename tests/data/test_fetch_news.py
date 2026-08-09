import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta


def test_update_news_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    mock_news = [
        MagicMock(
            symbols=["AAPL"],
            headline="Apple beats earnings",
            created_at=pd.Timestamp("2024-01-02 14:30:00"),
            source="Reuters",
        ),
        MagicMock(
            symbols=["MSFT"],
            headline="Microsoft raises guidance",
            created_at=pd.Timestamp("2024-01-02 10:15:00"),
            source="Bloomberg",
        ),
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter(mock_news))

    with patch.object(fetch_news, "_make_client") as mock_client_fn:
        mock_client_fn.return_value.get_news.return_value = mock_resp
        fetch_news.update_news(symbols=["AAPL", "MSFT"], target_date="2024-01-02")

    parquet_path = tmp_path / "2024-01-02.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert set(df.columns) >= {"ticker", "headline", "published_at", "source"}
    assert "AAPL" in df["ticker"].values


def test_update_news_skips_empty(tmp_path, monkeypatch):
    """If the API returns no news items, no file should be written."""
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter([]))

    with patch.object(fetch_news, "_make_client") as mock_client_fn:
        mock_client_fn.return_value.get_news.return_value = mock_resp
        fetch_news.update_news(symbols=["AAPL"], target_date="2024-01-02")

    assert not (tmp_path / "2024-01-02.parquet").exists()


def test_update_news_buckets_by_eastern_calendar_day(tmp_path, monkeypatch):
    """An article at 2024-01-03 01:00 UTC is 2024-01-02 20:00 ET — it must
    land in the Jan-2 file, not Jan-3, or it gets attributed to the wrong
    trading day downstream."""
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    mock_news = [
        MagicMock(
            symbols=["AAPL"],
            headline="Late-night wire story",
            created_at=pd.Timestamp("2024-01-03 01:00:00", tz="UTC"),
            source="Reuters",
        ),
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter(mock_news))

    with patch.object(fetch_news, "_make_client") as mock_client_fn:
        mock_client_fn.return_value.get_news.return_value = mock_resp
        # target_date is inclusive of the window, so use a date on/after Jan 3
        # to make sure the item is fetched, then verify it lands in Jan 2's file.
        fetch_news.update_news(symbols=["AAPL"], target_date="2024-01-03")

    assert (tmp_path / "2024-01-02.parquet").exists()
    assert not (tmp_path / "2024-01-03.parquet").exists()


def test_update_news_multi_ticker_item(tmp_path, monkeypatch):
    """A single news item that mentions two tickers should appear as two rows."""
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    mock_news = [
        MagicMock(
            symbols=["AAPL", "MSFT"],
            headline="Big tech rally",
            created_at=pd.Timestamp("2024-01-02 09:00:00"),
            source="WSJ",
        ),
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter(mock_news))

    with patch.object(fetch_news, "_make_client") as mock_client_fn:
        mock_client_fn.return_value.get_news.return_value = mock_resp
        fetch_news.update_news(symbols=["AAPL", "MSFT"], target_date="2024-01-02")

    df = pd.read_parquet(tmp_path / "2024-01-02.parquet")
    # One news item with two tickers → two rows
    assert len(df) == 2
    assert set(df["ticker"].values) == {"AAPL", "MSFT"}


def test_update_news_fetch_window_includes_target_day(tmp_path, monkeypatch):
    """The API's end bound is a midnight-anchored TIMESTAMP, so it must sit
    one day PAST the target date or the target day's own articles are
    excluded. This was the 'news is structurally two days stale' bug: with
    end = yesterday, the newest parquet any run could produce was D-2."""
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    with (
        patch.object(fetch_news, "_make_client"),
        patch.object(fetch_news, "_get_news_items", return_value=[]) as mock_get,
    ):
        fetch_news.update_news(symbols=["AAPL"], target_date="2024-01-02", lookback_days=7)

    _, _, start, end = mock_get.call_args.args
    assert start == "2023-12-26"
    assert end == "2024-01-03", "end bound must be the day AFTER the target date"


def test_update_news_merges_into_partial_day_file(tmp_path, monkeypatch):
    """A file written earlier in the day only holds part of that day's
    articles. A later fetch must UNION new articles into it — adding what's
    new, keeping what's already there, no duplicates."""
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    # The morning run already saved one article for Jan 2.
    ts_morning = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    existing = pd.DataFrame({
        "ticker": ["AAPL"], "headline": ["Apple beats earnings"],
        "published_at": [ts_morning], "source": ["Reuters"], "summary": [""],
    })
    existing.to_parquet(tmp_path / "2024-01-02.parquet", index=False)

    # The afternoon fetch returns the SAME article plus a new one.
    mock_news = [
        MagicMock(
            symbols=["AAPL"], headline="Apple beats earnings",
            created_at=ts_morning, source="Reuters", summary="",
        ),
        MagicMock(
            symbols=["MSFT"], headline="Microsoft afternoon story",
            created_at=pd.Timestamp("2024-01-02 19:00:00", tz="UTC"),
            source="Bloomberg", summary="",
        ),
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter(mock_news))

    with patch.object(fetch_news, "_make_client") as mock_client_fn:
        mock_client_fn.return_value.get_news.return_value = mock_resp
        fetch_news.update_news(symbols=["AAPL", "MSFT"], target_date="2024-01-02")

    df = pd.read_parquet(tmp_path / "2024-01-02.parquet")
    assert len(df) == 2, f"expected morning row + new afternoon row, got:\n{df}"
    assert set(df["headline"]) == {"Apple beats earnings", "Microsoft afternoon story"}


def test_load_recent_news_includes_anchor_day(tmp_path, monkeypatch):
    """The window must include the anchor day itself (i=0), not start at
    anchor-1 — otherwise 'today's news' never contains today and the 8:30 AM
    run can't see overnight headlines."""
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    anchor_day = pd.DataFrame({
        "ticker": ["AAPL"], "headline": ["Anchor-day headline"],
        "published_at": [pd.Timestamp("2024-03-02 08:00", tz="UTC")],
        "source": ["Reuters"], "summary": [""],
    })
    anchor_day.to_parquet(tmp_path / "2024-03-02.parquet")

    result = fetch_news.load_recent_news(days=3, as_of=date(2024, 3, 2))

    assert len(result) == 1
    assert result.iloc[0]["headline"] == "Anchor-day headline"


def test_load_recent_news_anchors_on_as_of(tmp_path, monkeypatch):
    """as_of lets load_recent_news load a window ending at a HISTORICAL date
    instead of real-world today. This is the root fix for the bug where the
    5-year feature matrix only ever saw news from the last couple of days
    near whenever the script happened to run: as_of lets the matrix builder
    say 'load news as if today were this old date', so news_score actually
    lines up with the historical dates being assembled.
    """
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    # Inside the as_of window (2024-03-02 minus 1..3 days = Feb29..Mar1).
    in_window = pd.DataFrame({
        "ticker": ["AAPL"], "headline": ["In-window headline"],
        "published_at": [pd.Timestamp("2024-03-01 14:00")],
        "source": ["Reuters"], "summary": [""],
    })
    in_window.to_parquet(tmp_path / "2024-03-01.parquet")

    # Outside the as_of window, but would be "3 days before real today" if
    # the function ignored as_of and used date.today() instead — a stale
    # real-world-anchored implementation would miss this test entirely
    # (since neither file lines up with today), which is exactly the bug.
    outside_window = pd.DataFrame({
        "ticker": ["AAPL"], "headline": ["Too-old headline"],
        "published_at": [pd.Timestamp("2024-01-01 14:00")],
        "source": ["Reuters"], "summary": [""],
    })
    outside_window.to_parquet(tmp_path / "2024-01-01.parquet")

    result = fetch_news.load_recent_news(days=3, as_of=date(2024, 3, 2))

    assert len(result) == 1
    assert result.iloc[0]["headline"] == "In-window headline"


def test_load_recent_news_defaults_to_real_today(tmp_path, monkeypatch):
    """With as_of=None (the default), behaviour must be unchanged from
    before this fix — anchored on real-world today. This is required by the
    intraday runner and daily_run, which genuinely want 'as of right now'.
    """
    from src.data import fetch_news
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    fresh = pd.DataFrame({
        "ticker": ["AAPL"], "headline": ["Fresh headline"],
        "published_at": [pd.Timestamp.now()],
        "source": ["Reuters"], "summary": [""],
    })
    fresh.to_parquet(tmp_path / f"{yesterday}.parquet")

    result = fetch_news.load_recent_news(days=3)

    assert len(result) == 1
    assert result.iloc[0]["headline"] == "Fresh headline"
