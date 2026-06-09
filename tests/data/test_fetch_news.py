import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import date


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
