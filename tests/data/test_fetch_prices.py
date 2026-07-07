import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


def make_mock_bars(ticker: str, dates: list) -> MagicMock:
    """Build a mock Alpaca bars response."""
    idx = pd.MultiIndex.from_tuples(
        [(ticker, pd.Timestamp(d)) for d in dates],
        names=["symbol", "timestamp"]
    )
    df = pd.DataFrame({
        "open":        [150.0] * len(dates),
        "high":        [155.0] * len(dates),
        "low":         [148.0] * len(dates),
        "close":       [152.0] * len(dates),
        "volume":      [1_000_000] * len(dates),
        "trade_count": [5000] * len(dates),
        "vwap":        [151.5] * len(dates),
    }, index=idx)
    mock_resp = MagicMock()
    mock_resp.df = df
    return mock_resp


def test_fetch_symbol_returns_dataframe(tmp_path):
    from src.data.fetch_prices import fetch_symbol
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = make_mock_bars(
        "AAPL", ["2024-01-02", "2024-01-03"]
    )
    df = fetch_symbol(mock_client, "AAPL", "2024-01-02", "2024-01-03")
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert df.index.name == "timestamp"
    assert len(df) == 2


def test_update_prices_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_prices
    monkeypatch.setattr(fetch_prices, "PRICES_DIR", tmp_path)
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = make_mock_bars(
        "AAPL", ["2024-01-02", "2024-01-03"]
    )
    with patch.object(fetch_prices, "_make_client", return_value=mock_client):
        fetch_prices.update_prices(symbols=["AAPL"])
    assert (tmp_path / "AAPL.parquet").exists()


def test_no_duplicates_on_rerun(tmp_path, monkeypatch):
    from src.data import fetch_prices
    monkeypatch.setattr(fetch_prices, "PRICES_DIR", tmp_path)
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = make_mock_bars(
        "AAPL", ["2024-01-02", "2024-01-03"]
    )
    with patch.object(fetch_prices, "_make_client", return_value=mock_client):
        fetch_prices.update_prices(symbols=["AAPL"])
        fetch_prices.update_prices(symbols=["AAPL"])
    df = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert df.index.duplicated().sum() == 0
