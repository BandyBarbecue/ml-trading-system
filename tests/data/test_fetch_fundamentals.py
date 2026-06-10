import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import config


def _mock_ticker(pb=42.5, gm=0.46):
    """Return a mock yf.Ticker whose .info dict contains the given values."""
    mock = MagicMock()
    mock.info = {"priceToBook": pb, "grossMargins": gm}
    return mock


def test_update_fundamentals_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")

    with patch("src.data.fetch_fundamentals.yf.Ticker", return_value=_mock_ticker()):
        fetch_fundamentals.update_fundamentals(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "fundamentals.parquet")
    assert set(df.columns) >= {"ticker", "date", "gross_profitability", "pb_ratio"}
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["gross_profitability"].iloc[0] == pytest.approx(0.46)
    assert df["pb_ratio"].iloc[0] == pytest.approx(42.5)


def test_update_fundamentals_handles_missing_values(tmp_path, monkeypatch):
    """When Yahoo returns None for a field, it is stored as None (not a crash)."""
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")

    mock = MagicMock()
    mock.info = {"priceToBook": None, "grossMargins": None}

    with patch("src.data.fetch_fundamentals.yf.Ticker", return_value=mock):
        fetch_fundamentals.update_fundamentals(symbols=["NEWIPO"])

    df = pd.read_parquet(tmp_path / "fundamentals.parquet")
    assert len(df) == 1
    assert pd.isna(df["pb_ratio"].iloc[0])
    assert pd.isna(df["gross_profitability"].iloc[0])


def test_update_fundamentals_deduplicates(tmp_path, monkeypatch):
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")

    with patch("src.data.fetch_fundamentals.yf.Ticker", return_value=_mock_ticker()):
        fetch_fundamentals.update_fundamentals(symbols=["AAPL"])
        fetch_fundamentals.update_fundamentals(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "fundamentals.parquet")
    assert len(df) == 1  # second call same date → upsert, not duplicate
