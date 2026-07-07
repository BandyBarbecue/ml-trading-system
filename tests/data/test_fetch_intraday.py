"""Tests for src/data/fetch_intraday.py

All tests mock the Alpaca client so we never make real network calls.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intraday_response(ticker: str, timestamps: list) -> MagicMock:
    """Build a fake Alpaca bars response with a (symbol, timestamp) MultiIndex."""
    idx = pd.MultiIndex.from_tuples(
        [(ticker, pd.Timestamp(ts, tz="UTC")) for ts in timestamps],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {
            "open":        [100.0] * len(timestamps),
            "high":        [105.0] * len(timestamps),
            "low":         [98.0]  * len(timestamps),
            "close":       [102.0] * len(timestamps),
            "volume":      [500_000] * len(timestamps),
            "vwap":        [101.5] * len(timestamps),
            "trade_count": [2000]  * len(timestamps),  # should be dropped by fetch_intraday
        },
        index=idx,
    )
    mock_resp = MagicMock()
    mock_resp.df = df
    return mock_resp


# ---------------------------------------------------------------------------
# test_fetch_intraday_returns_dataframe
# ---------------------------------------------------------------------------

def test_fetch_intraday_returns_dataframe():
    """mock client returns canned response → DataFrame with expected columns."""
    from src.data.fetch_intraday import fetch_intraday

    timestamps = [
        "2024-01-15 14:30:00",  # 09:30 ET = 14:30 UTC (regular open bar)
        "2024-01-15 15:00:00",
        "2024-01-15 15:30:00",
    ]
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = _make_intraday_response("AAPL", timestamps)

    df = fetch_intraday(mock_client, "AAPL", "2024-01-15")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 3

    # All expected OHLCV + vwap columns must be present
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        assert col in df.columns, f"Missing column: {col}"

    # trade_count is an Alpaca extra — we should have dropped it
    assert "trade_count" not in df.columns

    # Index should be a DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# test_fetch_intraday_empty_on_no_data
# ---------------------------------------------------------------------------

def test_fetch_intraday_empty_on_no_data():
    """mock client raises an exception → returns empty DataFrame (doesn't crash)."""
    from src.data.fetch_intraday import fetch_intraday

    mock_client = MagicMock()
    # Simulate any API error (e.g. APIError, network timeout, etc.)
    mock_client.get_stock_bars.side_effect = Exception("API error: no data for symbol")

    df = fetch_intraday(mock_client, "FAKE", "2024-01-15")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    # Even though it's empty, the column schema should be correct
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        assert col in df.columns


# ---------------------------------------------------------------------------
# test_update_intraday_saves_parquet
# ---------------------------------------------------------------------------

def test_update_intraday_saves_parquet(tmp_path, monkeypatch):
    """mock fetch_intraday → parquet file created at the correct path."""
    from src.data import fetch_intraday as fi_module

    # Redirect INTRADAY_DIR so we don't touch the real data/ folder
    monkeypatch.setattr(fi_module.config, "INTRADAY_DIR", tmp_path)
    monkeypatch.setattr(fi_module.config, "FALLBACK_SYMBOLS", ["AAPL"])

    # Build a small fake DataFrame to return from fetch_intraday
    fake_df = pd.DataFrame(
        {
            "open":   [100.0, 101.0],
            "high":   [105.0, 106.0],
            "low":    [98.0,  99.0],
            "close":  [102.0, 103.0],
            "volume": [500_000, 600_000],
            "vwap":   [101.5, 102.0],
        },
        index=pd.DatetimeIndex(
            ["2024-01-15 09:30:00-05:00", "2024-01-15 10:00:00-05:00"],
            name="timestamp",
        ),
    )

    # Patch fetch_intraday (the function inside the module) and _make_client
    with patch.object(fi_module, "fetch_intraday", return_value=fake_df) as mock_fetch, \
         patch.object(fi_module, "_make_client", return_value=MagicMock()):
        fi_module.update_intraday(symbols=["AAPL"], target_date="2024-01-15")

    expected_path = tmp_path / "AAPL" / "2024-01-15.parquet"
    assert expected_path.exists(), f"Expected parquet at {expected_path}"

    # Load it back and verify it round-trips correctly
    loaded = pd.read_parquet(expected_path)
    assert len(loaded) == 2
    assert "close" in loaded.columns
