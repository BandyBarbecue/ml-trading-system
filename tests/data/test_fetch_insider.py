import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import config

# yfinance's Ticker.insider_transactions returns a DataFrame with these
# columns (see the module docstring in src/data/fetch_insider.py for the
# full mapping). We build small mock DataFrames shaped the same way so we
# never hit the network in tests.


def _mock_insider_df(rows: list[dict]) -> pd.DataFrame:
    """Build a mock insider_transactions DataFrame from simple row dicts."""
    return pd.DataFrame(rows)


def _mock_ticker(df: pd.DataFrame) -> MagicMock:
    """Return a mock yf.Ticker whose .insider_transactions is *df*."""
    mock = MagicMock()
    mock.insider_transactions = df
    return mock


def test_update_insider_writes_parquet(tmp_path, monkeypatch):
    """Full pipeline: yfinance mock -> update_insider() -> parquet on disk."""
    from src.data import fetch_insider

    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")

    df = _mock_insider_df([
        {
            "Text":       "Purchase",
            "Position":   "Chief Executive Officer",
            "Start Date": pd.Timestamp("2024-05-01"),
            "Shares":     10000,
            "Value":      1850000.0,
        },
    ])

    with patch("src.data.fetch_insider.yf.Ticker", return_value=_mock_ticker(df)):
        fetch_insider.update_insider(symbols=["AAPL"])

    out_path = tmp_path / "insider.parquet"
    assert out_path.exists()
    result = pd.read_parquet(out_path)
    assert set(result.columns) >= {
        "ticker", "date", "insider_type", "transaction_type", "shares", "value_usd"
    }
    assert result["ticker"].iloc[0] == "AAPL"
    assert result["transaction_type"].iloc[0] == "P"
    # CEO gets 2x weight multiplier: 10000 shares * 2.0
    assert result["shares"].iloc[0] == pytest.approx(20000.0)


def test_fetch_ticker_insider_executive_role_weight():
    """Executive-role transactions (CEO/CFO/President) get a 2x weight."""
    from src.data.fetch_insider import _fetch_ticker_insider

    df = _mock_insider_df([
        {
            "Text":       "Purchase",
            "Position":   "Chief Executive Officer",
            "Start Date": pd.Timestamp("2024-05-01"),
            "Shares":     10000,
            "Value":      1850000.0,
        },
    ])

    with patch("src.data.fetch_insider.yf.Ticker", return_value=_mock_ticker(df)):
        rows = _fetch_ticker_insider("AAPL")

    assert len(rows) == 1
    assert rows[0]["transaction_type"] == "P"
    assert rows[0]["shares"] == pytest.approx(20000.0)         # 10000 * 2.0
    assert rows[0]["value_usd"] == pytest.approx(1850000.0 * 2.0)


def test_fetch_ticker_insider_director_no_weight():
    """Non-executive roles (e.g. plain Director) keep the 1x weight."""
    from src.data.fetch_insider import _fetch_ticker_insider

    df = _mock_insider_df([
        {
            "Text":       "Sale",
            "Position":   "Director",
            "Start Date": pd.Timestamp("2024-05-01"),
            "Shares":     500,
            "Value":      92500.0,
        },
    ])

    with patch("src.data.fetch_insider.yf.Ticker", return_value=_mock_ticker(df)):
        rows = _fetch_ticker_insider("AAPL")

    assert len(rows) == 1
    assert rows[0]["transaction_type"] == "S"
    assert rows[0]["shares"] == pytest.approx(500.0)     # 500 * 1.0, no exec weight
    assert rows[0]["value_usd"] == pytest.approx(92500.0)


def test_fetch_ticker_insider_ignores_non_signal_types():
    """Grants, option exercises, and gifts carry no signal and are skipped."""
    from src.data.fetch_insider import _fetch_ticker_insider

    df = _mock_insider_df([
        {
            "Text":       "Option Exercise",
            "Position":   "Chief Financial Officer",
            "Start Date": pd.Timestamp("2024-05-01"),
            "Shares":     2000,
            "Value":      0.0,
        },
        {
            "Text":       "Stock Gift",
            "Position":   "Director",
            "Start Date": pd.Timestamp("2024-05-02"),
            "Shares":     100,
            "Value":      0.0,
        },
        {
            "Text":       "Purchase",
            "Position":   "Director",
            "Start Date": pd.Timestamp("2024-05-03"),
            "Shares":     300,
            "Value":      55000.0,
        },
    ])

    with patch("src.data.fetch_insider.yf.Ticker", return_value=_mock_ticker(df)):
        rows = _fetch_ticker_insider("AAPL")

    # Only the "Purchase" row is a signal transaction — exercise and gift are dropped.
    assert len(rows) == 1
    assert rows[0]["transaction_type"] == "P"
