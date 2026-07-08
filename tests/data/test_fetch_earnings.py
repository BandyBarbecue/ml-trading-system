import pandas as pd
import pytest
from unittest.mock import patch
import json

MOCK_RESPONSE = [
    {"symbol": "AAPL", "date": "2024-02-01", "eps": 2.18, "epsEstimated": 2.10},
    {"symbol": "AAPL", "date": "2023-11-02", "eps": 1.46, "epsEstimated": 1.39},
]


def test_update_earnings_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = MOCK_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet")
    assert set(df.columns) >= {"ticker", "date", "actual_eps", "estimated_eps", "surprise_pct"}
    assert "AAPL" in df["ticker"].values
    # surprise_pct = (actual - estimated) / |estimated|
    row = df[df["ticker"] == "AAPL"].iloc[0]
    expected = (2.18 - 2.10) / abs(2.10)
    assert abs(row["surprise_pct"] - expected) < 1e-6


def test_update_earnings_writes_report_time_column(tmp_path, monkeypatch):
    """report_time must be persisted: map FMP's `time` field when present,
    fall back to "unknown" when absent or unrecognised."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    response = [
        # Explicit flags (FMP earnings-calendar style) map straight through
        {"symbol": "AAPL", "date": "2024-02-01", "eps": 2.18, "epsEstimated": 2.10, "time": "amc"},
        {"symbol": "AAPL", "date": "2023-11-02", "eps": 1.46, "epsEstimated": 1.39, "time": "BMO"},
        # Unrecognised value → unknown
        {"symbol": "AAPL", "date": "2023-08-03", "eps": 1.26, "epsEstimated": 1.19, "time": "--"},
        # Field missing entirely (current /earnings-surprises behaviour) → unknown
        {"symbol": "AAPL", "date": "2023-05-04", "eps": 1.52, "epsEstimated": 1.43},
    ]

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = response
        mock_get.return_value.raise_for_status = lambda: None
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet").set_index("date")
    assert "report_time" in df.columns
    assert df.loc[pd.Timestamp("2024-02-01"), "report_time"] == "amc"
    # Case-insensitive: "BMO" normalises to lowercase "bmo"
    assert df.loc[pd.Timestamp("2023-11-02"), "report_time"] == "bmo"
    assert df.loc[pd.Timestamp("2023-08-03"), "report_time"] == "unknown"
    assert df.loc[pd.Timestamp("2023-05-04"), "report_time"] == "unknown"
    # Only the three allowed values, ever
    assert set(df["report_time"].unique()) <= {"bmo", "amc", "unknown"}


def test_update_earnings_backfills_legacy_rows(tmp_path, monkeypatch):
    """A pre-existing parquet WITHOUT report_time (legacy) must come out of an
    update with the column present and legacy rows backfilled to "unknown"."""
    from src.data import fetch_earnings
    import config
    earnings_file = tmp_path / "earnings.parquet"
    monkeypatch.setattr(config, "EARNINGS_FILE", earnings_file)

    # Simulate a legacy file written before report_time existed
    legacy = pd.DataFrame({
        "ticker": ["MSFT"],
        "date": [pd.Timestamp("2023-10-24")],
        "actual_eps": [2.99],
        "estimated_eps": [2.65],
        "surprise_pct": [(2.99 - 2.65) / 2.65],
    })
    legacy.to_parquet(earnings_file, index=False)

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = MOCK_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None
        fetch_earnings.update_earnings(symbols=["AAPL"])  # MSFT skipped (already present)

    df = pd.read_parquet(earnings_file)
    assert "report_time" in df.columns
    # The legacy MSFT row was backfilled, the new AAPL rows got "unknown" too
    # (MOCK_RESPONSE has no time field).
    assert (df["report_time"] == "unknown").all()


def test_update_earnings_skips_missing_eps(tmp_path, monkeypatch):
    """Rows with None EPS values should be silently dropped."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    bad_response = [
        {"symbol": "AAPL", "date": "2024-02-01", "eps": None, "epsEstimated": 2.10},
        {"symbol": "AAPL", "date": "2023-11-02", "eps": 1.46, "epsEstimated": None},
        {"symbol": "AAPL", "date": "2023-08-03", "eps": 1.20, "epsEstimated": 1.15},
    ]

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = bad_response
        mock_get.return_value.raise_for_status = lambda: None
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet")
    # Only the third row (both values present) should survive
    assert len(df) == 1
    assert df.iloc[0]["actual_eps"] == 1.20


def test_update_earnings_skips_zero_estimated(tmp_path, monkeypatch):
    """epsEstimated == 0 should be dropped (division by zero guard)."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    response = [
        {"symbol": "AAPL", "date": "2024-02-01", "eps": 0.50, "epsEstimated": 0.0},
    ]

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = response
        mock_get.return_value.raise_for_status = lambda: None
        fetch_earnings.update_earnings(symbols=["AAPL"])

    # Nothing should be written when all rows are filtered out
    assert not (tmp_path / "earnings.parquet").exists()


def test_update_earnings_deduplicates(tmp_path, monkeypatch):
    """Running update twice should not create duplicate rows."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = MOCK_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None
        # Run twice — second run should see existing ticker and skip
        fetch_earnings.update_earnings(symbols=["AAPL"])

    # Reset monkeypatch binding so second call reads from the same tmp file
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    with patch("src.data.fetch_earnings.requests.get") as mock_get:
        mock_get.return_value.json.return_value = MOCK_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet")
    # Should still be 2 rows, not 4
    assert len(df) == 2
