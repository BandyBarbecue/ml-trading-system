"""Tests for the yfinance-based earnings fetcher.

We never hit Yahoo in tests: _fetch_symbol_earnings is patched to return a
synthetic frame shaped exactly like yfinance's get_earnings_dates() output —
a tz-aware US/Eastern DatetimeIndex named "Earnings Date" with columns
"EPS Estimate" and "Reported EPS".
"""

import pandas as pd
import numpy as np
from unittest.mock import patch


def _yf_frame(rows: list[tuple[str, float | None, float | None]]) -> pd.DataFrame:
    """Build a yfinance-style earnings frame.

    rows: list of (timestamp_str_ET, eps_estimate, reported_eps).
    """
    index = pd.DatetimeIndex(
        [pd.Timestamp(ts, tz="America/New_York") for ts, _, _ in rows],
        name="Earnings Date",
    )
    return pd.DataFrame(
        {
            "EPS Estimate": [est for _, est, _ in rows],
            "Reported EPS": [act for _, _, act in rows],
        },
        index=index,
    )


MOCK_FRAME = _yf_frame([
    ("2024-02-01 16:30", 2.10, 2.18),
    ("2023-11-02 16:30", 1.39, 1.46),
])


def test_update_earnings_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    with patch("src.data.fetch_earnings._fetch_symbol_earnings", return_value=MOCK_FRAME):
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet")
    assert set(df.columns) >= {"ticker", "date", "actual_eps", "estimated_eps", "surprise_pct"}
    assert "AAPL" in df["ticker"].values
    # surprise_pct = (actual - estimated) / |estimated|
    row = df[df["date"] == pd.Timestamp("2024-02-01")].iloc[0]
    expected = (2.18 - 2.10) / abs(2.10)
    assert abs(row["surprise_pct"] - expected) < 1e-6
    # Dates are stored tz-naive at midnight so they join cleanly against the
    # feature pipeline's daily index.
    assert df["date"].dt.tz is None


def test_update_earnings_derives_report_time_from_timestamp(tmp_path, monkeypatch):
    """report_time must be derived from the announcement timestamp:
    pre-market → bmo, post-close → amc, midnight (no time info) or
    mid-session oddities → unknown."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    frame = _yf_frame([
        ("2024-02-01 06:00", 2.10, 2.18),   # pre-market call → bmo
        ("2023-11-02 16:30", 1.39, 1.46),   # after the close → amc
        ("2023-08-03 13:00", 1.19, 1.26),   # mid-session quirk → unknown
        ("2023-05-04 00:00", 1.43, 1.52),   # midnight = date only → unknown
    ])

    with patch("src.data.fetch_earnings._fetch_symbol_earnings", return_value=frame):
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet").set_index("date")
    assert "report_time" in df.columns
    assert df.loc[pd.Timestamp("2024-02-01"), "report_time"] == "bmo"
    assert df.loc[pd.Timestamp("2023-11-02"), "report_time"] == "amc"
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

    def fake_fetch(ticker):
        # MSFT's stored report is old, so it IS re-queried — Yahoo just has
        # nothing new for it in this scenario.
        return MOCK_FRAME if ticker == "AAPL" else pd.DataFrame()

    with patch("src.data.fetch_earnings._fetch_symbol_earnings", side_effect=fake_fetch):
        fetch_earnings.update_earnings(symbols=["AAPL", "MSFT"])

    df = pd.read_parquet(earnings_file)
    assert "report_time" in df.columns
    # The legacy MSFT row was backfilled to "unknown"; the new AAPL rows got
    # real flags derived from their timestamps.
    assert df.loc[df["ticker"] == "MSFT", "report_time"].eq("unknown").all()
    assert set(df["report_time"].unique()) <= {"bmo", "amc", "unknown"}


def test_update_earnings_skips_fresh_tickers(tmp_path, monkeypatch):
    """A ticker whose newest stored report is recent must NOT be re-fetched —
    quarterly earnings can't have a new number that soon."""
    from src.data import fetch_earnings
    import config
    earnings_file = tmp_path / "earnings.parquet"
    monkeypatch.setattr(config, "EARNINGS_FILE", earnings_file)

    recent = pd.DataFrame({
        "ticker": ["MSFT"],
        "date": [pd.Timestamp.now().normalize() - pd.Timedelta(days=10)],
        "actual_eps": [2.99],
        "estimated_eps": [2.65],
        "surprise_pct": [(2.99 - 2.65) / 2.65],
        "report_time": ["amc"],
    })
    recent.to_parquet(earnings_file, index=False)

    with patch("src.data.fetch_earnings._fetch_symbol_earnings") as mock_fetch:
        fetch_earnings.update_earnings(symbols=["MSFT"])

    mock_fetch.assert_not_called()


def test_update_earnings_skips_missing_eps(tmp_path, monkeypatch):
    """Rows with NaN EPS values (future quarters, missing estimates) should be
    silently dropped."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    frame = _yf_frame([
        ("2024-05-01 16:30", 2.10, None),    # future quarter: no actual yet
        ("2023-11-02 16:30", None, 1.46),    # no estimate
        ("2023-08-03 16:30", 1.15, 1.20),    # complete row — survives
    ])

    with patch("src.data.fetch_earnings._fetch_symbol_earnings", return_value=frame):
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet")
    # Only the complete row should survive
    assert len(df) == 1
    assert df.iloc[0]["actual_eps"] == 1.20


def test_update_earnings_skips_zero_estimated(tmp_path, monkeypatch):
    """EPS Estimate == 0 should be dropped (division by zero guard)."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    frame = _yf_frame([
        ("2024-02-01 16:30", 0.0, 0.50),
    ])

    with patch("src.data.fetch_earnings._fetch_symbol_earnings", return_value=frame):
        fetch_earnings.update_earnings(symbols=["AAPL"])

    # Nothing should be written when all rows are filtered out
    assert not (tmp_path / "earnings.parquet").exists()


def test_update_earnings_deduplicates(tmp_path, monkeypatch):
    """Running update twice must not create duplicate (ticker, date) rows."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    with patch("src.data.fetch_earnings._fetch_symbol_earnings", return_value=MOCK_FRAME):
        # The stored rows are from 2023/2024 (stale), so the second run
        # re-fetches AAPL and gets the SAME rows back — dedup must collapse them.
        fetch_earnings.update_earnings(symbols=["AAPL"])
        fetch_earnings.update_earnings(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "earnings.parquet")
    # Should still be 2 rows, not 4
    assert len(df) == 2


def test_update_earnings_aborts_after_consecutive_failures(tmp_path, monkeypatch):
    """An unbroken failure streak (Yahoo throttling) must abort the update
    instead of hammering through the remaining universe."""
    from src.data import fetch_earnings
    import config
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")

    symbols = [f"T{i}" for i in range(fetch_earnings.MAX_CONSECUTIVE_FAILURES + 20)]

    with patch(
        "src.data.fetch_earnings._fetch_symbol_earnings",
        side_effect=ConnectionError("throttled"),
    ) as mock_fetch:
        fetch_earnings.update_earnings(symbols=symbols)

    # Aborted at the threshold — did not try all symbols
    assert mock_fetch.call_count == fetch_earnings.MAX_CONSECUTIVE_FAILURES
    assert not (tmp_path / "earnings.parquet").exists()
