import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import config


def test_revision_momentum_positive(tmp_path, monkeypatch):
    """When estimates are revised upward across a realistic quarterly gap,
    revision_momentum should be positive (before z-scoring).

    Real quarterly earnings reports land ~91-92 calendar days apart (a year
    divided by 4). REVISION_WINDOW is 120 days specifically so that two
    consecutive quarterly reports both land inside the window — with the
    old 90-day window, a 91-day gap would fall just *outside* it and
    revision_momentum would be NaN for nearly every real ticker.
    """
    from src.features import revision_features

    today = datetime(2024, 6, 1)
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": (today - timedelta(days=91)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.00, "actual_eps": 1.05, "surprise_pct": 5.0},
        {"ticker": "AAPL", "date": today.strftime("%Y-%m-%d"),
         "estimated_eps": 1.20, "actual_eps": 1.25, "surprise_pct": 4.2},
    ])
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")
    df.to_parquet(tmp_path / "earnings.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = revision_features.compute_revision_features(dates, symbols=["AAPL", "MSFT"])

    # With AAPL having upward revisions and MSFT having no data, AAPL gets a raw positive value.
    # With only 1 ticker having data (AAPL), z-scoring is not possible — raw value stored.
    assert not result.empty
    assert ("revision_momentum", "AAPL") in result.columns
    val = result.loc[today, ("revision_momentum", "AAPL")]
    assert not np.isnan(val), "91-day-apart quarterly reports must both fall inside the 120-day window"
    assert val > 0  # estimate rose from 1.00 to 1.20


def test_revision_momentum_cross_sectional(tmp_path, monkeypatch):
    """With two tickers, one with rising estimates and one with falling
    (each pair spaced a realistic ~91-92 days apart, one quarterly cycle),
    they should have opposite signs after z-scoring."""
    from src.features import revision_features

    today = datetime(2024, 6, 1)
    # AAPL: estimates rising over one quarterly cycle (~91 days)
    # MSFT: estimates falling over one quarterly cycle (~92 days)
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": (today - timedelta(days=91)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.00, "actual_eps": 1.05, "surprise_pct": 5.0},
        {"ticker": "AAPL", "date": today.strftime("%Y-%m-%d"),
         "estimated_eps": 1.20, "actual_eps": 1.25, "surprise_pct": 4.2},
        {"ticker": "MSFT", "date": (today - timedelta(days=92)).strftime("%Y-%m-%d"),
         "estimated_eps": 2.50, "actual_eps": 2.45, "surprise_pct": -2.0},
        {"ticker": "MSFT", "date": today.strftime("%Y-%m-%d"),
         "estimated_eps": 2.20, "actual_eps": 2.18, "surprise_pct": -0.9},
    ])
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")
    df.to_parquet(tmp_path / "earnings.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = revision_features.compute_revision_features(dates, symbols=["AAPL", "MSFT"])

    aapl_val = result.loc[today, ("revision_momentum", "AAPL")]
    msft_val = result.loc[today, ("revision_momentum", "MSFT")]
    # AAPL had upward revisions (+20%), MSFT downward (-12%) — opposite signs after z-scoring
    assert aapl_val > 0
    assert msft_val < 0


def test_revision_momentum_missing_file(tmp_path, monkeypatch):
    from src.features import revision_features

    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "no_earnings.parquet")
    dates = pd.DatetimeIndex([datetime(2024, 6, 1)])
    result = revision_features.compute_revision_features(dates, symbols=["AAPL"])
    assert result.empty
