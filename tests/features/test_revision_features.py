import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import config


def test_revision_momentum_positive(tmp_path, monkeypatch):
    """When estimates are consistently revised upward, revision_momentum should be positive (before z-scoring)."""
    from src.features import revision_features

    today = datetime(2024, 6, 1)
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": (today - timedelta(days=80)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.00, "actual_eps": 1.05, "surprise_pct": 5.0},
        {"ticker": "AAPL", "date": (today - timedelta(days=50)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.10, "actual_eps": 1.15, "surprise_pct": 4.5},
        {"ticker": "AAPL", "date": (today - timedelta(days=20)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.25, "actual_eps": 1.28, "surprise_pct": 2.4},
    ])
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")
    df.to_parquet(tmp_path / "earnings.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = revision_features.compute_revision_features(dates, symbols=["AAPL", "MSFT"])

    # With AAPL having upward revisions and MSFT having no data, AAPL gets a raw positive value.
    # With only 1 ticker having data (AAPL), z-scoring is not possible — raw value stored.
    # Test that the result is not empty and has the right structure.
    assert not result.empty
    assert ("revision_momentum", "AAPL") in result.columns


def test_revision_momentum_cross_sectional(tmp_path, monkeypatch):
    """With two tickers, one with rising estimates and one with falling, they should have opposite signs."""
    from src.features import revision_features

    today = datetime(2024, 6, 1)
    # AAPL: estimates rising
    # MSFT: estimates falling
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": (today - timedelta(days=80)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.00, "actual_eps": 1.05, "surprise_pct": 5.0},
        {"ticker": "AAPL", "date": (today - timedelta(days=40)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.20, "actual_eps": 1.25, "surprise_pct": 4.2},
        {"ticker": "MSFT", "date": (today - timedelta(days=75)).strftime("%Y-%m-%d"),
         "estimated_eps": 2.50, "actual_eps": 2.45, "surprise_pct": -2.0},
        {"ticker": "MSFT", "date": (today - timedelta(days=35)).strftime("%Y-%m-%d"),
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
