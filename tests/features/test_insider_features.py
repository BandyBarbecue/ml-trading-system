import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
import config


def test_buy_scores_positive_raw(tmp_path, monkeypatch):
    """With one ticker buying, insider_score should be positive (stored as raw, no z-score)."""
    from src.features import insider_features

    today = datetime(2024, 6, 1)
    df = pd.DataFrame([{
        "ticker": "AAPL", "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
        "insider_type": "ceo", "transaction_type": "P",
        "shares": 10000, "value_usd": 1_850_000.0,
    }])
    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")
    df.to_parquet(tmp_path / "insider.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = insider_features.compute_insider_features(dates, symbols=["AAPL"])
    assert not result.empty
    val = result.loc[today, ("insider_score", "AAPL")]
    assert val > 0  # net buy = positive


def test_cross_sectional_buy_vs_sell(tmp_path, monkeypatch):
    """AAPL buying and MSFT selling should result in AAPL > 0 and MSFT < 0 after z-scoring."""
    from src.features import insider_features

    today = datetime(2024, 6, 1)
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
         "insider_type": "ceo", "transaction_type": "P",
         "shares": 10000, "value_usd": 1_850_000.0},
        {"ticker": "MSFT", "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
         "insider_type": "cfo", "transaction_type": "S",
         "shares": 5000, "value_usd": 1_500_000.0},
    ])
    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")
    df.to_parquet(tmp_path / "insider.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = insider_features.compute_insider_features(dates, symbols=["AAPL", "MSFT"])

    aapl_val = result.loc[today, ("insider_score", "AAPL")]
    msft_val = result.loc[today, ("insider_score", "MSFT")]
    assert aapl_val > 0
    assert msft_val < 0


def test_transaction_lagged_by_two_business_days(tmp_path, monkeypatch):
    """A trade must not count on its own transaction date — Form 4 filings
    become public ~2 business days later. It should start counting once its
    effective (filing) date has arrived.
    """
    from src.features import insider_features

    trans_date = datetime(2024, 6, 3)  # Monday
    df = pd.DataFrame([{
        "ticker": "AAPL", "date": trans_date.strftime("%Y-%m-%d"),
        "insider_type": "ceo", "transaction_type": "P",
        "shares": 10000, "value_usd": 1_850_000.0,
    }])
    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")
    df.to_parquet(tmp_path / "insider.parquet", index=False)

    same_day  = trans_date                        # Mon: transaction date itself
    one_biz_day_later = trans_date + timedelta(days=1)   # Tue: filing still not public
    effective = trans_date + timedelta(days=2)     # Wed: +2 business days, now public

    dates = pd.DatetimeIndex([same_day, one_biz_day_later, effective])
    result = insider_features.compute_insider_features(dates, symbols=["AAPL"])

    assert np.isnan(result.loc[same_day, ("insider_score", "AAPL")]), (
        "Trade must not be visible on its own transaction date"
    )
    assert np.isnan(result.loc[one_biz_day_later, ("insider_score", "AAPL")]), (
        "Trade must not be visible before the +2 business day filing lag elapses"
    )
    val = result.loc[effective, ("insider_score", "AAPL")]
    assert not np.isnan(val) and val > 0, (
        "Trade must be visible once its effective (filing) date is reached"
    )


def test_empty_insider_file_returns_empty(tmp_path, monkeypatch):
    from src.features import insider_features

    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "no_insider.parquet")
    result = insider_features.compute_insider_features(
        pd.DatetimeIndex([datetime(2024, 6, 1)]), symbols=["AAPL"]
    )
    assert result.empty
