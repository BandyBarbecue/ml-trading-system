import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import config


def _make_fundamentals(ticker: str, filing_date: str, gp: float, pb: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker, "date": filing_date,
        "gross_profitability": gp, "pb_ratio": pb,
    }])


def test_45_day_lag_enforced(tmp_path, monkeypatch):
    """Filing dated 30 days ago should NOT appear; 60 days ago SHOULD appear.

    The recent (too-fresh) filing is given a GP *below* MSFT's while the
    valid old filing is given a GP *above* MSFT's — so if the 45-day lag
    were broken and the leaked 30-day filing got used instead, AAPL's
    z-score would flip negative. This makes the assertion on the public
    API's actual output (not just the internal _latest_valid() helper)
    a real check of the lag being enforced, not just a shape check.
    """
    from src.features import fundamental_features

    today = datetime(2024, 6, 1)
    recent_filing = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    old_filing    = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    df = pd.concat([
        _make_fundamentals("AAPL", recent_filing, 0.05, 40.0),  # too fresh — must be ignored
        _make_fundamentals("AAPL", old_filing,    0.44, 38.0),  # valid — should be used
        _make_fundamentals("MSFT", old_filing,    0.30, 20.0),
    ])
    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "f.parquet")
    df.to_parquet(tmp_path / "f.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = fundamental_features.compute_fundamental_features(dates, symbols=["AAPL", "MSFT"])

    # The recent filing (30 days) must not be used; only old_filing (60 days) is valid
    raw = fundamental_features._latest_valid(df, "AAPL", today, lag_days=45)
    assert raw is not None
    assert abs(raw["gross_profitability"] - 0.44) < 1e-6  # old_filing used, not recent

    # Now assert on the actual result from the public API: AAPL's old_filing
    # GP (0.44) is above MSFT's (0.30), so AAPL's cross-sectional z-score
    # must be positive and MSFT's negative. Leaking the recent filing (0.05,
    # below MSFT) would flip AAPL's sign negative — this catches that.
    gp_aapl = result.loc[today, ("gross_profitability", "AAPL")]
    gp_msft = result.loc[today, ("gross_profitability", "MSFT")]
    assert not np.isnan(gp_aapl)
    assert gp_aapl > 0
    assert gp_msft < 0


def test_z_scores_cross_sectional(tmp_path, monkeypatch):
    """With two tickers, z-scores should differ across tickers on the same date."""
    from src.features import fundamental_features

    today = datetime(2024, 6, 1)
    old_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    df = pd.concat([
        _make_fundamentals("AAPL", old_date, 0.60, 40.0),
        _make_fundamentals("MSFT", old_date, 0.30, 20.0),
    ])
    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "f.parquet")
    df.to_parquet(tmp_path / "f.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = fundamental_features.compute_fundamental_features(dates, symbols=["AAPL", "MSFT"])

    gp_aapl = result.loc[today, ("gross_profitability", "AAPL")]
    gp_msft = result.loc[today, ("gross_profitability", "MSFT")]
    assert abs(gp_aapl) > 0.1
    assert abs(gp_msft) > 0.1
    assert gp_aapl * gp_msft < 0  # opposite signs (one positive, one negative)


def test_missing_file_returns_empty(tmp_path, monkeypatch):
    from src.features import fundamental_features

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "no_file.parquet")
    dates = pd.DatetimeIndex([datetime(2024, 6, 1)])
    result = fundamental_features.compute_fundamental_features(dates, symbols=["AAPL"])
    assert result.empty
