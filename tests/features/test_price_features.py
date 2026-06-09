import pandas as pd
import numpy as np
import pytest
from src.features.price_features import (
    compute_momentum,
    compute_reversal,
    compute_vol,
    cross_section_zscore,
    compute_price_features,
)


def make_prices(n_days: int = 300, n_tickers: int = 5) -> dict[str, pd.DataFrame]:
    """Synthetic price data: random walks for each ticker."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    prices = {}
    for i, ticker in enumerate(["A", "B", "C", "D", "E"][:n_tickers]):
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n_days)))
        prices[ticker] = pd.DataFrame({"close": close}, index=dates)
    return prices


def test_momentum_skip_period():
    prices = make_prices()
    mom = compute_momentum(prices, window=21, skip=5)
    assert isinstance(mom, pd.DataFrame)
    # First (21+5) rows should be NaN
    assert mom.iloc[:25].isna().all().all()
    assert not mom.iloc[30:].isna().all().all()


def test_reversal_is_negative_5d_return():
    prices = make_prices()
    rev = compute_reversal(prices, window=5)
    # Check sign: reversal_1w = -(5-day return)
    closes = prices["A"]["close"]
    manual = -(closes / closes.shift(5) - 1)
    pd.testing.assert_series_equal(
        rev["A"].dropna(), manual.dropna(), check_names=False, rtol=1e-6
    )


def test_vol_is_std_of_returns():
    prices = make_prices()
    vol = compute_vol(prices, window=20)
    closes = prices["A"]["close"]
    manual_vol = closes.pct_change().rolling(20).std()
    pd.testing.assert_series_equal(
        vol["A"].dropna(), manual_vol.dropna(), check_names=False, rtol=1e-6
    )


def test_cross_section_zscore_unit_normal():
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    zscored = cross_section_zscore(df)
    row_means = zscored.mean(axis=1)
    assert (row_means.abs() < 1e-10).all()


def test_compute_price_features_shape():
    prices = make_prices(n_days=300, n_tickers=5)
    df = compute_price_features(prices)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Columns should be a 2-level MultiIndex: (feature, ticker)
    assert isinstance(df.columns, pd.MultiIndex)
    # All values should be z-scored: check no extreme values.
    # stack() on MultiIndex columns yields a DataFrame; use .values to get a
    # scalar max regardless of pandas version.
    assert float(df.stack(future_stack=True).abs().max().max()) < 10
