"""
Tests for feature_matrix.py — the module that assembles the final ML table,
including the forward-return target. This module previously had ZERO test
coverage, which is dangerous: the target is the one column where a
look-ahead bug would be *easy* to introduce and hardest to notice (it's
supposed to look forward — the risk is it also leaks into the features).

We use small, fully synthetic, deterministic price series (linear growth,
not random) so the expected cross-sectional rank of the forward return can
be computed by hand instead of re-deriving pandas' own rank() logic.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src.features.feature_matrix import build_feature_matrix


N_DAYS = 400
START = "2023-01-02"


def _make_synthetic_prices(n_days: int = N_DAYS, start: str = START):
    """Two tickers with different, deterministic linear growth rates.

    AAA grows 1.0/day, BBB grows 0.5/day — AAA's forward return is always
    larger than BBB's, so the expected cross-sectional rank (with only 2
    tickers) is always AAA=1.0, BBB=0.5. This lets test (i) below assert an
    exact expected value instead of just "not NaN".
    """
    dates = pd.bdate_range(start, periods=n_days)
    aaa = 100.0 + np.arange(n_days) * 1.0
    bbb = 100.0 + np.arange(n_days) * 0.5
    prices = {
        "AAA": pd.DataFrame({"close": aaa}, index=dates),
        "BBB": pd.DataFrame({"close": bbb}, index=dates),
    }
    return dates, prices


def _write_prices(dir_path, prices: dict[str, pd.DataFrame]) -> None:
    """Write each ticker's price DataFrame to <dir_path>/<TICKER>.parquet,
    matching the layout load_all_prices() expects (index=date, preserved)."""
    for ticker, df in prices.items():
        df.to_parquet(dir_path / f"{ticker}.parquet")


def _isolate_other_feature_sources(tmp_path, monkeypatch) -> None:
    """Redirect every non-price feature source (earnings, NLP, fundamentals,
    revisions, insider) to files/dirs that don't exist, so this test only
    exercises the price-feature + target logic in feature_matrix.py — the
    same monkeypatch-module-attribute pattern used throughout tests/features
    and tests/data (e.g. tests/data/test_fetch_prices.py, which patches
    fetch_prices.PRICES_DIR directly rather than config.PRICES_DIR).
    """
    from src.features import earnings_features, price_features
    from src.data import fetch_news

    # earnings_features binds `EARNINGS_FILE` at import time (`from config
    # import EARNINGS_FILE`), so config.EARNINGS_FILE alone won't redirect
    # it — patch both the module's own copy and the config source, since
    # revision_features.py reads `config.EARNINGS_FILE` live instead.
    monkeypatch.setattr(earnings_features, "EARNINGS_FILE", tmp_path / "no_earnings.parquet")
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "no_earnings.parquet")
    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "no_fundamentals.parquet")
    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "no_insider.parquet")
    # fetch_news binds `NEWS_DIR` at import time too.
    monkeypatch.setattr(fetch_news, "NEWS_DIR", tmp_path / "no_news")
    # price_features binds `PRICES_DIR` at import time; point it at tmp_path,
    # which is where _write_prices() puts the synthetic parquet files.
    monkeypatch.setattr(price_features, "PRICES_DIR", tmp_path)


def test_target_equals_cross_sectional_rank_of_forward_return(tmp_path, monkeypatch):
    """Target for day T must equal the cross-sectional rank of the
    T -> T+TARGET_HORIZON forward return, computed independently by hand."""
    dates, prices = _make_synthetic_prices()
    _write_prices(tmp_path, prices)
    _isolate_other_feature_sources(tmp_path, monkeypatch)

    T_idx = 300  # deep enough for mom_12m's 273-day warm-up to be satisfied
    T = dates[T_idx]
    horizon = config.TARGET_HORIZON

    fm = build_feature_matrix(["AAA", "BBB"], dates)
    assert not fm.empty

    # Hand-computed forward returns (NOT reusing pandas pct_change/rank):
    close_aaa_T   = 100.0 + T_idx * 1.0
    close_aaa_fut = 100.0 + (T_idx + horizon) * 1.0
    ret_aaa = (close_aaa_fut - close_aaa_T) / close_aaa_T

    close_bbb_T   = 100.0 + T_idx * 0.5
    close_bbb_fut = 100.0 + (T_idx + horizon) * 0.5
    ret_bbb = (close_bbb_fut - close_bbb_T) / close_bbb_T

    assert ret_aaa > ret_bbb  # sanity check on the synthetic setup itself

    # With exactly 2 tickers, percentile rank puts the winner at 1.0 and the
    # loser at 0.5 (rank 2/2 and 1/2 respectively).
    row_aaa = fm[(fm["date"] == T) & (fm["ticker"] == "AAA")].iloc[0]
    row_bbb = fm[(fm["date"] == T) & (fm["ticker"] == "BBB")].iloc[0]

    assert abs(row_aaa["target"] - 1.0) < 1e-9
    assert abs(row_bbb["target"] - 0.5) < 1e-9


def test_final_target_horizon_dates_have_nan_target(tmp_path, monkeypatch):
    """The last TARGET_HORIZON trading days have no future data available,
    so their target must be NaN — never a fabricated number."""
    dates, prices = _make_synthetic_prices()
    _write_prices(tmp_path, prices)
    _isolate_other_feature_sources(tmp_path, monkeypatch)

    fm = build_feature_matrix(["AAA", "BBB"], dates)
    assert not fm.empty

    tail_dates = dates[-config.TARGET_HORIZON:]
    tail_rows = fm[fm["date"].isin(tail_dates)]

    # Every ticker/date combination in the trailing window must be NaN.
    assert len(tail_rows) > 0
    assert tail_rows["target"].isna().all(), (
        "The final TARGET_HORIZON dates must have NaN target (future unknown)"
    )

    # And the day right before the trailing window must NOT be NaN, so this
    # test is actually pinning a boundary, not a blanket NaN bug.
    day_before_tail = dates[-config.TARGET_HORIZON - 1]
    row = fm[(fm["date"] == day_before_tail) & (fm["ticker"] == "AAA")]
    assert not row.empty
    assert not np.isnan(row.iloc[0]["target"])


def test_perturbing_future_prices_changes_only_targets(tmp_path, monkeypatch):
    """Perturbing a price AFTER day T must change T's target (which is
    supposed to look forward) but must NEVER change any feature value for
    day T (which must only look backward). This is the core no-look-ahead
    invariant for the assembled feature matrix.
    """
    dates, prices = _make_synthetic_prices()
    _write_prices(tmp_path, prices)
    _isolate_other_feature_sources(tmp_path, monkeypatch)

    symbols = ["AAA", "BBB"]
    fm_before = build_feature_matrix(symbols, dates)

    T_idx = 300
    T = dates[T_idx]
    horizon = config.TARGET_HORIZON
    # Perturb the EXACT future price that T's forward return depends on
    # (T + TARGET_HORIZON), so we know the target must change too — this
    # keeps the test from vacuously passing if targets happened not to move.
    perturb_idx = T_idx + horizon

    # AAA's forward return already beats BBB's (target=1.0), so merely
    # *increasing* AAA's future price further wouldn't change its rank —
    # decrease it instead, enough to flip the ranking (AAA's forward return
    # drops below BBB's), which forces target to actually move.
    perturbed_aaa = prices["AAA"].copy()
    perturbed_aaa.iloc[perturb_idx, 0] = perturbed_aaa.iloc[perturb_idx, 0] - 50.0
    perturbed_aaa.to_parquet(tmp_path / "AAA.parquet")

    fm_after = build_feature_matrix(symbols, dates)

    feature_cols = ["mom_1m", "mom_3m", "mom_12m", "rev_1w", "vol_20d"]

    for ticker in symbols:
        row_before = fm_before[(fm_before["date"] == T) & (fm_before["ticker"] == ticker)].iloc[0]
        row_after  = fm_after[(fm_after["date"] == T) & (fm_after["ticker"] == ticker)].iloc[0]

        # All backward-looking price features at T must be bit-for-bit
        # identical, no matter which ticker's future price we perturbed.
        for col in feature_cols:
            b, a = row_before[col], row_after[col]
            if pd.isna(b) and pd.isna(a):
                continue
            assert abs(b - a) < 1e-9, (
                f"{col} for {ticker} at day T changed after perturbing a future price "
                f"({perturb_idx - T_idx} days ahead) — this is a look-ahead leak"
            )

    # The target legitimately depends on the future, so it MUST change for
    # the perturbed ticker (AAA).
    target_before = fm_before[(fm_before["date"] == T) & (fm_before["ticker"] == "AAA")].iloc[0]["target"]
    target_after  = fm_after[(fm_after["date"] == T) & (fm_after["ticker"] == "AAA")].iloc[0]["target"]
    assert not np.isclose(target_before, target_after), (
        "Perturbing the exact future price used in T's forward return should change the target"
    )
