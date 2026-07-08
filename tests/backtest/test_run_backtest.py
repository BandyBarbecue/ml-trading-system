"""Tests for src/backtest/run_backtest.py

These tests use synthetic data and mocked ML / portfolio / risk calls so the
suite runs fast and requires no real market data.

Concepts tested
---------------
- Equity curve has the right length (covers all test-window trading days).
- NO data leakage: training max date < test min date for every fold.
- The returned dict contains all required keys.
- Portfolio starts at initial_value.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.backtest.run_backtest import run_backtest, compute_metrics, _max_drawdown, TEST_DAYS


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def make_feature_matrix(
    n_dates: int = 600,
    n_tickers: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal feature matrix suitable for run_backtest().

    The matrix contains:
      - ``date``   : business days starting 2018-01-01
      - ``ticker`` : T0 … T(n_tickers-1)
      - feature columns that match src.ml.model.FEATURE_COLS
      - ``target`` : random float in [0, 1]

    The minimum size required by run_backtest() is
    (MODEL_TRAIN_DAYS + TEST_DAYS) × n_tickers rows.
    With n_dates=600 and 5 tickers that is 3 000 rows — well above the minimum.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                # All feature columns that FEATURE_COLS expects
                "mom_1m":              float(rng.standard_normal()),
                "mom_3m":              float(rng.standard_normal()),
                "mom_12m":             float(rng.standard_normal()),
                "rev_1w":              float(rng.standard_normal()),
                "vol_20d":             float(abs(rng.standard_normal())),
                "news_score":          float(rng.uniform(-1, 1)),
                "supply_chain_score":  float(rng.uniform(-1, 1)),
                "sue":                 float(rng.standard_normal()),
                # target: cross-sectional rank proxy, must be in [0, 1]
                "target": float(rng.uniform(0, 1)),
            })

    return pd.DataFrame(rows)


def make_sector_map(feature_matrix: pd.DataFrame) -> dict[str, str]:
    """Assign a trivial sector to every ticker in the matrix.

    Uses FOUR dummy sectors, not two: with MAX_SECTOR_PCT = 0.30 the sector
    cap is only satisfiable when there are at least ceil(1 / 0.30) = 4
    sectors — k sectors can hold at most k x 30% of the book, so a 2-sector
    universe can never sum to 100% under a 30% cap. _apply_sector_cap now
    asserts convergence instead of silently renormalising the cap away, so
    an infeasible fixture universe would crash every rebalance.
    """
    tickers = feature_matrix["ticker"].unique().tolist()
    sectors = ["SectorA", "SectorB", "SectorC", "SectorD"]
    return {t: sectors[i % 4] for i, t in enumerate(tickers)}


def make_close_prices(feature_matrix: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic (date x ticker) close-price panel for run_backtest()'s
    `close_prices` parameter, covering every date/ticker in feature_matrix.

    run_backtest() now computes daily P&L from real close-to-close returns
    (see src/backtest/run_backtest.py) instead of the `target` column, so it
    needs an actual price panel. In production this comes from
    data/prices/<TICKER>.parquet; in tests we inject a synthetic geometric
    random walk instead — the exact price path doesn't matter, only that it
    is genuine day-over-day price data.
    """
    rng = np.random.default_rng(seed)
    dates = sorted(feature_matrix["date"].unique())
    tickers = sorted(feature_matrix["ticker"].unique())
    daily_rets = rng.normal(loc=0.0003, scale=0.01, size=(len(dates), len(tickers)))
    prices = 100.0 * np.cumprod(1 + daily_rets, axis=0)
    return pd.DataFrame(prices, index=pd.DatetimeIndex(dates), columns=tickers)


# ---------------------------------------------------------------------------
# Sentinel objects for mock train / predict
# ---------------------------------------------------------------------------

class _SentinelModel:
    """A placeholder that proves the mock train() was called."""
    pass


def _mock_train(feature_matrix, feature_weights=None, save=False):
    """Records the training dates so leakage tests can inspect them."""
    return _SentinelModel()


def _mock_predict(model, feature_matrix):
    """Returns equal scores for all tickers (neutral prediction)."""
    tickers = feature_matrix["ticker"].values
    return pd.Series(
        np.full(len(tickers), 0.5),
        index=tickers,
        name="predicted_score",
    )


# ---------------------------------------------------------------------------
# Test: equity curve has the correct length
# ---------------------------------------------------------------------------

class TestEquityCurveLength:
    """The equity curve should cover exactly the trading days in the test windows."""

    def test_equity_curve_has_correct_length(self, monkeypatch, tmp_path):
        """Equity curve length equals the number of trading days across all test windows."""
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        # Patch train and predict so the test doesn't need LightGBM to run.
        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        result = run_backtest(
            fm, sector_map, initial_value=100_000.0, log_path=tmp_path / "decisions.jsonl",
            close_prices=make_close_prices(fm),
        )
        eq = result["equity_curve"]

        # Calculate expected number of test-window days.
        all_dates = sorted(fm["date"].unique())
        n_dates = len(all_dates)
        train_days = config.MODEL_TRAIN_DAYS
        expected_test_days = 0
        i = train_days
        while i + TEST_DAYS <= n_dates:
            expected_test_days += TEST_DAYS
            i += TEST_DAYS

        assert isinstance(eq, pd.Series), "equity_curve must be a pd.Series"
        assert len(eq) == expected_test_days, (
            f"Expected {expected_test_days} days in equity curve, got {len(eq)}"
        )


# ---------------------------------------------------------------------------
# Test: NO data leakage
# ---------------------------------------------------------------------------

class TestNoDataLeakage:
    """Training slice must end strictly before the test slice begins every fold."""

    def test_no_data_leakage(self, monkeypatch, tmp_path):
        """For every training call the max training date < min test date."""
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        # Record (max_train_date, min_test_date) each time train() is called.
        leakage_log: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        all_dates = sorted(fm["date"].unique())
        train_days = config.MODEL_TRAIN_DAYS
        call_index = [0]  # mutable counter accessible inside closure

        def recording_train(feature_matrix_slice, feature_weights=None, save=False):
            """Record the date range of every training slice."""
            fold_i = call_index[0]
            call_index[0] += 1

            # Determine the test start for this fold from the global date list.
            i = train_days + fold_i * TEST_DAYS
            if i + TEST_DAYS <= len(all_dates):
                test_start = all_dates[i]
                train_max = feature_matrix_slice["date"].max()
                leakage_log.append((train_max, test_start))

            return _SentinelModel()

        monkeypatch.setattr("src.backtest.run_backtest.train", recording_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        run_backtest(
            fm, sector_map, initial_value=100_000.0, log_path=tmp_path / "decisions.jsonl",
            close_prices=make_close_prices(fm),
        )

        assert len(leakage_log) > 0, "No training calls were recorded — check test setup."

        for train_max, test_start in leakage_log:
            assert train_max < test_start, (
                f"LEAKAGE: training max date {train_max} is NOT strictly less than "
                f"test start {test_start}."
            )


# ---------------------------------------------------------------------------
# Test: NO look-ahead in the P&L path (real returns + one-day fill lag)
# ---------------------------------------------------------------------------

class TestNoPriceLookaheadInPnL:
    """The equity curve at day T must be identical no matter what prices do
    strictly AFTER day T.

    This is the P&L-path counterpart to TestNoDataLeakage above, which only
    covers the model's train/test split. It catches two distinct bugs a
    model-split-only check cannot see:
      1. Using the (20-day-forward) `target` column as "today's return" —
         that literally reads future prices into today's equity value.
      2. Same-day fills — earning day T's return on weights decided from
         day T's own close, instead of lagging them to day T+1.
    If either bug were reintroduced, changing prices after day T would
    change the equity value AT day T (or earlier), because the future price
    change would (directly or indirectly, via a different `target` rank)
    feed into a decision or return that lands on/before T.
    """

    def test_perturbing_future_prices_does_not_change_past_equity(self, monkeypatch, tmp_path):
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        all_dates = sorted(fm["date"].unique())
        # T is a day inside the first test window (which starts at index
        # MODEL_TRAIN_DAYS = 504), with plenty of test-window days both
        # before and after it.
        t_index = config.MODEL_TRAIN_DAYS + 15
        day_t = all_dates[t_index]

        prices_a = make_close_prices(fm, seed=7)
        prices_b = prices_a.copy()
        # Perturb prices STRICTLY AFTER day_t only — a large, obviously
        # different shock, so any leakage would show up unmistakably.
        future_mask = prices_b.index > day_t
        prices_b.loc[future_mask] = prices_b.loc[future_mask] * 5.0 + 17.0

        result_a = run_backtest(
            fm, sector_map, initial_value=100_000.0,
            log_path=tmp_path / "decisions_a.jsonl", close_prices=prices_a,
        )
        result_b = run_backtest(
            fm, sector_map, initial_value=100_000.0,
            log_path=tmp_path / "decisions_b.jsonl", close_prices=prices_b,
        )

        eq_a = result_a["equity_curve"]
        eq_b = result_b["equity_curve"]

        # Every day up to and including day_t must be untouched by the
        # future price shock.
        past_and_present = eq_a.index[eq_a.index <= day_t]
        assert len(past_and_present) > 1, "Test setup produced no pre-T equity history."
        pd.testing.assert_series_equal(
            eq_a.loc[past_and_present],
            eq_b.loc[past_and_present],
            check_exact=False,
            atol=1e-6,
        )

        # Sanity check the perturbation actually DOES matter later on —
        # otherwise this test would trivially pass even with the old bugs.
        after = eq_a.index[eq_a.index > day_t]
        assert len(after) > 0, "Test setup produced no post-T equity history."
        assert not np.allclose(eq_a.loc[after].values, eq_b.loc[after].values), (
            "Perturbing future prices had no effect anywhere — the P&L path "
            "may not be using close_prices at all."
        )


# ---------------------------------------------------------------------------
# Test: returned dict has all required keys
# ---------------------------------------------------------------------------

class TestMetricsShape:
    """The returned dict must contain all documented keys."""

    REQUIRED_KEYS = {"equity_curve", "sharpe", "max_drawdown", "total_return", "n_rebalances"}

    def test_metrics_shapes(self, monkeypatch, tmp_path):
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        result = run_backtest(
            fm, sector_map, initial_value=100_000.0, log_path=tmp_path / "decisions.jsonl",
            close_prices=make_close_prices(fm),
        )

        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in result dict: {missing}"

    def test_metric_types(self, monkeypatch, tmp_path):
        """Spot-check that each value has the expected Python type."""
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        result = run_backtest(
            fm, sector_map, initial_value=100_000.0, log_path=tmp_path / "decisions.jsonl",
            close_prices=make_close_prices(fm),
        )

        assert isinstance(result["equity_curve"], pd.Series)
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["max_drawdown"], float)
        assert isinstance(result["total_return"], float)
        assert isinstance(result["n_rebalances"], int)


# ---------------------------------------------------------------------------
# Test: portfolio starts at initial_value
# ---------------------------------------------------------------------------

class TestPositiveInitialValue:
    """The first entry in the equity curve must equal initial_value."""

    def test_initial_value_respected(self, monkeypatch, tmp_path):
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        start_capital = 50_000.0
        result = run_backtest(
            fm, sector_map, initial_value=start_capital,
            log_path=tmp_path / "decisions.jsonl",
            close_prices=make_close_prices(fm),
        )

        # The equity curve's first value is recorded on the first test-window day.
        # Before any rebalance fires the portfolio value is still start_capital.
        first_value = result["equity_curve"].iloc[0]
        assert abs(first_value - start_capital) < 1.0, (
            f"Portfolio first value should be close to {start_capital}, got {first_value}"
        )

    def test_default_initial_value(self, monkeypatch, tmp_path):
        """Omitting initial_value defaults to 100_000."""
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        result = run_backtest(
            fm, sector_map, log_path=tmp_path / "decisions.jsonl",
            close_prices=make_close_prices(fm),
        )

        assert abs(result["equity_curve"].iloc[0] - 100_000.0) < 1.0


# ---------------------------------------------------------------------------
# Test: compute_metrics unit tests (no mocking needed)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Unit tests for the standalone metrics helpers."""

    def test_flat_curve_zero_return(self):
        """A flat equity curve should produce total_return ~0 and sharpe=0."""
        dates = pd.bdate_range("2022-01-03", periods=50)
        curve = pd.Series(100_000.0, index=dates)
        m = compute_metrics(curve)
        assert abs(m["total_return"]) < 1e-9
        assert m["sharpe"] == 0.0    # std of daily_returns is 0 → no division
        assert m["max_drawdown"] == 0.0

    def test_monotonically_rising_curve(self):
        """A curve that rises every day has zero max drawdown and positive return."""
        dates = pd.bdate_range("2022-01-03", periods=100)
        curve = pd.Series(np.linspace(100_000, 120_000, 100), index=dates)
        m = compute_metrics(curve)
        assert m["total_return"] == pytest.approx(0.20, abs=1e-3)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_known_case(self):
        """Manually verifiable drawdown: drop from 100 → 80 = 20% drawdown."""
        values = [100, 110, 90, 80, 85, 95, 105]
        curve = pd.Series(
            [float(v) for v in values],
            index=pd.bdate_range("2022-01-03", periods=len(values)),
        )
        dd = _max_drawdown(curve)
        # Peak is 110, trough is 80 → drawdown = (110-80)/110 ≈ 0.2727
        expected = (110 - 80) / 110
        assert abs(dd - expected) < 1e-6

    def test_total_return_positive(self):
        """Curve ending higher than it started → positive total return."""
        dates = pd.bdate_range("2022-01-03", periods=50)
        # Starts at 100k, ends at 115k.
        values = np.linspace(100_000, 115_000, 50)
        curve = pd.Series(values, index=dates)
        m = compute_metrics(curve)
        assert m["total_return"] > 0


# ---------------------------------------------------------------------------
# Test: missing 'target' column raises immediately
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Bad inputs should raise ValueError early, not silently mis-behave."""

    def test_missing_target_column_raises(self, monkeypatch, tmp_path):
        fm = make_feature_matrix(n_dates=600, n_tickers=5)
        fm = fm.drop(columns=["target"])  # deliberately break the matrix
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        with pytest.raises(ValueError, match="target"):
            run_backtest(fm, sector_map, log_path=tmp_path / "decisions.jsonl")

    def test_insufficient_data_raises(self, monkeypatch, tmp_path):
        """If the feature matrix is too short, a clear error should be raised."""
        # Only 100 dates — far fewer than MODEL_TRAIN_DAYS + TEST_DAYS = 525.
        fm = make_feature_matrix(n_dates=100, n_tickers=5)
        sector_map = make_sector_map(fm)

        monkeypatch.setattr("src.backtest.run_backtest.train", _mock_train)
        monkeypatch.setattr("src.backtest.run_backtest.predict", _mock_predict)

        with pytest.raises(ValueError, match="MODEL_TRAIN_DAYS"):
            run_backtest(fm, sector_map, log_path=tmp_path / "decisions.jsonl")
