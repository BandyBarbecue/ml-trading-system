import json
import pandas as pd
import numpy as np
import pytest
from src.ml.ic_tracker import compute_ic, get_feature_weights, ICTracker


def make_correlated(n=100, rho=0.4):
    np.random.seed(0)
    x = np.random.randn(n)
    noise = np.random.randn(n)
    y = rho * x + np.sqrt(1 - rho**2) * noise
    return pd.Series(x), pd.Series(y)


def test_compute_ic_positive_correlation():
    x, y = make_correlated(rho=0.5)
    ic = compute_ic(x, y)
    assert ic > 0.2  # should detect meaningful correlation


def test_compute_ic_negative_correlation():
    x, y = make_correlated(rho=-0.5)
    ic = compute_ic(x, y)
    assert ic < -0.2


def test_compute_ic_random_is_near_zero():
    np.random.seed(42)
    x = pd.Series(np.random.randn(200))
    y = pd.Series(np.random.randn(200))
    ic = compute_ic(x, y)
    assert abs(ic) < 0.2


def test_get_feature_weights_sum_to_one():
    tracker = ICTracker()
    # Manually set IC history
    tracker._ic_history = {
        "mom_1m": [0.05, 0.04, 0.06],
        "mom_12m": [0.07, 0.06, 0.08],
        "news_score": [-0.01, 0.00, -0.01],  # near-zero
        "sue": [0.06, 0.05, 0.06],
    }
    weights = get_feature_weights(tracker)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    tracker = ICTracker()
    tracker._ic_history = {"mom_1m": [0.05, 0.06], "sue": [0.03]}
    path = tmp_path / "ic_tracker.json"
    tracker.save(path)
    loaded = ICTracker.load_or_create(path)
    assert loaded._ic_history == tracker._ic_history


def test_load_or_create_starts_fresh_if_no_file(tmp_path):
    tracker = ICTracker.load_or_create(tmp_path / "nonexistent.json")
    assert tracker._ic_history == {}


def test_retirement_flag_near_zero_ic():
    from config import IC_RETIRE_MONTHS, IC_RETIRE_THRESH
    tracker = ICTracker()
    tracker._ic_history = {
        "mom_1m": [IC_RETIRE_THRESH - 0.001] * (IC_RETIRE_MONTHS + 1),
    }
    warnings = tracker.get_retirement_warnings()
    assert "mom_1m" in warnings


# ---------------------------------------------------------------------------
# Task 11a: compute_ic `window` restricts the computation to the most recent
# `window` distinct dates (so IC is dynamic, not a full-history average).
# ---------------------------------------------------------------------------

def make_two_regime_panel(n_dates_per_regime=40, n_stocks=30, seed=1):
    """Build a long-format panel with two regimes:
      - OLD regime (first half of dates): feature is ANTI-correlated with returns.
      - RECENT regime (second half): feature is strongly POSITIVELY correlated.
    A windowed IC over only the recent dates must come out positive; a
    full-history IC would be dragged toward zero/negative by the old regime.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=2 * n_dates_per_regime)
    feats, rets, date_col = [], [], []
    for i, d in enumerate(dates):
        x = rng.standard_normal(n_stocks)
        sign = 1.0 if i >= n_dates_per_regime else -1.0
        y = sign * x + 0.1 * rng.standard_normal(n_stocks)
        feats.append(x)
        rets.append(y)
        date_col.extend([d] * n_stocks)
    feature = pd.Series(np.concatenate(feats))
    fwd_ret = pd.Series(np.concatenate(rets))
    dates_series = pd.Series(date_col)
    return feature, fwd_ret, dates_series, n_dates_per_regime


def test_compute_ic_window_uses_only_recent_dates():
    feature, fwd_ret, dates, n_per_regime = make_two_regime_panel()

    # Windowed to the recent regime only → strongly positive.
    ic_windowed = compute_ic(feature, fwd_ret, dates=dates, window=n_per_regime)
    assert ic_windowed > 0.5

    # Full history (window=None) mixes both regimes → pulled toward zero.
    ic_full = compute_ic(feature, fwd_ret, dates=dates, window=None)
    assert abs(ic_full) < ic_windowed


def test_compute_ic_window_ignored_without_dates():
    """Backward compatibility: without `dates`, window has no effect — the
    result is identical to the plain two-argument call."""
    x, y = make_correlated(rho=0.5)
    assert compute_ic(x, y, window=10) == compute_ic(x, y)


def test_compute_ic_default_window_matches_ic_window_config():
    """The default `window` is config.IC_WINDOW — pin it so a silent change
    to the signature default gets caught."""
    import inspect
    from config import IC_WINDOW
    sig = inspect.signature(compute_ic)
    assert sig.parameters["window"].default == IC_WINDOW


# ---------------------------------------------------------------------------
# Task 11b: ICTracker.update() is idempotent per calendar month — daily calls
# within one month blend into ONE observation instead of appending, so
# "IC_RETIRE_MONTHS consecutive near-zero observations" really means months.
# ---------------------------------------------------------------------------

def test_update_same_month_does_not_append():
    tracker = ICTracker()
    tracker.update("mom_1m", 0.04, as_of="2026-07-01")
    tracker.update("mom_1m", 0.08, as_of="2026-07-02")
    tracker.update("mom_1m", 0.06, as_of="2026-07-15")
    # Three daily calls in July → still exactly ONE monthly observation.
    assert len(tracker._ic_history["mom_1m"]) == 1
    # In-place running mean of the month's values: (0.04+0.08+0.06)/3 = 0.06
    assert tracker._ic_history["mom_1m"][0] == pytest.approx(0.06)


def test_update_new_month_appends():
    tracker = ICTracker()
    tracker.update("mom_1m", 0.04, as_of="2026-06-30")
    tracker.update("mom_1m", 0.08, as_of="2026-07-01")
    assert len(tracker._ic_history["mom_1m"]) == 2
    assert tracker._ic_history["mom_1m"] == [0.04, 0.08]


def test_daily_updates_do_not_trigger_premature_retirement():
    """The original bug: 3 near-zero DAILY updates flagged a feature as
    '3 months near-zero'. With monthly idempotency, a whole month of bad
    days is only ONE observation — no warning until the month count is real."""
    from config import IC_RETIRE_MONTHS, IC_RETIRE_THRESH
    tracker = ICTracker()
    bad_ic = IC_RETIRE_THRESH / 2  # near-zero
    # IC_RETIRE_MONTHS(+2) consecutive bad DAYS, all inside one month.
    for day in range(1, IC_RETIRE_MONTHS + 3):
        tracker.update("mom_1m", bad_ic, as_of=f"2026-07-{day:02d}")
    assert tracker.get_retirement_warnings() == [], (
        "A few bad days within ONE month must not count as months of decay."
    )
    # But genuinely bad MONTHS still trigger the warning.
    for month in range(8, 8 + IC_RETIRE_MONTHS):
        tracker.update("mom_1m", bad_ic, as_of=f"2026-{month:02d}-01")
    assert "mom_1m" in tracker.get_retirement_warnings()


def test_update_defaults_to_today_when_as_of_omitted():
    """Omitting as_of uses the real current date — two same-day calls must
    collapse into one observation (this is the daily_run.py call pattern)."""
    tracker = ICTracker()
    tracker.update("mom_1m", 0.02)
    tracker.update("mom_1m", 0.04)
    assert len(tracker._ic_history["mom_1m"]) == 1
    assert tracker._ic_history["mom_1m"][0] == pytest.approx(0.03)


def test_save_load_preserves_month_state(tmp_path):
    """Month-tracking state must survive a save/load cycle: daily_run.py
    reloads the tracker from disk every morning, so if this state were lost,
    every day would look like a 'new month' and the old bug would return."""
    path = tmp_path / "ic_tracker.json"
    tracker = ICTracker()
    tracker.update("mom_1m", 0.04, as_of="2026-07-01")
    tracker.save(path)

    reloaded = ICTracker.load_or_create(path)
    reloaded.update("mom_1m", 0.08, as_of="2026-07-02")
    # Same month across a reload → still one observation, mean updated.
    assert len(reloaded._ic_history["mom_1m"]) == 1
    assert reloaded._ic_history["mom_1m"][0] == pytest.approx(0.06)


def test_load_or_create_reads_legacy_format(tmp_path):
    """Files written before month-tracking existed were a bare
    {feature: [ic, ...]} dict. They must still load."""
    path = tmp_path / "ic_tracker.json"
    path.write_text(json.dumps({"mom_1m": [0.05, 0.06], "sue": [0.03]}))
    tracker = ICTracker.load_or_create(path)
    assert tracker._ic_history == {"mom_1m": [0.05, 0.06], "sue": [0.03]}
    assert tracker._last_update_month == {}
