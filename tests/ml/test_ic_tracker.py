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


# ---------------------------------------------------------------------------
# compute_ic None-vs-0.0 contract: "no data" must be distinguishable from a
# genuinely-measured zero IC (see ic_tracker.py's docstring for the "why").
# ---------------------------------------------------------------------------

def test_compute_ic_returns_none_for_insufficient_rows():
    """Fewer than 10 overlapping (feature, target) rows -> None, not 0.0."""
    x = pd.Series(np.random.randn(5))
    y = pd.Series(np.random.randn(5))
    assert compute_ic(x, y) is None


def test_compute_ic_returns_none_when_no_overlapping_observations():
    """The exact bug scenario: a feature and target that share no index
    (e.g. a feature that has never had a matching target date) must come
    back as None ('never measured'), not 0.0 ('measured, no relationship')."""
    x = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    y = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    assert compute_ic(x, y) is None


def test_get_feature_weights_sum_to_one():
    tracker = ICTracker()
    # Manually set IC history
    tracker._ic_history = {
        "mom_1m": [0.05, 0.04, 0.06],
        "mom_12m": [0.07, 0.06, 0.08],
        "news_score": [-0.01, 0.00, -0.01],  # near-zero
        "sue": [0.06, 0.05, 0.06],
    }
    weights, no_data = get_feature_weights(tracker)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert no_data == []


# ---------------------------------------------------------------------------
# get_feature_weights: features with NO measured (non-None) IC ever must be
# excluded from the normalisation and reported separately, and the weight
# basis for the rest must be the latest NON-None entry (skipping trailing
# no-data months).
# ---------------------------------------------------------------------------

def test_get_feature_weights_excludes_all_none_history():
    tracker = ICTracker()
    tracker._ic_history = {
        "mom_1m": [0.05, 0.04, 0.06],
        "news_score": [None, None, None],  # never had overlapping data
    }
    weights, no_data = get_feature_weights(tracker)
    assert "news_score" not in weights
    assert no_data == ["news_score"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_get_feature_weights_uses_latest_non_none_ic():
    """A feature whose most recent month has no data (None) but DOES have
    older measured months must still be weighted using the latest measured
    value, not silently dropped."""
    tracker = ICTracker()
    tracker._ic_history = {
        "mom_1m": [0.05, 0.04, None],
    }
    weights, no_data = get_feature_weights(tracker)
    assert "mom_1m" in weights
    assert no_data == []


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


def test_load_or_create_reads_legacy_zero_fallback_unchanged():
    """Legacy files may contain literal 0.0 entries written by the OLD
    compute_ic fallback (before this fix). No migration: those zeros must
    load exactly as-is, not be reinterpreted as None."""
    tracker = ICTracker()
    tracker._ic_history = {"news_score": [0.0, 0.0, 0.0]}
    # Sanity: a legacy zero is a real (if uninformative) measured value —
    # get_recent_ic must return it, not None.
    assert tracker.get_recent_ic("news_score") == 0.0


# ---------------------------------------------------------------------------
# ICTracker.update() None-handling: a month with no overlapping data (i.e.
# compute_ic returned None) must be recorded as null/None, never folded into
# the running mean as if it were a measured 0.0.
# ---------------------------------------------------------------------------

def test_update_records_none_for_no_data():
    tracker = ICTracker()
    tracker.update("news_score", None, as_of="2026-07-01")
    assert tracker._ic_history["news_score"] == [None]


def test_update_none_does_not_corrupt_running_mean_same_month():
    """A no-data day in the middle of a month with real data must not drag
    the running mean toward 0.0 — it should simply be skipped."""
    tracker = ICTracker()
    tracker.update("mom_1m", 0.05, as_of="2026-07-01")
    tracker.update("mom_1m", None, as_of="2026-07-15")
    assert tracker._ic_history["mom_1m"] == [0.05]


def test_update_first_real_value_after_none_in_same_month():
    """No-data days early in the month, followed by a real measurement later
    in the SAME month, must set the value (not blend with a phantom 0.0)."""
    tracker = ICTracker()
    tracker.update("mom_1m", None, as_of="2026-07-01")
    tracker.update("mom_1m", 0.05, as_of="2026-07-15")
    assert tracker._ic_history["mom_1m"] == [0.05]


def test_update_none_new_month_appends_none():
    tracker = ICTracker()
    tracker.update("news_score", 0.05, as_of="2026-06-30")
    tracker.update("news_score", None, as_of="2026-07-01")
    assert tracker._ic_history["news_score"] == [0.05, None]


def test_save_load_roundtrip_preserves_none(tmp_path):
    """JSON null <-> Python None must survive a save/load cycle unchanged —
    daily_run.py reloads the tracker from disk every morning."""
    tracker = ICTracker()
    tracker.update("news_score", None, as_of="2026-07-01")
    tracker.update("mom_1m", 0.05, as_of="2026-07-01")
    path = tmp_path / "ic_tracker.json"
    tracker.save(path)

    loaded = ICTracker.load_or_create(path)
    assert loaded._ic_history["news_score"] == [None]
    assert loaded._ic_history["mom_1m"] == [0.05]


# ---------------------------------------------------------------------------
# get_recent_ic: must skip None ("no data") entries and return the latest
# genuinely MEASURED value, or None if nothing has ever been measured.
# ---------------------------------------------------------------------------

def test_get_recent_ic_skips_trailing_none():
    tracker = ICTracker()
    tracker._ic_history = {"mom_1m": [0.05, 0.08, None]}
    assert tracker.get_recent_ic("mom_1m") == pytest.approx(0.08)


def test_get_recent_ic_returns_none_when_never_measured():
    tracker = ICTracker()
    tracker._ic_history = {"news_score": [None, None]}
    assert tracker.get_recent_ic("news_score") is None


def test_get_recent_ic_returns_none_for_unknown_feature():
    """No history at all (feature never updated) -> None, matching the
    'never measured' contract rather than the old 0.0 default."""
    tracker = ICTracker()
    assert tracker.get_recent_ic("brand_new_feature") is None
