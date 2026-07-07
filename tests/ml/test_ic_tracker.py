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
