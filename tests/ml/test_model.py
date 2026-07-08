import pandas as pd
import numpy as np
import pytest
from src.ml import model as model_module
from src.ml.model import train, predict, walk_forward_backtest, FEATURE_WEIGHT_FLOOR


def make_feature_matrix(n_dates=300, n_tickers=50, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            ticker = f"T{i:03d}"
            feat_vals = np.random.randn(7)
            # Target slightly correlated with first feature (mom_1m)
            target = np.clip(0.5 + 0.2 * feat_vals[0] + 0.1 * np.random.randn(), 0, 1)
            rows.append({
                "date": d, "ticker": ticker,
                "mom_1m": feat_vals[0], "mom_3m": feat_vals[1],
                "mom_12m": feat_vals[2], "rev_1w": feat_vals[3],
                "vol_20d": feat_vals[4], "news_score": feat_vals[5],
                "supply_chain_score": feat_vals[6], "sue": np.random.randn(),
                "target": target,
            })
    return pd.DataFrame(rows)


def test_train_returns_model(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    fm = make_feature_matrix()
    model = train(fm)
    assert model is not None


def test_predict_returns_series(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    fm = make_feature_matrix()
    model = train(fm)
    today = fm[fm["date"] == fm["date"].max()]
    scores = predict(model, today)
    assert isinstance(scores, pd.Series)
    assert len(scores) == today["ticker"].nunique()
    # Scores should be between 0 and 1
    assert scores.between(0, 1).all()


def test_walk_forward_returns_dataframe():
    fm = make_feature_matrix(n_dates=150, n_tickers=30)
    results = walk_forward_backtest(fm, train_days=60, test_days=21)
    assert isinstance(results, pd.DataFrame)
    assert "predicted_score" in results.columns
    assert "actual_rank" in results.columns
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Task 2: purge/embargo — no training row's target horizon may overlap the
# test window. See walk_forward_backtest()'s embargo comment for the "why".
# ---------------------------------------------------------------------------

def test_walk_forward_embargoes_target_horizon(monkeypatch):
    """Pin the leakage fix: every training date used in a fold must be more
    than TARGET_HORIZON trading days before that fold's test start date.

    Before the fix, train_dates = dates[i-train_days : i] — the last
    TARGET_HORIZON training rows have targets computed from prices that
    extend into the test window. This test would have failed against that
    version (gap would be as small as 1, not > TARGET_HORIZON).
    """
    import config

    fm = make_feature_matrix(n_dates=150, n_tickers=30)
    all_dates = sorted(fm["date"].unique())
    date_to_idx = {d: idx for idx, d in enumerate(all_dates)}

    recorded_train_dates: list[list] = []
    original_train = model_module.train

    def recording_train(feature_matrix_slice, feature_weights=None, params=None, save=True):
        """Record which dates were actually used for training, then delegate
        to the real train() (with save=False) so predict() downstream still
        gets a genuinely fitted model."""
        recorded_train_dates.append(sorted(feature_matrix_slice["date"].unique()))
        return original_train(
            feature_matrix_slice, feature_weights=feature_weights, params=params, save=False
        )

    monkeypatch.setattr(model_module, "train", recording_train)

    train_days = 60
    test_days = 21
    model_module.walk_forward_backtest(fm, train_days=train_days, test_days=test_days)

    assert recorded_train_dates, "No training calls were recorded — check test wiring."

    # Recompute the test_start for each fold the same way walk_forward_backtest does.
    i = train_days
    fold = 0
    while i + test_days <= len(all_dates):
        test_start = all_dates[i]
        train_dates_used = recorded_train_dates[fold]
        assert train_dates_used, f"Fold {fold} trained on zero dates."
        for d in train_dates_used:
            gap = date_to_idx[test_start] - date_to_idx[d]
            assert gap > config.TARGET_HORIZON, (
                f"LEAKAGE: fold {fold} trained on {d}, only {gap} trading days "
                f"before test start {test_start} — must be > TARGET_HORIZON "
                f"({config.TARGET_HORIZON})."
            )
        fold += 1
        i += test_days


# ---------------------------------------------------------------------------
# Task 9: train() accepts optional hyperparameter overrides (from Optuna via
# src/ml/tuner.py::load_latest_params()) and merges them into LightGBM kwargs.
# ---------------------------------------------------------------------------

def test_train_params_override_reaches_fitted_model(tmp_path, monkeypatch):
    """A params override must actually change the fitted regressor's config,
    proving train() -> LGBMRegressor(**merged_params) really uses it."""
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    fm = make_feature_matrix(n_dates=60, n_tickers=10)

    model = train(fm, params={"n_estimators": 50}, save=False)

    assert model.n_estimators == 50
    # Keys NOT mentioned in the override keep their function defaults.
    assert model.get_params()["verbose"] == -1
    assert model.get_params()["n_jobs"] == -1


def test_train_without_params_keeps_defaults(tmp_path, monkeypatch):
    """Omitting `params` entirely must not change existing behaviour."""
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    fm = make_feature_matrix(n_dates=60, n_tickers=10)

    model = train(fm, save=False)
    assert model.n_estimators == 300  # the function's built-in default


# ---------------------------------------------------------------------------
# Task 11c: feature_weights now performs feature SELECTION (drop features
# below FEATURE_WEIGHT_FLOOR), not a no-op constant sample_weight.
# ---------------------------------------------------------------------------

def test_train_drops_low_weight_features(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    fm = make_feature_matrix(n_dates=60, n_tickers=10)

    feature_weights = {
        "mom_1m": 0.9,                        # well above the floor — kept
        "sue": FEATURE_WEIGHT_FLOOR - 0.01,   # below the floor — dropped
    }
    model = train(fm, feature_weights=feature_weights, save=False)

    assert "sue" not in model.feature_name_
    assert "mom_1m" in model.feature_name_
    # Features absent from feature_weights are "unknown", not zero-weight —
    # they must be kept.
    assert "mom_3m" in model.feature_name_


def test_train_feature_weights_none_keeps_all_available_features(tmp_path, monkeypatch):
    """Baseline: with feature_weights=None every available feature is used —
    confirms the drop behaviour above is specific to the weights, not a
    general regression."""
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    fm = make_feature_matrix(n_dates=60, n_tickers=10)

    model = train(fm, feature_weights=None, save=False)
    assert "sue" in model.feature_name_
    assert "mom_1m" in model.feature_name_
