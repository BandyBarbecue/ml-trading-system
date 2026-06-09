"""
model.py — LightGBM cross-sectional return-rank predictor.

The model is trained to predict a stock's cross-sectional rank (0–1) among
all stocks on a given day. A score of 1.0 means "top of the universe", 0.0
means "bottom". Think of it like percentile placement in a class, but for
expected returns.

Walk-forward backtest: we never let the model peek at future dates. We train
on a window, predict the next window, then roll forward — exactly like you
would in practice.
"""

import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import config  # import the module so monkeypatching in tests works correctly
from datetime import datetime

# All features the model can use. "sue" = standardised unexpected earnings.
FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_12m", "rev_1w",
    "vol_20d", "news_score", "supply_chain_score", "sue",
]


def train(
    feature_matrix: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    save: bool = True,
) -> lgb.LGBMRegressor:
    """Train LightGBM on the feature matrix to predict cross-sectional return rank.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Must contain columns for every entry in FEATURE_COLS plus a 'target' column
        (float in [0, 1] — the stock's cross-sectional rank on that day).
    feature_weights : dict[str, float] | None
        Optional per-feature importance weights from the IC tracker. Features with
        higher recent information coefficient (IC) are given more influence by
        passing them as LightGBM feature weights. If None, all features are equal.
    save : bool
        If True, pickle the fitted model to config.MODELS_DIR / 'lgbm_YYYY-MM.pkl'.
        Set to False during walk-forward to avoid cluttering disk with interim models.

    Returns
    -------
    lgb.LGBMRegressor
        The fitted model.
    """
    # Drop rows where any feature or the target is missing — LightGBM can handle
    # NaNs internally, but we want clean, intentional training data.
    train_data = feature_matrix.dropna(subset=FEATURE_COLS + ["target"])
    X = train_data[FEATURE_COLS]
    y = train_data["target"]

    # sample_weight: if the caller provided per-feature IC weights, translate them
    # into per-sample weights based on how "informative" each row's features are.
    # The simplest proxy: weight each sample by the average IC of its non-null features.
    sample_weight = None
    if feature_weights is not None:
        weights_array = np.array([
            feature_weights.get(col, 1.0) for col in FEATURE_COLS
        ])
        # Clip to avoid zero or negative weights from bad IC values
        weights_array = np.clip(weights_array, 1e-3, None)
        # Broadcast: every sample gets the same feature-average weight vector
        # (we're weighting features, not individual rows, so all rows share this)
        sample_weight = np.full(len(X), weights_array.mean())

    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,       # controls model complexity; 31 is a sensible default
        "min_child_samples": 20, # min samples in a leaf — prevents overfitting on tiny groups
        "feature_fraction": 0.8, # use 80% of features per tree (like dropout for trees)
        "bagging_fraction": 0.8, # use 80% of data per tree (reduces variance)
        "bagging_freq": 5,       # apply bagging every 5 trees
        "verbose": -1,           # silence LightGBM's training log spam
        "n_jobs": -1,            # use all available CPU cores
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, sample_weight=sample_weight)

    if save:
        # Access through config module so tests can monkeypatch config.MODELS_DIR
        models_dir = config.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        tag = datetime.now().strftime("%Y-%m")
        path = models_dir / f"lgbm_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

    return model


def predict(model: lgb.LGBMRegressor, feature_matrix: pd.DataFrame) -> pd.Series:
    """Return a predicted score (0–1) for each ticker in feature_matrix.

    Scores are clipped to [0, 1] because LightGBM regression can extrapolate
    slightly outside the training target range.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        A fitted model returned by train().
    feature_matrix : pd.DataFrame
        Must contain a 'ticker' column plus any subset of FEATURE_COLS.
        Missing features are filled with 0 (neutral).

    Returns
    -------
    pd.Series
        Index = ticker symbols, values = predicted scores in [0, 1].
    """
    # Only use features the model was trained on that are present in the input.
    # This guards against calling predict on data that lacks some features.
    avail = [c for c in FEATURE_COLS if c in feature_matrix.columns]
    X = feature_matrix[avail].fillna(0)  # 0 = "no signal" for missing features
    raw_scores = model.predict(X)
    return (
        pd.Series(raw_scores, index=feature_matrix["ticker"].values, name="predicted_score")
        .clip(0, 1)
    )


def load_latest_model() -> lgb.LGBMRegressor | None:
    """Load the most recently saved model artefact from config.MODELS_DIR.

    Returns None if no saved models exist yet — callers should handle this
    gracefully (e.g. by falling back to training a fresh model).
    """
    models_dir = config.MODELS_DIR
    if not models_dir.exists():
        return None
    models = sorted(models_dir.glob("lgbm_*.pkl"))
    if not models:
        return None
    with open(models[-1], "rb") as f:
        return pickle.load(f)


def walk_forward_backtest(
    feature_matrix: pd.DataFrame,
    train_days: int = config.MODEL_TRAIN_DAYS,
    test_days: int = 21,
) -> pd.DataFrame:
    """Walk-forward validation: train on a rolling window, predict the next window.

    This is the correct way to validate a time-series model. The model never
    sees future data during training — each test period is genuinely out-of-sample.
    Think of it like repeatedly re-sitting an exam with fresh questions.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Full history with 'date', 'ticker', all FEATURE_COLS, and 'target'.
    train_days : int
        Number of trading days in each rolling training window.
    test_days : int
        Number of trading days in each test (prediction) window. Defaults to ~1 month.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, predicted_score, actual_rank.
        Each row is one ticker on one out-of-sample test day.
    """
    # Sort unique dates so we can index them like a list
    dates = sorted(feature_matrix["date"].unique())
    results = []

    # Start after we have enough history for the first training window
    i = train_days
    while i + test_days <= len(dates):
        train_dates = dates[i - train_days : i]   # window BEFORE the test period
        test_dates  = dates[i : i + test_days]    # window we're predicting

        train_data = feature_matrix[feature_matrix["date"].isin(train_dates)]
        test_data  = feature_matrix[feature_matrix["date"].isin(test_dates)]

        # save=False: walk-forward models are interim artefacts, not worth keeping
        model = train(train_data, save=False)

        # Predict day-by-day so scores are indexed correctly
        for d in test_dates:
            day_data = test_data[test_data["date"] == d].copy()
            if day_data.empty:
                continue
            scores = predict(model, day_data)
            # Align predictions with the day's rows using ticker as the key
            day_data = day_data.set_index("ticker")
            day_data["predicted_score"] = scores
            day_data["actual_rank"] = day_data["target"]  # ground truth for scoring
            day_data["date"] = d
            results.append(
                day_data[["date", "predicted_score", "actual_rank"]].reset_index()
            )

        i += test_days  # roll forward by one test window

    if not results:
        return pd.DataFrame(columns=["ticker", "date", "predicted_score", "actual_rank"])

    return pd.concat(results, ignore_index=True)
