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
    "gross_profitability", "pb_ratio",
    "revision_momentum", "insider_score",
]


def train(
    feature_matrix: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    excluded_features: list[str] | None = None,
    params: dict | None = None,
    save: bool = True,
) -> lgb.LGBMRegressor:
    """Train LightGBM on the feature matrix to predict cross-sectional return rank.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Must contain columns for every entry in FEATURE_COLS plus a 'target' column
        (float in [0, 1] — the stock's cross-sectional rank on that day).
    feature_weights : dict[str, float] | None
        Optional per-feature IC-derived weights from the IC tracker (see
        src/ml/ic_tracker.py::get_feature_weights), typically already
        regime-adjusted and renormalised by daily_run.py Step 4. Informational
        only here — kept on the signature for callers/logging/future use, but
        train() does NOT use its magnitude to decide which features to drop
        (see excluded_features below for why that changed).
    excluded_features : list[str] | None
        Feature names to drop from training entirely. This is the ONLY thing
        that decides feature selection now. WHY: the old design dropped a
        feature when its *normalised weight share* fell below a floor
        (FEATURE_WEIGHT_FLOOR = 0.05) — but a feature's normalised share
        depends on how concentrated OTHER features' IC is, not on its own
        merit (if two features hold 85% of total IC mass, everything else
        gets squeezed below 5% regardless of whether it's individually
        predictive). The keep/drop decision now happens exactly ONCE, in
        daily_run.py Step 4, by comparing each feature's own latest measured
        IC against config.IC_RETIRE_THRESH (the same space the IC retirement
        warning already uses) — the resolved list is passed in here so there
        is a single source of truth for "why was this feature dropped."
        Features absent from this list are kept, including features absent
        from feature_weights entirely ("unknown" — e.g. no IC data yet — is
        not the same as "measured and weak", so they aren't punished).
        If None (the default), every available feature is used.
    params : dict | None
        Optional LightGBM hyperparameter overrides (e.g. from
        src/ml/tuner.py::load_latest_params()). Any key here overrides the
        corresponding default below; keys not present keep their default.
        If None, the defaults are used unchanged.
    save : bool
        If True, pickle the fitted model to config.MODELS_DIR / 'lgbm_YYYY-MM.pkl'.
        Set to False during walk-forward to avoid cluttering disk with interim models.

    Returns
    -------
    lgb.LGBMRegressor
        The fitted model.
    """
    # Use every feature column that exists and has at least some data.
    # Features that are entirely NaN (e.g. news_score before news files exist)
    # are excluded from the feature set entirely.
    avail_features = [
        c for c in FEATURE_COLS
        if c in feature_matrix.columns and feature_matrix[c].notna().any()
    ]

    # Feature SELECTION is driven solely by excluded_features now (see the
    # docstring above for why the old feature_weights-magnitude floor was
    # removed). LightGBM's sklearn API has no per-FEATURE weight argument
    # anyway — only `sample_weight`, which weights entire ROWS — so dropping
    # unwanted columns from X before fit() is the only way to keep the model
    # from wasting tree splits on features known to be noise right now.
    if excluded_features:
        avail_features = [c for c in avail_features if c not in excluded_features]

    # Only drop rows where the TARGET is unknown — we can't train without knowing
    # the outcome. Missing FEATURE values are filled with 0 (neutral signal) rather
    # than dropping the row: news_score is legitimately 0 for dates with no news,
    # and dropping those rows would reduce a 2-year training window to 3 days.
    train_data = feature_matrix.dropna(subset=["target"])[avail_features + ["target"]].copy()
    train_data[avail_features] = train_data[avail_features].fillna(0)
    X = train_data[avail_features]
    y = train_data["target"]
    if X.empty:
        raise ValueError(
            "No training rows remain after dropping rows without a target (or "
            f"after excluded_features filtering) — available features: {avail_features}"
        )

    default_params = {
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
    # Caller-supplied params (e.g. from Optuna tuning) override the defaults
    # above key-by-key. Dict unpacking order matters here: default_params is
    # spread first, then params, so params' values win on any key collision —
    # any key params does NOT mention (like verbose/n_jobs) keeps its default.
    merged_params = {**default_params, **params} if params else default_params

    model = lgb.LGBMRegressor(**merged_params)
    model.fit(X, y)

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
    # When the feature matrix spans multiple dates (the full training history),
    # score only the most recent date.  build_portfolio expects one score per
    # unique ticker — not one per (date, ticker) pair.
    fm = feature_matrix
    if "date" in fm.columns and fm["date"].nunique() > 1:
        fm = fm[fm["date"] == fm["date"].max()]

    # Use the exact feature list the model was trained with.
    # model.feature_name_ records which columns were present at fit time — this
    # handles the case where not all FEATURE_COLS had data when training ran.
    train_cols = getattr(model, "feature_name_", FEATURE_COLS)
    # reindex selects the right columns in the right order; fillna(0) treats any
    # missing column as "no signal" rather than crashing.
    X = fm.reindex(columns=train_cols).fillna(0)
    raw_scores = model.predict(X)
    return (
        pd.Series(raw_scores, index=fm["ticker"].values, name="predicted_score")
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
        # PURGE / EMBARGO: each row's target is a TARGET_HORIZON-day FORWARD
        # return rank (see feature_matrix.py). A training row dated d actually
        # "knows" prices up to d + TARGET_HORIZON. If we trained on the last
        # TARGET_HORIZON dates right before the test window, those rows' targets
        # would overlap with (partly be computed from) prices inside the test
        # window — the model would be trained on information from the future
        # it's about to be tested on. We embargo the last TARGET_HORIZON dates
        # of the training window so every training target's horizon ends
        # strictly before the test window begins. Explicit train/test boundary,
        # per project rule #2 (no data leakage, ever).
        train_dates = dates[i - train_days : i - config.TARGET_HORIZON]
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
