"""
tuner.py — Monthly Optuna hyperparameter search for the LightGBM model.

Each month we run a short Optuna study to find LightGBM hyperparameters that
minimise MSE on a held-out time slice (the last 20% of dates in the feature
matrix). Think of it like adjusting the tuning knobs on a racing car before
each race — we try many combinations quickly, keep the best one, but only
replace last month's settings if the new ones are actually better.

Walk-forward discipline is preserved: validation data is always *later* in
time than training data, so there is no look-ahead bias.
"""

import json
import optuna
import numpy as np
import lightgbm as lgb
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error

import config  # import the module so monkeypatching in tests works correctly
from src.ml.model import FEATURE_COLS

# Silence Optuna's per-trial progress spam — we only care about the final result
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Single Optuna trial: fit a LightGBM model with suggested params, return MSE.

    Optuna calls this function many times, each time with a different set of
    hyperparameter suggestions. We return the validation MSE so Optuna can
    learn which regions of the parameter space are promising.
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "n_jobs": -1,
        # Optuna suggests values for each hyperparameter in the specified range
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,  # fixed: apply bagging every 5 trees
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return float(mean_squared_error(y_val, preds))


def run_tuning(feature_matrix: pd.DataFrame) -> dict:
    """Run Optuna hyperparameter search and return the best params found.

    Steps:
    1. Split the feature matrix 80/20 by time (no shuffling — always train
       on earlier dates, validate on later dates).
    2. Run `config.OPTUNA_TRIALS` trials to minimise validation MSE.
    3. If an existing params file is found (from last month), compare the
       new params against those. Keep whichever performs better on the same
       validation slice.
    4. Save the winner to `config.MODELS_DIR / params_YYYY-MM.json` and
       return it as a dict.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Must contain FEATURE_COLS, a 'target' column, and a 'date' column.

    Returns
    -------
    dict
        Hyperparameter dict ready to pass to lgb.LGBMRegressor(**params).
    """
    # Only use features that are actually present in this DataFrame — the same
    # pattern as model.py.  Test fixtures may supply fewer columns than the full
    # FEATURE_COLS list, so we never demand columns that aren't there.
    avail_features = [c for c in FEATURE_COLS if c in feature_matrix.columns]

    # Drop rows with missing features or target — clean data only
    data = feature_matrix.dropna(subset=avail_features + ["target"])
    dates = sorted(data["date"].unique())

    # 80/20 time-based split: earlier 80% for training, later 20% for validation.
    # We never shuffle because financial data is time-ordered — shuffling would
    # let the model train on "future" data, which is look-ahead bias.
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    train_df = data[data["date"].isin(train_dates)]
    val_df = data[data["date"].isin(val_dates)]

    X_train = train_df[avail_features].fillna(0)
    y_train = train_df["target"]
    X_val = val_df[avail_features].fillna(0)
    y_val = val_df["target"]

    # Run the hyperparameter search. Optuna's TPE sampler (default) builds a
    # probabilistic model of which params work well, so later trials are smarter
    # than random — a bit like educated guessing that gets better over time.
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=config.OPTUNA_TRIALS,
    )

    best_params = study.best_params
    best_score = study.best_value  # MSE on validation slice

    # Check whether last month's params (if saved) outperform the new ones.
    # Access config.MODELS_DIR through the module so tests can monkeypatch it.
    models_dir = config.MODELS_DIR
    existing = sorted(models_dir.glob("params_*.json")) if models_dir.exists() else []
    if existing:
        prev = json.loads(existing[-1].read_text())
        # Retrain with previous params on the same training data for a fair comparison
        prev_model = lgb.LGBMRegressor(**{**prev, "verbose": -1, "n_jobs": -1})
        prev_model.fit(X_train, y_train)
        prev_score = float(mean_squared_error(y_val, prev_model.predict(X_val)))
        if prev_score <= best_score:
            # Previous params are at least as good — don't overwrite, just return them
            print(
                f"[tuner] Previous params better "
                f"(MSE {prev_score:.4f} vs {best_score:.4f}). Keeping."
            )
            return prev

    # New params won (or no previous params exist) — save to disk
    models_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y-%m")
    path = models_dir / f"params_{tag}.json"
    path.write_text(json.dumps(best_params, indent=2))
    print(f"[tuner] New params saved to {path.name} (MSE {best_score:.4f})")
    return best_params


def load_latest_params() -> dict | None:
    """Load the most recently saved hyperparameter file, or None if none exist.

    Returns
    -------
    dict | None
        Hyperparameter dict, or None if config.MODELS_DIR has no params_*.json files.
    """
    models_dir = config.MODELS_DIR
    if not models_dir.exists():
        return None
    files = sorted(models_dir.glob("params_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())
