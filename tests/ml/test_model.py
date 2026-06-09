import pandas as pd
import numpy as np
import pytest
from src.ml.model import train, predict, walk_forward_backtest


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
