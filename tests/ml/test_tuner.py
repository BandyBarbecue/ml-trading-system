import pandas as pd
import numpy as np
import json
import pytest
from src.ml.model import train, predict


def make_fm(n_dates=120, n_tickers=30):
    np.random.seed(1)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            vals = np.random.randn(8)
            rows.append({"date": d, "ticker": f"T{i:02d}",
                "mom_1m": vals[0], "mom_3m": vals[1], "mom_12m": vals[2],
                "rev_1w": vals[3], "vol_20d": vals[4], "news_score": vals[5],
                "supply_chain_score": vals[6], "sue": vals[7],
                "target": np.clip(0.5 + 0.2*vals[0] + 0.1*np.random.randn(), 0, 1)})
    return pd.DataFrame(rows)


def test_run_tuning_returns_params(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "OPTUNA_TRIALS", 3)  # fast for tests
    from src.ml.tuner import run_tuning
    fm = make_fm()
    params = run_tuning(fm)
    assert isinstance(params, dict)
    assert "num_leaves" in params
    assert "learning_rate" in params
    assert any(tmp_path.glob("params_*.json"))


def test_params_saved_to_json(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "OPTUNA_TRIALS", 3)
    from src.ml.tuner import run_tuning
    fm = make_fm()
    run_tuning(fm)
    json_files = list(tmp_path.glob("params_*.json"))
    assert len(json_files) >= 1
    params = json.loads(json_files[0].read_text())
    assert "num_leaves" in params
