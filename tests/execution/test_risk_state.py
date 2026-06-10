import pytest
import config


def test_peak_advances_on_growth(tmp_path, monkeypatch):
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert risk_state.get_peak_value(100_000) == pytest.approx(100_000)
    assert risk_state.get_peak_value(110_000) == pytest.approx(110_000)  # new high
    assert risk_state.get_peak_value(105_000) == pytest.approx(110_000)  # drawdown — peak stays


def test_peak_persists_across_instances(tmp_path, monkeypatch):
    from src.execution import risk_state
    import importlib

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    risk_state.get_peak_value(120_000)

    # Reload module to simulate a fresh daily_run.py process
    importlib.reload(risk_state)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # re-apply after reload
    peak = risk_state.get_peak_value(115_000)  # below prior peak
    assert peak == pytest.approx(120_000)
