import pandas as pd
import pytest
from datetime import date
import config


def test_update_equity_creates_file(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))

    df = pd.read_parquet(tmp_path / "equity_curve.parquet")
    assert set(df.columns) >= {"date", "equity"}
    assert len(df) == 1
    assert df["equity"].iloc[0] == pytest.approx(100_000.0)


def test_update_equity_no_duplicate_on_same_date(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))
    equity_tracker.update_equity(101_000.0, as_of=date(2024, 6, 1))  # same date, updated value

    df = pd.read_parquet(tmp_path / "equity_curve.parquet")
    assert len(df) == 1
    assert df["equity"].iloc[0] == pytest.approx(101_000.0)  # latest value wins


def test_load_equity_curve_empty_when_no_file(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = equity_tracker.load_equity_curve()
    assert df.empty
