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


def test_update_equity_multiple_dates_accumulate(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))
    equity_tracker.update_equity(101_000.0, as_of=date(2024, 6, 2))
    equity_tracker.update_equity(102_000.0, as_of=date(2024, 6, 3))
    df = pd.read_parquet(tmp_path / "equity_curve.parquet")
    assert len(df) == 3
    assert list(df["equity"]) == pytest.approx([100_000.0, 101_000.0, 102_000.0])


# ---------------------------------------------------------------------------
# load_previous_equity — used by daily_run.py's real daily P&L calc (#14)
# ---------------------------------------------------------------------------

def test_load_previous_equity_none_when_no_history(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert equity_tracker.load_previous_equity("2024-06-01") is None


def test_load_previous_equity_returns_most_recent_prior_row(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))
    equity_tracker.update_equity(101_000.0, as_of=date(2024, 6, 2))
    equity_tracker.update_equity(103_000.0, as_of=date(2024, 6, 3))

    # "Today" is 2024-06-03 → most recent PRIOR row is 2024-06-02 (101,000).
    prev = equity_tracker.load_previous_equity(date(2024, 6, 3))
    assert prev == pytest.approx(101_000.0)


def test_load_previous_equity_none_when_only_todays_row_exists(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))

    # No row strictly BEFORE 2024-06-01 exists.
    assert equity_tracker.load_previous_equity(date(2024, 6, 1)) is None


# ---------------------------------------------------------------------------
# Corrupt-file safety (#17a)
# ---------------------------------------------------------------------------

def test_corrupt_equity_curve_is_preserved_not_wiped(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = tmp_path / "equity_curve.parquet"
    path.write_bytes(b"this is not a real parquet file")

    df = equity_tracker.load_equity_curve()
    assert df.empty

    corrupt_files = list(tmp_path.glob("equity_curve.parquet.corrupt-*"))
    assert len(corrupt_files) == 1, f"Expected exactly one preserved corrupt file, got {corrupt_files}"
    assert corrupt_files[0].read_bytes() == b"this is not a real parquet file"


# ---------------------------------------------------------------------------
# Atomic write safety (#17a)
# ---------------------------------------------------------------------------

def test_atomic_write_failure_preserves_old_file(tmp_path, monkeypatch):
    """If the write fails partway through (simulated os.replace failure), the
    previously-written parquet file must be left completely intact."""
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))

    path = tmp_path / "equity_curve.parquet"
    original_bytes = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(equity_tracker.os, "replace", boom)

    with pytest.raises(OSError):
        equity_tracker.update_equity(999_000.0, as_of=date(2024, 6, 2))

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob("equity_curve.parquet.*.tmp")) == []
