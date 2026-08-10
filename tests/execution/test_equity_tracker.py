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
# load_previous_equity staleness guard (2026-08-10 kill-switch false-fire fix)
# ---------------------------------------------------------------------------

def test_load_previous_equity_none_when_prior_row_28_days_old(tmp_path, monkeypatch):
    """The exact shape of the 2026-08-10 bug: the only prior row is 28
    calendar days old (a month of failed runs). That is far wider than any
    legitimate gap between consecutive trading days, so it must be rejected
    rather than treated as "yesterday"."""
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 7, 13))

    prev = equity_tracker.load_previous_equity(date(2026, 8, 10))
    assert prev is None


def test_load_previous_equity_returns_value_when_prior_row_1_day_old(tmp_path, monkeypatch):
    """An ordinary consecutive trading day (1 calendar day back) must still
    work — the guard should not reject legitimate same-week gaps."""
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 8, 6))

    prev = equity_tracker.load_previous_equity(date(2026, 8, 7))
    assert prev == pytest.approx(100_000.0)


def test_load_previous_equity_returns_value_when_prior_row_3_days_old(tmp_path, monkeypatch):
    """Friday -> Monday is a 3-calendar-day gap between consecutive trading
    days and must still be accepted (it's within the 4-day default)."""
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    # Friday 2026-08-07 -> "today" is Monday 2026-08-10.
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 8, 7))

    prev = equity_tracker.load_previous_equity(date(2026, 8, 10))
    assert prev == pytest.approx(100_000.0)


def test_load_previous_equity_max_age_days_none_disables_guard(tmp_path, monkeypatch):
    """max_age_days=None must reproduce the old, unguarded behaviour for
    callers that explicitly want the raw most-recent-prior-row value."""
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 7, 13))

    prev = equity_tracker.load_previous_equity(date(2026, 8, 10), max_age_days=None)
    assert prev == pytest.approx(100_000.0)


def test_load_previous_equity_2026_08_10_regression(tmp_path, monkeypatch):
    """REGRESSION for the 2026-08-10 false kill-switch fire.

    Reproduces the real curve exactly: rows on 2026-07-08 and 2026-07-13
    (both 100,000.0), an as-of date of 2026-08-10, and an account value of
    97,940.69 (the true one-day move was only -0.62%, but dividing by the
    28-day-old row produced -2.06%, breaching DAILY_LOSS_LIMIT=0.02).

    We assert on the durable contract — the staleness guard rejects the
    stale row — rather than re-deriving the exact percentage, since the
    guard (not a specific pnl number) is what prevents recurrence."""
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 7, 8))
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 7, 13))

    prev_equity = equity_tracker.load_previous_equity(date(2026, 8, 10))

    # The guard must reject the 28-day-old row outright...
    assert prev_equity is None

    # ...which means daily_run.py's fallback chain lands on 0.0, and
    # daily_pnl_pct is never computed from the stale row at all — so it
    # cannot breach -0.02 (the exact bug that liquidated the book).
    account_value = 97_940.69
    daily_pnl_pct = 0.0
    if prev_equity is not None and prev_equity != 0:
        daily_pnl_pct = (account_value - prev_equity) / prev_equity
    assert daily_pnl_pct > -0.02


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
