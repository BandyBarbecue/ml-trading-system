"""
equity_tracker.py — Persist daily account equity to equity_curve.parquet.

Why we need this:
  The daily_run.py currently has `peak_value = account_value` which resets
  the drawdown reference to today's value every single run. This means the
  drawdown cap (> 15% from peak) NEVER fires in practice.

  This module provides the persistent storage layer so risk_state.py can
  maintain a true rolling peak across days, and so daily_run.py can compute
  a real day-over-day P&L for the kill switch instead of a hardcoded 0.0.

  Secondary use: the attribution module needs a real equity curve to compute
  monthly portfolio P&L vs SPY.
"""

import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

_COLUMNS = ["date", "equity"]


# ---------------------------------------------------------------------------
# Low-level read/write helpers (atomic write, corrupt-file safety)
# ---------------------------------------------------------------------------

def _read_equity_curve(path: Path) -> pd.DataFrame:
    """Load equity_curve.parquet, tolerating corruption without losing history.

    A crash mid-write (or a disk error) can leave the parquet file truncated
    or unreadable. Silently starting over in that case would erase the whole
    equity history the drawdown/kill-switch/attribution logic depends on —
    so instead we rename the bad file aside (<name>.corrupt-<timestamp>) so
    it can be inspected later, log at ERROR, and only then start fresh.
    """
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = path.with_name(f"{path.name}.corrupt-{stamp}")
        try:
            path.rename(corrupt_path)
        except OSError as rename_exc:
            logger.error("Could not preserve corrupt %s: %s", path, rename_exc)
        logger.error(
            "equity_curve.parquet was corrupt/unreadable (%s) — preserved as %s, starting fresh.",
            exc, corrupt_path,
        )
        return pd.DataFrame(columns=_COLUMNS)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically: write to a temp file in the same
    directory, then os.replace() it over the target.

    Why: os.replace() is atomic on both POSIX and Windows — a crash mid-write
    can never leave a half-written (corrupt) equity_curve.parquet behind.
    Readers always see either the fully-written old file or the fully-written
    new one, never something in between.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)  # to_parquet wants to open the path itself; we just needed a unique name
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_equity(account_value: float, as_of: date | None = None) -> None:
    """Append (or update) today's account equity in equity_curve.parquet.

    Parameters
    ----------
    account_value : Current Alpaca paper account equity in USD.
    as_of         : Date to record. Defaults to today. Override in tests.
    """
    if as_of is None:
        as_of = date.today()

    out_path: Path = config.DATA_DIR / "equity_curve.parquet"
    existing = _read_equity_curve(out_path)

    # Use pd.Timestamp for consistent date handling
    as_of_ts = pd.Timestamp(as_of)

    if not existing.empty:
        existing = existing.copy()
        existing["date"] = pd.to_datetime(existing["date"])
        # Remove any existing row for this date (so we always have the latest value)
        existing = existing[existing["date"] != as_of_ts]

    new_row = pd.DataFrame([{"date": as_of_ts, "equity": account_value}])

    # Skip pd.concat entirely when existing is empty: concatenating a
    # non-empty frame with an empty-but-typed one still trips pandas'
    # FutureWarning about dtype inference changing in a future version, even
    # though the result is identical to just using new_row directly.
    combined = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    _atomic_write_parquet(combined, out_path)


def load_equity_curve() -> pd.DataFrame:
    """Load the full equity curve. Returns empty DataFrame if file doesn't exist
    or is corrupt (corrupt files are preserved aside, not silently discarded —
    see _read_equity_curve)."""
    out_path: Path = config.DATA_DIR / "equity_curve.parquet"
    return _read_equity_curve(out_path)


def load_previous_equity(before: date | str) -> float | None:
    """Return the most recent recorded equity value strictly BEFORE `before`.

    Used by daily_run.py to compute today's real daily P&L for the kill
    switch: (account_value - prev_equity) / prev_equity. Returns None when
    there is no prior history (e.g. the very first run ever) — the caller
    treats that as "can't compute a daily P&L yet" and uses 0.0.

    Parameters
    ----------
    before : The date to compare against (exclusive) — normally "today"
             (either a date or a "YYYY-MM-DD" string).
    """
    curve = load_equity_curve()
    if curve.empty:
        return None

    curve = curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    cutoff = pd.Timestamp(before)

    prior_rows = curve[curve["date"] < cutoff].sort_values("date")
    if prior_rows.empty:
        return None
    return float(prior_rows["equity"].iloc[-1])
