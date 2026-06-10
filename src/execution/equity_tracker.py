"""
equity_tracker.py — Persist daily account equity to equity_curve.parquet.

Why we need this:
  The daily_run.py currently has `peak_value = account_value` which resets
  the drawdown reference to today's value every single run. This means the
  drawdown cap (> 15% from peak) NEVER fires in practice.

  This module provides the persistent storage layer so risk_state.py can
  maintain a true rolling peak across days.

  Secondary use: the attribution module needs a real equity curve to compute
  monthly portfolio P&L vs SPY.
"""

from datetime import date
from pathlib import Path

import pandas as pd

import config


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

    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
        except Exception as exc:
            import warnings
            warnings.warn(f"equity_curve.parquet unreadable ({exc}); starting fresh.")
            existing = pd.DataFrame(columns=["date", "equity"])
    else:
        existing = pd.DataFrame(columns=["date", "equity"])

    # Use pd.Timestamp for consistent date handling
    as_of_ts = pd.Timestamp(as_of)
    existing["date"] = pd.to_datetime(existing["date"])

    # Remove any existing row for this date (so we always have the latest value)
    existing = existing[existing["date"] != as_of_ts]

    new_row = pd.DataFrame([{"date": as_of_ts, "equity": account_value}])
    combined = pd.concat([existing, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)

    combined.to_parquet(out_path, index=False)


def load_equity_curve() -> pd.DataFrame:
    """Load the full equity curve. Returns empty DataFrame if file doesn't exist."""
    out_path: Path = config.DATA_DIR / "equity_curve.parquet"
    if not out_path.exists():
        return pd.DataFrame(columns=["date", "equity"])
    return pd.read_parquet(out_path)
