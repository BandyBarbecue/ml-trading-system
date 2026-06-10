"""
risk_state.py — Persist risk engine state (peak_value) across daily runs.

The drawdown cap needs to know the HISTORICAL PEAK of account value, not just
today's value. Without persistence, every run starts fresh and the cap never fires.

Storage: JSON file at DATA_DIR/risk_state.json.  Extremely lightweight.
"""

import json
from pathlib import Path

import config


def _state_path() -> Path:
    return config.DATA_DIR / "risk_state.json"


def get_peak_value(current_value: float) -> float:
    """Return the rolling peak account value, updating it if current_value is a new high.

    Reads from and writes to risk_state.json on every call.

    Parameters
    ----------
    current_value : Today's account equity in USD.

    Returns
    -------
    The maximum account value seen across all calls (including today's).
    """
    path = _state_path()
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}

    try:
        stored_peak = float(state.get("peak_value", 0.0))
    except (TypeError, ValueError):
        stored_peak = 0.0
    new_peak = max(stored_peak, current_value)

    state["peak_value"] = new_peak
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))

    return new_peak
