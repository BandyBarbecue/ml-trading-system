"""
risk_state.py — Persist risk engine state (peak_value, daily halt flag) across
daily runs.

The drawdown cap needs to know the HISTORICAL PEAK of account value, not just
today's value. Without persistence, every run starts fresh and the cap never
fires.

This module also persists a "halted for today" flag: if the kill switch fires
(daily loss too severe), we record today's ET date here. If Task Scheduler
(or a human) reruns daily_run.py / intraday_runner.py again later the SAME
day — e.g. because the first run crashed after the kill switch fired, or
someone manually restarts it — the halt flag makes the rerun refuse to trade
instead of quietly resuming. Analogy: like a fire alarm that stays tripped
until someone resets it by hand, rather than silently rearming itself the
next time the building's sensors run their routine check.

Storage: JSON file at DATA_DIR/risk_state.json.  Extremely lightweight.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _state_path() -> Path:
    return config.DATA_DIR / "risk_state.json"


# ---------------------------------------------------------------------------
# Low-level read/write helpers (atomic write, corrupt-file safety)
# ---------------------------------------------------------------------------

def _read_state(path: Path) -> dict:
    """Load risk_state.json, tolerating corruption without losing history.

    A crash mid-write (or a disk error) can leave the file truncated or not
    valid JSON. Silently starting over in that case would erase the
    peak_value the drawdown cap depends on — so instead we rename the bad
    file aside (<name>.corrupt-<timestamp>) so it can be inspected later,
    log at ERROR, and only then start fresh.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = path.with_name(f"{path.name}.corrupt-{stamp}")
        try:
            path.rename(corrupt_path)
        except OSError as rename_exc:
            logger.error("Could not preserve corrupt %s: %s", path, rename_exc)
        logger.error(
            "risk_state.json was corrupt/unreadable (%s) — preserved as %s, starting fresh.",
            exc, corrupt_path,
        )
        return {}


def _atomic_write_state(path: Path, state: dict) -> None:
    """Write JSON atomically: write to a temp file in the same directory,
    then os.replace() it over the target.

    Why: os.replace() is atomic on both POSIX and Windows — a crash or power
    loss mid-write can never leave a half-written (corrupt) risk_state.json
    behind. Readers always see either the fully-written old file or the
    fully-written new one, never something in between.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(state))
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API — peak value (drawdown tracking)
# ---------------------------------------------------------------------------

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
    state = _read_state(path)

    try:
        stored_peak = float(state.get("peak_value", 0.0))
    except (TypeError, ValueError):
        stored_peak = 0.0
    new_peak = max(stored_peak, current_value)

    state["peak_value"] = new_peak
    _atomic_write_state(path, state)

    return new_peak


# ---------------------------------------------------------------------------
# Public API — daily halt flag (restart-safe kill switch)
# ---------------------------------------------------------------------------

def set_halted(date_str: str) -> None:
    """Record that trading is halted for the ET calendar date `date_str` (YYYY-MM-DD).

    Called from daily_run.py's Step 6 the moment the risk engine's decisions
    include a "kill_switch" action. Persisting this (rather than just acting
    on it in-memory for the current process) is what makes the halt
    restart-safe: a crashed-then-retried run, or a manually re-triggered
    run, will see the flag via is_halted() and refuse to trade again today.
    """
    path = _state_path()
    state = _read_state(path)
    state["halted_date"] = date_str
    _atomic_write_state(path, state)
    logger.error("Trading halted for %s (kill switch fired) — recorded in risk_state.json.", date_str)


def is_halted(date_str: str) -> bool:
    """Return True if trading was halted (kill switch fired) for `date_str` (YYYY-MM-DD, ET)."""
    path = _state_path()
    state = _read_state(path)
    return state.get("halted_date") == date_str
