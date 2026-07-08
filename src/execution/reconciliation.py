"""
reconciliation.py — Weekly paper-account vs. backtest reconciliation report.

Why this exists:
  The Alpaca paper account and the backtester are two SEPARATE measurements
  of the *same* strategy — one from live (paper) order fills, one from a
  simulation. If they diverge, there are exactly two possible explanations:

    1. The backtest is "flattering itself" — it's leaking future information,
       under-costing trades, or otherwise producing returns the real system
       could never actually earn.
    2. The live implementation has drifted from what the backtest simulates
       (a bug in daily_run.py, a stale model, a risk-engine mismatch, a data
       feed difference, etc.).

  Either way, something needs fixing. Comparing the two curves every week is
  the cheapest overfitting detector available: if paper trading quietly stops
  matching the backtest, that's a signal to investigate BEFORE trusting any
  future backtest number.

Analogy: this is like re-weighing yourself on two different scales every
Friday. If they've always agreed and start disagreeing, you don't know yet
which scale broke — but you know something did, and that's worth flagging
before you trust either reading again.

Run standalone with:
    python -m src.execution.reconciliation
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Trading decisions (and "today" for the report filename) are always judged
# in US/Eastern — mirrors daily_run.py's _today() / intraday_runner._now_et().
_ET = ZoneInfo("America/New_York")

# Verdict thresholds — see build_report()'s docstring for what a WARN means.
_CUM_DELTA_WARN = 0.02   # |cumulative return delta| > 2 percentage points
_CORR_WARN      = 0.7    # daily-return correlation below this is "not tracking"

_MIN_PAPER_ROWS = 10     # need at least this many paper-equity rows to bother comparing


def _now_et() -> datetime:
    """Current datetime in US/Eastern. Extracted so tests can monkeypatch it
    (mirrors daily_run._today())."""
    return datetime.now(tz=_ET)


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def _simulate_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Run a fresh walk-forward backtest and return its equity curve,
    restricted to dates in [start, end].

    Deviation from the original plan worth flagging: run_backtest()'s walk-
    forward loop ALWAYS trains its own model internally on a rolling
    config.MODEL_TRAIN_DAYS window before predicting each TEST_DAYS block —
    its public API has no "just predict with this one saved model" entry
    point (see src/backtest/run_backtest.py's run_backtest() docstring). So
    load_latest_model() cannot be threaded into the simulation itself; it is
    used in build_report() only as a cheap existence check before paying for
    this (expensive) call.

    Because run_backtest() needs at least MODEL_TRAIN_DAYS + TEST_DAYS
    trading days of feature history just to produce ONE walk-forward step,
    we build a feature matrix that reaches back far enough before `start`,
    then slice the resulting equity curve down to the requested window.

    Raises whatever build_feature_matrix()/run_backtest() raise (empty
    feature matrix, insufficient history, etc.) — the caller (build_report)
    is responsible for catching this so reconciliation can never break the
    daily pipeline.
    """
    from src.data.sp500_universe import get_symbols, get_sector_map
    from src.features.feature_matrix import build_feature_matrix
    from src.backtest.run_backtest import run_backtest, TEST_DAYS

    symbols    = get_symbols()
    sector_map = get_sector_map()

    # The walk-forward loop only ever simulates dates in chunks of TEST_DAYS
    # (21 trading days). Round the requested window up to a whole number of
    # TEST_DAYS chunks so the loop actually reaches every date we want to
    # compare — a partial trailing chunk shorter than TEST_DAYS would be
    # left un-simulated entirely (see run_backtest()'s while-loop condition
    # `i + TEST_DAYS <= n_dates`).
    span_trading_days = max(int(np.busday_count(start.date(), end.date())) + 1, 1)
    test_span = math.ceil(span_trading_days / TEST_DAYS) * TEST_DAYS
    periods   = config.MODEL_TRAIN_DAYS + test_span

    dates = pd.bdate_range(end=end, periods=periods)
    feature_matrix = build_feature_matrix(symbols, dates)

    sim = run_backtest(feature_matrix, sector_map)
    equity_curve: pd.Series = sim["equity_curve"]

    return equity_curve[(equity_curve.index >= start) & (equity_curve.index <= end)]


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _atomic_write_text(text: str, path: Path) -> None:
    """Write text atomically: temp file in the same dir + os.replace(), so a
    crash mid-write never leaves a half-written report (matches
    equity_tracker._atomic_write_parquet's house style)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(lookback_days: int = 60) -> str | None:
    """Compare the real paper equity curve against a fresh backtest
    simulation over the trailing ``lookback_days`` and write a markdown
    reconciliation report.

    Never raises — reconciliation is a diagnostic, not a trading decision,
    so any failure (missing history, simulation blowing up, etc.) is logged
    and this function returns None instead of propagating.

    Args:
        lookback_days: How many trailing rows of the paper equity curve to
            compare against (roughly trading days, since equity_tracker only
            records one row per trading day).

    Returns:
        The markdown report text, or None if there wasn't enough data to
        build a comparison.
    """
    from src.execution import equity_tracker

    paper_curve = equity_tracker.load_equity_curve()
    if len(paper_curve) < _MIN_PAPER_ROWS:
        logger.info(
            "Reconciliation: not enough paper history yet (%d rows, need %d).",
            len(paper_curve), _MIN_PAPER_ROWS,
        )
        return None

    paper_curve = paper_curve.copy()
    paper_curve["date"] = pd.to_datetime(paper_curve["date"])
    paper_curve = paper_curve.sort_values("date")

    window = paper_curve.tail(lookback_days)
    start, end = window["date"].iloc[0], window["date"].iloc[-1]

    # Cheap precondition: no saved model at all means run_backtest() would
    # fail deep inside train() anyway — skip the expensive simulation call
    # entirely. See _simulate_window()'s docstring for why this is the only
    # place load_latest_model() can be used (run_backtest has no "predict
    # with THIS model" entry point).
    from src.ml.model import load_latest_model
    if load_latest_model() is None:
        logger.info("Reconciliation: no saved model yet — skipping simulation.")
        return None

    try:
        sim_curve = _simulate_window(start, end)
    except Exception as exc:
        # MUST NEVER let a simulation failure break the daily pipeline —
        # log the exception TYPE (not necessarily the full trace — this
        # runs unattended) and bail out gracefully.
        logger.warning(
            "Reconciliation: simulation failed (%s: %s) — skipping this week.",
            type(exc).__name__, exc,
        )
        return None

    if sim_curve is None or sim_curve.empty:
        logger.info("Reconciliation: simulation produced no data for the window — skipping.")
        return None

    # ------------------------------------------------------------------
    # Align the two series on their common dates (inner join).
    # ------------------------------------------------------------------
    paper_series = window.set_index("date")["equity"].astype(float)
    sim_series   = sim_curve.astype(float)
    sim_series.index = pd.to_datetime(sim_series.index)

    common_dates = paper_series.index.intersection(sim_series.index).sort_values()
    if len(common_dates) < 5:
        logger.info(
            "Reconciliation: too few overlapping dates (%d) between paper and "
            "sim — skipping.", len(common_dates),
        )
        return None

    paper_aligned = paper_series.loc[common_dates]
    sim_aligned   = sim_series.loc[common_dates]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    paper_returns = paper_aligned.pct_change().dropna()
    sim_returns   = sim_aligned.pct_change().dropna()
    ret_dates     = paper_returns.index.intersection(sim_returns.index)
    paper_returns = paper_returns.loc[ret_dates]
    sim_returns   = sim_returns.loc[ret_dates]

    correlation = float(paper_returns.corr(sim_returns)) if len(ret_dates) >= 2 else float("nan")

    # Annualised tracking error: std of the day-by-day return DIFFERENCE,
    # scaled by sqrt(252) the same way a daily volatility is annualised.
    return_diff     = paper_returns - sim_returns
    tracking_error  = float(return_diff.std(ddof=1) * np.sqrt(252)) if len(ret_dates) >= 2 else float("nan")

    paper_cum_return = float(paper_aligned.iloc[-1] / paper_aligned.iloc[0] - 1)
    sim_cum_return   = float(sim_aligned.iloc[-1] / sim_aligned.iloc[0] - 1)
    cumulative_delta = paper_cum_return - sim_cum_return

    worst_date      = return_diff.abs().idxmax()
    worst_magnitude = float(return_diff.loc[worst_date])

    # WARN if either the cumulative paths have drifted apart by more than 2
    # percentage points, OR the daily returns barely correlate at all
    # (including the NaN case — e.g. one series is perfectly flat — which is
    # itself a reason to distrust the comparison, not a reason to pass it).
    is_warn = (
        abs(cumulative_delta) > _CUM_DELTA_WARN
        or pd.isna(correlation)
        or correlation < _CORR_WARN
    )
    verdict = "WARN" if is_warn else "OK"

    today_et = _now_et().date()

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    verdict_note = (
        "A WARN means the paper account and the backtest have meaningfully "
        "diverged: either the backtest is still flattering itself (leakage "
        "the anti-leakage guards missed, under-costed fills, an optimistic "
        "fill-timing assumption) or the live implementation has drifted from "
        "what the backtest simulates (stale model, a daily_run.py bug, a "
        "risk-engine mismatch, a data feed difference). Either way, treat "
        "any future backtest number with extra suspicion until this is "
        "understood."
        if is_warn else
        "The two measurements of the strategy still agree closely — no "
        "action needed."
    )

    lines = [
        f"# Reconciliation Report — {today_et}",
        "",
        f"**Window:** {start.date()} → {end.date()} "
        f"({len(common_dates)} overlapping trading days)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Daily-return correlation | {correlation:.3f} |",
        f"| Annualised tracking error | {tracking_error:.2%} |",
        f"| Paper cumulative return | {paper_cum_return:+.2%} |",
        f"| Backtest cumulative return | {sim_cum_return:+.2%} |",
        f"| Cumulative delta (paper − backtest) | {cumulative_delta:+.2%} |",
        f"| Worst single-day divergence | {worst_magnitude:+.2%} on {worst_date.date()} |",
        "",
        f"## Verdict: {verdict}",
        "",
        verdict_note,
        "",
    ]
    report = "\n".join(lines)

    out_path = config.DATA_DIR / "reconciliation" / f"{today_et}.md"
    _atomic_write_text(report, out_path)

    print(f"[reconciliation] {today_et}: verdict={verdict}  "
          f"corr={correlation:.3f}  tracking_error={tracking_error:.2%}  "
          f"cum_delta={cumulative_delta:+.2%}")
    print(f"[reconciliation] report written to {out_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    result = build_report()
    if result is None:
        print("[reconciliation] No report generated (see log above for why).")
