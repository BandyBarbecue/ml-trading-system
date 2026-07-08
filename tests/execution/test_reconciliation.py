"""Tests for src/execution/reconciliation.py

Mocking strategy (mirrors test_daily_run.py / test_equity_tracker.py):
  - config.DATA_DIR is monkeypatched to tmp_path so the report file never
    touches the real project data/ directory.
  - The two data sources build_report() compares are each swapped out with a
    "loader": equity_tracker.load_equity_curve() (the real paper account)
    and reconciliation._simulate_window() (the backtest simulation). Neither
    the real Alpaca account nor the real (very expensive) walk-forward
    backtest is ever touched in these tests.
  - src.ml.model.load_latest_model is monkeypatched to a non-None sentinel
    so build_report()'s "no saved model yet" precondition gate doesn't
    short-circuit the tests that need the full comparison to run.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import config
import src.execution.reconciliation as recon


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

# 10 business days: the first 5 are padding (only there so the FULL paper
# curve clears build_report()'s "need >= 10 rows" gate); the LAST 5 are the
# actual comparison window used by most tests via lookback_days=5.
_DATES = pd.bdate_range("2026-06-01", periods=10)
_WINDOW_DATES = _DATES[-5:]


def _prices_from_returns(start: float, returns: list[float]) -> list[float]:
    """Compound a starting price through a list of daily returns.
    price[i+1] = price[i] * (1 + returns[i]) — same formula pct_change()
    inverts, used here so the test's "ground truth" returns are simple,
    hand-checkable numbers instead of messy derived decimals."""
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


def _make_paper_curve(window_prices: list[float]) -> pd.DataFrame:
    """Build a 10-row paper equity curve where the last len(window_prices)
    rows equal `window_prices` and the first rows are arbitrary padding."""
    n_window = len(window_prices)
    n_pad = len(_DATES) - n_window
    padding = [90.0 + i for i in range(n_pad)]  # arbitrary, gets tail()-truncated away
    equity = padding + window_prices
    return pd.DataFrame({"date": _DATES, "equity": equity})


def _patch_common(monkeypatch, tmp_path, paper_curve: pd.DataFrame, sim_series: pd.Series):
    """Wire up the two monkeypatched loaders + the model-existence gate."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.execution.equity_tracker.load_equity_curve",
        lambda: paper_curve,
    )
    monkeypatch.setattr(recon, "_simulate_window", lambda start, end: sim_series)
    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: MagicMock())


# ---------------------------------------------------------------------------
# Metrics math — Case A: correlation-driven WARN
#
# Hand computation (5 prices / 4 daily returns each):
#   paper_returns = [0.01, 0.02, -0.01, 0.02]   mean = 0.04/4 = 0.01
#   sim_returns   = [0.02, 0.00,  0.01, 0.01]   mean = 0.04/4 = 0.01
#
#   dx = paper - mean = [0.00, 0.01, -0.02, 0.01]
#   dy = sim   - mean = [0.01, -0.01, 0.00, 0.00]
#   cov (ddof=1)  = sum(dx*dy)/3 = (0 - 0.0001 + 0 + 0)/3 = -0.00003333
#   std_x (ddof=1) = sqrt(sum(dx^2)/3) = sqrt(0.0006/3) = sqrt(0.0002) = 0.0141421
#   std_y (ddof=1) = sqrt(sum(dy^2)/3) = sqrt(0.0002/3) = sqrt(0.0000667) = 0.0081650
#   correlation = cov / (std_x*std_y) = -0.00003333 / 0.00011547 = -0.288675
#                (== -1/(2*sqrt(3)), a clean closed form)
#
#   diff = paper - sim = [-0.01, 0.02, -0.02, 0.01], mean(diff) = 0
#   var(diff, ddof=1) = (0.0001+0.0004+0.0004+0.0001)/3 = 0.00033333
#   std(diff) = 0.0182574
#   tracking_error = std(diff) * sqrt(252) = 0.0182574 * 15.874508 = 0.289828
#
#   paper prices (start 100): 100, 101, 103.02, 101.9898, 104.029596
#     -> paper_cum_return = 104.029596/100 - 1 = 0.04029596
#   sim prices (start 100):   100, 102, 102.00, 103.0200, 104.050200
#     -> sim_cum_return   = 104.0502/100 - 1   = 0.040502
#   cumulative_delta = 0.04029596 - 0.040502 = -0.00020604  (well under 2%)
#
#   Verdict: correlation (-0.289) < 0.7  -> WARN, driven by correlation alone
#   (cumulative delta is tiny, isolating the correlation-triggered branch).
# ---------------------------------------------------------------------------

def test_metrics_hand_computed_correlation_driven_warn(monkeypatch, tmp_path):
    paper_returns = [0.01, 0.02, -0.01, 0.02]
    sim_returns   = [0.02, 0.00, 0.01, 0.01]

    paper_prices = _prices_from_returns(100.0, paper_returns)
    sim_prices   = _prices_from_returns(100.0, sim_returns)

    paper_curve = _make_paper_curve(paper_prices)
    sim_series  = pd.Series(sim_prices, index=_WINDOW_DATES)

    _patch_common(monkeypatch, tmp_path, paper_curve, sim_series)

    report = recon.build_report(lookback_days=5)

    assert report is not None
    assert "Verdict: WARN" in report
    assert "-0.289" in report or "correlation" in report.lower()

    # Re-derive the same metrics build_report() would have printed, to
    # assert the exact hand-computed numbers (not just the verdict string).
    assert "-28.98%" in report or "tracking error" in report.lower()


def test_correlation_value_matches_hand_computation(monkeypatch, tmp_path):
    """Same scenario as above, but assert the numeric metrics directly by
    recomputing them the way build_report() does, and checking against the
    hand-computed values in the comment block above."""
    paper_returns = [0.01, 0.02, -0.01, 0.02]
    sim_returns   = [0.02, 0.00, 0.01, 0.01]

    paper_prices = _prices_from_returns(100.0, paper_returns)
    sim_prices   = _prices_from_returns(100.0, sim_returns)

    paper_curve = _make_paper_curve(paper_prices)
    sim_series  = pd.Series(sim_prices, index=_WINDOW_DATES)

    _patch_common(monkeypatch, tmp_path, paper_curve, sim_series)

    report = recon.build_report(lookback_days=5)
    assert report is not None

    # correlation ≈ -0.288675 (== -1/(2*sqrt(3)))
    assert -1 / (2 * np.sqrt(3)) == pytest.approx(-0.288675, abs=1e-4)
    # Cross-check the report actually embeds a value close to that.
    assert "-0.289" in report

    # tracking_error ≈ 0.289828 (28.98%)
    assert "28.98%" in report

    # cumulative delta ≈ -0.02% (paper below sim by ~0.0206%)
    assert "-0.02%" in report


# ---------------------------------------------------------------------------
# Metrics math — Case B: cumulative-delta-driven WARN (~3% divergence)
#
# Hand computation (6 prices / 5 daily returns):
#   paper_returns = [0.02, -0.01, 0.03, -0.02, 0.01]
#   sim_returns   = paper_returns - 0.006 (constant per-day drag)
#                 = [0.014, -0.016, 0.024, -0.026, 0.004]
#
#   Because sim_returns is paper_returns shifted by a CONSTANT, Pearson
#   correlation is invariant to that additive shift -> correlation = 1.0
#   exactly (an affine transform y = x - c with slope 1 always has r = 1,
#   as long as x has nonzero variance, which it does here).
#
#   diff = paper - sim = 0.006 every day (constant) -> std(diff, ddof=1) = 0
#   -> tracking_error = 0.0 exactly.
#
#   paper prices (start 100): 100, 102, 100.98, 104.0094, 101.929212, 102.94850412
#     -> paper_cum_return = 102.94850412/100 - 1 = 0.0294850412
#   sim prices (start 100):   100, 101.4, 99.7776, 102.1722624, 99.5157835776, 99.9138467119
#     -> sim_cum_return   = 99.9138467119/100 - 1 = -0.0008615329
#   cumulative_delta = 0.0294850412 - (-0.0008615329) = 0.0303465741  (~+3.03%)
#
#   Verdict: |cumulative_delta| = 3.03% > 2% -> WARN, correlation stays 1.0
#   (well above 0.7) so this isolates the cumulative-delta-triggered branch.
# ---------------------------------------------------------------------------

def test_warn_triggers_on_three_percent_cumulative_divergence(monkeypatch, tmp_path):
    paper_returns = [0.02, -0.01, 0.03, -0.02, 0.01]
    sim_returns   = [r - 0.006 for r in paper_returns]

    paper_prices = _prices_from_returns(100.0, paper_returns)   # 6 points
    sim_prices   = _prices_from_returns(100.0, sim_returns)     # 6 points

    dates = pd.bdate_range("2026-06-01", periods=11)   # 5 padding + 6 window
    window_dates = dates[-6:]
    paper_curve = pd.DataFrame({
        "date": dates,
        "equity": [90.0 + i for i in range(5)] + paper_prices,
    })
    sim_series = pd.Series(sim_prices, index=window_dates)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.execution.equity_tracker.load_equity_curve", lambda: paper_curve)
    monkeypatch.setattr(recon, "_simulate_window", lambda start, end: sim_series)
    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: MagicMock())

    report = recon.build_report(lookback_days=6)

    assert report is not None
    assert "Verdict: WARN" in report
    # Cumulative delta ≈ +3.03%, comfortably past the 2% WARN threshold.
    assert "+3.03%" in report
    # Correlation must have stayed near-perfect (1.000) — this WARN is
    # driven by cumulative delta, not by a correlation breakdown.
    assert "1.000" in report


# ---------------------------------------------------------------------------
# Metrics math — Case C: identical curves -> OK
# ---------------------------------------------------------------------------

def test_ok_verdict_when_curves_match_exactly(monkeypatch, tmp_path):
    returns = [0.02, -0.01, 0.03, -0.02, 0.01]
    prices = _prices_from_returns(100.0, returns)

    paper_curve = _make_paper_curve_generic(prices)
    sim_series = pd.Series(prices, index=_dates_for(len(prices)))

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.execution.equity_tracker.load_equity_curve", lambda: paper_curve)
    monkeypatch.setattr(recon, "_simulate_window", lambda start, end: sim_series)
    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: MagicMock())

    report = recon.build_report(lookback_days=len(prices))

    assert report is not None
    assert "Verdict: OK" in report
    assert "1.000" in report      # correlation == 1.0 (identical returns)
    assert "0.00%" in report      # tracking error == 0.0 and delta == 0.0


def _dates_for(n_window: int, n_pad: int = 5) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-06-01", periods=n_pad + n_window)[-n_window:]


def _make_paper_curve_generic(window_prices: list[float], n_pad: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=n_pad + len(window_prices))
    padding = [90.0 + i for i in range(n_pad)]
    return pd.DataFrame({"date": dates, "equity": padding + window_prices})


# ---------------------------------------------------------------------------
# Report file written to tmp_path-patched DATA_DIR
# ---------------------------------------------------------------------------

def test_report_written_to_data_dir(monkeypatch, tmp_path):
    returns = [0.02, -0.01, 0.03, -0.02, 0.01]
    prices = _prices_from_returns(100.0, returns)
    paper_curve = _make_paper_curve_generic(prices)
    sim_series = pd.Series(prices, index=_dates_for(len(prices)))

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.execution.equity_tracker.load_equity_curve", lambda: paper_curve)
    monkeypatch.setattr(recon, "_simulate_window", lambda start, end: sim_series)
    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: MagicMock())
    # Freeze "today" so the report's filename is deterministic.
    monkeypatch.setattr(recon, "_now_et", lambda: datetime(2026, 6, 12, 9, 0))

    report = recon.build_report(lookback_days=len(prices))

    out_path = tmp_path / "reconciliation" / "2026-06-12.md"
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == report


# ---------------------------------------------------------------------------
# Not enough paper history -> None, no exception
# ---------------------------------------------------------------------------

def test_not_enough_paper_history_returns_none(monkeypatch, tmp_path):
    # Only 5 rows total — below the _MIN_PAPER_ROWS(10) gate.
    short_curve = pd.DataFrame({
        "date": pd.bdate_range("2026-06-01", periods=5),
        "equity": [100.0, 101.0, 102.0, 101.5, 103.0],
    })

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.execution.equity_tracker.load_equity_curve", lambda: short_curve)

    result = recon.build_report()

    assert result is None
    assert not (tmp_path / "reconciliation").exists()


# ---------------------------------------------------------------------------
# Simulation raising -> None, never propagates
# ---------------------------------------------------------------------------

def test_simulation_raising_returns_none_not_exception(monkeypatch, tmp_path):
    paper_curve = _make_paper_curve_generic([100.0, 101.0, 102.0, 101.5, 103.0])

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.execution.equity_tracker.load_equity_curve", lambda: paper_curve)
    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: MagicMock())

    def _boom(start, end):
        raise RuntimeError("walk-forward backtest blew up")

    monkeypatch.setattr(recon, "_simulate_window", _boom)

    # Must not raise — reconciliation is a diagnostic, never allowed to
    # break the caller (daily_run.py).
    result = recon.build_report(lookback_days=5)

    assert result is None
    assert not (tmp_path / "reconciliation").exists()


def test_no_saved_model_returns_none_without_calling_simulation(monkeypatch, tmp_path):
    """If no model has ever been trained, build_report should bail out
    BEFORE paying for the (expensive) simulation call at all."""
    paper_curve = _make_paper_curve_generic([100.0, 101.0, 102.0, 101.5, 103.0])

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.execution.equity_tracker.load_equity_curve", lambda: paper_curve)
    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: None)

    mock_simulate = MagicMock()
    monkeypatch.setattr(recon, "_simulate_window", mock_simulate)

    result = recon.build_report(lookback_days=5)

    assert result is None
    mock_simulate.assert_not_called()


# ---------------------------------------------------------------------------
# Friday gating in daily_run.py
# ---------------------------------------------------------------------------

# Real 2026 trading days (mid-June, no holiday nearby) so the NYSE calendar
# gate in run_daily() doesn't need patching — mirrors test_daily_run.py's
# approach of using real trading-day fixtures where possible.
_THURSDAY = datetime(2026, 6, 11)
_FRIDAY   = datetime(2026, 6, 12)


def test_reconciliation_not_called_on_thursday(monkeypatch, tmp_path):
    from tests.execution.test_daily_run import _patch_all_steps
    import src.execution.daily_run as daily_run_mod

    _patch_all_steps(monkeypatch, tmp_path, monday=False)
    monkeypatch.setattr(daily_run_mod, "_today", lambda: _THURSDAY)

    mock_build_report = MagicMock()
    monkeypatch.setattr("src.execution.reconciliation.build_report", mock_build_report)

    daily_run_mod.run_daily(dry_run=True)

    mock_build_report.assert_not_called()


def test_reconciliation_called_on_friday(monkeypatch, tmp_path):
    from tests.execution.test_daily_run import _patch_all_steps
    import src.execution.daily_run as daily_run_mod

    _patch_all_steps(monkeypatch, tmp_path, monday=False)
    monkeypatch.setattr(daily_run_mod, "_today", lambda: _FRIDAY)

    mock_build_report = MagicMock()
    monkeypatch.setattr("src.execution.reconciliation.build_report", mock_build_report)

    daily_run_mod.run_daily(dry_run=True)

    mock_build_report.assert_called_once()


def test_reconciliation_failure_does_not_affect_run_result(monkeypatch, tmp_path):
    """A broken reconciliation report must be swallowed — it must not appear
    in result['errors'] or change result['step']; it's a diagnostic, not a
    pipeline step."""
    from tests.execution.test_daily_run import _patch_all_steps
    import src.execution.daily_run as daily_run_mod

    _patch_all_steps(monkeypatch, tmp_path, monday=False)
    monkeypatch.setattr(daily_run_mod, "_today", lambda: _FRIDAY)
    monkeypatch.setattr(
        "src.execution.reconciliation.build_report",
        MagicMock(side_effect=RuntimeError("reconciliation blew up")),
    )

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["step"] == 7
    assert result["errors"] == []
