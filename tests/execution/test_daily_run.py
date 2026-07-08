"""Tests for src/execution/daily_run.py

These tests verify the *wiring* of the orchestrator — that each step calls
the right functions with the right arguments.  The individual modules are
fully mocked; we never touch the network, file system, or Alpaca API here.

Mocking strategy:
  All heavy dependencies are imported lazily inside run_daily(), so we patch
  them at their canonical import path (e.g. "src.data.fetch_prices.update_prices").
  We also patch the _today() helper to control the day-of-week / day-of-month
  seen by the orchestrator.

  config.DATA_DIR is monkeypatched to a per-test tmp_path so the new
  state-file interactions added for the kill-switch/halt/lock fixes
  (risk_state.json, equity_curve.parquet, run.lock, target_weights.json)
  never touch the real project data/ directory during a test run.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

import config

# We import the module under test early so that monkeypatching its _today
# helper works correctly even when the module is already cached.
import src.execution.daily_run as daily_run_mod


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A Monday that is NOT the 1st of the month (so we can test independently)
MONDAY_DATE  = datetime(2026, 6, 8)   # Monday, 8 June 2026
# A Tuesday (portfolio construction should be skipped)
TUESDAY_DATE = datetime(2026, 6, 9)   # Tuesday, 9 June 2026
# A Monday that IS the 1st of the month
FIRST_MONDAY = datetime(2026, 6, 1)   # Monday, 1 June 2026
# A Tuesday that IS the 1st of the month
FIRST_TUESDAY = datetime(2026, 6, 2)  # Hmm, 2026-06-01 is a Monday — use March
FIRST_NON_MONDAY = datetime(2026, 3, 1)  # Tuesday, 1 March 2026

# Minimal pd.Series to simulate scores / weights / trades
_FAKE_SCORES  = pd.Series({"AAPL": 0.9, "MSFT": 0.8}, name="predicted_score")
_FAKE_WEIGHTS = pd.Series({"AAPL": 0.6, "MSFT": 0.4}, name="weights")
_FAKE_TRADES  = pd.Series({"AAPL": 0.6, "MSFT": 0.4}, name="trades")


def _patch_all_steps(monkeypatch, tmp_path, *, monday=True, first_of_month=False,
                     model=None, scores=None, weights=None, trades=None):
    """
    Helper: monkeypatch every heavy import inside run_daily() with lightweight
    mocks, then return a dict of the mock objects so tests can assert on them.

    All data-fetch and ML functions become no-ops by default.

    config.DATA_DIR is redirected to tmp_path so risk_state.json,
    equity_curve.parquet, run.lock, and target_weights.json are all written
    into an isolated per-test directory — the halt flag / lock / peak value
    / etc. logic runs for REAL against tmp_path rather than being mocked
    away, which gives genuine coverage of that plumbing while staying
    completely isolated from the real project data/ directory.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    # Pick the right date sentinel
    if monday and first_of_month:
        fake_date = FIRST_MONDAY
    elif monday:
        fake_date = MONDAY_DATE
    elif first_of_month:
        fake_date = FIRST_NON_MONDAY
    else:
        fake_date = TUESDAY_DATE

    monkeypatch.setattr(daily_run_mod, "_today", lambda: fake_date)

    # ---- Step 1 mocks ----
    mock_update_prices   = MagicMock()
    mock_update_news     = MagicMock()
    mock_update_vix      = MagicMock()
    mock_update_earnings = MagicMock()

    mock_update_fundamentals = MagicMock()
    mock_update_insider      = MagicMock()
    mock_update_macro        = MagicMock()

    monkeypatch.setattr("src.data.fetch_prices.update_prices",   mock_update_prices)
    monkeypatch.setattr("src.data.fetch_news.update_news",       mock_update_news)
    monkeypatch.setattr("src.data.fetch_vix.update_vix",         mock_update_vix)
    monkeypatch.setattr("src.data.fetch_earnings.update_earnings", mock_update_earnings)
    monkeypatch.setattr("src.data.fetch_fundamentals.update_fundamentals", mock_update_fundamentals)
    monkeypatch.setattr("src.data.fetch_insider.update_insider",           mock_update_insider)
    monkeypatch.setattr("src.data.fetch_macro.update_macro",               mock_update_macro)

    # ---- Step 3 mocks ----
    mock_feature_matrix = MagicMock(return_value=pd.DataFrame({"ticker": ["AAPL"],
                                                                 "target": [0.5]}))
    mock_get_symbols    = MagicMock(return_value=["AAPL", "MSFT"])
    mock_get_sector_map = MagicMock(return_value={"AAPL": "Tech", "MSFT": "Tech"})
    mock_ic_tracker     = MagicMock()  # ICTracker instance
    mock_ic_tracker_cls = MagicMock(return_value=mock_ic_tracker)
    mock_get_fw         = MagicMock(return_value={})

    monkeypatch.setattr("src.features.feature_matrix.build_feature_matrix",
                        mock_feature_matrix)
    monkeypatch.setattr("src.data.sp500_universe.get_symbols",   mock_get_symbols)
    monkeypatch.setattr("src.data.sp500_universe.get_sector_map", mock_get_sector_map)
    monkeypatch.setattr("src.ml.ic_tracker.ICTracker",            mock_ic_tracker_cls)
    monkeypatch.setattr("src.ml.ic_tracker.get_feature_weights",  mock_get_fw)

    # ---- Step 4 mocks ----
    _model        = model or MagicMock()
    _scores       = scores if scores is not None else _FAKE_SCORES
    mock_predict  = MagicMock(return_value=_scores)
    mock_train    = MagicMock(return_value=_model)
    mock_load_mdl = MagicMock(return_value=_model)
    mock_load_params = MagicMock(return_value={})

    monkeypatch.setattr("src.ml.model.predict",            mock_predict)
    monkeypatch.setattr("src.ml.model.train",              mock_train)
    monkeypatch.setattr("src.ml.model.load_latest_model",  mock_load_mdl)
    monkeypatch.setattr("src.ml.tuner.load_latest_params", mock_load_params)

    # ---- Step 5 mocks ----
    _weights       = weights if weights is not None else _FAKE_WEIGHTS
    _trades        = trades if trades is not None else _FAKE_TRADES
    mock_build_pf  = MagicMock(return_value=_weights)
    mock_comp_tr   = MagicMock(return_value=_trades)

    monkeypatch.setattr("src.portfolio.construction.build_portfolio", mock_build_pf)
    monkeypatch.setattr("src.portfolio.construction.compute_trades",  mock_comp_tr)

    # Helpers used by Step 5 & 6
    monkeypatch.setattr(daily_run_mod, "_get_current_weights",
                        lambda: pd.Series(dtype=float))
    monkeypatch.setattr(daily_run_mod, "_load_recent_returns",
                        lambda symbols, lookback=60: pd.DataFrame())
    monkeypatch.setattr(daily_run_mod, "_get_account_value",
                        lambda: 100_000.0)

    # ---- Step 6 mocks ----
    mock_engine_instance = MagicMock()
    mock_engine_instance.check.return_value = (_weights, [])
    mock_engine_cls = MagicMock(return_value=mock_engine_instance)
    mock_load_vix   = MagicMock(return_value=pd.DataFrame({"vix": [20.0]}))

    monkeypatch.setattr("src.risk.engine.RiskEngine",       mock_engine_cls)
    monkeypatch.setattr("src.data.fetch_vix.load_vix",      mock_load_vix)

    # ---- Step 7 mocks ----
    mock_make_client      = MagicMock(return_value=MagicMock())
    mock_submit_rebalance = MagicMock(return_value=[{"ticker": "AAPL", "side": "buy",
                                                     "shares": 10, "status": "accepted"}])
    mock_get_acct_val     = MagicMock(return_value=100_000.0)

    monkeypatch.setattr("src.execution.paper_trader._make_trading_client",
                        mock_make_client)
    monkeypatch.setattr("src.execution.paper_trader.submit_rebalance",
                        mock_submit_rebalance)
    monkeypatch.setattr("src.execution.paper_trader.get_account_value",
                        mock_get_acct_val)

    return {
        "update_prices":   mock_update_prices,
        "update_news":     mock_update_news,
        "update_vix":      mock_update_vix,
        "update_earnings": mock_update_earnings,
        "build_feature_matrix": mock_feature_matrix,
        "get_symbols":     mock_get_symbols,
        "predict":         mock_predict,
        "train":           mock_train,
        "load_latest_model": mock_load_mdl,
        "load_latest_params": mock_load_params,
        "build_portfolio": mock_build_pf,
        "compute_trades":  mock_comp_tr,
        "submit_rebalance": mock_submit_rebalance,
        "engine_check":    mock_engine_instance.check,
    }


# ---------------------------------------------------------------------------
# test_dry_run_skips_orders
# ---------------------------------------------------------------------------

def test_dry_run_skips_orders(monkeypatch, tmp_path):
    """dry_run=True must produce an empty orders list and still reach step 7."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True)

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["orders"] == [], "dry_run must not submit any orders"
    assert result["step"] == 7,    "pipeline must complete all 7 steps"
    mocks["submit_rebalance"].assert_not_called()


# ---------------------------------------------------------------------------
# test_result_has_all_keys
# ---------------------------------------------------------------------------

def test_result_has_all_keys(monkeypatch, tmp_path):
    """The returned dict must contain every documented key."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    result = daily_run_mod.run_daily(dry_run=True)

    required_keys = {"date", "step", "scores", "weights", "trades",
                     "orders", "risk_decisions", "errors"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


# ---------------------------------------------------------------------------
# test_result_date_matches_today
# ---------------------------------------------------------------------------

def test_result_date_matches_today(monkeypatch, tmp_path):
    """result['date'] must match the date returned by _today()."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    # _today is already patched to MONDAY_DATE = 2026-06-08
    result = daily_run_mod.run_daily(dry_run=True)

    assert result["date"] == "2026-06-08"


# ---------------------------------------------------------------------------
# test_monday_triggers_portfolio_construction
# ---------------------------------------------------------------------------

def test_monday_triggers_portfolio_construction(monkeypatch, tmp_path):
    """On a Monday, build_portfolio must be called."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True)

    daily_run_mod.run_daily(dry_run=True)

    mocks["build_portfolio"].assert_called_once()


# ---------------------------------------------------------------------------
# test_non_monday_skips_portfolio
# ---------------------------------------------------------------------------

def test_non_monday_skips_portfolio(monkeypatch, tmp_path):
    """On a Tuesday (or any non-Monday), build_portfolio must NOT be called."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=False)

    result = daily_run_mod.run_daily(dry_run=True)

    mocks["build_portfolio"].assert_not_called()
    assert result["weights"] is None
    assert result["trades"]  is None


# ---------------------------------------------------------------------------
# test_first_of_month_triggers_retrain
# ---------------------------------------------------------------------------

def test_first_of_month_triggers_retrain(monkeypatch, tmp_path):
    """On the 1st of the month, train() must be called."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True, first_of_month=True)

    daily_run_mod.run_daily(dry_run=True)

    mocks["train"].assert_called_once()
    # load_latest_model is also called (to check if existing model exists),
    # then train() is called for the retrain — both are correct

    # #16(a): params=load_latest_params() must be passed through to train().
    mocks["load_latest_params"].assert_called_once()
    _, train_kwargs = mocks["train"].call_args
    assert "params" in train_kwargs


# ---------------------------------------------------------------------------
# test_non_first_skips_retrain
# ---------------------------------------------------------------------------

def test_non_first_skips_retrain(monkeypatch, tmp_path):
    """On a non-1st day, load_latest_model() must be called (not train)."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True, first_of_month=False)

    daily_run_mod.run_daily(dry_run=True)

    mocks["load_latest_model"].assert_called_once()
    mocks["train"].assert_not_called()


# ---------------------------------------------------------------------------
# test_live_run_submits_orders
# ---------------------------------------------------------------------------

def test_live_run_submits_orders(monkeypatch, tmp_path):
    """dry_run=False on a Monday with trades → submit_rebalance is called."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True)

    result = daily_run_mod.run_daily(dry_run=False)

    mocks["submit_rebalance"].assert_called_once()
    # orders list should be non-empty (our mock returns one order)
    assert len(result["orders"]) == 1


# ---------------------------------------------------------------------------
# test_step_counter_reaches_7
# ---------------------------------------------------------------------------

def test_step_counter_reaches_7(monkeypatch, tmp_path):
    """After a clean run, step must equal 7 regardless of dry_run."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["step"] == 7


# ---------------------------------------------------------------------------
# test_errors_key_empty_on_clean_run
# ---------------------------------------------------------------------------

def test_errors_key_empty_on_clean_run(monkeypatch, tmp_path):
    """When all steps succeed, the errors list must be empty."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["errors"] == [], f"Unexpected errors: {result['errors']}"


# ---------------------------------------------------------------------------
# test_step1_failure_is_recorded_but_pipeline_continues
# ---------------------------------------------------------------------------

def test_step1_failure_is_recorded_but_pipeline_continues(monkeypatch, tmp_path):
    """If Step 1 (data fetch) raises, the error is captured but step 7 is reached."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True)

    # Make update_prices raise
    mocks["update_prices"].side_effect = RuntimeError("Network down")

    result = daily_run_mod.run_daily(dry_run=True)

    # At least one error recorded
    assert any("Step 1" in e for e in result["errors"])
    # Pipeline must still finish (step advanced past 1)
    assert result["step"] == 7


# ---------------------------------------------------------------------------
# test_scores_populated_when_model_available
# ---------------------------------------------------------------------------

def test_scores_populated_when_model_available(monkeypatch, tmp_path):
    """If a model is available, result['scores'] must be a non-None pd.Series."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["scores"] is not None
    assert isinstance(result["scores"], pd.Series)


# ---------------------------------------------------------------------------
# test_weights_none_on_non_monday
# ---------------------------------------------------------------------------

def test_weights_none_on_non_monday(monkeypatch, tmp_path):
    """On a non-Monday, weights and trades must remain None (no rebalance)."""
    _patch_all_steps(monkeypatch, tmp_path, monday=False)

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["weights"] is None
    assert result["trades"]  is None


# ---------------------------------------------------------------------------
# test_target_weights_persisted_on_monday  (#15b wiring)
# ---------------------------------------------------------------------------

def test_target_weights_persisted_on_monday(monkeypatch, tmp_path):
    """Step 5 must persist the final weights to target_weights.json so
    intraday_runner.py can size against real HRP targets."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    daily_run_mod.run_daily(dry_run=True)

    path = tmp_path / "target_weights.json"
    assert path.exists()
    import json
    payload = json.loads(path.read_text())
    assert payload["date"] == "2026-06-08"
    assert payload["weights"] == {"AAPL": pytest.approx(0.6), "MSFT": pytest.approx(0.4)}


# ---------------------------------------------------------------------------
# Kill switch / halt / restart-safety  (#14)
# ---------------------------------------------------------------------------

def test_kill_switch_fires_on_real_loss_and_halts_and_blocks_restart(monkeypatch, tmp_path):
    """A real -3% day (worse than DAILY_LOSS_LIMIT=2%) must trip the REAL
    RiskEngine's kill switch, persist a halt flag for today, and a same-day
    restart must then refuse to run at all — the core of the #14 fix."""
    from src.risk.engine import RiskEngine as RealRiskEngine
    from src.execution import equity_tracker, risk_state

    _patch_all_steps(monkeypatch, tmp_path, monday=True)
    # Use the REAL risk engine for this test instead of the base fixture's mock,
    # so the kill switch's actual threshold logic is exercised.
    monkeypatch.setattr("src.risk.engine.RiskEngine", RealRiskEngine)

    # Seed "yesterday's" equity so today's daily P&L can be computed for real.
    equity_tracker.update_equity(100_000.0, as_of=date(2026, 6, 5))
    # Today's account value is down 3% from that — should trip the 2% kill switch.
    monkeypatch.setattr(daily_run_mod, "_get_account_value", lambda: 97_000.0)

    result = daily_run_mod.run_daily(dry_run=True)

    assert any(d.get("action") == "kill_switch" for d in result["risk_decisions"]), (
        f"Expected a kill_switch decision, got: {result['risk_decisions']}"
    )
    assert risk_state.is_halted("2026-06-08") is True

    # A same-day restart (e.g. Task Scheduler retrying after a crash) must
    # refuse to run at all.
    result2 = daily_run_mod.run_daily(dry_run=True)
    assert result2.get("halted") is True
    assert result2["step"] == 0
    assert any("halted" in e.lower() for e in result2["errors"])


def test_no_kill_switch_no_halt_flag(monkeypatch, tmp_path):
    """On an ordinary day (no loss), no halt flag should be recorded and a
    second run on the same (fake) day should proceed normally."""
    from src.execution import risk_state

    _patch_all_steps(monkeypatch, tmp_path, monday=True)  # default mock engine fires no decisions

    daily_run_mod.run_daily(dry_run=True)

    assert risk_state.is_halted("2026-06-08") is False

    result2 = daily_run_mod.run_daily(dry_run=True)
    assert "halted" not in result2
    assert result2["step"] == 7


# ---------------------------------------------------------------------------
# Run lock  (#18c)
# ---------------------------------------------------------------------------

def test_run_daily_refuses_to_start_while_lock_held(monkeypatch, tmp_path):
    """If run.lock is already held by another (simulated) run, run_daily must
    exit immediately without running any pipeline steps."""
    mocks = _patch_all_steps(monkeypatch, tmp_path, monday=True)

    from src.execution.paper_trader import acquire_run_lock
    assert acquire_run_lock() is True  # simulate another run already in progress

    result = daily_run_mod.run_daily(dry_run=True)

    assert result["step"] == 0
    assert any("lock" in e.lower() for e in result["errors"])
    mocks["build_portfolio"].assert_not_called()


def test_run_daily_releases_lock_after_completion(monkeypatch, tmp_path):
    """After a normal run finishes, run.lock must be released so a subsequent
    run can acquire it again."""
    _patch_all_steps(monkeypatch, tmp_path, monday=True)

    daily_run_mod.run_daily(dry_run=True)

    assert not (tmp_path / "run.lock").exists()
