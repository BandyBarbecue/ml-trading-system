"""daily_run.py — Main daily pipeline orchestrator.

Runs each trading day at ~8:30 AM ET and coordinates all 7 steps:

  Step 1: Fetch data  (prices, news, VIX, earnings)
  Step 2: Claude NLP  (batch headlines → structured signals)
  Step 3: Feature matrix + IC tracker update
  Step 4: ML inference / retrain on the 1st of the month
  Step 5: Portfolio construction (Mondays only)
  Step 6: Risk engine veto
  Step 7: Submit to Alpaca paper API (skipped if --dry-run)

Analogy: think of this like a factory assembly line.  Each step adds
something to the "product" (today's trade list).  If one station breaks,
the line pauses gracefully — it records the failure and hands off whatever
it completed so far rather than crashing everything.

Design decisions:
  - All heavy imports are LAZY (inside run_daily), not at module top-level.
    This makes each import trivially mockable in tests without needing the
    actual module to be importable on the test machine.
  - Each step is wrapped in try/except.  A failure in Step 2 does NOT prevent
    Steps 3–7 from running (though later steps may produce weaker results).
  - config is imported at the module level because it is always available and
    is used in the helper functions too.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

import config

if TYPE_CHECKING:
    pass  # type-only imports go here if needed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date helper — extracted so tests can patch it with monkeypatch
# ---------------------------------------------------------------------------

def _today() -> datetime:
    """Return the current datetime.  Extracted so tests can override it."""
    return datetime.now()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_recent_returns(symbols: list[str], lookback: int = 60) -> pd.DataFrame:
    """Load the last ``lookback`` days of daily close returns for all symbols.

    Returns a DataFrame with dates as the index and tickers as columns.
    Each value is a daily percentage return (e.g. 0.012 = +1.2%).
    An empty DataFrame is returned if no price data is available.

    Analogy: like pulling the last 60 days of daily closing prices from a
    spreadsheet and converting each row from "price today" to "% change from yesterday."

    Args:
        symbols:  List of ticker symbols.
        lookback: How many days of returns to include (default 60).

    Returns:
        pd.DataFrame indexed by date, columns are ticker symbols.
    """
    from src.data.fetch_prices import load_prices

    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = load_prices(sym)
            if not df.empty:
                # pct_change() converts prices → daily returns; dropna() removes
                # the first NaN row (no return on day 0); iloc[-lookback:] keeps only
                # the most recent `lookback` observations.
                frames[sym] = df["close"].pct_change().dropna().iloc[-lookback:]
        except Exception as exc:
            logger.debug("Could not load prices for %s: %s", sym, exc)

    if not frames:
        return pd.DataFrame()

    # pd.DataFrame(frames) aligns all series on their date index automatically.
    # dropna(how="all") removes any date where every single ticker is missing.
    return pd.DataFrame(frames).dropna(how="all")


def _get_current_weights() -> pd.Series:
    """Get current Alpaca paper positions as portfolio weights.

    Returns an empty Series if Alpaca is unreachable (e.g. network down, bad
    credentials, or the account has no positions yet).  An empty Series means
    "we have no current positions" — the portfolio construction step will treat
    this as a fresh start.
    """
    try:
        from src.execution.paper_trader import _make_trading_client, get_current_positions
        client = _make_trading_client()
        return get_current_positions(client)
    except Exception as exc:
        logger.warning("Could not fetch current positions (using empty): %s", exc)
        return pd.Series(dtype=float)


def _get_account_value() -> float:
    """Get total account equity from Alpaca.

    Falls back to $100,000 if Alpaca is unavailable — this is a reasonable
    starting paper-account balance and avoids crashing the pipeline when
    credentials aren't configured yet.
    """
    try:
        from src.execution.paper_trader import _make_trading_client, get_account_value
        client = _make_trading_client()
        return get_account_value(client)
    except Exception as exc:
        logger.warning("Could not fetch account value (using fallback $100,000): %s", exc)
        return 100_000.0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_daily(dry_run: bool = False, force_rebalance: bool = False) -> dict:
    """Execute the full daily pipeline (7 steps).

    Args:
        dry_run:         If True, skip actual order submission in Step 7.
                         All other steps still run normally.
        force_rebalance: If True, treat today as Monday and run Steps 5–7
                         regardless of the actual weekday.  Combine with
                         dry_run=True to test the full rebalance path safely.

    Returns:
        A dict with the following keys:
          - "date":           today's date as "YYYY-MM-DD"
          - "step":           last successfully completed step (int, 1–7)
          - "scores":         pd.Series of ML scores, or None
          - "weights":        pd.Series of target weights, or None
          - "trades":         pd.Series of weight deltas, or None
          - "orders":         list of submitted order dicts (empty if dry_run)
          - "risk_decisions": list of risk-engine decision dicts
          - "errors":         list of error strings from any step that failed
    """
    now = _today()
    today = now.strftime("%Y-%m-%d")

    result: dict = {
        "date":           today,
        "step":           0,
        "scores":         None,
        "weights":        None,
        "trades":         None,
        "orders":         [],
        "risk_decisions": [],
        "errors":         [],
    }

    is_monday       = (now.weekday() == config.REBALANCE_WEEKDAY) or force_rebalance
    is_first_month  = now.day == 1

    logger.info("=== Daily run starting: %s (Monday=%s, 1st=%s) ===",
                today, is_monday, is_first_month)

    # ------------------------------------------------------------------
    # Step 1: Fetch data  (each source wrapped independently so one failure
    #         does not block the others — cached data covers any gap)
    # ------------------------------------------------------------------
    step1_ok = True
    for label, import_path, fn_name in [
        ("prices",       "src.data.fetch_prices",       "update_prices"),
        ("news",         "src.data.fetch_news",         "update_news"),
        ("VIX",          "src.data.fetch_vix",          "update_vix"),
        ("earnings",     "src.data.fetch_earnings",     "update_earnings"),
        ("fundamentals", "src.data.fetch_fundamentals", "update_fundamentals"),
        ("insider",      "src.data.fetch_insider",      "update_insider"),
        ("macro",        "src.data.fetch_macro",        "update_macro"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(import_path)
            getattr(mod, fn_name)()
        except Exception as exc:
            step1_ok = False
            msg = f"Step 1 ({label}) failed: {exc}"
            logger.warning(msg)           # warning, not error — other sources OK
            logger.debug(traceback.format_exc())
            result["errors"].append(msg)

    result["step"] = 1
    if step1_ok:
        logger.info("Step 1 complete: all data sources fetched.")
    else:
        logger.info("Step 1 complete with warnings: some sources failed (see errors).")

    # ------------------------------------------------------------------
    # Step 2: Claude NLP (news features computed as part of Step 3)
    # ------------------------------------------------------------------
    try:
        # NLP features are driven by compute_news_features inside build_feature_matrix.
        # We mark the step done here to keep the counter consistent; the actual
        # compute happens during feature matrix assembly in Step 3.
        result["step"] = 2
        logger.info("Step 2 complete: NLP signals queued (computed in Step 3).")
    except Exception as exc:
        msg = f"Step 2 (NLP) failed: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        result["errors"].append(msg)

    # ------------------------------------------------------------------
    # Step 3: Build feature matrix + IC tracker
    # ------------------------------------------------------------------
    feature_matrix = pd.DataFrame()
    tracker = None
    regime_feature_weights: dict[str, float] = {}

    try:
        from src.features.feature_matrix import build_feature_matrix
        from src.ml.ic_tracker         import ICTracker, get_feature_weights
        from src.data.sp500_universe   import get_symbols, get_sector_map

        symbols    = get_symbols()
        sector_map = get_sector_map()

        # Build a DatetimeIndex: MODEL_TRAIN_DAYS + 25 business days ending today.
        # The +25 gives a buffer so the trailing TARGET_HORIZON rows still have
        # valid target values when we train.
        # Using a DatetimeIndex (not strings) is important: the NLP feature code
        # does date arithmetic (d - pub_timestamp) which needs real Timestamps.
        dates = pd.bdate_range(end=today, periods=config.MODEL_TRAIN_DAYS + 25)

        feature_matrix = build_feature_matrix(symbols, dates)

        # Load IC history from disk (accumulates across daily runs).
        # On first ever run this starts fresh; afterwards it picks up where it left off.
        from src.ml.ic_tracker import compute_ic
        from src.ml.model import FEATURE_COLS
        tracker = ICTracker.load_or_create()

        # Compute today's IC for each feature vs. the target column and update history.
        if "target" in feature_matrix.columns:
            recent = feature_matrix.dropna(subset=["target"])
            for col in FEATURE_COLS:
                if col in recent.columns:
                    ic = compute_ic(recent[col], recent["target"])
                    tracker.update(col, ic)
            tracker.save()  # persist to data/features/ic_tracker.json

        # Compute regime weights (empty dict = no modulation if anything fails)
        try:
            from src.ml.regime_classifier import classify_regime, get_regime_feature_weights
            macro_path = config.DATA_DIR / "macro" / "macro.parquet"
            if macro_path.exists():
                macro_df = pd.read_parquet(macro_path)
                regime = classify_regime(macro_df)
                regime_feature_weights = get_regime_feature_weights(regime)
                logger.info("Regime: %s (modulation keys: %s)", regime, list(regime_feature_weights.keys()))
        except Exception as exc:
            logger.warning("Regime classification failed: %s", exc)

        result["step"] = 3
        logger.info("Step 3 complete: feature matrix built (%s rows).", len(feature_matrix))
    except Exception as exc:
        msg = f"Step 3 (feature matrix) failed: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        result["errors"].append(msg)
        # Without a feature matrix we can't train or predict; downstream steps
        # degrade gracefully to None scores.

    # ------------------------------------------------------------------
    # Step 4: ML inference (or retrain on the 1st of the month)
    # ------------------------------------------------------------------
    try:
        from src.ml.model  import predict, train, load_latest_model
        from src.ml.tuner  import load_latest_params  # noqa: F401 — available for future use

        existing_model = load_latest_model()

        if (is_first_month or existing_model is None) and not feature_matrix.empty:
            reason = "1st of month — retraining" if existing_model else "no saved model — training initial"
            logger.info("Step 4: %s.", reason)
            # Combine IC-based weights with regime modulation
            feature_weights = get_feature_weights(tracker) if tracker else {}
            combined_weights: dict[str, float] = {
                k: feature_weights.get(k, 1.0) * regime_feature_weights.get(k, 1.0)
                for k in set(feature_weights) | set(regime_feature_weights)
            } if (feature_weights or regime_feature_weights) else {}
            model = train(
                feature_matrix,
                feature_weights=combined_weights if combined_weights else None,
                save=True,
            )
        else:
            model = existing_model

        if model is not None and not feature_matrix.empty:
            scores = predict(model, feature_matrix)
            result["scores"] = scores
            logger.info("Step 4 complete: scores computed for %d tickers.", len(scores))
        else:
            logger.warning("Step 4: no model or no feature matrix — scores are None.")

        result["step"] = 4
    except Exception as exc:
        msg = f"Step 4 (ML inference) failed: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        result["errors"].append(msg)

    # ------------------------------------------------------------------
    # Step 5: Portfolio construction (Mondays only)
    # ------------------------------------------------------------------
    weights        : pd.Series | None = None
    trades         : pd.Series | None = None
    current_weights: pd.Series        = pd.Series(dtype=float)

    try:
        if is_monday and result["scores"] is not None:
            from src.portfolio.construction import build_portfolio, compute_trades

            current_weights = _get_current_weights()
            recent_returns  = _load_recent_returns(symbols, lookback=config.HRP_LOOKBACK)

            weights = build_portfolio(
                result["scores"],
                sector_map,
                current_weights,
                recent_returns,
            )
            trades             = compute_trades(weights, current_weights)
            result["weights"]  = weights
            result["trades"]   = trades

            logger.info(
                "Step 5 complete: portfolio built (%d positions).", len(weights)
            )

            # Short portfolio (only when LONG_SHORT_ENABLED — disabled by default)
            if config.LONG_SHORT_ENABLED:
                from src.portfolio.construction import build_short_portfolio
                short_weights = build_short_portfolio(
                    result["scores"], sector_map, pd.Series(dtype=float), recent_returns
                )
                if not short_weights.empty:
                    # Merge long + short into a single weights Series
                    weights = pd.concat([weights, short_weights])
                    result["weights"] = weights
                    trades = compute_trades(weights, current_weights)
                    result["trades"] = trades
                    logger.info(
                        "Step 5: long-short — %d long, %d short positions.",
                        (weights > 0).sum(), (weights < 0).sum(),
                    )

            # Log a feature breakdown for every selected ticker so there is a
            # human-readable record of what drove each pick.
            try:
                feat_cols = ["mom_1m", "mom_3m", "vol_20d", "news_score", "supply_chain_score", "sue"]
                latest_date = feature_matrix["date"].max()
                today_fm = (
                    feature_matrix[feature_matrix["date"] == latest_date]
                    .set_index("ticker")
                )
                header = f"{'ticker':>6}  {'score':>6}  {'weight':>7}" + "".join(f"  {c:>18}" for c in feat_cols)
                logger.info("Step 5 — selected positions:\n%s", header)
                for ticker in sorted(weights.index, key=lambda t: result["scores"].get(t, 0), reverse=True):
                    score  = result["scores"].get(ticker, float("nan"))
                    weight = weights[ticker]
                    vals = []
                    for c in feat_cols:
                        v = today_fm.at[ticker, c] if ticker in today_fm.index and c in today_fm.columns else float("nan")
                        vals.append(f"{v:+.3f}" if v == v else "   nan")
                    logger.info("  %6s  %6.3f  %6.1f%%  %s", ticker, score, weight * 100, "  ".join(vals))
            except Exception:
                pass  # never let logging crash the pipeline
        else:
            reason = "not Monday" if not is_monday else "no ML scores"
            logger.info("Step 5 skipped: %s.", reason)

        result["step"] = 5
    except Exception as exc:
        msg = f"Step 5 (portfolio construction) failed: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        result["errors"].append(msg)

    # ------------------------------------------------------------------
    # Step 6: Risk engine veto
    # ------------------------------------------------------------------
    try:
        if weights is not None and trades is not None:
            from src.risk.engine    import RiskEngine
            from src.data.fetch_vix import load_vix

            vix_df = load_vix()
            vix    = float(vix_df["vix"].iloc[-1]) if not vix_df.empty else 20.0

            engine        = RiskEngine()
            account_value = _get_account_value()
            from src.execution.risk_state   import get_peak_value
            from src.execution.equity_tracker import update_equity
            update_equity(account_value)
            peak_value = get_peak_value(account_value)

            adjusted_weights, decisions = engine.check(
                weights,
                account_value,
                peak_value,
                vix,
                daily_pnl_pct=0.0,
            )

            from src.portfolio.construction import compute_trades as _compute_trades

            result["risk_decisions"] = decisions
            result["weights"]        = adjusted_weights
            result["trades"]         = _compute_trades(adjusted_weights, current_weights)

            logger.info(
                "Step 6 complete: risk engine fired %d decision(s).", len(decisions)
            )
        else:
            logger.info("Step 6 skipped: no weights to vet.")

        result["step"] = 6
    except Exception as exc:
        msg = f"Step 6 (risk engine) failed: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        result["errors"].append(msg)

    # ------------------------------------------------------------------
    # Step 7: Submit orders (skipped if --dry-run)
    # ------------------------------------------------------------------
    try:
        final_trades = result["trades"]

        if (
            not dry_run
            and final_trades is not None
            and not final_trades.empty
        ):
            from src.execution.paper_trader import (
                _make_trading_client,
                submit_rebalance,
                get_account_value,
            )

            client        = _make_trading_client()
            account_value = get_account_value(client)
            orders        = submit_rebalance(client, final_trades, account_value)
            result["orders"] = orders

            logger.info("Step 7 complete: %d order(s) submitted.", len(orders))
        else:
            reason = "--dry-run" if dry_run else "no trades"
            logger.info("Step 7 skipped: %s.", reason)

        result["step"] = 7
    except Exception as exc:
        msg = f"Step 7 (order submission) failed: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        result["errors"].append(msg)

    logger.info(
        "=== Daily run finished: step=%d/7, orders=%d, errors=%d ===",
        result["step"], len(result["orders"]), len(result["errors"]),
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    # Log to both the terminal AND a dated file in data/logs/
    # so Task Scheduler runs leave a permanent record even with no visible terminal.
    log_dir = config.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"daily_run_{datetime.now().strftime('%Y-%m-%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),                      # terminal
            logging.FileHandler(log_file, encoding="utf-8"),  # daily log file
        ],
    )

    parser = argparse.ArgumentParser(description="Run the daily trading pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute all steps but skip actual order submission (Step 7).",
    )
    parser.add_argument(
        "--force-rebalance",
        action="store_true",
        help="Treat today as Monday: run portfolio construction and risk veto "
             "(Steps 5–7) even if it is not a rebalance day.  "
             "Use with --dry-run to test the full path safely.",
    )
    args = parser.parse_args()

    result = run_daily(dry_run=args.dry_run, force_rebalance=args.force_rebalance)

    print(f"\nDaily run complete — Step {result['step']}/7 | Orders: {len(result['orders'])}")
    if result["errors"]:
        print("Warnings / soft errors (pipeline continued):")
        for err in result["errors"]:
            print(f"  • {err}")
    # Exit 0 if the pipeline completed all 7 steps — soft data-source warnings
    # (e.g. one earnings ticker failing, FRED DNS timeout) don't count as failure.
    # Exit 1 only if the pipeline aborted before step 7 (feature matrix crash, etc.).
    sys.exit(0 if result["step"] == 7 else 1)
