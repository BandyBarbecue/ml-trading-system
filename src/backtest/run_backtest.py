"""Walk-forward backtest for the ML-based trading pipeline.

This module simulates the full daily trading pipeline on historical data using
a rolling walk-forward scheme.  At each step:
  - Train LightGBM on the most recent MODEL_TRAIN_DAYS of data.
  - Predict scores for the next 21 trading days (1 month).
  - Rebalance the portfolio every Monday using those scores.
  - Apply risk checks before accepting any new weights.
  - Simulate daily P&L and accumulate an equity curve.

Walk-forward analogy: imagine studying for an exam using only notes from the
past two years, sitting the exam, then moving forward one month and repeating.
You never study the exam questions before sitting them — that is the anti-leakage
guarantee this module enforces.

Run from trading-research/ with:
    python -m src.backtest.run_backtest
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is importable regardless of where this is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.ml.model import train, predict
from src.portfolio.construction import build_portfolio
from src.risk.engine import RiskEngine

# Number of trading days in each test (prediction) window.
TEST_DAYS = 21   # roughly one calendar month of trading

# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

# 7 basis points round-trip: this covers broker commission, bid-ask spread,
# and market-impact on a typical mid-cap stock.  1 basis point = 0.01%, so
# 7 bps = 0.07% of notional traded.  Each *side* of the trade costs ~3.5 bps.
TRANSACTION_COST_BPS = 7


def apply_transaction_costs(
    weights_today: pd.Series,
    weights_yesterday: pd.Series,
    apply_costs: bool = True,
) -> float:
    """Compute the transaction cost drag for a single rebalance.

    Analogy: every time you rearrange your portfolio, you pay a small fee to
    the broker proportional to *how much* you moved, not *how large* the
    portfolio is.  If you swap 10% of the portfolio into new stocks you pay
    7 bps on that 10%, not on the full 100%.

    Cost formula:
        turnover   = sum(|weight_today - weight_yesterday|) / 2
        cost_drag  = turnover × (7 / 10000)

    The /2 avoids double-counting: selling stock A and buying stock B both
    appear in the |delta| sum, but they are one round-trip, not two.

    Parameters
    ----------
    weights_today : pd.Series
        Target portfolio weights after rebalancing (index = ticker).
    weights_yesterday : pd.Series
        Portfolio weights before rebalancing (index = ticker).
    apply_costs : bool
        If False, returns 0.0 immediately (used with --no-costs flag).

    Returns
    -------
    float
        Fractional cost to subtract from the period return.
        Example: 0.0007 means a 0.07% drag on portfolio value.
    """
    if not apply_costs:
        return 0.0
    # Align the two series; treat missing tickers in yesterday as 0 (new position).
    delta = weights_today.fillna(0) - weights_yesterday.fillna(0)
    turnover = delta.abs().sum() / 2.0
    return turnover * (TRANSACTION_COST_BPS / 10_000)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Close-price panel for P&L simulation
# ---------------------------------------------------------------------------

def _load_close_price_panel(tickers: list[str]) -> pd.DataFrame:
    """Load a (date x ticker) close-price panel for the given tickers.

    This reuses src/features/price_features.load_all_prices() — the SAME
    price-loading function src/features/feature_matrix.py calls to build the
    momentum/reversal/volatility features. We load it independently here
    because build_feature_matrix() does not carry raw close prices through
    into its output (only derived features + the forward-looking `target`
    rank survive), and the `target` column is exactly what we must NOT use
    for daily P&L (see run_backtest()'s docstring for why).

    Returns
    -------
    pd.DataFrame
        Index = date, columns = ticker, values = close price. Empty
        DataFrame if none of the tickers have a parquet file on disk yet.
    """
    from src.features.price_features import load_all_prices

    prices = load_all_prices(tickers)
    if not prices:
        return pd.DataFrame()
    close_wide = pd.DataFrame({t: df["close"] for t, df in prices.items()})
    return close_wide.sort_index()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown as a positive fraction.

    Drawdown measures how far the portfolio fell from its running peak before
    recovering (or ending the period).  A 15% drawdown means the portfolio
    dropped 15% from its highest point.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value indexed by date.

    Returns
    -------
    float
        Positive number, e.g. 0.15 means a 15% peak-to-trough decline.
    """
    peak = equity_curve.cummax()           # running high-water mark
    drawdown = (equity_curve - peak) / peak  # fraction below peak (always <= 0)
    return float(-drawdown.min())           # flip sign so result is positive


def compute_metrics(equity_curve: pd.Series) -> dict:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value indexed by date.  Must have at least 2 data points.

    Returns
    -------
    dict with keys:
        sharpe       – annualised Sharpe ratio (assumes 252 trading days/year).
        max_drawdown – maximum peak-to-trough decline as a positive fraction.
        total_return – total return as a fraction (e.g. 0.12 = +12%).
    """
    daily_returns = equity_curve.pct_change().dropna()

    std = daily_returns.std()
    # Avoid division by zero if there is no volatility (e.g. single-day curve).
    sharpe = float((daily_returns.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    max_dd = _max_drawdown(equity_curve)
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    return {"sharpe": sharpe, "max_drawdown": max_dd, "total_return": total_return}


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def run_backtest(
    feature_matrix: pd.DataFrame,
    sector_map: dict[str, str],
    initial_value: float = 100_000.0,
    log_path: Path | None = None,
    apply_costs: bool = True,
    close_prices: pd.DataFrame | None = None,
) -> dict:
    """Run a walk-forward backtest of the ML pipeline on historical data.

    Walk-forward logic
    ------------------
    Training window : config.MODEL_TRAIN_DAYS trading days (504 = 2 years).
    Test window     : TEST_DAYS trading days (21 = ~1 month).
    Step            : TEST_DAYS days (non-overlapping test windows).

    At each step the model is trained on rows where date < test_start.
    Predictions are made only on rows where date >= test_start.
    This is the strict anti-leakage guarantee.

    Data leakage guard
    ------------------
    The function asserts at runtime that every training slice ends strictly
    before the corresponding test slice begins.  If you accidentally pass
    overlapping data the assertion will catch it immediately.

    P&L simulation — real returns, lagged by one day
    --------------------------------------------------
    Daily P&L is computed from REAL close-to-close simple returns, not from
    the `target` column. `target` is a TARGET_HORIZON-day (20-day) FORWARD
    cross-sectional return rank (see feature_matrix.py) — using it (even
    re-centred to [-0.5, +0.5]) as "today's return" would let the equity
    curve realise returns that are actually computed from prices up to 20
    days in the future. That is a severe form of look-ahead bias: an older
    version of this function did exactly that.

    We also apply a one-day fill lag: weights decided using day T's close
    (score, HRP, risk checks — all computed from information available AT
    T's close) cannot start earning returns until T+1. In code:
        portfolio_return(T+1) = sum(weights_as_of_T * simple_return(T+1))
    Each day's loop iteration therefore applies TODAY's return using
    yesterday's decided weights FIRST, and only THEN (if it's a rebalance
    day) decides new weights that take effect starting tomorrow. Same-day
    fills (deciding weights off T's close and having them earn T's own
    return) are unrealistic — you cannot trade on a closing price before it
    prints — and were the second leakage bug this function had.

    The close prices used for these returns come from the SAME source
    src/features/feature_matrix.py used to build the price-based features
    (mom_1m, vol_20d, ...) — see close_prices below.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Tidy long-format with columns:
          ``date``  (datetime), ``ticker`` (str), feature columns, ``target`` (float).
    sector_map : dict[str, str]
        Maps ticker → GICS sector name.
    initial_value : float
        Starting portfolio value in dollars.  Default $100,000.
    log_path : Path or None
        Where the RiskEngine should write decision logs.  Defaults to
        ``config.DATA_DIR / "decisions_backtest.jsonl"``.
    apply_costs : bool
        If True (default), deduct 7bps round-trip transaction costs on each
        rebalance.  Pass False (--no-costs flag) for a frictionless comparison.
    close_prices : pd.DataFrame or None
        Optional (date x ticker) close-price panel to use for daily P&L,
        covering (at least) the tickers and dates in `feature_matrix`. If
        None (default), it is loaded automatically from
        data/prices/<TICKER>.parquet via
        src.features.price_features.load_all_prices() — the same source
        feature_matrix.py uses. Tests inject a synthetic panel here instead
        of touching disk.

    Returns
    -------
    dict with keys:
        equity_curve  – pd.Series indexed by date, values = portfolio value ($).
        sharpe        – annualised Sharpe ratio.
        max_drawdown  – maximum peak-to-trough decline (positive fraction).
        total_return  – total return as a fraction.
        n_rebalances  – number of Monday rebalances that fired.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    if "target" not in feature_matrix.columns:
        raise ValueError(
            "feature_matrix must contain a 'target' column (cross-sectional return rank)."
        )
    if "date" not in feature_matrix.columns or "ticker" not in feature_matrix.columns:
        raise ValueError("feature_matrix must contain 'date' and 'ticker' columns.")

    # Sort by date so our index-based slicing is unambiguous.
    fm = feature_matrix.copy()
    fm["date"] = pd.to_datetime(fm["date"])
    fm = fm.sort_values("date").reset_index(drop=True)

    # All unique trading days in chronological order.
    all_dates: list[pd.Timestamp] = sorted(fm["date"].unique())
    n_dates = len(all_dates)

    train_days = config.MODEL_TRAIN_DAYS  # 504

    # Need at least one full train window plus one test window.
    if n_dates < train_days + TEST_DAYS:
        raise ValueError(
            f"feature_matrix has {n_dates} unique dates but needs at least "
            f"{train_days + TEST_DAYS} (MODEL_TRAIN_DAYS + TEST_DAYS)."
        )

    # ------------------------------------------------------------------
    # 0b. Load the REAL close-price panel used for daily P&L (see the
    # "P&L simulation" section of this function's docstring for why this
    # replaced using `target` as a return proxy).
    # ------------------------------------------------------------------
    tickers = fm["ticker"].unique().tolist()
    if close_prices is None:
        close_prices = _load_close_price_panel(tickers)
    if close_prices is None or close_prices.empty:
        raise ValueError(
            "No close-price data available to simulate daily P&L. Either "
            "pass `close_prices` explicitly (a date x ticker DataFrame), or "
            "make sure data/prices/<TICKER>.parquet exists for the tickers "
            "in feature_matrix."
        )
    # Simple (not log) daily returns: (close_t / close_{t-1}) - 1.
    returns_panel = close_prices.sort_index().pct_change()

    # ------------------------------------------------------------------
    # 1. Walk-forward loop
    # ------------------------------------------------------------------
    equity_curve: dict[pd.Timestamp, float] = {}
    portfolio_value = initial_value
    peak_value = initial_value
    held_weights = pd.Series(dtype=float)     # weights currently EARNING returns (decided as of yesterday's close)
    prev_weights = pd.Series(dtype=float)     # weights *before* each rebalance (for turnover cost calc)
    n_rebalances = 0

    risk_log = log_path if log_path is not None else config.DATA_DIR / "decisions_backtest.jsonl"
    risk_engine = RiskEngine(log_path=risk_log)

    # i points to the first date of the test window.
    i = train_days
    while i + TEST_DAYS <= n_dates:
        train_dates = all_dates[i - train_days : i]   # [t-504, t)
        test_dates  = all_dates[i : i + TEST_DAYS]    # [t, t+21)

        test_start = test_dates[0]
        train_end  = train_dates[-1]

        # ------------------------------------------------------------------
        # LEAKAGE GUARD: training data must end strictly before test starts.
        # ------------------------------------------------------------------
        assert train_end < test_start, (
            f"Data leakage detected! Training max date {train_end} is not "
            f"strictly less than test start {test_start}."
        )

        # ------------------------------------------------------------------
        # 2. Train model on training slice only.
        # ------------------------------------------------------------------
        train_slice = fm[fm["date"].isin(set(train_dates))]
        model = train(train_slice, save=False)

        # ------------------------------------------------------------------
        # 3. Simulate each day in the test window.
        # ------------------------------------------------------------------
        for day in test_dates:
            day_slice = fm[fm["date"] == day].copy()
            if day_slice.empty:
                continue

            # ---- Daily P&L: apply TODAY's real return using YESTERDAY's weights ----
            # This is the one-day fill lag (see docstring). `held_weights` were
            # decided using a PRIOR day's close — never today's — so it's
            # legitimate to let them earn today's close-to-close return. On the
            # very first test day held_weights is still empty (no decision has
            # been made yet), so no return is applied.
            if not held_weights.empty and day in returns_panel.index:
                day_returns = returns_panel.loc[day]
                held = held_weights.index.intersection(day_returns.index)
                if len(held) > 0:
                    daily_pnl = float(
                        (held_weights[held] * day_returns[held].fillna(0.0)).sum()
                    )
                    portfolio_value *= (1 + daily_pnl)

            # ---- Monday rebalance: DECIDE new weights from today's close ----
            # pd.Timestamp.weekday() returns 0 for Monday. These new weights
            # are stored into held_weights at the end of this block but will
            # not earn any return until tomorrow's iteration of this loop —
            # today's return (above) already used the OLD held_weights.
            if day.weekday() == config.REBALANCE_WEEKDAY:
                scores = predict(model, day_slice)

                # Build a prices-proxy from recent target values so HRP has
                # something to work with (shape: dates × tickers).
                # We use the last HRP_LOOKBACK days from the training window.
                # NOTE: this HRP covariance input still uses the target-rank
                # proxy (not real returns) — that is a separate, pre-existing
                # simplification and out of scope for this fix, which only
                # covers the P&L simulation path below.
                hrp_dates = train_dates[-config.HRP_LOOKBACK:]
                prices_proxy = (
                    fm[fm["date"].isin(set(hrp_dates))]
                    .pivot(index="date", columns="ticker", values="target")
                    .fillna(0)
                )
                # Only keep tickers that appear in scores to avoid shape mismatches.
                common = [t for t in scores.index if t in prices_proxy.columns]
                if common:
                    prices_proxy = prices_proxy[common]

                try:
                    proposed_weights = build_portfolio(
                        scores=scores,
                        sector_map=sector_map,
                        current_weights=held_weights,
                        prices=prices_proxy,
                    )
                except (ValueError, KeyError) as exc:
                    # ValueError: e.g. build_portfolio's explicit "too few
                    # tickers" check, or scipy's linkage()/squareform()
                    # rejecting a degenerate (NaN/zero-variance) distance
                    # matrix. KeyError: a ticker referenced by `scores` is
                    # missing from prices_proxy/sector_map. Anything else
                    # (e.g. a genuine bug) should still crash loudly instead
                    # of being silently swallowed here.
                    logger.warning(
                        "build_portfolio failed on %s (%s: %s) — keeping "
                        "existing weights.",
                        day.date(), type(exc).__name__, exc,
                    )
                    proposed_weights = held_weights

                # ---- Risk check ----
                # Estimate today's daily P&L (0 on first day). portfolio_value
                # already reflects today's return (applied above), so this
                # correctly measures how far today's move has taken us from peak.
                daily_pnl_pct = (portfolio_value - peak_value) / peak_value if equity_curve else 0.0

                # Snapshot weights BEFORE the risk check so we can measure
                # how much the portfolio actually changed after approval.
                prev_weights = held_weights.copy()

                new_weights, _ = risk_engine.check(
                    proposed_weights=proposed_weights,
                    current_value=portfolio_value,
                    peak_value=peak_value,
                    vix=20.0,          # neutral VIX placeholder (no live VIX in backtest)
                    daily_pnl_pct=daily_pnl_pct,
                )
                n_rebalances += 1

                # ---- Transaction cost drag ----
                # Deduct 7bps on the notional traded at this rebalance.
                # This reduces portfolio value directly, simulating broker fees
                # and bid-ask spread. Costs are paid TODAY (the trade executes
                # at today's close) even though the new weights only start
                # earning returns tomorrow.
                cost_drag = apply_transaction_costs(
                    new_weights, prev_weights, apply_costs=apply_costs
                )
                portfolio_value *= (1 - cost_drag)

                # held_weights now becomes "yesterday's weights" from the
                # perspective of the NEXT loop iteration (tomorrow).
                held_weights = new_weights

            equity_curve[day] = portfolio_value

            # Advance peak tracker for drawdown calculation.
            if portfolio_value > peak_value:
                peak_value = portfolio_value

        i += TEST_DAYS  # slide the window forward by one test period

    # ------------------------------------------------------------------
    # 4. Build output
    # ------------------------------------------------------------------
    if not equity_curve:
        raise ValueError("Backtest produced no equity curve — check input size.")

    equity_series = pd.Series(equity_curve).sort_index()
    metrics = compute_metrics(equity_series)

    return {
        "equity_curve": equity_series,
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
        "n_rebalances": n_rebalances,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Minimal CLI runner — loads the pre-built feature matrix and runs the backtest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for the ML trading pipeline."
    )
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="Run backtest without transaction costs (for comparison).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running the full backtest.",
    )
    args = parser.parse_args()

    apply_costs = not args.no_costs
    cost_label = (
        "no costs (--no-costs)" if args.no_costs
        else f"transaction cost model: {TRANSACTION_COST_BPS}bps"
    )
    logger.info("Backtest config: %s", cost_label)

    # Exit here if this is just a config check — no data loading needed.
    if args.dry_run:
        sys.exit(0)

    features_path = config.FEATURES_DIR / "feature_matrix.parquet"
    if not features_path.exists():
        print(
            f"[run_backtest] Feature matrix not found at {features_path}.\n"
            "Run `python -m src.features.feature_matrix` first."
        )
        return

    print(f"[run_backtest] Loading feature matrix from {features_path} …")
    fm = pd.read_parquet(features_path)

    # Build a minimal sector map from the tickers present in the data.
    # In production this would come from sp500_tickers.csv or the FMP API.
    tickers = fm["ticker"].unique().tolist()
    sector_map = {t: "Unknown" for t in tickers}

    print(f"[run_backtest] Running walk-forward backtest …")
    result = run_backtest(fm, sector_map, apply_costs=apply_costs)

    ec = result["equity_curve"]
    print(f"\n  Period       : {ec.index[0].date()} → {ec.index[-1].date()}")
    print(f"  Total return : {result['total_return']:+.2%}")
    print(f"  Sharpe ratio : {result['sharpe']:.3f}")
    print(f"  Max drawdown : {result['max_drawdown']:.2%}")
    print(f"  Rebalances   : {result['n_rebalances']}")
    print(f"\n  Final value  : ${ec.iloc[-1]:,.2f}  (started at ${ec.iloc[0]:,.2f})")


if __name__ == "__main__":
    main()
