"""Phase 4: Walk-forward optimisation with out-of-sample holdout.

THE CORE IDEA
=============
A normal backtest fits parameters to the full dataset and then measures
performance on that *same* dataset -- the strategy looks great because it
was tuned to the data it's being graded on. That's overfitting.

Walk-forward solves this by never letting the optimizer see the test data:

    |---- TRAIN (120 days) ----|-- TEST (40 days) --|
                                    -> slide ->
         |---- TRAIN (120 days) ----|-- TEST (40 days) --|
                                         -> slide ->
              |---- TRAIN (120 days) ----|-- TEST (40 days) --|

In each fold:
  1. Try all (short_window, long_window) combinations on the TRAIN slice.
  2. Pick the pair with the highest Sharpe ratio.
  3. Backtest ONLY those parameters on the TEST slice (never seen before).
  4. Record the test result -- this is the honest, out-of-sample performance.

The average of the TEST results tells you how the strategy *actually* holds up
beyond the data it was trained on.

Run from trading-research/ with:
    python -m src.backtest.walk_forward
"""

import itertools
import sys
import os
from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.strategies.ma_crossover import ma_crossover_signals

INIT_CASH = 10_000

# Walk-forward window sizes (in trading days).
TRAIN_DAYS = 120   # length of each optimisation window
TEST_DAYS  = 40    # length of each evaluation window; also the step size

# Parameter grid to search over.
# Only combinations where short_window < long_window are valid.
SHORT_WINDOWS = [10, 15, 20, 25]
LONG_WINDOWS  = [30, 40, 50, 60, 75]


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_short: int
    best_long: int
    train_sharpe: float
    test_return_pct: float
    test_sharpe: float
    test_trades: int


def _optimise(close_train: pd.Series) -> tuple[int, int, float]:
    """Try all valid parameter pairs on the training slice.

    Returns (best_short_window, best_long_window, best_sharpe).
    If no parameters produce any signals, returns the strategy defaults.
    """
    best_sharpe = float("-inf")
    best_short = SHORT_WINDOWS[0]
    best_long  = LONG_WINDOWS[-1]

    for short_w, long_w in itertools.product(SHORT_WINDOWS, LONG_WINDOWS):
        if short_w >= long_w:
            continue

        entries, exits = ma_crossover_signals(close_train, short_w, long_w)

        # Skip if no trades fire -- Sharpe is undefined with zero trades.
        if entries.sum() == 0:
            continue

        pf = vbt.Portfolio.from_signals(
            close_train, entries, exits, init_cash=INIT_CASH, freq="1D"
        )
        sharpe = pf.stats()["Sharpe Ratio"]
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_short = short_w
            best_long = long_w

    return best_short, best_long, best_sharpe


def _test_fold(
    close: pd.Series,
    test_start_idx: int,
    test_end_idx: int,
    short_w: int,
    long_w: int,
) -> tuple[float, float, int]:
    """Evaluate a parameter pair on a test window.

    WHY we prepend warmup bars: a 75-day moving average needs 75 price bars
    before it produces a valid number. A 40-day test window alone would give
    NaN MAs for the first 74 bars -- no signals would ever fire. So we borrow
    the last `long_w` bars from the training window just to warm up the MA,
    then slice the signals back to the test window only. No future data leaks
    in -- MAs are always computed looking *backwards*.

    Returns (total_return_pct, sharpe_ratio, total_trades).
    """
    # Include warmup bars from before the test window to prime the MA.
    warmup_start = max(0, test_start_idx - long_w)
    n_warmup = test_start_idx - warmup_start

    combined = close.iloc[warmup_start:test_end_idx]
    entries_all, exits_all = ma_crossover_signals(combined, short_w, long_w)

    # Slice to test portion only.
    test_close   = combined.iloc[n_warmup:]
    test_entries = entries_all.iloc[n_warmup:]
    test_exits   = exits_all.iloc[n_warmup:]

    if test_entries.sum() == 0:
        # No signals in the test window -- hold cash, 0% return.
        return 0.0, 0.0, 0

    pf = vbt.Portfolio.from_signals(
        test_close, test_entries, test_exits, init_cash=INIT_CASH, freq="1D"
    )
    stats = pf.stats()
    return (
        float(stats["Total Return [%]"]),
        float(stats["Sharpe Ratio"]),
        int(stats["Total Trades"]),
    )


def walk_forward_symbol(symbol: str) -> list[FoldResult]:
    """Run the full walk-forward loop for one symbol.

    Returns one FoldResult per fold.
    """
    csv_path = config.DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No data for {symbol}. Run `python -m src.data.fetch_data` first."
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close = df["close"]
    n = len(close)

    results: list[FoldResult] = []
    fold = 0
    start = 0

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end  = start + TRAIN_DAYS
        test_end   = train_end + TEST_DAYS

        close_train = close.iloc[start:train_end]

        best_short, best_long, train_sharpe = _optimise(close_train)
        ret, sharpe, trades = _test_fold(close, train_end, test_end, best_short, best_long)

        results.append(FoldResult(
            fold=fold + 1,
            train_start=str(close.index[start].date()),
            train_end=str(close.index[train_end - 1].date()),
            test_start=str(close.index[train_end].date()),
            test_end=str(close.index[test_end - 1].date()),
            best_short=best_short,
            best_long=best_long,
            train_sharpe=round(train_sharpe, 3),
            test_return_pct=round(ret, 2),
            test_sharpe=round(sharpe, 3),
            test_trades=trades,
        ))

        start += TEST_DAYS
        fold += 1

    return results


def print_symbol_report(symbol: str, folds: list[FoldResult]) -> None:
    if not folds:
        print(f"  {symbol}: not enough data for a single fold.")
        return

    print(f"\n{'='*72}")
    print(f"  Walk-Forward Results: {symbol}")
    print(f"  Train={TRAIN_DAYS}d  Test={TEST_DAYS}d  |  "
          f"Param grid: short={SHORT_WINDOWS}  long={LONG_WINDOWS}")
    print(f"{'='*72}")
    print(f"  {'Fold':<5} {'Train period':<25} {'Test period':<25} {'Best MA':<10} "
          f"{'Train SR':<10} {'Test Ret%':<10} {'Test SR':<10} {'Trades'}")
    print(f"  {'-'*67}")

    for r in folds:
        ma_label = f"{r.best_short}/{r.best_long}"
        train_period = f"{r.train_start} -> {r.train_end}"
        test_period  = f"{r.test_start} -> {r.test_end}"
        print(f"  {r.fold:<5} {train_period:<25} {test_period:<25} {ma_label:<10} "
              f"{r.train_sharpe:<10.3f} {r.test_return_pct:<10.2f} {r.test_sharpe:<10.3f} {r.test_trades}")

    avg_ret    = sum(r.test_return_pct for r in folds) / len(folds)
    avg_sharpe = sum(r.test_sharpe for r in folds) / len(folds)
    total_trades = sum(r.test_trades for r in folds)

    print(f"\n  Out-of-sample summary ({len(folds)} folds):")
    print(f"    Avg test return:  {avg_ret:+.2f}%  per {TEST_DAYS}-day period")
    print(f"    Avg test Sharpe:  {avg_sharpe:.3f}")
    print(f"    Total OOS trades: {total_trades}")

    if avg_ret > 0:
        print(f"\n  Interpretation: on average, the optimised parameters earned "
              f"{avg_ret:+.2f}% per test window on data they were never trained on.")
    else:
        print(f"\n  Interpretation: even after optimisation, the strategy lost "
              f"{avg_ret:.2f}% per test window out-of-sample. This is common -- "
              f"simple MA crossover is often not profitable after transaction costs.")


def main() -> None:
    print(f"\n{'='*72}")
    print(f"  Phase 4 -- Walk-Forward Optimisation")
    print(f"  Optimise MA parameters on training windows, evaluate out-of-sample.")
    print(f"  TRAIN={TRAIN_DAYS} days  |  TEST={TEST_DAYS} days  |  "
          f"Step={TEST_DAYS} days")
    print(f"{'='*72}")

    for symbol in config.SYMBOLS:
        try:
            folds = walk_forward_symbol(symbol)
            print_symbol_report(symbol, folds)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {e}")

    print(f"\n  Done. Walk-forward complete for {len(config.SYMBOLS)} symbols.\n")


if __name__ == "__main__":
    main()
