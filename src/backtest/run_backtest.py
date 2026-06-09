"""Phase 1: backtest the moving-average crossover strategy using vectorbt.

Run from the project root (trading-research/) with:

    python -m src.backtest.run_backtest

What this script does:
1. Loads the CSV for each symbol from data/.
2. Passes the closing prices into the MA crossover strategy to get entry/exit signals.
3. Feeds those signals into vectorbt, which simulates trades and tracks a portfolio.
4. Prints a stats table and saves an equity-curve plot for each symbol.

"Equity curve" = a line graph of your portfolio value over time. If you started
with £10,000 and it ends at £9,200, you lost money — but now you have a machine
that can *measure* that loss, which is the whole point.
"""

import sys
import os

import pandas as pd
import vectorbt as vbt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.strategies.ma_crossover import ma_crossover_signals

# Starting cash for the simulated portfolio, in dollars.
INIT_CASH = 10_000


def backtest_symbol(symbol: str) -> vbt.Portfolio:
    """Load data for one symbol, run the strategy, return a vectorbt Portfolio."""
    csv_path = config.DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No data file for {symbol}. Run `python -m src.data.fetch_data` first."
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close = df["close"]

    entries, exits = ma_crossover_signals(close)

    # vectorbt simulates the portfolio: it buys on entry signals, sells on exits.
    # freq="1D" tells it bars are daily so it can annualise stats correctly.
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=INIT_CASH,
        freq="1D",
    )
    return portfolio


def main() -> None:
    print(f"{'='*60}")
    print(f"  MA Crossover Backtest  (short=20d, long=50d)")
    print(f"  Universe: {config.SYMBOLS}")
    print(f"  Period:   {config.START_DATE.date()} → {config.END_DATE.date()}")
    print(f"  Capital:  ${INIT_CASH:,}")
    print(f"{'='*60}\n")

    for symbol in config.SYMBOLS:
        print(f"--- {symbol} ---")
        pf = backtest_symbol(symbol)

        stats = pf.stats()
        # Pull out the numbers that matter most for a first look.
        print(f"  Total Return:    {stats['Total Return [%]']:.2f}%")
        print(f"  Max Drawdown:    {stats['Max Drawdown [%]']:.2f}%")
        print(f"  Win Rate:        {stats['Win Rate [%]']:.2f}%")
        print(f"  Total Trades:    {int(stats['Total Trades'])}")
        print(f"  Sharpe Ratio:    {stats['Sharpe Ratio']:.3f}")
        print()

        # Save equity curve plot as an HTML file (opens in browser).
        plot_path = config.DATA_DIR / f"{symbol}_equity_curve.html"
        pf.plot().write_html(str(plot_path))
        print(f"  Equity curve saved → {plot_path.relative_to(config.PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
