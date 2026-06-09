"""Moving-average crossover strategy.

The idea: use two moving averages of different lengths — a "short" one that
reacts quickly to price changes and a "long" one that moves slowly. When the
fast line crosses *above* the slow line, the stock may be gaining momentum →
buy signal. When it crosses *below* → sell signal.

This is called a "golden cross" (buy) / "death cross" (sell) in trading jargon.
It's deliberately simple — the point is to have something to backtest, not to
make money.
"""

import pandas as pd


def ma_crossover_signals(
    close: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
) -> tuple[pd.Series, pd.Series]:
    """Compute buy/sell signals for a moving-average crossover strategy.

    Args:
        close: Daily closing prices for one symbol, indexed by timestamp.
        short_window: Lookback days for the fast moving average (default 20 ≈ 1 month).
        long_window:  Lookback days for the slow moving average (default 50 ≈ 2.5 months).

    Returns:
        entries: Boolean Series — True on the bar where a buy signal fires.
        exits:   Boolean Series — True on the bar where a sell signal fires.

    Note on NaN handling: the first `long_window - 1` rows have no long MA yet,
    so those bars produce NaN and are treated as no-signal by vectorbt.
    """
    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()

    # A crossover happens on bar N when the relationship *changed* vs bar N-1.
    # shift(1) gives us yesterday's value so we can compare.
    entries = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    exits   = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

    return entries, exits
