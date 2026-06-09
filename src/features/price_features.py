"""
price_features.py — cross-sectional price-based feature engineering.

Computes momentum (1m/3m/12m with skip periods), short-term mean reversion,
and realised volatility for a universe of tickers, then z-scores each feature
cross-sectionally (i.e. each row is normalised across tickers so the average
score on any given day is zero).

Why cross-sectional z-scoring?
  Think of it like grading on a curve: instead of asking "is AAPL's momentum
  positive?" we ask "is AAPL's momentum above average for today's universe?".
  This removes market-wide effects — on a day when every stock rips, we still
  want to rank stocks *relative* to each other, not reward the whole market.
"""

import pandas as pd
import numpy as np
from config import PRICES_DIR


def compute_momentum(
    prices: dict[str, pd.DataFrame],
    window: int,
    skip: int,
) -> pd.DataFrame:
    """Compute momentum = return over [t-window-skip : t-skip] for each ticker.

    The ``skip`` period (usually 1 week = 5 trading days) prevents short-term
    reversal from contaminating the momentum signal.  Academic research shows
    the most-recent week often *reverses*, so we skip it and look further back.

    Args:
        prices: mapping of ticker → DataFrame with a "close" column.
        window: how many trading days to measure the return over.
        skip: how many recent days to exclude from the window.

    Returns:
        DataFrame indexed by date; columns are ticker symbols.
        Rows 0..(window+skip-1) are NaN — not enough history yet.
    """
    result = {}
    for ticker, df in prices.items():
        close = df["close"]
        # close.shift(skip) = price skip days ago
        # close.shift(window + skip) = price (window+skip) days ago
        # ratio - 1 = return over the window, excluding the skip period
        ret = close.shift(skip) / close.shift(window + skip) - 1
        result[ticker] = ret
    return pd.DataFrame(result)


def compute_reversal(
    prices: dict[str, pd.DataFrame],
    window: int = 5,
) -> pd.DataFrame:
    """Short-term mean reversion: negative of the window-day return.

    The idea: stocks that have fallen sharply over the past week tend to
    *bounce back* in the short run (and vice versa).  By negating the return
    we get a signal that is positive for recent losers (expected bouncers).

    Args:
        prices: mapping of ticker → DataFrame with a "close" column.
        window: lookback in trading days (default 5 = 1 week).

    Returns:
        DataFrame indexed by date; columns are ticker symbols.
    """
    result = {}
    for ticker, df in prices.items():
        close = df["close"]
        ret = close / close.shift(window) - 1
        result[ticker] = -ret  # negative: recent losers expected to bounce
    return pd.DataFrame(result)


def compute_vol(
    prices: dict[str, pd.DataFrame],
    window: int = 20,
) -> pd.DataFrame:
    """Realised volatility: rolling standard deviation of daily log returns.

    Uses pct_change (≈ log return for small moves) and a rolling window.
    Higher vol → more uncertainty about the stock's price path.

    Args:
        prices: mapping of ticker → DataFrame with a "close" column.
        window: rolling window in trading days (default 20 = 1 month).

    Returns:
        DataFrame indexed by date; columns are ticker symbols.
    """
    result = {}
    for ticker, df in prices.items():
        daily_ret = df["close"].pct_change()
        result[ticker] = daily_ret.rolling(window).std()
    return pd.DataFrame(result)


def cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across tickers (cross-sectional normalisation).

    For every date (row), subtracts the cross-sectional mean and divides by
    the cross-sectional std.  Result: each row has mean ≈ 0, std ≈ 1.

    Rows where all tickers are NaN, or where std == 0, stay NaN (we replace
    zero std with NaN to avoid divide-by-zero producing inf values).

    Args:
        df: DataFrame with dates as index, tickers as columns.

    Returns:
        DataFrame of the same shape with z-scored values.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.subtract(mean, axis=0).divide(std, axis=0)


def load_all_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load close prices for all symbols from parquet files in PRICES_DIR.

    Silently skips any ticker whose parquet file does not exist yet.

    Args:
        symbols: list of ticker strings (e.g. ["AAPL", "MSFT"]).

    Returns:
        dict mapping ticker → DataFrame with at least a "close" column.
    """
    prices: dict[str, pd.DataFrame] = {}
    for ticker in symbols:
        path = PRICES_DIR / f"{ticker}.parquet"
        if path.exists():
            prices[ticker] = pd.read_parquet(path)[["close"]]
    return prices


def compute_price_features(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute all price features and return a wide DataFrame.

    Output shape: index = date, columns = MultiIndex of (feature, ticker).

    Features computed (all cross-sectionally z-scored):
      mom_1m   — 1-month momentum  (21-day return, skip last 5 days)
      mom_3m   — 3-month momentum  (63-day return, skip last 5 days)
      mom_12m  — 12-month momentum (252-day return, skip last 21 days)
      rev_1w   — 1-week mean reversion (negative 5-day return)
      vol_20d  — 20-day realised volatility

    Rows that are entirely NaN after z-scoring are dropped.

    Args:
        prices: mapping of ticker → DataFrame with a "close" column.

    Returns:
        Wide DataFrame with MultiIndex columns ``(feature, ticker)``.
    """
    raw: dict[str, pd.DataFrame] = {
        "mom_1m":  compute_momentum(prices, window=21,  skip=5),
        "mom_3m":  compute_momentum(prices, window=63,  skip=5),
        "mom_12m": compute_momentum(prices, window=252, skip=21),
        "rev_1w":  compute_reversal(prices, window=5),
        "vol_20d": compute_vol(prices, window=20),
    }

    frames: list[pd.DataFrame] = []
    for name, df in raw.items():
        zscored = cross_section_zscore(df)
        # Attach a second level to the columns so we know which feature each is
        zscored.columns = pd.MultiIndex.from_product([[name], zscored.columns])
        frames.append(zscored)

    combined = pd.concat(frames, axis=1)
    # Drop rows where every single value is NaN (warm-up period at the start)
    return combined.dropna(how="all")
