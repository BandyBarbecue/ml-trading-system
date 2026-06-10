"""
fundamental_features.py — Quality factors from FMP key-metrics.

Two signals:
  gross_profitability  — Novy-Marx (2013) quality factor. High GP stocks
                         persistently outperform. Approximated as grossProfitMargin.
  pb_ratio             — Price-to-book. Low P/B = value factor.

Reporting lag:
  Financials are filed with the SEC up to 45 days after quarter end.
  We only use filings where the filing date is at least LAG_DAYS before the
  current date. This prevents look-ahead bias.

Output: wide DataFrame with MultiIndex columns (feature_name, ticker),
        same shape as price_features output so feature_matrix.py can merge them.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import config

LAG_DAYS = 45


def _latest_valid(
    df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    lag_days: int = LAG_DAYS,
) -> dict | None:
    """Return the most recent filing for ticker that predates as_of by lag_days.

    Args:
        df:       Full fundamentals DataFrame with columns ticker, date, features.
        ticker:   The stock symbol to look up.
        as_of:    The current date we're computing features for.
        lag_days: Minimum number of days a filing must predate as_of. Default 45.

    Returns:
        dict of the row's values, or None if no valid filing exists.
    """
    cutoff = as_of - pd.Timedelta(days=lag_days)
    subset = df[(df["ticker"] == ticker) & (pd.to_datetime(df["date"]) <= cutoff)]
    if subset.empty:
        return None
    row = subset.sort_values("date").iloc[-1]
    return row.to_dict()


def compute_fundamental_features(
    dates: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return wide DataFrame of fundamental features, cross-sectionally z-scored.

    Parameters
    ----------
    dates   : DatetimeIndex of trading dates to compute features for.
    symbols : Tickers to include.  If None, loads full S&P 500 from config.

    Returns
    -------
    DataFrame with MultiIndex columns (feature_name, ticker), index = date.
    Missing values are NaN — the model fills with 0 (neutral signal).
    """
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    fund_path: Path = config.FUNDAMENTALS_FILE
    if not fund_path.exists():
        return pd.DataFrame()

    fund_df = pd.read_parquet(fund_path)

    feature_names = ["gross_profitability", "pb_ratio"]
    records: dict[tuple, list] = {(f, t): [] for f in feature_names for t in symbols}
    index_dates = []

    for date in dates:
        index_dates.append(date)
        row_vals: dict[str, dict[str, float]] = {f: {} for f in feature_names}

        for ticker in symbols:
            snap = _latest_valid(fund_df, ticker, date)
            if snap is None:
                continue
            for feat in feature_names:
                val = snap.get(feat)
                if val is not None and not np.isnan(float(val)):
                    row_vals[feat][ticker] = float(val)

        # Cross-sectional z-score each feature across all tickers on this date.
        # Analogy: like grading on a curve — we rank stocks relative to each
        # other on each day, not against their own historical values.
        for feat in feature_names:
            vals = pd.Series(row_vals[feat])
            if len(vals) >= 2:
                z = (vals - vals.mean()) / (vals.std() + 1e-9)
            else:
                z = vals * np.nan  # can't z-score a single value meaningfully
            for ticker in symbols:
                records[(feat, ticker)].append(z.get(ticker, np.nan))

    if not index_dates:
        return pd.DataFrame()

    result = pd.DataFrame(records, index=pd.DatetimeIndex(index_dates))
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
