"""
revision_features.py — Analyst estimate revision momentum.

When sell-side analysts raise their EPS estimates, they are updating their
models with new information. A series of upward revisions signals improving
business fundamentals. Academic literature (Chan et al. 1996) shows this
predicts 3-6 month outperformance.

We compute revision_momentum as the sum of consecutive relative EPS estimate
changes over the last REVISION_WINDOW days:
  revision = sum((est_t / est_{t-1}) - 1) for consecutive reports in window

Positive score = analysts raising estimates; negative = cutting them.
Cross-sectionally z-scored so the daily average is always zero.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import config

# Quarterly earnings reports land ~91-92 calendar days apart (a year / 4).
# A 90-day window is *shorter* than that real-world gap, so the previous
# report usually falls just outside it and revision_momentum ends up NaN
# for nearly every real ticker (it needs >= 2 reports in-window). 120 days
# comfortably covers one quarterly gap with room to spare for reporting
# schedule jitter.
REVISION_WINDOW = 120  # days of earnings history to include


def compute_revision_features(
    dates: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return wide DataFrame of revision_momentum, cross-sectionally z-scored.

    Parameters
    ----------
    dates   : DatetimeIndex of dates to compute features for.
    symbols : Tickers. If None, loads S&P 500 from config.

    Returns
    -------
    DataFrame with MultiIndex columns (revision_momentum, ticker), index = date.
    Empty DataFrame if earnings file does not exist.
    """
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    earn_path: Path = config.EARNINGS_FILE
    if not earn_path.exists():
        return pd.DataFrame()

    earn_df = pd.read_parquet(earn_path)
    earn_df["date"] = pd.to_datetime(earn_df["date"])
    earn_df = earn_df.sort_values(["ticker", "date"])

    records: dict[tuple, list] = {("revision_momentum", t): [] for t in symbols}
    index_dates = []

    for date in dates:
        index_dates.append(date)
        cutoff_start = date - pd.Timedelta(days=REVISION_WINDOW)
        raw: dict[str, float] = {}

        for ticker in symbols:
            t_df = earn_df[
                (earn_df["ticker"] == ticker)
                & (earn_df["date"] >= cutoff_start)
                & (earn_df["date"] <= date)
            ].copy()

            if len(t_df) < 2:
                # Need at least two reports to compute a revision
                continue

            t_df = t_df.sort_values("date")
            estimates = t_df["estimated_eps"].values
            # Sum of consecutive relative revisions:
            # (est_t - est_{t-1}) / |est_{t-1}|
            # Adding 1e-9 to denominator prevents division by zero if estimate is 0.
            revisions = np.diff(estimates) / (np.abs(estimates[:-1]) + 1e-9)
            raw[ticker] = float(revisions.sum())

        # Cross-sectional z-score across all tickers with data on this date.
        # Analogy: like grading on a curve — we rank stocks relative to each
        # other on each day, not against their own historical values.
        vals = pd.Series(raw)
        if len(vals) >= 2:
            z = (vals - vals.mean()) / (vals.std() + 1e-9)
        else:
            # Can't z-score a single value — keep raw value so the model at
            # least sees the direction even if magnitude isn't relative to peers.
            z = vals

        for ticker in symbols:
            records[("revision_momentum", ticker)].append(z.get(ticker, np.nan))

    if not index_dates:
        return pd.DataFrame()

    result = pd.DataFrame(records, index=pd.DatetimeIndex(index_dates))
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
