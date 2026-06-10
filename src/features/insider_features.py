"""
insider_features.py — Insider transaction signal.

Aggregates recent insider buys/sells into a net score:
  insider_score = sum(value_usd for buys * recency_weight)
                - sum(value_usd for sells * recency_weight)
  over the last INSIDER_WINDOW days

Recency weight: transactions yesterday matter more than 89 days ago.
  weight = 1.0 - days_ago / INSIDER_WINDOW

Then cross-sectionally z-scored. Positive = insiders buying net.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import config

INSIDER_WINDOW = 90  # days


def compute_insider_features(
    dates: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return wide DataFrame of insider_score, cross-sectionally z-scored."""
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    insider_path: Path = config.INSIDER_FILE
    if not insider_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(insider_path)
    df["date"] = pd.to_datetime(df["date"])

    records: dict[tuple, list] = {("insider_score", t): [] for t in symbols}
    index_dates = []

    for date in dates:
        index_dates.append(date)
        cutoff = date - pd.Timedelta(days=INSIDER_WINDOW)
        raw: dict[str, float] = {}

        for ticker in symbols:
            sub = df[
                (df["ticker"] == ticker)
                & (df["date"] >= cutoff)
                & (df["date"] <= date)
            ].copy()
            if sub.empty:
                continue

            sub["days_ago"] = (date - sub["date"]).dt.days
            sub["weight"] = 1.0 - sub["days_ago"] / INSIDER_WINDOW

            buys  = sub[sub["transaction_type"] == "P"]
            sells = sub[sub["transaction_type"] == "S"]
            buy_val  = (buys["value_usd"] * buys["weight"]).sum()
            sell_val = (sells["value_usd"] * sells["weight"]).sum()
            raw[ticker] = float(buy_val - sell_val)

        vals = pd.Series(raw)
        if len(vals) >= 2:
            z = (vals - vals.mean()) / (vals.std() + 1e-9)
        else:
            z = vals

        for ticker in symbols:
            records[("insider_score", ticker)].append(z.get(ticker, np.nan))

    if not index_dates:
        return pd.DataFrame()

    result = pd.DataFrame(records, index=pd.DatetimeIndex(index_dates))
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
