"""
earnings_features.py — Standardised Unexpected Earnings (SUE) feature engineering.

What is SUE?
  SUE = (actual_eps - estimated_eps) / std(last_N_surprises)

  Think of it like a standardised test score for earnings: we divide the raw
  "surprise" by how *volatile* that company's surprises usually are.  A company
  that routinely swings ±10% needs a bigger beat to be impressive than one that
  is always within ±1%.

Why does this matter for trading?
  Post-Earnings Announcement Drift (PEAD) is one of the most replicated anomalies
  in academic finance: stocks that beat expectations keep drifting upward for weeks,
  and those that miss keep drifting downward.  SUE quantifies *how big* a beat/miss
  was, which predicts how strong the subsequent drift will be.

Signal decay:
  The drift doesn't last forever.  We model it as a linear decay: the signal is
  at full strength on the announcement date and decays to zero over DECAY_DAYS
  calendar days.  After that, the signal is set to NaN (no information).
"""

import pandas as pd
import numpy as np
from config import EARNINGS_FILE

DECAY_DAYS = 60    # linear decay window in calendar days
MIN_HISTORY = 3    # minimum past earnings reports required to compute SUE


def compute_sue(earnings: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute a daily SUE score for each ticker across the given date range.

    For each date ``d`` and each ticker, we:
      1. Find all past earnings reports for that ticker on or before ``d``.
      2. Require at least MIN_HISTORY reports (otherwise return NaN — not enough
         data to estimate the std of surprises reliably).
      3. Compute SUE = surprise_pct_latest / std(last_8_surprise_pcts).
      4. Apply a linear decay based on how many *calendar days* have elapsed
         since the most recent announcement.  At 0 days: full signal.
         At DECAY_DAYS: NaN (signal expired).

    Args:
        earnings: DataFrame with columns
            ticker (str), date (datetime), surprise_pct (float).
            surprise_pct = (actual - estimated) / |estimated|, or equivalent.
        dates: DatetimeIndex of trading dates to evaluate.

    Returns:
        DataFrame with ``dates`` as index and tickers as columns.
        Values are decayed SUE scores; NaN where insufficient history or expired.
    """
    if earnings.empty:
        # Nothing to compute — return an empty DataFrame with the right index
        return pd.DataFrame(index=dates)

    result_rows = []
    tickers = earnings["ticker"].unique()

    for d in dates:
        row: dict[str, float] = {}
        for ticker in tickers:
            # All earnings for this ticker strictly BEFORE date d (not <=),
            # sorted newest-first so iloc[0] is the most recent report.
            # Why strict "<": most S&P 500 companies report earnings After
            # Market Close (AMC), so a report dated day T is not public
            # information until day T+1's session. Using "<=" would let day
            # T's own announcement leak into day T's feature (look-ahead
            # bias) at full decay strength (days_since == 0 → decay 1.0).
            # A proper fix would track a real BMO/AMC flag per report and
            # only exclude same-day AMC reports — that's a future improvement;
            # excluding the whole announcement day is the conservative, safe
            # default in the meantime.
            hist = (
                earnings[
                    (earnings["ticker"] == ticker) & (earnings["date"] < d)
                ]
                .sort_values("date", ascending=False)
            )

            # Not enough history to compute a reliable std — return NaN.
            # Analogy: you wouldn't trust a "average" based on one data point.
            if len(hist) < MIN_HISTORY:
                row[ticker] = np.nan
                continue

            latest = hist.iloc[0]

            # Use up to the 8 most recent surprises for the std calculation.
            # We cap at 8 because older reports (> 2 years ago) may reflect
            # a very different business environment.
            past_surprises = hist["surprise_pct"].iloc[:8]
            std = past_surprises.std(ddof=1)   # sample std (ddof=1 is standard)

            if std == 0 or np.isnan(std):
                # Zero std means every surprise was identical — can't normalise.
                row[ticker] = np.nan
                continue

            raw_sue = latest["surprise_pct"] / std

            # Linear decay: how stale is the most recent announcement?
            # We use calendar days (not trading days) because earnings drift
            # is a calendar-time phenomenon.
            days_since = (d - latest["date"]).days

            if days_since >= DECAY_DAYS:
                # Signal has fully decayed — treat as no information.
                row[ticker] = np.nan
            else:
                # decay_factor goes from 1.0 (day 0) down to ~0 (day DECAY_DAYS-1).
                # Example: on day 30 of a 60-day window, decay_factor = 0.5.
                decay_factor = 1.0 - (days_since / DECAY_DAYS)
                row[ticker] = raw_sue * decay_factor

        result_rows.append(row)

    return pd.DataFrame(result_rows, index=dates)


def compute_earnings_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Load earnings data from disk and return cross-sectionally z-scored SUE.

    This is the top-level entry point used by the feature pipeline.  It:
      1. Loads the earnings parquet file (or returns empty if it doesn't exist yet).
      2. Computes SUE via compute_sue().
      3. Z-scores cross-sectionally so values are comparable across tickers
         on any given date (mean ≈ 0, std ≈ 1 per row).

    Args:
        dates: DatetimeIndex of trading dates to evaluate.

    Returns:
        DataFrame with ``dates`` as index, tickers as columns, z-scored SUE values.
    """
    # Import here to avoid circular imports at module level
    from src.features.price_features import cross_section_zscore

    if EARNINGS_FILE.exists():
        earnings = pd.read_parquet(EARNINGS_FILE)
    else:
        earnings = pd.DataFrame(
            columns=["ticker", "date", "actual_eps", "estimated_eps", "surprise_pct"]
        )

    sue = compute_sue(earnings, dates)
    return cross_section_zscore(sue)
