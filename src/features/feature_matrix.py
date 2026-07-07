"""
feature_matrix.py — Assembles all feature sources into one tidy DataFrame for ML.

Why a single "feature matrix"?
  The ML model (LightGBM) expects a single table where each row is one
  (date, ticker) observation and each column is a feature.  We call this
  "long format" or "tidy" format.  Think of it like a spreadsheet where:
    - Each row = one stock on one day
    - Each column = one input signal (momentum, volatility, news score, etc.)
    - The last column = the target (what we're trying to predict)

  Without this assembly step, we'd have several separate DataFrames in different
  shapes that the model can't consume directly.

Target variable:
  We predict the *cross-sectional rank* of the 20-day forward return.
  "Rank" rather than raw return because:
    1. We care about *relative* performance (which stocks beat the market).
    2. Rank is bounded [0, 1] — it's more stable to model than raw % returns
       which can have fat-tailed distributions.
  A rank of 1.0 = top performer on that day; 0.0 = bottom performer.

NaN target:
  The most recent TARGET_HORIZON trading days will always have NaN targets
  because the future hasn't happened yet.  This is intentional and correct —
  the model should never be trained on rows where the target is unknown.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from config import PRICES_DIR, TARGET_HORIZON

log = logging.getLogger(__name__)


def build_feature_matrix(
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble all features into a tidy long-format DataFrame for ML.

    Loads price features, earnings features, and NLP features, then
    joins them into a single table keyed by (date, ticker).

    Output columns:
        date, ticker, mom_1m, mom_3m, mom_12m, rev_1w, vol_20d,
        news_score, supply_chain_score, sue, target

    The ``target`` column is the 20-day forward cross-sectional return rank
    (percentile 0–1).  Rows at the trailing edge of the date range will have
    NaN targets (no future data available yet) — do NOT train on these.

    Args:
        symbols: List of ticker strings (e.g. ["AAPL", "MSFT"]).
        dates:   DatetimeIndex of trading dates to include in the matrix.

    Returns:
        Long-format DataFrame.  Empty DataFrame if no price data is available.
    """
    # Lazy imports: keep module-level imports minimal to avoid circular deps
    # and slow startup.  These modules are only loaded when this function runs.
    from src.features.price_features import compute_price_features, load_all_prices
    from src.features.earnings_features import compute_earnings_features
    from src.features.nlp_features import compute_news_features
    from src.features.fundamental_features import compute_fundamental_features
    from src.features.revision_features    import compute_revision_features
    from src.features.insider_features     import compute_insider_features

    # Normalise dates to a tz-naive DatetimeIndex so all downstream code
    # (price lookups, NLP date arithmetic) uses real Timestamps, not strings.
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    if dates.tzinfo is not None:
        dates = dates.tz_localize(None)

    # --- 1. Price features ---
    # load_all_prices silently skips any symbol whose parquet doesn't exist yet.
    prices = load_all_prices(symbols)
    if not prices:
        # No price data at all — nothing to assemble.
        return pd.DataFrame()

    # price_feats has MultiIndex columns: (feature_name, ticker)
    # e.g. ("mom_1m", "AAPL"), ("vol_20d", "MSFT"), etc.
    price_feats = compute_price_features(prices)

    # --- 2. Earnings features ---
    # Returns wide DataFrame: index=date, columns=ticker, values=z-scored SUE.
    earn_feats = compute_earnings_features(dates)

    # --- 3. NLP features ---
    # Returns wide DataFrame with MultiIndex columns: (feature_name, ticker).
    nlp_feats = compute_news_features(dates)

    # --- 3b. Fundamental features (with 45-day lag) ---
    fund_feats = compute_fundamental_features(dates, symbols)

    # --- 3c. Revision momentum ---
    rev_feats = compute_revision_features(dates, symbols)

    # --- 3d. Insider score ---
    insider_feats = compute_insider_features(dates, symbols)

    # --- 4. Compute target: 20-day forward cross-sectional return rank ---
    # Build a (date × ticker) close-price matrix.
    close_wide = pd.DataFrame({t: prices[t]["close"] for t in prices})

    # pct_change(n) = return over n days.
    # shift(-n) shifts the series *back* by n days — so for any date d, we get
    # the return from d to d+n (i.e. the future return as seen from date d).
    # Analogy: if you're standing at day 0 looking forward, shift(-20) puts
    # the 20-day-ahead value at your current position.
    fwd_ret = close_wide.pct_change(TARGET_HORIZON, fill_method=None).shift(-TARGET_HORIZON)

    # rank(pct=True) gives percentile ranks across the universe for each date.
    # A stock ranked 0.9 beat 90% of the universe over the next 20 days.
    target = fwd_ret.rank(axis=1, pct=True)

    # --- 5. Melt into tidy long format ---
    # This loop is the slow part: ~529 dates × 500 tickers = ~264k iterations.
    # Pure Python + per-cell .loc[] is not fast, but it's easy to follow.
    rows: list[dict] = []
    n_dates = len(dates)
    for i, date in enumerate(dates):
        if i > 0 and i % 100 == 0:
            log.info("feature matrix: %d/%d dates assembled", i, n_dates)
        if date not in price_feats.index:
            # This date has no price features (e.g. before warm-up period ends).
            continue

        for ticker in symbols:
            row: dict = {"date": date, "ticker": ticker}

            # Price features — stored as (feature, ticker) MultiIndex columns.
            for feat in ["mom_1m", "mom_3m", "mom_12m", "rev_1w", "vol_20d"]:
                try:
                    row[feat] = price_feats.loc[date, (feat, ticker)]
                except KeyError:
                    # This ticker has no data for this feature on this date.
                    row[feat] = np.nan

            # Earnings SUE — wide format with tickers as columns.
            try:
                row["sue"] = (
                    earn_feats.loc[date, ticker]
                    if not earn_feats.empty and ticker in earn_feats.columns
                    else np.nan
                )
            except KeyError:
                row["sue"] = np.nan

            # NLP features — MultiIndex columns: (feature_name, ticker).
            for feat in ["news_score", "supply_chain_score"]:
                try:
                    row[feat] = (
                        nlp_feats.loc[date, (feat, ticker)]
                        if not nlp_feats.empty and (feat, ticker) in nlp_feats.columns
                        else np.nan
                    )
                except KeyError:
                    row[feat] = np.nan

            # Fundamental features — MultiIndex columns: (feature_name, ticker).
            for feat in ["gross_profitability", "pb_ratio"]:
                try:
                    row[feat] = (
                        fund_feats.loc[date, (feat, ticker)]
                        if not fund_feats.empty and (feat, ticker) in fund_feats.columns
                        else np.nan
                    )
                except KeyError:
                    row[feat] = np.nan

            # Revision momentum
            try:
                row["revision_momentum"] = (
                    rev_feats.loc[date, ("revision_momentum", ticker)]
                    if not rev_feats.empty and ("revision_momentum", ticker) in rev_feats.columns
                    else np.nan
                )
            except KeyError:
                row["revision_momentum"] = np.nan

            # Insider score
            try:
                row["insider_score"] = (
                    insider_feats.loc[date, ("insider_score", ticker)]
                    if not insider_feats.empty and ("insider_score", ticker) in insider_feats.columns
                    else np.nan
                )
            except KeyError:
                row["insider_score"] = np.nan

            # Target: forward cross-sectional return rank (0–1).
            # NaN for the most recent TARGET_HORIZON dates (future not yet known).
            try:
                row["target"] = (
                    target.loc[date, ticker]
                    if ticker in target.columns
                    else np.nan
                )
            except KeyError:
                row["target"] = np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point — builds and saves the full-history feature matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import config
    from src.data.sp500_universe import get_symbols

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    symbols = get_symbols()
    # Full date range from first available data to today — gives the walk-forward
    # backtest the maximum possible history.
    dates = pd.bdate_range(start=config.START_DATE, end=config.END_DATE)
    log.info("Building feature matrix: %d symbols × %d dates …", len(symbols), len(dates))

    fm = build_feature_matrix(symbols, dates)

    if fm.empty:
        log.error("Feature matrix is empty — check that price parquets exist in data/prices/.")
        sys.exit(1)

    out_path = config.FEATURES_DIR / "feature_matrix.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fm.to_parquet(out_path, index=False)

    log.info("Saved %d rows to %s", len(fm), out_path)
    log.info("Unique dates: %d  |  Unique tickers: %d", fm["date"].nunique(), fm["ticker"].nunique())
