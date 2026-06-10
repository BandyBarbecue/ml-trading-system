"""
Portfolio Construction
======================
Takes ML scores for each ticker, selects the top N positions, sizes them using
Hierarchical Risk Parity, then enforces three risk guardrails in order:

  1. Sector cap  — no GICS sector can exceed MAX_SECTOR_PCT of the portfolio.
  2. Position cap — no single stock can exceed MAX_POSITION_PCT.
  3. Turnover cap — total two-way turnover is limited to TURNOVER_CAP.

Analogy: think of the portfolio as a pizza.  HRP cuts the initial slices by
risk.  The sector cap says "no one topping can cover more than 30% of the
pizza."  The position cap says "no single slice can be bigger than 8%."  The
turnover cap says "don't swap out more than 40% of the pizza's toppings in one
go — change gradually."
"""

import config
import pandas as pd
import numpy as np
from src.portfolio.hrp import hrp_weights


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_sector_cap(weights: pd.Series, sector_map: dict[str, str]) -> pd.Series:
    """Scale down any sector whose total weight exceeds MAX_SECTOR_PCT.

    If sector X has 40% but the cap is 30%, every ticker in sector X is
    multiplied by 30/40 = 0.75 so the sector total hits exactly 30%.
    Weights are then renormalised so they still sum to 1.
    """
    w = weights.copy()
    for sector in set(sector_map.values()):
        tickers_in_sector = [t for t in w.index if sector_map.get(t) == sector]
        sector_total = w[tickers_in_sector].sum()
        if sector_total > config.MAX_SECTOR_PCT:
            # Scale every ticker in the sector proportionally down to the cap
            scale = config.MAX_SECTOR_PCT / sector_total
            w[tickers_in_sector] *= scale
    # Renormalise so all weights still sum to 1.0
    total = w.sum()
    if total > 0:
        w /= total
    return w


def _apply_position_cap(weights: pd.Series) -> pd.Series:
    """Clip any single position to MAX_POSITION_PCT, redistribute excess.

    Works iteratively: set anything above the cap to the cap, collect the
    excess weight, then spread it proportionally across tickers that are still
    below the cap.  Repeat until nothing is above the cap (or 10 iterations
    as a safety limit against floating-point drift).
    """
    w = weights.copy()
    cap = config.MAX_POSITION_PCT

    for _ in range(10):
        over_mask = w > cap
        if not over_mask.any():
            break

        # Total weight that has been clipped off
        excess = (w[over_mask] - cap).sum()
        # Set over-cap tickers to exactly the cap
        w[over_mask] = cap

        # Recipients: tickers still below the cap
        under_mask = w < cap
        under_total = w[under_mask].sum()

        if under_total > 0:
            # Distribute excess proportionally to their current weights
            w[under_mask] += excess * (w[under_mask] / under_total)
        # else: all tickers are at the cap — nowhere to redistribute, so just stop

    # Final normalisation to absorb floating-point drift
    total = w.sum()
    if total > 0:
        w /= total
    return w


def _apply_turnover_cap(
    target: pd.Series,
    current: pd.Series,
    scores: pd.Series,
) -> pd.Series:
    """Limit two-way turnover to TURNOVER_CAP.

    Turnover = sum(|target - current|) / 2.

    When turnover already fits within the cap, return `target` unchanged.

    When it exceeds the cap, we greedily accept the *largest* individual
    changes first (they carry the most information) until the cap is hit,
    then revert the rest to their current weights.  The result is renormalised
    to sum to 1.

    Note: if `current` is empty (first day), skip the cap entirely.
    """
    if current.empty:
        return target

    # Align: tickers in target or current; fill missing with 0
    all_tickers = target.index.union(current.index)
    t = target.reindex(all_tickers, fill_value=0.0)
    c = current.reindex(all_tickers, fill_value=0.0)

    delta = (t - c).abs()
    turnover = delta.sum() / 2.0

    if turnover <= config.TURNOVER_CAP:
        return t  # already within budget

    # Sort tickers by size of change (largest first — highest priority)
    priority_order = delta.sort_values(ascending=False).index

    budget_used = 0.0
    accepted = pd.Series(False, index=all_tickers)

    for ticker in priority_order:
        change_cost = delta[ticker] / 2.0  # two-way cost of this one change
        if budget_used + change_cost <= config.TURNOVER_CAP:
            accepted[ticker] = True
            budget_used += change_cost
        # else: leave this ticker at its current weight

    # Build the result: accepted tickers take their target weight,
    # rejected tickers stay at their current weight
    result = c.copy()
    result[accepted] = t[accepted]

    # Renormalise to 1.0 (the mix of kept current + accepted target may not sum to 1)
    total = result.sum()
    if total > 0:
        result /= total

    # Drop zero-weight tickers that were never in target or current
    return result[result > 0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_portfolio(
    scores: pd.Series,
    sector_map: dict[str, str],
    current_weights: pd.Series,
    prices: pd.DataFrame,
) -> pd.Series:
    """Build a target-weight portfolio from ML scores.

    Parameters
    ----------
    scores : pd.Series
        ML model scores indexed by ticker.  Higher = more bullish.
    sector_map : dict[str, str]
        Maps ticker → GICS sector name (e.g. "AAPL" → "Information Technology").
    current_weights : pd.Series
        Current portfolio weights indexed by ticker.  Pass an empty Series on
        the first day (no existing positions).
    prices : pd.DataFrame
        Recent daily *returns* (not price levels) with dates as the index and
        tickers as columns.  Used to estimate the covariance for HRP.
        Should already be filtered to the candidate universe.

    Returns
    -------
    pd.Series
        Target weights indexed by ticker, summing to 1.0.
    """
    # ------------------------------------------------------------------
    # Step 1: Select top N tickers by score
    # ------------------------------------------------------------------
    n = config.N_POSITIONS
    top_tickers = scores.nlargest(n).index.tolist()

    # ------------------------------------------------------------------
    # Step 2: HRP weighting
    # ------------------------------------------------------------------
    # Filter price returns to only the selected tickers (and only columns
    # that actually exist in the prices DataFrame to avoid KeyErrors)
    available = [t for t in top_tickers if t in prices.columns]
    if len(available) == 0:
        raise ValueError("None of the top-N tickers are present in the prices DataFrame.")

    returns_subset = prices[available].dropna(how="all")
    raw_weights = hrp_weights(returns_subset)  # pd.Series indexed by ticker

    # ------------------------------------------------------------------
    # Step 3: Sector cap
    # ------------------------------------------------------------------
    weights = _apply_sector_cap(raw_weights, sector_map)

    # ------------------------------------------------------------------
    # Step 4: Position cap
    # ------------------------------------------------------------------
    weights = _apply_position_cap(weights)

    # ------------------------------------------------------------------
    # Step 5: Renormalise (belt-and-suspenders after the caps)
    # ------------------------------------------------------------------
    weights = weights / weights.sum()

    # ------------------------------------------------------------------
    # Step 6: Turnover cap
    # ------------------------------------------------------------------
    weights = _apply_turnover_cap(weights, current_weights, scores)

    return weights


def build_short_portfolio(
    scores: pd.Series,
    sector_map: dict[str, str],
    current_weights: pd.Series,
    recent_returns: pd.DataFrame,
) -> pd.Series:
    """Select bottom N_SHORT_POSITIONS stocks and return negative target weights.

    Equal-weights the bottom N stocks with a per-name cap of MAX_SHORT_PCT.
    All weights are negative (short positions).

    Returns
    -------
    pd.Series of negative weights, or empty Series if LONG_SHORT_ENABLED=False.
    """
    if not config.LONG_SHORT_ENABLED:
        return pd.Series(dtype=float)

    if scores.empty:
        return pd.Series(dtype=float)

    bottom = scores.nsmallest(config.N_SHORT_POSITIONS)
    n = len(bottom)
    if n == 0:
        return pd.Series(dtype=float)

    # Equal-weight initial sizing for shorts (negative = short)
    raw_weights = pd.Series(-1.0 / n, index=bottom.index)

    # Apply per-name cap: no single short position > MAX_SHORT_PCT
    raw_weights = raw_weights.clip(-config.MAX_SHORT_PCT, 0)

    return raw_weights


def compute_trades(
    target_weights: pd.Series,
    current_weights: pd.Series,
) -> pd.Series:
    """Compute the trade delta needed to move from current to target weights.

    Parameters
    ----------
    target_weights : pd.Series
        Desired portfolio weights (indexed by ticker).
    current_weights : pd.Series
        Current portfolio weights (indexed by ticker).

    Returns
    -------
    pd.Series
        Weight delta per ticker.  Positive = buy, negative = sell.
        Tickers that appear in only one series are treated as 0 in the other.
    """
    return target_weights.subtract(current_weights, fill_value=0.0)
