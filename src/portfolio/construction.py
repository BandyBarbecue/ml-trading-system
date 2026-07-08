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
    multiplied by 30/40 = 0.75 so the sector total hits exactly 30% —
    but then renormalising once (dividing everyone by the new, smaller
    total so the portfolio still sums to 1.0) scales sector X back UP
    again, proportionally with everyone else, pushing it back over the
    cap. Proven case: sector X at 60% with a 30% cap becomes 42.9% after
    one cap-then-renormalise pass, not 30%.

    Analogy: it's like turning down one loud speaker in a room, then
    turning UP the master volume to compensate for the room feeling
    quieter overall — the speaker you just turned down creeps back up too.

    Fix: instead of renormalising, move the trimmed-off weight into sectors
    that still have headroom below the cap. If every sector is full (e.g.
    only 3 sectors exist, and 3 x 30% = 90% < 100%), the leftover weight is
    deliberately NOT reinvested — it stays as cash. Holding cash is always
    an acceptable outcome for a risk limit; breaching the cap (or crashing)
    is not.
    """
    w = weights.copy()
    cap = config.MAX_SECTOR_PCT
    tol = 1e-9
    sectors = set(sector_map.values())
    max_iterations = 20

    def sector_tickers(sector: str) -> list[str]:
        return [t for t in w.index if sector_map.get(t) == sector]

    for _ in range(max_iterations):
        # 1. Trim every over-cap sector down to exactly the cap, pooling
        #    the trimmed-off weight ("excess") for redistribution.
        excess = 0.0
        for sector in sectors:
            tickers = sector_tickers(sector)
            sector_total = float(w[tickers].sum())
            if sector_total > cap + tol:
                w[tickers] *= cap / sector_total
                excess += sector_total - cap

        if excess <= tol:
            break  # nothing was over the cap — done

        # 2. Hand the excess to sectors with room left, in proportion to how
        #    much room each has. Capping the hand-out at each sector's
        #    headroom means a receiving sector can never itself go over the
        #    cap, so this converges instead of ping-ponging.
        headrooms = {}
        for sector in sectors:
            # A sector with no tickers actually held can't receive weight —
            # skip it (sector_map often covers the whole universe, while the
            # portfolio only holds the selected top N).
            if not sector_tickers(sector):
                continue
            room = cap - float(w[sector_tickers(sector)].sum())
            if room > tol:
                headrooms[sector] = room
        total_headroom = sum(headrooms.values())
        if total_headroom <= tol:
            break  # every sector is full — the excess stays as cash

        to_distribute = min(excess, total_headroom)
        for sector, room in headrooms.items():
            tickers = sector_tickers(sector)
            share = to_distribute * (room / total_headroom)
            sector_total = float(w[tickers].sum())
            if sector_total > tol:
                # Pro-rata within the sector, preserving relative sizes.
                w[tickers] += share * (w[tickers] / sector_total)
            else:
                # Sector has tickers but ~zero weight: split equally.
                w[tickers] += share / len(tickers)
        # Any excess beyond total headroom is simply not reinvested (cash).

    # Final safety trim: after the bounded loop, hard-scale any sector still
    # a hair over the cap (floating-point dust). The tiny shortfall becomes
    # cash — never crash a live rebalance over an infeasible cap.
    for sector in sectors:
        tickers = sector_tickers(sector)
        sector_total = float(w[tickers].sum())
        if sector_total > cap + tol:
            w[tickers] *= cap / sector_total
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

    # Final hard safety clip: covers float drift or a ticker still (barely)
    # over the cap if the loop above hit its 10-iteration limit.
    w[w > cap] = cap

    # Only renormalise back to sum=1.0 if doing so does NOT push anything
    # back over the cap. Dividing by a total < 1.0 inflates every weight —
    # exactly the "renormalise undoes the cap" bug this task exists to fix
    # (see src/risk/engine.py for the proven case). If redistribution above
    # left no room (every ticker sitting at the cap) the total will be
    # < 1.0; that gap is left unallocated — i.e. held as cash — rather than
    # forced back into already-capped positions.
    total = w.sum()
    if total > 0:
        candidate = w / total
        if (candidate <= cap + 1e-9).all():
            w = candidate
        # else: leave `w` as-is; (1 - total) sits in cash.
    return w


def _apply_turnover_cap(
    target: pd.Series,
    current: pd.Series,
) -> pd.Series:
    """Limit two-way turnover to TURNOVER_CAP.

    Turnover = sum(|target - current|) / 2.

    When turnover already fits within the cap, return `target` unchanged.
    When it exceeds the cap, ALL trade deltas are shrunk by the same factor
    (TURNOVER_CAP / turnover) so the portfolio moves partway from `current`
    toward `target` and lands exactly on the turnover budget.

    Analogy: you planned a 100-mile road trip but only have fuel for 40
    miles — so you drive 40% of the way along the SAME route, rather than
    picking a few favourite stops and teleporting to them.

    Why proportional shrink rather than the old greedy accept/reject +
    renormalise?  Two proven defects in the old approach:
      1. The final renormalisation rescaled the *rejected* tickers too,
         moving them off the current weights we had deliberately frozen,
         and could push turnover back over the cap — the same "renormalise
         undoes the control" bug fixed in the sector/position caps above.
      2. Even without renormalising, accepting big buys while rejecting the
         offsetting sells makes the weights sum to MORE than 1.0 — that's
         leverage, which this long-only book must never have.
    The proportional shrink has neither problem, provably: the result is
    the convex mix (1-s)*current + s*target with s in (0, 1), so it stays
    non-negative, its turnover is exactly s * original = TURNOVER_CAP, and
    its total sits between sum(current) and sum(target) — never above 1.0.
    If `target` holds some cash (sum < 1, e.g. after a cap couldn't
    redistribute), the result holds proportionally less of that cash; no
    renormalisation is needed or done.

    Note: if `current` is empty (first day), skip the cap entirely —
    there's no existing book to protect from churn.
    """
    if current.empty:
        return target

    # Align: tickers in target or current; fill missing with 0
    all_tickers = target.index.union(current.index)
    t = target.reindex(all_tickers, fill_value=0.0)
    c = current.reindex(all_tickers, fill_value=0.0)

    delta = t - c
    turnover = float(delta.abs().sum()) / 2.0

    if turnover <= config.TURNOVER_CAP:
        return t  # already within budget

    # Shrink every delta by the same factor so total turnover lands exactly
    # on the cap. s < 1 by construction here (turnover > cap).
    s = config.TURNOVER_CAP / turnover
    result = c + delta * s

    # Drop dust: tickers whose weight is (numerically) zero were never in
    # target or current, or fully exited.
    return result[result > 1e-12]


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
        Target weights indexed by ticker. Sums to 1.0 in the normal case,
        but may sum to slightly less when a cap could not redistribute all
        excess weight or the turnover cap froze some tickers — the
        shortfall is deliberately held as cash rather than forced back
        into capped positions.
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

    # NOTE: no blind renormalisation between the caps and the turnover cap.
    # The old "Step 5" divided the weights by their sum, which inflates
    # every weight and can push capped positions straight back over their
    # limits — the exact bug the caps exist to prevent. Each cap helper now
    # handles its own normalisation safely; any shortfall below 1.0 is
    # intentional cash.

    # ------------------------------------------------------------------
    # Step 5: Turnover cap
    # ------------------------------------------------------------------
    weights = _apply_turnover_cap(weights, current_weights)

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
