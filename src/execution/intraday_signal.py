"""Entry timing and event-driven trade signals using intraday bars.

Two responsibilities:
1. get_entry_signal — given 30-min bars for a ticker the daily ML model
   already selected, decide *when* to enter (immediately, partially, end-of-day,
   or skip).
2. get_event_signal — if Claude's NLP pipeline flagged a high-confidence news
   event today, return an immediate trade dict regardless of the daily schedule.
"""

import math
import pandas as pd

import config


# ---------------------------------------------------------------------------
# VWAP helper
# ---------------------------------------------------------------------------

def compute_vwap(bars: pd.DataFrame) -> float:
    """Volume-Weighted Average Price over all rows in *bars*.

    Think of VWAP as the "fairness price" — it weights each bar's close by
    how much volume traded at that price.  A close above VWAP means buyers
    have been in control all day; below means sellers.

    Formula: sum(close * volume) / sum(volume)

    Returns NaN if *bars* is empty (no volume to weight by).
    """
    if bars.empty:
        return float("nan")

    total_volume = bars["volume"].sum()
    if total_volume == 0:
        return float("nan")

    return float((bars["close"] * bars["volume"]).sum() / total_volume)


# ---------------------------------------------------------------------------
# Entry timing signal
# ---------------------------------------------------------------------------

def get_entry_signal(
    bars: pd.DataFrame,
    current_bar_idx: int,
) -> str:
    """Decide how to size the entry based on intraday momentum.

    Analogy: think of the trading day as a race start.
    - Bar 0 (the gun): just go — enter full size immediately.
    - Later bars: check if you're ahead of pace (price ≥ VWAP).
      * If yes → full entry.
      * If no but still early → half entry (toe in the water; wait for proof).
      * If past the entry window → leave it for end-of-day or skip.

    Parameters
    ----------
    bars:             All 30-min bars fetched so far today (may have many rows).
    current_bar_idx:  Zero-based index of the bar we're acting on right now.

    Returns
    -------
    "full"  — enter the full planned position immediately.
    "half"  — enter half now, review again next bar.
    "eod"   — wait until end-of-day to act (or skip if it's too late).
    "skip"  — (reserved for future logic; currently unreachable via this path).
    """
    # Rule 1: Market open — always enter at the open bar.
    if current_bar_idx == 0:
        return "full"

    # Safety: if idx is out of range, treat as past the window.
    if current_bar_idx >= len(bars):
        return "eod"

    current_price = bars.iloc[current_bar_idx]["close"]
    vwap = compute_vwap(bars)

    # Rule 2: Price at or above VWAP → momentum confirmed, enter full.
    if not math.isnan(vwap) and current_price >= vwap:
        return "full"

    # Rule 3: Past the entry window (e.g. > 90 min into the day) → eod.
    if current_bar_idx >= config.ENTRY_WINDOW_BARS:
        return "eod"

    # Rule 4: Within the window but no confirmation yet → partial entry.
    return "half"


# ---------------------------------------------------------------------------
# Event-driven signal
# ---------------------------------------------------------------------------

def get_event_signal(
    news_signals: list[dict],
    ticker: str,
) -> dict | None:
    """Return an event-driven trade dict if a qualifying event exists.

    "Qualifying" means:
    - The event is for *ticker* (exact match).
    - confidence >= config.EVENT_CONFIDENCE_MIN (0.85 by default).
    - magnitude == "large" (small/medium events don't justify a surprise trade).

    If multiple events qualify, we pick the highest-confidence one — the
    strongest signal wins.

    Parameters
    ----------
    news_signals:
        List of dicts from Claude's NLP pipeline, each expected to have keys:
        ticker, confidence (float), magnitude (str), direction (str).
        direction should be "long"/"short" or "bullish"/"bearish" — we
        normalise "bullish" → "long" and "bearish" → "short".
    ticker:
        The ticker we're checking events for.

    Returns
    -------
    A trade dict like:
        {"ticker": "AAPL", "direction": "long", "size": 0.02}
    or None if no qualifying event found.
    """
    # Normalise direction labels so callers don't need to worry about it.
    _DIRECTION_MAP = {"bullish": "long", "bearish": "short"}

    qualifying = [
        sig for sig in news_signals
        if (
            sig.get("ticker") == ticker
            and sig.get("confidence", 0.0) >= config.EVENT_CONFIDENCE_MIN
            and sig.get("magnitude") == "large"
        )
    ]

    if not qualifying:
        return None

    # Best signal = highest confidence.
    best = max(qualifying, key=lambda s: s["confidence"])

    raw_direction = best.get("direction", "long")
    direction = _DIRECTION_MAP.get(raw_direction, raw_direction)  # keep "long"/"short" as-is

    return {
        "ticker": ticker,
        "direction": direction,
        "size": config.EVENT_POSITION_MAX,
    }
