"""
regime_classifier.py — Classify the current market regime using FRED data + Claude.

Regime labels and their feature weight modulation:
  bull_trending  — momentum up-weighted, reversal down-weighted
  bull_volatile  — volatility and quality up-weighted
  bear           — low-vol, quality, value up-weighted; momentum down-weighted
  sideways       — no modulation
  crisis         — heavy defensive weighting; momentum near-zero

Why Claude rather than pure rules?
  FRED series are lagging indicators. Claude can integrate the directionality
  of changes (is unemployment rising or falling?) and the interactions between
  signals (inverted curve + rising unemployment is much more serious than either
  alone). It adds interpretation that threshold rules miss.
"""

import json
import logging
from typing import Literal

import anthropic
import pandas as pd

import config

log = logging.getLogger(__name__)

VALID_REGIMES = {"bull_trending", "bull_volatile", "bear", "sideways", "crisis"}
RegimeLabel   = Literal["bull_trending", "bull_volatile", "bear", "sideways", "crisis"]

# Per-regime IC weight multipliers. Keys match FEATURE_COLS in model.py.
# A multiplier of 2.0 means "treat this feature as twice as important as its
# raw IC suggests." 1.0 = no modulation. Missing keys default to 1.0.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "bull_trending": {
        "mom_1m": 1.5, "mom_3m": 1.5, "mom_12m": 2.0,
        "rev_1w": 0.5, "vol_20d": 0.8,
        "gross_profitability": 1.0, "news_score": 1.2,
    },
    "bull_volatile": {
        "mom_1m": 1.0, "mom_3m": 1.0, "mom_12m": 1.0,
        "vol_20d": 1.5, "gross_profitability": 1.3,
        "news_score": 1.3, "insider_score": 1.5,
    },
    "bear": {
        "mom_1m": 0.5, "mom_3m": 0.5, "mom_12m": 0.3,
        "vol_20d": 2.0, "gross_profitability": 2.0,
        "pb_ratio": 1.5, "rev_1w": 1.5,
    },
    "sideways": {},
    "crisis": {
        "mom_1m": 0.3, "mom_3m": 0.3, "mom_12m": 0.2,
        "vol_20d": 2.5, "gross_profitability": 2.0,
        "insider_score": 2.0,
    },
}

REGIME_PROMPT = """You are a macro economist classifying the current market regime.

Recent FRED indicators (latest values):
{macro_summary}

Classify the current regime as exactly ONE of:
  bull_trending  — positive trend, normal yield curve, healthy employment
  bull_volatile  — positive trend but warning signals (inverted curve or rising unemployment)
  bear           — recession signals present (inverted curve + rising unemployment)
  sideways       — mixed signals, no clear direction
  crisis         — acute stress (inverted curve AND rising unemployment AND high financial stress)

Return ONLY a JSON object: {{"regime": "<label>", "reasoning": "<1-2 sentences>"}}
"""


def classify_regime(macro_df: pd.DataFrame) -> RegimeLabel:
    """Classify the current regime using the latest FRED data + Claude Haiku.

    Parameters
    ----------
    macro_df : DataFrame with columns date, t10y2y, unrate, fedfunds.

    Returns
    -------
    One of VALID_REGIMES. Defaults to "sideways" on any error.
    """
    if macro_df.empty:
        return "sideways"

    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    latest = macro_df.sort_values("date").iloc[-1]

    prev_30d = macro_df[
        macro_df["date"] >= (latest["date"] - pd.Timedelta(days=30))
    ]
    unrate_change = (
        latest["unrate"] - prev_30d["unrate"].iloc[0]
        if len(prev_30d) > 1 and "unrate" in macro_df.columns
        else 0.0
    )

    t10y2y_val   = float(latest.get("t10y2y",   0.0))
    unrate_val   = float(latest.get("unrate",    4.0))
    fedfunds_val = float(latest.get("fedfunds",  5.0))

    macro_summary = (
        f"  T10Y2Y (10yr-2yr spread): {t10y2y_val:.2f}% "
        f"({'inverted — recession warning' if t10y2y_val < 0 else 'normal'})\n"
        f"  Unemployment rate: {unrate_val:.1f}% "
        f"({'rising {:.1f}pp over 30d'.format(unrate_change) if unrate_change > 0.1 else 'stable/falling'})\n"
        f"  Fed funds rate: {fedfunds_val:.2f}%"
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.CLAUDE_NLP_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": REGIME_PROMPT.format(macro_summary=macro_summary)}],
        )
        parsed = json.loads(response.content[0].text)
        regime = parsed.get("regime", "sideways")
        if regime not in VALID_REGIMES:
            regime = "sideways"
        log.info("Regime classified as '%s': %s", regime, parsed.get("reasoning", ""))
        return regime
    except Exception as exc:
        log.warning("Regime classification failed, defaulting to sideways: %s", exc)
        return "sideways"


def get_regime_feature_weights(regime: RegimeLabel) -> dict[str, float]:
    """Return per-feature IC weight multipliers for the given regime.

    Returns empty dict for 'sideways' (no modulation needed).
    Any feature not mentioned gets an implicit multiplier of 1.0.
    """
    return dict(REGIME_WEIGHTS.get(regime, {}))
