"""
nlp_features.py — Claude Haiku NLP extraction pipeline.

Sends financial headlines to Claude in batches, parses structured JSON
responses, and aggregates the results into daily per-ticker feature scores.

Two output features per ticker per day:
  news_score         — direct sentiment signal for the primary ticker
  supply_chain_score — second-order effect on suppliers/peers

Both are cross-sectionally z-scored so the daily average is always zero
(same "grade on a curve" approach as price features).

Cache design (why it exists):
  Claude API calls cost money and time.  Every processed headline is stored
  in a JSON file keyed by headline text.  On re-runs, already-seen headlines
  are served from disk — no API call needed.  Think of it like a lookup table:
  "have I already looked up this sentence?  If yes, return the stored answer."
"""

import json
import anthropic
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    CLAUDE_NLP_MODEL,
    NLP_BATCH_SIZE,
    NLP_DECAY_DAYS,
    FEATURES_DIR,
)

# Module-level path to the cache file.  Tests monkeypatch this attribute to
# redirect reads/writes to a temporary directory so the real cache is untouched.
CACHE_FILE: Path = FEATURES_DIR / "nlp_cache.json"

# The system prompt we send Claude for every batch of headlines.
# We ask for strict JSON output so we can parse it programmatically.
EXTRACTION_PROMPT = """\
You are a financial signal extractor. Given a list of financial news items, \
return a JSON array where each element corresponds to one item (in order).

For each item you receive a HEADLINE and optionally a SUMMARY with more context. \
Use the summary to improve your analysis when available.

Extract:
- headline: the original headline text (copy verbatim)
- event_type: one of [earnings_beat, earnings_miss, guidance_raise, guidance_cut, \
  ma_announced, ceo_change, fda_approval, fda_rejection, rate_decision, macro_data, other]
- primary_ticker: the main ticker symbol affected
- direction: bullish | bearish | neutral
- magnitude: small | medium | large
- time_horizon: intraday | 1week | 1month | quarter
- confidence: float 0.0-1.0
- affected_tickers: list of {ticker, direction, reason} for supply-chain or peer effects

Return ONLY the JSON array, no other text.

News items:
"""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    """Read the on-disk cache.  Returns an empty dict if the file doesn't exist."""
    # We read CACHE_FILE at call time (not at import time) so that monkeypatching
    # the module-level CACHE_FILE in tests actually redirects the reads.
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    """Write the in-memory cache dict back to disk as JSON."""
    # Same reason as _load_cache: reference the module-level name so tests work.
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _direction_to_score(direction: str) -> float:
    """Convert a text direction label to a signed float.

    bullish →  1.0  (positive signal)
    bearish → -1.0  (negative signal)
    neutral →  0.0  (no signal)
    """
    return {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get(direction, 0.0)


def _magnitude_to_weight(magnitude: str) -> float:
    """Convert a text magnitude label to a multiplicative weight.

    Think of it as a volume dial: small events count half, large events 1.5×.
    """
    return {"small": 0.5, "medium": 1.0, "large": 1.5}.get(magnitude, 1.0)


# ---------------------------------------------------------------------------
# Batch formatting
# ---------------------------------------------------------------------------

def _format_batch(batch: list[dict]) -> str:
    """Format a batch of news items for the Claude prompt.

    Each item is a dict with at least a "headline" key and optionally a
    "summary" key.  When a summary is present it is appended on the next
    line so Claude has extra context to improve extraction quality.
    Summary text is truncated to 300 characters to keep token usage bounded.
    """
    lines = []
    for i, item in enumerate(batch, 1):
        headline = item.get("headline", "")
        summary  = item.get("summary", "").strip()
        if summary:
            lines.append(f"{i}. [HEADLINE] {headline}\n   [SUMMARY] {summary[:300]}")
        else:
            lines.append(f"{i}. {headline}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_signals(headlines: pd.DataFrame) -> list[dict]:
    """Send headlines to Claude Haiku and return structured signals.

    Processes headlines in batches of NLP_BATCH_SIZE.  Already-seen headlines
    (matched by text) are returned from the on-disk cache without API calls.

    Args:
        headlines: DataFrame with at least a "headline" column (str).
                   Typically also has ticker, published_at, source columns.

    Returns:
        List of signal dicts, one per headline that was found in the cache
        or successfully parsed from Claude's response.  Each dict has keys:
        headline, event_type, primary_ticker, direction, magnitude,
        time_horizon, confidence, affected_tickers.
    """
    cache = _load_cache()

    # Build a deduplicated list of (headline, summary) dicts.
    # The same headline often appears on multiple rows (one per ticker it mentions),
    # so we work on unique headline texts only — saves ~65% of API calls.
    # We include summary so _format_batch can send richer context to Claude.
    summary_col = headlines["summary"] if "summary" in headlines.columns else pd.Series(
        [""] * len(headlines), index=headlines.index
    )
    # Deduplicate by headline text; keep the first summary seen for each.
    seen: dict[str, str] = {}
    for h, s in zip(headlines["headline"], summary_col):
        if h not in seen:
            seen[h] = s if isinstance(s, str) else ""

    # Cache key = headline + "|" + first 200 chars of summary so that if the
    # article body changes (e.g. Alpaca backfills a summary) we re-extract.
    def _cache_key(headline: str, summary: str) -> str:
        return headline + "|" + summary[:200]

    uncached_items = [
        {"headline": h, "summary": s}
        for h, s in seen.items()
        if _cache_key(h, s) not in cache
    ]

    if uncached_items:
        # Build the Anthropic client.  The constructor reads ANTHROPIC_API_KEY
        # from the environment automatically (set in .env via python-dotenv).
        # In tests, `patch("src.features.nlp_features.anthropic.Anthropic")` intercepts
        # this call before any real network traffic occurs.
        client = anthropic.Anthropic()

        for i in range(0, len(uncached_items), NLP_BATCH_SIZE):
            batch = uncached_items[i : i + NLP_BATCH_SIZE]

            # Build the user message: prompt + formatted headline/summary items.
            # _format_batch includes [HEADLINE] / [SUMMARY] markers when a summary
            # is available, giving Claude richer context for extraction.
            prompt = EXTRACTION_PROMPT + _format_batch(batch)

            # 8192 tokens gives ~400 chars per signal × 20-signal batch with room
            # to spare.  2048 was too tight — batches >~5 headlines got truncated
            # mid-JSON and fell into the JSONDecodeError except, producing 0 signals.
            resp = client.messages.create(
                model=CLAUDE_NLP_MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the JSON array Claude returns.
            # Claude sometimes wraps the array in ```json ... ``` markdown fences —
            # strip those before parsing so json.loads sees raw JSON.
            try:
                raw_text = resp.content[0].text.strip()
                if raw_text.startswith("```"):
                    # drop opening fence line and closing fence
                    lines = raw_text.splitlines()
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    raw_text = "\n".join(lines).strip()
                parsed: list[dict] = json.loads(raw_text)
            except (json.JSONDecodeError, IndexError, AttributeError):
                parsed = []

            # Store each signal in the cache, keyed by headline + summary snippet.
            # Match signals back to batch items by position (Claude returns in order).
            for j, signal in enumerate(parsed):
                if j < len(batch):
                    item = batch[j]
                    key = _cache_key(item["headline"], item["summary"])
                    if key:
                        cache[key] = signal

        # Persist the updated cache to disk so the next call can skip these.
        _save_cache(cache)

    # Return signals for all requested headlines that are now in the cache.
    # We preserve the original row order from the input DataFrame.
    result = []
    for h, s in zip(headlines["headline"], summary_col):
        s_str = s if isinstance(s, str) else ""
        key = _cache_key(h, s_str)
        if key in cache:
            result.append(cache[key])
    return result


def _aggregate_scores(
    signals: list[dict],
    dates: pd.DatetimeIndex,
    headlines: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Aggregate signals into daily news_score and supply_chain_score per ticker.

    For each date in *dates*, we look back at all headlines published within
    the last NLP_DECAY_DAYS days.  Older headlines are ignored entirely.
    Headlines published today get a 2× recency bonus; older ones get 1×.

    The raw score formula for a primary ticker signal is:
        direction_score × magnitude_weight × confidence × recency_weight

    Supply-chain tickers get half the primary score, adjusted for their own
    direction relative to the primary event.

    Args:
        signals:   List of signal dicts from extract_signals().
        dates:     Business-day DatetimeIndex to compute scores for.
        headlines: Original DataFrame (provides published_at timestamps).

    Returns:
        Dict with keys "news_score" and "supply_chain_score", each a DataFrame
        indexed by *dates* with tickers as columns.
    """
    # Build a quick lookup: headline text → publication timestamp
    pub_times: dict[str, pd.Timestamp] = dict(
        zip(headlines["headline"], headlines["published_at"])
    )

    news_rows: list[dict[str, float]] = []
    sc_rows: list[dict[str, float]] = []

    for d in dates:
        news_day: dict[str, float] = {}
        sc_day: dict[str, float] = {}

        for sig in signals:
            pub = pub_times.get(sig.get("headline", ""))
            if pub is None:
                continue

            pub_ts = pd.Timestamp(pub)
            # Normalise to midnight and strip timezone so arithmetic works regardless
            # of whether the publication timestamp is tz-aware (UTC from Alpaca) or
            # d is a tz-naive Timestamp (from pd.bdate_range).
            pub_date = pub_ts.normalize()
            if pub_date.tzinfo is not None:
                pub_date = pub_date.tz_localize(None)
            d_naive = pd.Timestamp(d)
            if d_naive.tzinfo is not None:
                d_naive = d_naive.tz_localize(None)
            # days_ago: how many calendar days before date d was this published?
            days_ago = (d_naive - pub_date).days

            # Skip headlines that haven't happened yet (future) or are too old.
            if days_ago < 0 or days_ago >= NLP_DECAY_DAYS:
                continue

            # Recency weight: today's news is twice as impactful as yesterday's.
            # This is a simple step-function decay; more sophisticated versions
            # could use exponential decay, but KISS (Keep It Simple) for now.
            recency = 2.0 if days_ago == 0 else 1.0

            # Combine all three signal dimensions into a single signed score.
            score = (
                _direction_to_score(sig.get("direction", "neutral"))
                * _magnitude_to_weight(sig.get("magnitude", "medium"))
                * sig.get("confidence", 0.5)
                * recency
            )

            # Accumulate the primary ticker's news score (sum if multiple articles).
            ticker = sig.get("primary_ticker", "")
            if ticker:
                news_day[ticker] = news_day.get(ticker, 0.0) + score

            # Supply-chain / peer effects: each affected ticker gets a secondary
            # score.  We scale by 0.5 (second-order effect is weaker) and adjust
            # the sign based on the affected ticker's own direction.
            for affected in sig.get("affected_tickers", []):
                at = affected.get("ticker", "")
                if not at:
                    continue
                # Example: Apple beats → TSMC (supplier) is also bullish.
                # We multiply direction scores so both-bullish → positive,
                # opposite directions → negative (contrarian effect).
                at_score = (
                    score
                    * _direction_to_score(affected.get("direction", "neutral"))
                    * 0.5
                )
                sc_day[at] = sc_day.get(at, 0.0) + at_score

        news_rows.append(news_day)
        sc_rows.append(sc_day)

    # Build DataFrames.  fillna(0) so tickers with no signal on a given day
    # get a neutral score rather than NaN (which would be dropped in z-scoring).
    news_df = pd.DataFrame(news_rows, index=dates).fillna(0.0)
    sc_df   = pd.DataFrame(sc_rows,   index=dates).fillna(0.0)

    return {"news_score": news_df, "supply_chain_score": sc_df}


def compute_news_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Load recent headlines, extract signals, aggregate and z-score.

    This is the top-level entry point called by the feature pipeline.
    It glues together: news loading → Claude extraction → aggregation → z-scoring.

    Args:
        dates: Business-day DatetimeIndex for which to compute features.

    Returns:
        Wide DataFrame with MultiIndex columns (feature_name, ticker).
        Returns an empty DataFrame if no news is available.
    """
    # Lazy imports to avoid circular dependencies and keep module load fast.
    from src.features.price_features import cross_section_zscore
    from src.data.fetch_news import load_recent_news

    headlines = load_recent_news(days=NLP_DECAY_DAYS)
    if headlines.empty:
        return pd.DataFrame(index=dates)

    signals = extract_signals(headlines)
    if not signals:
        return pd.DataFrame(index=dates)

    scores = _aggregate_scores(signals, dates, headlines)

    # Z-score cross-sectionally: each day, a ticker's score is normalised
    # relative to the full universe average (same as price features).
    news_z = cross_section_zscore(scores["news_score"])
    sc_z   = cross_section_zscore(scores["supply_chain_score"])

    # Attach a feature-name level to the columns so downstream code can
    # select by feature name: df["news_score"]["AAPL"]
    news_z.columns = pd.MultiIndex.from_product([["news_score"],   news_z.columns])
    sc_z.columns   = pd.MultiIndex.from_product([["supply_chain_score"], sc_z.columns])

    return pd.concat([news_z, sc_z], axis=1)
