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
    NLP_API_WINDOW_DAYS,
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


def _cache_key(headline: str, summary: str) -> str:
    """Cache key = headline + "|" + first 200 chars of summary.

    Including the summary snippet means that if the article body changes
    (e.g. Alpaca backfills a summary later) we re-extract with the richer
    context. Module-level (not nested) so the one-off cache migration script
    and tests can build keys in exactly the same format.
    """
    return headline + "|" + summary[:200]


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

def extract_signals(
    headlines: pd.DataFrame,
    allow_api: bool = True,
    api_cutoff: pd.Timestamp | None = None,
) -> list[dict]:
    """Send headlines to Claude Haiku and return structured signals.

    Processes headlines in batches of NLP_BATCH_SIZE.  Already-seen headlines
    (matched by text) are returned from the on-disk cache without API calls.

    Args:
        headlines: DataFrame with at least a "headline" column (str).
                   Typically also has ticker, published_at, source columns.
        allow_api: When False, never calls the Claude API — only headlines
                   already present in the on-disk cache are returned;
                   anything uncached is silently dropped (same as if it were
                   never found). This is used by compute_news_features() for
                   OLD historical headlines, so that widening the matrix
                   build's date window can only ever surface signals Claude
                   already scored on some earlier run — it can never trigger
                   fresh (paid) extraction for stale news.
        api_cutoff: Optional timestamp guard on PAID extraction. Uncached
                   headlines published BEFORE this moment are treated as
                   cache-only (dropped, never sent to the API), even when
                   allow_api=True. This is the circuit-breaker for the
                   2026-07-10 incident: a cache-key change made ~2 years of
                   already-paid-for headlines look uncached, and the run
                   tried to re-extract all of them at once. With a cutoff,
                   the worst case is a few days of headlines. None disables
                   the guard (old behaviour).

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
    pub_col = (
        headlines["published_at"]
        if "published_at" in headlines.columns
        else pd.Series([None] * len(headlines), index=headlines.index)
    )
    # Deduplicate by headline text; keep the first summary/timestamp seen for each.
    seen: dict[str, str] = {}
    pub_times: dict[str, pd.Timestamp | None] = {}
    for h, s, p in zip(headlines["headline"], summary_col, pub_col):
        if h not in seen:
            seen[h] = s if isinstance(s, str) else ""
            pub_times[h] = p

    def _is_too_old(headline: str) -> bool:
        """True if this headline predates api_cutoff (so paid extraction is barred)."""
        if api_cutoff is None:
            return False
        pub = pub_times.get(headline)
        if pub is None:
            return False  # no timestamp to judge by — treat as fresh
        pub_ts = pd.Timestamp(pub)
        # Compare in UTC; parquet timestamps are tz-aware UTC, test fixtures
        # are sometimes naive — localise defensively, same as _aggregate_scores.
        if pub_ts.tzinfo is None:
            pub_ts = pub_ts.tz_localize("UTC")
        cutoff = api_cutoff if api_cutoff.tzinfo else api_cutoff.tz_localize("UTC")
        return pub_ts < cutoff

    uncached_items = [
        {"headline": h, "summary": s}
        for h, s in seen.items()
        if _cache_key(h, s) not in cache and not _is_too_old(h)
    ]

    # allow_api=False means: serve cache hits only, never touch the network.
    # Uncached headlines simply produce no signal (they're dropped below).
    if uncached_items and allow_api:
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

            # Match signals back to batch items. Claude is asked to echo the
            # headline verbatim, so match on that first — it survives Claude
            # skipping or reordering items. Fall back to position only when
            # the echoed headline doesn't match anything we sent.
            by_headline = {item["headline"]: item for item in batch}
            for j, signal in enumerate(parsed):
                if not isinstance(signal, dict):
                    continue  # malformed element (e.g. a bare string) — skip
                item = by_headline.get(signal.get("headline", ""))
                if item is None and j < len(batch):
                    item = batch[j]
                if item is None:
                    continue
                cache[_cache_key(item["headline"], item["summary"])] = signal

            # Any batch item Claude answered the batch for but returned no
            # entry on is a deliberate skip — cache a sentinel so we stop
            # paying to re-send it on every future run. If the WHOLE batch
            # failed to parse (parsed == []), that's a transient error
            # (truncated JSON, API hiccup): cache nothing, retry next run.
            if parsed:
                for item in batch:
                    key = _cache_key(item["headline"], item["summary"])
                    if key not in cache:
                        cache[key] = {"_no_signal": True}

            # Persist after EVERY batch, not once at the end. Each batch is
            # paid-for API work — the 2026-07-10 run crashed at batch ~70 of
            # 145 and lost all 69 completed batches because saving only
            # happened after the full loop.
            _save_cache(cache)

    # Return signals for all requested headlines that are now in the cache.
    # We preserve the original row order from the input DataFrame.
    # Sentinel entries ({"_no_signal": true}) mean "already asked Claude,
    # nothing useful here" — they suppress re-extraction above but must not
    # leak into the signal list.
    result = []
    for h, s in zip(headlines["headline"], summary_col):
        s_str = s if isinstance(s, str) else ""
        signal = cache.get(_cache_key(h, s_str))
        if isinstance(signal, dict) and not signal.get("_no_signal"):
            result.append(signal)
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
            # fetch_news stores Alpaca timestamps, which are UTC. Defensively
            # assume UTC if we ever see a tz-naive timestamp (e.g. from an
            # older cache) so the ET conversion below is still correct.
            if pub_ts.tzinfo is None:
                pub_ts = pub_ts.tz_localize("UTC")
            pub_et = pub_ts.tz_convert("US/Eastern")

            # US equity markets close at 16:00 ET. A headline published after
            # that cutoff was not public before that trading day's close, so
            # it can't be "today's" news for feature purposes — it belongs to
            # the NEXT calendar day's window. Without this, a 17:15 ET
            # headline would count at 2x strength for a day whose close had
            # already happened before the news broke (look-ahead bias).
            close_cutoff = pub_et.normalize() + pd.Timedelta(hours=16)
            if pub_et >= close_cutoff:
                effective_date = pub_et.normalize() + pd.Timedelta(days=1)
            else:
                effective_date = pub_et.normalize()
            # Strip timezone so arithmetic works against d, which is a
            # tz-naive Timestamp (from pd.bdate_range).
            effective_date = effective_date.tz_localize(None)

            d_naive = pd.Timestamp(d)
            if d_naive.tzinfo is not None:
                d_naive = d_naive.tz_localize(None)
            # days_ago: how many calendar days before date d does the
            # effective (post-close-cutoff-adjusted) news date fall?
            days_ago = (d_naive - effective_date).days

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
            # sig.get(..., []) only supplies the default when the key is
            # MISSING — if Claude returns "affected_tickers": null, .get()
            # happily returns None and the for-loop below crashes with a
            # TypeError. `or []` catches both the missing-key and
            # explicit-null cases.
            for affected in (sig.get("affected_tickers") or []):
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


def compute_news_features(dates: pd.DatetimeIndex, allow_api: bool = True) -> pd.DataFrame:
    """Load recent headlines, extract signals, aggregate and z-score.

    This is the top-level entry point called by the feature pipeline.
    It glues together: news loading → Claude extraction → aggregation → z-scoring.

    Args:
        dates: Business-day DatetimeIndex for which to compute features.
        allow_api: If False, never call Claude for uncached headlines (see
            extract_signals) — only already-cached signals are used.

    Returns:
        Wide DataFrame with MultiIndex columns (feature_name, ticker).
        Returns an empty DataFrame if no news is available.
    """
    # Lazy imports to avoid circular dependencies and keep module load fast.
    from src.features.price_features import cross_section_zscore
    from src.data.fetch_news import load_recent_news

    # Span needed: from (earliest date - decay window) through the latest date,
    # so _aggregate_scores has real headlines available for EVERY date in
    # `dates`, not just the trailing few days near whenever this function
    # happens to be called. Anchoring on dates.max() (not date.today()) is
    # what makes this correct for historical matrix builds — as_of=None
    # would silently fall back to today and reproduce the exact bug.
    span_days = (dates.max() - dates.min()).days + NLP_DECAY_DAYS + 1
    headlines = load_recent_news(days=span_days, as_of=dates.max().date())
    if headlines.empty:
        return pd.DataFrame(index=dates)

    # Paid extraction is only allowed for headlines from the last
    # NLP_API_WINDOW_DAYS of the window being built. Runs happen daily, so
    # anything older was already extracted (and cached) by an earlier run —
    # if it LOOKS uncached, something is wrong (e.g. a cache-key change),
    # and silently re-buying 2 years of extraction is never the right answer.
    api_cutoff = dates.max() - pd.Timedelta(days=NLP_API_WINDOW_DAYS)
    if api_cutoff.tzinfo is None:  # bdate_range gives tz-naive dates
        api_cutoff = api_cutoff.tz_localize("UTC")
    signals = extract_signals(headlines, allow_api=allow_api, api_cutoff=api_cutoff)
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
