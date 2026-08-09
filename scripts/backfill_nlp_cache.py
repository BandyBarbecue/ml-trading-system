"""One-off, controlled backfill of historical NLP signals.

Why this exists (2026-07-10 post-mortem): the news fetcher used to deliver
files up to two days late, and extraction only ever paid for *fresh*
headlines — so headlines that arrived late were never extracted at all.
About 2,600 historical headlines in data/news/ have no cached signal, which
leaves news_score/supply_chain_score empty for many past dates the model
trains on.

The daily run will NOT backfill these by design: extract_signals' api_cutoff
circuit-breaker limits paid extraction to the last NLP_API_WINDOW_DAYS, so a
deliberate decision (running this script) is the only way to spend money on
old headlines.

Safety properties:
  - Dry run by default: prints how many headlines/batches it would buy and
    the rough cost, then exits. Nothing is spent without --yes.
  - Crash-safe: extract_signals saves the cache after every batch, so if the
    run dies (credits, network, Ctrl+C) everything completed so far is kept,
    and re-running only buys what's still missing. Re-running is always safe.

Usage (from trading-research/):
    .venv\\Scripts\\python.exe scripts\\backfill_nlp_cache.py         # dry run
    .venv\\Scripts\\python.exe scripts\\backfill_nlp_cache.py --yes   # spend
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NEWS_DIR, NLP_BATCH_SIZE  # noqa: E402
from src.features.nlp_features import CACHE_FILE, _cache_key, extract_signals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("backfill")


def load_all_news() -> pd.DataFrame:
    """Every news parquet, newest first — matching extract_signals' dedup order."""
    frames = [pd.read_parquet(f) for f in sorted(NEWS_DIR.glob("*.parquet"), reverse=True)]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true",
                        help="actually call the Claude API (default: dry run)")
    args = parser.parse_args()

    headlines = load_all_news()
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

    # Reproduce extract_signals' view of what is uncached (first summary wins).
    seen: dict[str, str] = {}
    summaries = headlines["summary"] if "summary" in headlines.columns else [""] * len(headlines)
    for h, s in zip(headlines["headline"], summaries):
        if h not in seen:
            seen[h] = s if isinstance(s, str) else ""
    uncached = sum(1 for h, s in seen.items() if _cache_key(h, s) not in cache)
    batches = -(-uncached // NLP_BATCH_SIZE)

    log.info("news files: %d, unique headlines: %d, uncached: %d",
             len(list(NEWS_DIR.glob("*.parquet"))), len(seen), uncached)
    log.info("would send ~%d batches (rough cost $%.2f-$%.2f at Haiku pricing)",
             batches, uncached * 0.001, uncached * 0.002)

    if not args.yes:
        log.info("DRY RUN — nothing sent. Re-run with --yes to spend.")
        return

    # api_cutoff=None deliberately disables the freshness circuit-breaker:
    # buying old headlines is the entire point of this script.
    signals = extract_signals(headlines, allow_api=True, api_cutoff=None)
    log.info("done: %d signals now available; cache saved incrementally at %s",
             len(signals), CACHE_FILE)


if __name__ == "__main__":
    main()
