"""
fetch_fundamentals.py — Pull gross profitability and P/B ratio from FMP.

FMP endpoint: GET /key-metrics/{symbol}?limit=20&apikey=...
Returns a list of quarterly snapshots newest-first.

45-day reporting lag rule: financials filed with the SEC up to 45 days after
quarter end. We store the raw filing date and enforce the lag in the feature
function, not here. The data layer just stores what FMP reports.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)


def update_fundamentals(symbols: list[str] | None = None) -> None:
    """Fetch FMP key-metrics for each symbol and append to fundamentals.parquet.

    Uses config.FUNDAMENTALS_FILE as the target path — override via monkeypatch
    in tests (same pattern as fetch_earnings.py).
    """
    if not config.FMP_API_KEY:
        log.warning("FMP_API_KEY not set — skipping fundamentals update.")
        return

    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    out_path: Path = config.FUNDAMENTALS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    existing_keys = set(
        zip(existing["ticker"], existing["date"])
    ) if not existing.empty else set()

    rows = []
    for sym in symbols:
        url = f"{config.FMP_BASE_URL}/key-metrics/{sym}"
        params = {"limit": 20, "apikey": config.FMP_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            for snap in r.json():
                date_str = snap.get("date", "")
                if (sym, date_str) in existing_keys:
                    continue
                rows.append({
                    "ticker":              sym,
                    "date":                date_str,
                    "gross_profitability": snap.get("grossProfitMargin"),
                    "pb_ratio":            snap.get("priceToBookRatio"),
                })
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                log.error("FMP API authentication failed — aborting. Check FMP_API_KEY.")
                return  # abort entire update
            log.warning("fundamentals HTTP error for %s: %s", sym, exc)
        except Exception as exc:
            log.warning("fundamentals fetch failed for %s: %s", sym, exc)

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    combined.to_parquet(out_path, index=False)
    log.info("Fundamentals: wrote %d rows to %s", len(combined), out_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s — %(message)s")
    from src.data.sp500_universe import get_symbols
    update_fundamentals(get_symbols())
