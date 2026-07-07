"""
fetch_macro.py — Fetch macro indicators from FRED API.

Series used:
  T10Y2Y  — 10-year minus 2-year Treasury spread. Negative = inverted yield
             curve, historically reliable recession predictor.
  UNRATE  — US unemployment rate. Rising rate = potential recession.
  FEDFUNDS — Federal funds rate. High rates = tighter financial conditions.

FRED is completely free. No API key needed for 120 requests/min.
Optional FRED_API_KEY in .env raises the rate limit.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)
FRED_SERIES = ["T10Y2Y", "UNRATE", "FEDFUNDS"]


def _fetch_series(series_id: str, start_date: str) -> pd.Series:
    """Fetch one FRED series as a date-indexed Series."""
    params = {
        "series_id":         series_id,
        "observation_start": start_date,
        "file_type":         "json",
        "sort_order":        "asc",
    }
    if config.FRED_API_KEY:
        params["api_key"] = config.FRED_API_KEY

    r = requests.get(config.FRED_BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    observations = r.json().get("observations", [])
    data = {
        obs["date"]: float(obs["value"])
        for obs in observations
        if obs["value"] != "."  # FRED uses "." for missing values
    }
    return pd.Series(data, name=series_id.lower())


def update_macro() -> None:
    """Fetch recent FRED data and append to macro.parquet."""
    out_path: Path = config.DATA_DIR / "macro" / "macro.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    frames: dict[str, pd.Series] = {}
    for series in FRED_SERIES:
        try:
            frames[series.lower()] = _fetch_series(series, start_date)
        except Exception as exc:
            log.warning("FRED fetch failed for %s: %s", series, exc)

    if not frames:
        return

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # Forward-fill monthly series (UNRATE, FEDFUNDS change infrequently) to daily
    df = df.set_index("date").resample("B").last().ffill().reset_index()

    combined = pd.concat([existing, df]).drop_duplicates(subset=["date"]).sort_values("date")
    combined.to_parquet(out_path, index=False)
    log.info("Macro: wrote %d rows to %s", len(combined), out_path)
