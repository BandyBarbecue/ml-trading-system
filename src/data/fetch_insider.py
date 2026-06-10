"""
fetch_insider.py — Fetch SEC EDGAR Form 4 (insider transactions).

SEC EDGAR is completely free and requires no API key.  We use the EDGAR
submissions API to find Form 4 filings for each ticker.

Form 4 transaction codes:
  P = Purchase (bullish signal)
  S = Sale (bearish signal)
  A, F = Grants/withheld for taxes — excluded (not open-market trades)

Executive trades (CEO/CFO/President) get a 2x weight multiplier because
executive insider trades carry more information content.

CIK map: EDGAR identifies companies by CIK (Central Index Key), not ticker.
We build a CIK map from EDGAR's company_tickers.json file.
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "TradingResearch edward.mcandrew20@gmail.com"}
SIGNAL_CODES = {"P", "S"}  # Purchase, Sale
EXECUTIVE_TITLES = {"ceo", "cfo", "president", "chief executive", "chief financial"}


def _get_cik_map() -> dict[str, str]:
    """Download the SEC company → CIK mapping (cached in DATA_DIR)."""
    cache = config.DATA_DIR / "sec_cik_map.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    cik_map = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in raw.values()
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cik_map))
    return cik_map


def _parse_form4(xml_text: str, ticker: str) -> list[dict]:
    """Extract transaction rows from a Form 4 XML document."""
    rows = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return rows

    rel = root.find(".//reportingOwnerRelationship")
    is_officer = (rel.findtext("isOfficer", "0") == "1") if rel is not None else False
    title = (rel.findtext("officerTitle", "") or "").lower() if rel is not None else ""
    is_executive = any(t in title for t in EXECUTIVE_TITLES)
    role_weight = 2.0 if (is_officer and is_executive) else 1.0
    insider_type = title if title else ("officer" if is_officer else "director")

    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el   = txn.find(".//transactionCode")
        date_el   = txn.find(".//transactionDate/value")
        shares_el = txn.find(".//transactionShares/value")
        price_el  = txn.find(".//transactionPricePerShare/value")

        if code_el is None or date_el is None:
            continue
        code = (code_el.text or "").strip()
        if code not in SIGNAL_CODES:
            continue

        try:
            shares = float((shares_el.text or "0").replace(",", "")) if shares_el is not None else 0.0
            price  = float((price_el.text or "0").replace(",", "")) if price_el is not None else 0.0
        except ValueError:
            continue

        rows.append({
            "ticker":           ticker,
            "date":             date_el.text.strip(),
            "insider_type":     insider_type,
            "transaction_type": code,
            "shares":           shares * role_weight,
            "value_usd":        shares * price * role_weight,
        })
    return rows


def update_insider(
    symbols: list[str] | None = None,
    cik_map: dict[str, str] | None = None,
) -> None:
    """Fetch recent Form 4 filings for each symbol and append to insider.parquet."""
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    if cik_map is None:
        cik_map = _get_cik_map()

    out_path: Path = config.INSIDER_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()

    rows = []
    for sym in symbols:
        cik = cik_map.get(sym)
        if not cik:
            log.debug("No CIK for %s — skipping.", sym)
            continue
        try:
            url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            recent = data.get("filings", {}).get("recent", {})
            forms  = recent.get("form", [])
            dates  = recent.get("filingDate", [])
            docs   = recent.get("primaryDocument", [])

            for form, fdate, doc in zip(forms, dates, docs):
                if form != "4":
                    continue
                # Build the direct URL to the filing document using EDGAR archives
                cik_int = int(cik)
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{doc}"
                try:
                    dr = requests.get(doc_url, headers=HEADERS, timeout=10)
                    dr.raise_for_status()
                    rows.extend(_parse_form4(dr.text, sym))
                except Exception as exc:
                    log.debug("Form 4 parse failed for %s %s: %s", sym, doc, exc)
        except Exception as exc:
            log.warning("EDGAR fetch failed for %s: %s", sym, exc)

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["ticker", "date", "insider_type", "transaction_type", "shares"]
    )
    combined.to_parquet(out_path, index=False)
    log.info("Insider: wrote %d rows to %s", len(combined), out_path)
