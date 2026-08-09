"""modal_probe.py — Phase 0 de-risk probe. Run BEFORE committing to the migration.

The single biggest unknown in moving this system to Modal is whether the three
external data sources still answer from a datacenter IP. yfinance in particular
is a Yahoo scraper, not an official API, and cloud IP ranges get rate-limited
far more aggressively than home broadband. The daily pipeline makes ~1000-1500
yfinance calls, so if Yahoo blocks Modal's egress the migration is dead on
arrival and we need a different data source first.

This probe costs a fraction of a cent and takes under a minute.

    pip install modal
    modal setup                       # one-time browser auth
    modal run modal_probe.py

Read the output before writing any other Modal code.
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "yfinance>=0.2",
    "alpaca-py>=0.33",
    "anthropic>=0.34",
    "requests>=2.31",
    "pandas>=2.0",
    # get_earnings_dates parses HTML; without a parser it raises ImportError,
    # which an earlier run misread as a Yahoo failure.
    "lxml>=5.0",
    "html5lib>=1.1",
    "beautifulsoup4>=4.11",
)

app = modal.App("trading-probe", image=image)


@app.function(secrets=[modal.Secret.from_name("trading-secrets")], timeout=600)
def probe() -> dict:
    import os
    import time

    results: dict = {}

    # --- 0. Where are we egressing from? --------------------------------
    # Yahoo's rate limiter keys on IP, so knowing the ASN helps interpret a 429.
    import requests
    try:
        r = requests.get("https://ipinfo.io/json", timeout=15)
        info = r.json()
        results["egress"] = f"{info.get('ip')} / {info.get('org')} / {info.get('country')}"
    except Exception as exc:
        results["egress"] = f"UNKNOWN ({type(exc).__name__}: {exc})"

    # --- 1. yfinance: the make-or-break test ----------------------------
    # 30 tickers at the same 0.05s pacing the real fetcher uses. If this
    # succeeds cleanly we can extrapolate to 501; if it 429s early, we know.
    import yfinance as yf

    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
        "MRK", "ABBV", "KO", "PEP", "COST", "ADBE", "CRM", "ACN", "MCD",
        "TMO", "CSCO", "INTC",
    ]
    # Score the fields fetch_fundamentals.py ACTUALLY reads. An earlier version
    # of this probe gated on info["marketCap"], which the pipeline never touches
    # -- it lives in a different module of Yahoo's quoteSummary payload, so a
    # ticker can lack marketCap while still returning both fields we need. That
    # bug, plus an arbitrary 90% threshold, produced a false "Yahoo is throttling
    # this IP" FAIL on 2026-08-09. A same-day full 503-ticker sweep from Modal
    # returned 0/503 missing priceToBook and 1/503 missing grossMargins.
    ok, failed, empty = 0, [], []
    t0 = time.time()
    for sym in tickers:
        try:
            info = yf.Ticker(sym).info or {}
            if info.get("priceToBook") is not None and info.get("grossMargins") is not None:
                ok += 1
            else:
                empty.append(sym)
        except Exception as exc:
            failed.append(f"{sym}:{type(exc).__name__}")
        time.sleep(0.05)
    elapsed = time.time() - t0

    results["yfinance_info"] = (
        f"{ok}/{len(tickers)} ok, {len(empty)} partial, {len(failed)} raised "
        f"in {elapsed:.1f}s -- partial={empty[:5]} raised={failed[:5]}"
    )
    # Real throttling shows up as RAISED exceptions and a rate that climbs as
    # the run proceeds. Occasional partial payloads do not block the pipeline:
    # fetch_fundamentals writes None for a missing field rather than dropping
    # the ticker, and only a raised exception loses the row entirely.
    results["yfinance_verdict"] = (
        "PASS" if not failed and ok >= len(tickers) * 0.75
        else "FAIL -- exceptions raised, investigate before migrating"
    )

    # yfinance is also the VIX source, via a different endpoint (download vs info)
    try:
        vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=False)
        results["yfinance_vix"] = f"{len(vix)} rows" if vix is not None else "None"
    except Exception as exc:
        results["yfinance_vix"] = f"FAILED ({type(exc).__name__}: {exc})"

    # --- 2. Alpaca: market data + trading ---------------------------------
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
        )
        acct = client.get_account()
        positions = client.get_all_positions()
        results["alpaca_trading"] = (
            f"OK -- equity={acct.equity}, {len(positions)} positions, status={acct.status}"
        )
    except Exception as exc:
        results["alpaca_trading"] = f"FAILED ({type(exc).__name__}: {exc})"

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta

        dc = StockHistoricalDataClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
        )
        req = StockBarsRequest(
            symbol_or_symbols=["SPY"],
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=10),
        )
        bars = dc.get_stock_bars(req).df
        results["alpaca_data"] = f"OK -- {len(bars)} SPY daily bars"
    except Exception as exc:
        results["alpaca_data"] = f"FAILED ({type(exc).__name__}: {exc})"

    # --- 3. Anthropic: confirms the key AND that credits are topped up ----
    try:
        import anthropic
        c = anthropic.Anthropic()
        msg = c.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        results["anthropic"] = f"OK -- {msg.content[0].text.strip()!r}"
    except Exception as exc:
        results["anthropic"] = f"FAILED ({type(exc).__name__}: {exc})"

    # --- 4. FRED + Wikipedia (the two requests-based sources) -------------
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "T10Y2Y",
                "api_key": os.environ.get("FRED_API_KEY", ""),
                "file_type": "json",
                "limit": 5,
            },
            timeout=15,
        )
        results["fred"] = f"HTTP {r.status_code}"
    except Exception as exc:
        results["fred"] = f"FAILED ({type(exc).__name__}: {exc})"

    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        results["wikipedia"] = f"HTTP {r.status_code}, {len(r.content)} bytes"
    except Exception as exc:
        results["wikipedia"] = f"FAILED ({type(exc).__name__}: {exc})"

    # --- 5. Container environment ----------------------------------------
    import sys
    from datetime import datetime as _dt
    results["python"] = sys.version.split()[0]
    results["tz_env"] = os.environ.get("TZ", "<unset>")
    # This is the value config.END_DATE would freeze at, and what all 22 naive
    # datetime.now() call sites would see.
    results["naive_now"] = _dt.now().isoformat(timespec="seconds")

    return results


@app.local_entrypoint()
def main():
    out = probe.remote()
    print("\n" + "=" * 68)
    print("MODAL PHASE-0 PROBE")
    print("=" * 68)
    for k, v in out.items():
        print(f"{k:>18} : {v}")
    print("=" * 68)
    verdict = out.get("yfinance_verdict", "")
    if verdict.startswith("PASS"):
        print("yfinance works from Modal -- migration can proceed as planned.")
    else:
        print("yfinance is throttled from Modal -- STOP. Replace the yfinance")
        print("sources (fundamentals/insider/earnings/VIX) before migrating.")
    print()
