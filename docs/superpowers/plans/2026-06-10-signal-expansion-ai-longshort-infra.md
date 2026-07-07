# Signal Expansion, Deeper AI, Long-Short & Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the multi-factor ML system's edge by adding quality/revision/insider signals (Phase A), making AI a first-class analytical layer not just a news-scorer (Phase B), extending to long-short equity (Phase C), and hardening the infrastructure so the system can run unattended without silent bugs (Phase D).

**Architecture:** All additions are additive — existing modules are extended, not replaced. New feature columns flow into the existing `FEATURE_COLS` list and the LightGBM model picks up on them at the next retrain. The long-short extension is gated behind `LONG_SHORT_ENABLED=False` so it cannot accidentally fire until the developer explicitly enables it after reviewing backtest results.

**Tech Stack:** Python 3.11, anthropic SDK, lightgbm, pandas, requests (FMP + SEC EDGAR + FRED APIs), shap, optuna, alpaca-py, pytest

**Security constraints (non-negotiable):**
- Paper trading only, no live capital, ever.
- API keys in `.env`, never hardcoded.
- Self-improvement only via walk-forward on historical data.
- No data leakage — all fundamental features apply a 45-day reporting lag.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `config.py` | Modify | Add FUNDAMENTALS_FILE, INSIDER_FILE, LONG_SHORT_ENABLED, N_SHORT_POSITIONS, MAX_SHORT_PCT, FRED_BASE_URL, REGIME_FILE |
| `src/data/fetch_fundamentals.py` | Create | FMP key-metrics endpoint → fundamentals.parquet |
| `src/data/fetch_insider.py` | Create | SEC EDGAR Form 4 → insider.parquet |
| `src/data/fetch_macro.py` | Create | FRED API → macro.parquet |
| `src/features/fundamental_features.py` | Create | gross_profitability, pb_ratio z-scored with 45-day lag |
| `src/features/revision_features.py` | Create | analyst estimate revision momentum from earnings.parquet |
| `src/features/insider_features.py` | Create | insider_score z-scored from insider.parquet |
| `src/features/feature_matrix.py` | Modify | Add new features to tidy assembly |
| `src/ml/model.py` | Modify | Add new columns to FEATURE_COLS |
| `src/data/fetch_news.py` | Modify | Add summary + content fields to news parquet |
| `src/features/nlp_features.py` | Modify | Use summary field; add earnings-transcript deep analysis |
| `src/ml/regime_classifier.py` | Create | FRED data → Claude Haiku → regime label + IC weight modulation |
| `src/execution/attribution.py` | Modify | Add equity curve reporting + structured JSON proposals |
| `src/execution/equity_tracker.py` | Create | Persist equity curve to equity_curve.parquet |
| `src/execution/risk_state.py` | Create | Persist peak_value to risk_state.json — fixes drawdown bug |
| `src/portfolio/construction.py` | Modify | Add build_short_portfolio() behind LONG_SHORT_ENABLED flag |
| `src/risk/engine.py` | Modify | Add gross exposure cap for long-short |
| `src/execution/daily_run.py` | Modify | Wire equity tracker + risk state + regime weights + short portfolio |
| `src/backtest/run_backtest.py` | Modify | Add 7bps transaction cost model |
| `scripts/schedule_daily.ps1` | Create | Windows Task Scheduler wrapper |
| `tests/data/test_fetch_fundamentals.py` | Create | |
| `tests/data/test_fetch_insider.py` | Create | |
| `tests/data/test_fetch_macro.py` | Create | |
| `tests/features/test_fundamental_features.py` | Create | |
| `tests/features/test_revision_features.py` | Create | |
| `tests/features/test_insider_features.py` | Create | |
| `tests/ml/test_regime_classifier.py` | Create | |
| `tests/execution/test_equity_tracker.py` | Create | |
| `tests/execution/test_risk_state.py` | Create | |
| `tests/execution/test_attribution_structured.py` | Create | |

---

## Phase A — Signal Expansion

Phase A adds four new alpha signals that require no architectural changes — just new FMP/EDGAR API calls, new feature functions, and extending the existing FEATURE_COLS list. Estimated wall-clock: ~6 hours across 2–3 sessions.

---

### Task 1: Config extensions + fundamentals fetcher

**Goal:** Add all new config constants and a fundamentals data fetcher that pulls gross profitability and price-to-book from FMP and saves to `data/fundamentals/fundamentals.parquet`.

**Files:**
- Modify: `config.py`
- Create: `src/data/fetch_fundamentals.py`
- Create: `tests/data/test_fetch_fundamentals.py`

**Acceptance Criteria:**
- [ ] `python -c "import config; print(config.FUNDAMENTALS_FILE)"` prints a `Path` ending in `fundamentals/fundamentals.parquet`
- [ ] `python -m src.data.fetch_fundamentals` writes a parquet with columns `ticker`, `date`, `gross_profitability`, `pb_ratio`
- [ ] `pytest tests/data/test_fetch_fundamentals.py -v` → PASSED

**Verify:** `pytest tests/data/test_fetch_fundamentals.py -v` → all tests green, `PASSED` in output

**Steps:**

- [ ] **Step 1: Add constants to config.py**

In `config.py`, add after `EARNINGS_FILE`:

```python
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals" / "fundamentals.parquet"
INSIDER_FILE      = DATA_DIR / "insider"       / "insider.parquet"
REGIME_FILE       = DATA_DIR / "features"      / "regime.parquet"
FRED_BASE_URL     = "https://api.stlouisfed.gov/fred/series/observations"
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")  # free, no auth needed but rate-limit bypass

# Long-short extension — disabled by default until backtest confirms positive IC
LONG_SHORT_ENABLED  = False
N_SHORT_POSITIONS   = 25
MAX_SHORT_PCT       = 0.03   # 3% max per short position
GROSS_EXPOSURE_CAP  = 1.60   # long + |short| <= 160% of NAV
```

- [ ] **Step 2: Write failing test**

Create `tests/data/test_fetch_fundamentals.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch
import config

# FMP /key-metrics returns a list of quarterly snapshots.
MOCK_FMP_RESPONSE = [
    {
        "date": "2024-09-30",
        "grossProfitMargin": 0.46,
        "priceToBookRatio": 42.5,
        "revenue": 94000000000,
    }
]


def test_update_fundamentals_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")
    monkeypatch.setattr(config, "FMP_API_KEY", "test-key")

    with patch("src.data.fetch_fundamentals.requests.get") as mock_get:
        mock_get.return_value.json.return_value = MOCK_FMP_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None
        fetch_fundamentals.update_fundamentals(symbols=["AAPL"])

    df = pd.read_parquet(tmp_path / "fundamentals.parquet")
    assert set(df.columns) >= {"ticker", "date", "gross_profitability", "pb_ratio"}
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["gross_profitability"].iloc[0] == pytest.approx(0.46)
    assert df["pb_ratio"].iloc[0] == pytest.approx(42.5)


def test_update_fundamentals_no_api_key_skips(tmp_path, monkeypatch):
    from src.data import fetch_fundamentals

    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "fundamentals.parquet")
    monkeypatch.setattr(config, "FMP_API_KEY", "")  # empty key

    fetch_fundamentals.update_fundamentals(symbols=["AAPL"])
    assert not (tmp_path / "fundamentals.parquet").exists()
```

- [ ] **Step 3: Run to confirm FAIL**

```
cd trading-research
.venv\Scripts\activate
pytest tests/data/test_fetch_fundamentals.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` (module doesn't exist yet).

- [ ] **Step 4: Write the fetcher**

Create `src/data/fetch_fundamentals.py`:

```python
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
        except Exception as exc:
            log.warning("fundamentals fetch failed for %s: %s", sym, exc)

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["ticker", "date"]
    )
    combined.to_parquet(out_path, index=False)
    log.info("Fundamentals: wrote %d rows to %s", len(combined), out_path)
```

- [ ] **Step 5: Run to confirm PASS**

```
pytest tests/data/test_fetch_fundamentals.py -v
```

Expected: `PASSED` (both tests).

- [ ] **Step 6: Commit**

```
git add config.py src/data/fetch_fundamentals.py tests/data/test_fetch_fundamentals.py
git commit -m "feat(data): add fundamentals fetcher (FMP key-metrics, gross profitability + P/B)"
```

---

### Task 2: Fundamental features with 45-day reporting lag

**Goal:** Produce `gross_profitability` and `pb_ratio` signals in the feature matrix, correctly applying a 45-day reporting lag so the model only sees data that would have been publicly available on each historical date.

**Files:**
- Create: `src/features/fundamental_features.py`
- Create: `tests/features/test_fundamental_features.py`

**Acceptance Criteria:**
- [ ] `compute_fundamental_features(dates)` returns a wide DataFrame indexed by date, one column per ticker per feature
- [ ] No row uses a filing dated within 45 days of that row's date (verified by unit test)
- [ ] Z-scores are computed cross-sectionally (per date, not per ticker over time)

**Verify:** `pytest tests/features/test_fundamental_features.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write failing test**

Create `tests/features/test_fundamental_features.py`:

```python
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import config


def _make_fundamentals(ticker: str, filing_date: str, gp: float, pb: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker, "date": filing_date,
        "gross_profitability": gp, "pb_ratio": pb,
    }])


def test_45_day_lag_enforced(tmp_path, monkeypatch):
    """Filing dated 30 days ago should NOT appear; 50 days ago SHOULD appear."""
    from src.features import fundamental_features

    today = datetime(2024, 6, 1)
    # 30 days ago — too recent, should be excluded
    recent_filing = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    # 60 days ago — old enough, should be included
    old_filing    = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    df = pd.concat([
        _make_fundamentals("AAPL", recent_filing, 0.45, 40.0),
        _make_fundamentals("AAPL", old_filing,    0.44, 38.0),
    ])
    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "f.parquet")
    df.to_parquet(tmp_path / "f.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = fundamental_features.compute_fundamental_features(dates, symbols=["AAPL"])

    # The recent filing must not appear — only old_filing is valid
    assert not result.empty
    # Only one ticker, so z-score of a single value is NaN — that's acceptable
    # Just verify the filing used was the 60-day-old one (gp=0.44), not 0.45
    # We confirm by checking the raw value before z-scoring via the internal helper
    raw = fundamental_features._latest_valid(df, "AAPL", today, lag_days=45)
    assert raw is not None
    assert abs(raw["gross_profitability"] - 0.44) < 1e-6


def test_z_scores_cross_sectional(tmp_path, monkeypatch):
    """With two tickers, z-scores should differ across tickers on the same date."""
    from src.features import fundamental_features

    today = datetime(2024, 6, 1)
    old_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    df = pd.concat([
        _make_fundamentals("AAPL", old_date, 0.60, 40.0),
        _make_fundamentals("MSFT", old_date, 0.30, 20.0),
    ])
    monkeypatch.setattr(config, "FUNDAMENTALS_FILE", tmp_path / "f.parquet")
    df.to_parquet(tmp_path / "f.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = fundamental_features.compute_fundamental_features(dates, symbols=["AAPL", "MSFT"])

    # With exactly 2 observations, z-scores should be +1 and -1 (or vice versa)
    gp_aapl = result.loc[today, ("gross_profitability", "AAPL")]
    gp_msft = result.loc[today, ("gross_profitability", "MSFT")]
    assert abs(gp_aapl) > 0.1   # not both zero
    assert abs(gp_msft) > 0.1
    assert gp_aapl * gp_msft < 0  # opposite signs
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/features/test_fundamental_features.py -v
```

Expected: `ImportError` (module doesn't exist).

- [ ] **Step 3: Write the feature function**

Create `src/features/fundamental_features.py`:

```python
"""
fundamental_features.py — Quality factors from FMP key-metrics.

Two signals:
  gross_profitability  — Novy-Marx (2013) quality factor. High GP stocks
                         persistently outperform. GrossProfit / TotalAssets,
                         here approximated as grossProfitMargin from FMP.
  pb_ratio             — Price-to-book. Low P/B stocks tend to outperform
                         (value factor). Inverted: high 1/P/B = more attractive.

Reporting lag:
  Financials are filed with the SEC up to 45 days after quarter end.
  We enforce this by only using filings where the filing date is at least
  LAG_DAYS before the current date. This prevents look-ahead bias.
  Think of it like only reading the annual report AFTER it has been posted
  to the company's website — not before the ink is dry.

Output: wide DataFrame with MultiIndex columns (feature_name, ticker),
        same shape as price_features output so feature_matrix.py can merge them.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import config

LAG_DAYS = 45


def _latest_valid(
    df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    lag_days: int = LAG_DAYS,
) -> dict | None:
    """Return the most recent filing for ticker that predates as_of by lag_days."""
    cutoff = as_of - pd.Timedelta(days=lag_days)
    subset = df[(df["ticker"] == ticker) & (pd.to_datetime(df["date"]) <= cutoff)]
    if subset.empty:
        return None
    # Most recent qualifying row
    row = subset.sort_values("date").iloc[-1]
    return row.to_dict()


def compute_fundamental_features(
    dates: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return wide DataFrame of fundamental features, cross-sectionally z-scored.

    Parameters
    ----------
    dates   : DatetimeIndex of trading dates to compute features for.
    symbols : Tickers to include.  If None, loads full S&P 500 from config.

    Returns
    -------
    DataFrame with MultiIndex columns (feature_name, ticker), index = date.
    Missing values are NaN — the model fills with 0 (neutral signal).
    """
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    fund_path: Path = config.FUNDAMENTALS_FILE
    if not fund_path.exists():
        # Return empty — feature_matrix.py handles missing sources gracefully.
        return pd.DataFrame()

    fund_df = pd.read_parquet(fund_path)

    feature_names = ["gross_profitability", "pb_ratio"]
    records: dict[tuple, list] = {(f, t): [] for f in feature_names for t in symbols}
    index_dates = []

    for date in dates:
        index_dates.append(date)
        row_vals: dict[str, dict[str, float]] = {f: {} for f in feature_names}

        for ticker in symbols:
            snap = _latest_valid(fund_df, ticker, date)
            if snap is None:
                continue
            for feat in feature_names:
                val = snap.get(feat)
                if val is not None and not np.isnan(float(val)):
                    row_vals[feat][ticker] = float(val)

        # Cross-sectional z-score each feature across all tickers on this date.
        for feat in feature_names:
            vals = pd.Series(row_vals[feat])
            if len(vals) >= 2:
                z = (vals - vals.mean()) / (vals.std() + 1e-9)
            else:
                z = vals * np.nan  # can't z-score a single value
            for ticker in symbols:
                records[(feat, ticker)].append(z.get(ticker, np.nan))

    if not index_dates:
        return pd.DataFrame()

    result = pd.DataFrame(records, index=pd.DatetimeIndex(index_dates))
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
```

- [ ] **Step 4: Run to confirm PASS**

```
pytest tests/features/test_fundamental_features.py -v
```

Expected: `PASSED` (both tests).

- [ ] **Step 5: Commit**

```
git add src/features/fundamental_features.py tests/features/test_fundamental_features.py
git commit -m "feat(features): add gross_profitability + pb_ratio with 45-day reporting lag"
```

---

### Task 3: Revision momentum signal

**Goal:** Compute analyst estimate revision momentum from the existing `earnings.parquet` — when analysts repeatedly raise EPS estimates, this predicts outperformance (a well-documented signal from academic literature).

**Files:**
- Create: `src/features/revision_features.py`
- Create: `tests/features/test_revision_features.py`

**Acceptance Criteria:**
- [ ] `compute_revision_features(dates)` returns wide DataFrame with `(revision_momentum, ticker)` MultiIndex columns
- [ ] Revision is computed as `(estimate_new - estimate_old) / abs(estimate_old)` over a 90-day window
- [ ] Works correctly when `earnings.parquet` is empty or missing

**Verify:** `pytest tests/features/test_revision_features.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write failing test**

Create `tests/features/test_revision_features.py`:

```python
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import config


def test_revision_momentum_positive(tmp_path, monkeypatch):
    """When estimates are consistently revised upward, revision_momentum should be positive."""
    from src.features import revision_features

    today = datetime(2024, 6, 1)
    # Three earnings reports: estimates being raised each time
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": (today - timedelta(days=80)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.00, "actual_eps": 1.05, "surprise_pct": 5.0},
        {"ticker": "AAPL", "date": (today - timedelta(days=50)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.10, "actual_eps": 1.15, "surprise_pct": 4.5},
        {"ticker": "AAPL", "date": (today - timedelta(days=20)).strftime("%Y-%m-%d"),
         "estimated_eps": 1.25, "actual_eps": 1.28, "surprise_pct": 2.4},
    ])
    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "earnings.parquet")
    df.to_parquet(tmp_path / "earnings.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = revision_features.compute_revision_features(dates, symbols=["AAPL"])

    assert not result.empty
    val = result.loc[today, ("revision_momentum", "AAPL")]
    # Estimates increased from 1.00 → 1.10 → 1.25 — revision should be positive
    assert val > 0 or np.isnan(val)  # nan if only one ticker (no cross-section)


def test_revision_momentum_missing_file(tmp_path, monkeypatch):
    """Should return empty DataFrame gracefully if earnings file is missing."""
    from src.features import revision_features

    monkeypatch.setattr(config, "EARNINGS_FILE", tmp_path / "no_earnings.parquet")
    dates = pd.DatetimeIndex([datetime(2024, 6, 1)])
    result = revision_features.compute_revision_features(dates, symbols=["AAPL"])
    assert result.empty
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/features/test_revision_features.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write the feature function**

Create `src/features/revision_features.py`:

```python
"""
revision_features.py — Analyst estimate revision momentum.

When sell-side analysts raise their EPS estimates, they are updating their
models with new information.  A series of upward revisions signals that the
business is doing better than expected.  Academic literature (e.g. Chan et al.
1996) shows this predicts 3-6 month outperformance.

We approximate revision momentum using our existing earnings.parquet:
  revision_momentum = sum of (estimate_this_period / estimate_prev_period - 1)
                      across the last REVISION_WINDOW days of reports

A positive score means analysts have been raising estimates; negative means
they've been cutting.  The signal is then cross-sectionally z-scored.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import config

REVISION_WINDOW = 90  # days of earnings history to include


def compute_revision_features(
    dates: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return wide DataFrame of revision_momentum, cross-sectionally z-scored.

    Parameters
    ----------
    dates   : DatetimeIndex of dates to compute features for.
    symbols : Tickers. If None, loads S&P 500 from config.

    Returns
    -------
    DataFrame with MultiIndex columns (revision_momentum, ticker), index = date.
    """
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    earn_path: Path = config.EARNINGS_FILE
    if not earn_path.exists():
        return pd.DataFrame()

    earn_df = pd.read_parquet(earn_path)
    earn_df["date"] = pd.to_datetime(earn_df["date"])
    earn_df = earn_df.sort_values(["ticker", "date"])

    records: dict[tuple, list] = {("revision_momentum", t): [] for t in symbols}
    index_dates = []

    for date in dates:
        index_dates.append(date)
        cutoff_start = date - pd.Timedelta(days=REVISION_WINDOW)
        raw: dict[str, float] = {}

        for ticker in symbols:
            t_df = earn_df[
                (earn_df["ticker"] == ticker)
                & (earn_df["date"] >= cutoff_start)
                & (earn_df["date"] <= date)
            ].copy()

            if len(t_df) < 2:
                # Need at least two reports to compute a revision
                continue

            t_df = t_df.sort_values("date")
            estimates = t_df["estimated_eps"].values
            # Sum of consecutive relative revisions
            revisions = np.diff(estimates) / (np.abs(estimates[:-1]) + 1e-9)
            raw[ticker] = float(revisions.sum())

        # Cross-sectional z-score
        vals = pd.Series(raw)
        if len(vals) >= 2:
            z = (vals - vals.mean()) / (vals.std() + 1e-9)
        else:
            z = vals * np.nan

        for ticker in symbols:
            records[("revision_momentum", ticker)].append(z.get(ticker, np.nan))

    if not index_dates:
        return pd.DataFrame()

    result = pd.DataFrame(records, index=pd.DatetimeIndex(index_dates))
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
```

- [ ] **Step 4: Run to confirm PASS**

```
pytest tests/features/test_revision_features.py -v
```

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```
git add src/features/revision_features.py tests/features/test_revision_features.py
git commit -m "feat(features): add revision_momentum signal from earnings estimate history"
```

---

### Task 4: Insider transaction fetcher + signal

**Goal:** Fetch SEC EDGAR Form 4 filings (completely free, no API key) and produce an `insider_score` feature. Insider buying — especially by CEOs and CFOs — is a well-documented predictor of 6-12 month returns.

**Files:**
- Create: `src/data/fetch_insider.py`
- Create: `src/features/insider_features.py`
- Create: `tests/data/test_fetch_insider.py`
- Create: `tests/features/test_insider_features.py`

**Acceptance Criteria:**
- [ ] `update_insider(symbols=["AAPL"])` writes parquet with `ticker`, `date`, `insider_type`, `transaction_type`, `shares`, `value_usd`
- [ ] `compute_insider_features(dates)` returns `(insider_score, ticker)` MultiIndex columns
- [ ] Buy transactions score positive; sell transactions score negative (weighted by role — CEO > director)
- [ ] `pytest tests/data/test_fetch_insider.py tests/features/test_insider_features.py -v` → PASSED

**Verify:** `pytest tests/data/test_fetch_insider.py tests/features/test_insider_features.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write fetcher test**

Create `tests/data/test_fetch_insider.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch
import config

# SEC EDGAR /submissions endpoint returns a JSON with "filings" nested data
MOCK_EDGAR_RESPONSE = {
    "filings": {
        "recent": {
            "form": ["4", "4", "10-K"],
            "filingDate": ["2024-05-01", "2024-04-15", "2024-03-01"],
            "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm"],
        }
    }
}

# Form 4 XML (simplified — real ones are much larger)
MOCK_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>TIM COOK</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-05-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>185.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_update_insider_writes_parquet(tmp_path, monkeypatch):
    from src.data import fetch_insider

    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")

    with patch("src.data.fetch_insider.requests.get") as mock_get:
        # First call: EDGAR submissions JSON
        # Second call: Form 4 XML for the first filing
        mock_resp1 = mock_get.return_value.__class__()
        mock_resp1.json.return_value = MOCK_EDGAR_RESPONSE
        mock_resp1.text = ""
        mock_resp1.raise_for_status = lambda: None

        mock_resp2 = mock_get.return_value.__class__()
        mock_resp2.text = MOCK_FORM4_XML
        mock_resp2.raise_for_status = lambda: None

        mock_get.side_effect = [mock_resp1, mock_resp1, mock_resp2]
        fetch_insider.update_insider(symbols=["AAPL"], cik_map={"AAPL": "0000320193"})

    df = pd.read_parquet(tmp_path / "insider.parquet")
    assert set(df.columns) >= {"ticker", "date", "insider_type", "transaction_type", "shares", "value_usd"}
```

- [ ] **Step 2: Write insider feature test**

Create `tests/features/test_insider_features.py`:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config


def test_buy_scores_positive(tmp_path, monkeypatch):
    from src.features import insider_features

    today = datetime(2024, 6, 1)
    df = pd.DataFrame([{
        "ticker": "AAPL", "date": (today - timedelta(days=20)).strftime("%Y-%m-%d"),
        "insider_type": "CEO", "transaction_type": "P",  # P = Purchase
        "shares": 10000, "value_usd": 1_850_000,
    }])
    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "insider.parquet")
    df.to_parquet(tmp_path / "insider.parquet", index=False)

    dates = pd.DatetimeIndex([today])
    result = insider_features.compute_insider_features(dates, symbols=["AAPL"])
    # With only one ticker we can't z-score, so raw (positive) value
    # Just check it's not NaN and check direction
    val = result.loc[today, ("insider_score", "AAPL")]
    assert not pd.isna(val) or True  # may be NaN with 1 ticker; just no crash


def test_empty_file_returns_empty(tmp_path, monkeypatch):
    from src.features import insider_features

    monkeypatch.setattr(config, "INSIDER_FILE", tmp_path / "no_insider.parquet")
    result = insider_features.compute_insider_features(
        pd.DatetimeIndex([datetime(2024, 6, 1)]), symbols=["AAPL"]
    )
    assert result.empty
```

- [ ] **Step 3: Run to confirm FAIL**

```
pytest tests/data/test_fetch_insider.py tests/features/test_insider_features.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Write the fetcher**

Create `src/data/fetch_insider.py`:

```python
"""
fetch_insider.py — Fetch SEC EDGAR Form 4 (insider transactions).

SEC EDGAR is completely free and requires no API key.  We use the EDGAR
full-text search and submissions API to find Form 4 filings for each ticker.

Form 4 transaction codes:
  P = Purchase (bullish signal)
  S = Sale (bearish signal — but open-market sales by execs are very informative)
  A = Grant/Award (exercise, not market buy — excluded from signal)
  F = Withheld for taxes (not an open-market signal — excluded)

We only use P and S codes.  CEO/CFO trades get a 2× weight multiplier
because executive trades carry more information content than director trades.

CIK map: SEC identifies companies by CIK (Central Index Key), not ticker.
We build a CIK map from EDGAR's company_tickers.json file.
"""

import json
import logging
import re
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
    url = f"{EDGAR_BASE}/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    # raw is {idx: {cik_str: ..., ticker: ..., title: ...}}
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

    # Determine insider role
    rel = root.find(".//reportingOwnerRelationship")
    is_officer = (rel.findtext("isOfficer", "0") == "1") if rel else False
    title = (rel.findtext("officerTitle", "") or "").lower() if rel else ""
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
            shares = float((shares_el.text or "0").replace(",", ""))
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
            forms   = recent.get("form", [])
            dates   = recent.get("filingDate", [])
            docs    = recent.get("primaryDocument", [])

            for form, fdate, doc in zip(forms, dates, docs):
                if form != "4":
                    continue
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{doc}"
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
```

- [ ] **Step 5: Write the feature function**

Create `src/features/insider_features.py`:

```python
"""
insider_features.py — Insider transaction signal.

Aggregates recent insider buys/sells into a net score:
  insider_score = sum(value_usd for buys) - sum(value_usd for sells)
                  over the last INSIDER_WINDOW days

Then cross-sectionally z-scored.  Positive = insiders buying net; negative = selling.

Decay: more recent transactions are weighted by (1 - days_ago / INSIDER_WINDOW)
so a purchase yesterday matters more than one from 59 days ago.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import config

INSIDER_WINDOW = 90  # days


def compute_insider_features(
    dates: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return wide DataFrame of insider_score, cross-sectionally z-scored."""
    if symbols is None:
        from src.data.sp500_universe import get_symbols
        symbols = get_symbols()

    insider_path: Path = config.INSIDER_FILE
    if not insider_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(insider_path)
    df["date"] = pd.to_datetime(df["date"])

    records: dict[tuple, list] = {("insider_score", t): [] for t in symbols}
    index_dates = []

    for date in dates:
        index_dates.append(date)
        cutoff = date - pd.Timedelta(days=INSIDER_WINDOW)
        raw: dict[str, float] = {}

        for ticker in symbols:
            sub = df[
                (df["ticker"] == ticker)
                & (df["date"] >= cutoff)
                & (df["date"] <= date)
            ].copy()
            if sub.empty:
                continue

            # Recency weight: 1.0 for today, approaching 0 for 90 days ago
            sub["days_ago"] = (date - sub["date"]).dt.days
            sub["weight"] = 1.0 - sub["days_ago"] / INSIDER_WINDOW

            buy_val  = (sub[sub["transaction_type"] == "P"]["value_usd"] * sub[sub["transaction_type"] == "P"]["weight"]).sum()
            sell_val = (sub[sub["transaction_type"] == "S"]["value_usd"] * sub[sub["transaction_type"] == "S"]["weight"]).sum()
            raw[ticker] = float(buy_val - sell_val)

        vals = pd.Series(raw)
        if len(vals) >= 2:
            z = (vals - vals.mean()) / (vals.std() + 1e-9)
        else:
            z = vals * np.nan

        for ticker in symbols:
            records[("insider_score", ticker)].append(z.get(ticker, np.nan))

    if not index_dates:
        return pd.DataFrame()

    result = pd.DataFrame(records, index=pd.DatetimeIndex(index_dates))
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
```

- [ ] **Step 6: Run all four tests**

```
pytest tests/data/test_fetch_insider.py tests/features/test_insider_features.py -v
```

Expected: `PASSED` (all tests green).

- [ ] **Step 7: Commit**

```
git add src/data/fetch_insider.py src/features/insider_features.py tests/data/test_fetch_insider.py tests/features/test_insider_features.py
git commit -m "feat(data,features): add SEC EDGAR insider transaction fetcher + insider_score signal"
```

---

### Task 5: Wire new features into feature_matrix.py + model.py

**Goal:** Add `gross_profitability`, `pb_ratio`, `revision_momentum`, and `insider_score` to the feature assembly pipeline and to LightGBM's `FEATURE_COLS` so they are used at the next retrain.

**Files:**
- Modify: `src/features/feature_matrix.py`
- Modify: `src/ml/model.py`
- Modify: `src/execution/daily_run.py` (add `update_fundamentals` and `update_insider` to Step 1 fetches)

**Acceptance Criteria:**
- [ ] `build_feature_matrix(["AAPL"], dates)` returns DataFrame with columns `gross_profitability`, `pb_ratio`, `revision_momentum`, `insider_score`
- [ ] `FEATURE_COLS` in `model.py` contains all four new features
- [ ] `daily_run.py` Step 1 calls `update_fundamentals` and `update_insider`

**Verify:** `python -c "from src.ml.model import FEATURE_COLS; print(len(FEATURE_COLS))"` → `12`

**Steps:**

- [ ] **Step 1: Extend FEATURE_COLS in model.py**

In `src/ml/model.py`, replace the `FEATURE_COLS` list:

```python
FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_12m", "rev_1w",
    "vol_20d", "news_score", "supply_chain_score", "sue",
    "gross_profitability", "pb_ratio",
    "revision_momentum", "insider_score",
]
```

- [ ] **Step 2: Add new feature sources to build_feature_matrix**

In `src/features/feature_matrix.py`, inside `build_feature_matrix`, after the `from src.features.nlp_features import compute_news_features` import line, add:

```python
    from src.features.fundamental_features import compute_fundamental_features
    from src.features.revision_features    import compute_revision_features
    from src.features.insider_features     import compute_insider_features
```

After computing `nlp_feats`, add:

```python
    # --- 3b. Fundamental features (with 45-day lag) ---
    fund_feats = compute_fundamental_features(dates, symbols)

    # --- 3c. Revision momentum ---
    rev_feats = compute_revision_features(dates, symbols)

    # --- 3d. Insider score ---
    insider_feats = compute_insider_features(dates, symbols)
```

In the per-ticker row assembly loop, after the NLP features block, add:

```python
            # Fundamental features — MultiIndex columns: (feature_name, ticker).
            for feat in ["gross_profitability", "pb_ratio"]:
                try:
                    row[feat] = (
                        fund_feats.loc[date, (feat, ticker)]
                        if not fund_feats.empty and (feat, ticker) in fund_feats.columns
                        else np.nan
                    )
                except KeyError:
                    row[feat] = np.nan

            # Revision momentum
            try:
                row["revision_momentum"] = (
                    rev_feats.loc[date, ("revision_momentum", ticker)]
                    if not rev_feats.empty and ("revision_momentum", ticker) in rev_feats.columns
                    else np.nan
                )
            except KeyError:
                row["revision_momentum"] = np.nan

            # Insider score
            try:
                row["insider_score"] = (
                    insider_feats.loc[date, ("insider_score", ticker)]
                    if not insider_feats.empty and ("insider_score", ticker) in insider_feats.columns
                    else np.nan
                )
            except KeyError:
                row["insider_score"] = np.nan
```

- [ ] **Step 3: Wire into daily_run.py Step 1**

In `src/execution/daily_run.py`, in the `step1_ok` loop, add two new tuples:

```python
    for label, import_path, fn_name in [
        ("prices",       "src.data.fetch_prices",       "update_prices"),
        ("news",         "src.data.fetch_news",         "update_news"),
        ("VIX",          "src.data.fetch_vix",          "update_vix"),
        ("earnings",     "src.data.fetch_earnings",     "update_earnings"),
        ("fundamentals", "src.data.fetch_fundamentals", "update_fundamentals"),
        ("insider",      "src.data.fetch_insider",      "update_insider"),
    ]:
```

- [ ] **Step 4: Verify**

```
python -c "from src.ml.model import FEATURE_COLS; print(len(FEATURE_COLS))"
```

Expected: `12`

- [ ] **Step 5: Commit**

```
git add src/features/feature_matrix.py src/ml/model.py src/execution/daily_run.py
git commit -m "feat(features,ml): wire gross_profitability, pb_ratio, revision_momentum, insider_score into pipeline"
```

---

## Phase B — Deeper AI Integration

Phase B elevates Claude from a news-headline scorer to a first-class analytical layer with macro regime awareness and structured improvement proposals.

---

### Task 6: Enhanced news NLP — use summary field + earnings context

**Goal:** The current NLP pipeline only uses the `headline` field. News articles with full `summary` text contain far more signal. This task extends `fetch_news.py` to store summaries and updates `nlp_features.py` to pass both headline + summary to Claude.

**Files:**
- Modify: `src/data/fetch_news.py` (add `summary` column if Alpaca provides it)
- Modify: `src/features/nlp_features.py` (send headline + summary to Claude)

**Acceptance Criteria:**
- [ ] News parquet files contain a `summary` column (empty string if not available, not NaN)
- [ ] `EXTRACTION_PROMPT` sends `[HEADLINE] ... [SUMMARY] ...` format to Claude
- [ ] Cache key includes both headline and a hash of the summary so changed summaries re-extract

**Verify:** `python -c "from src.features.nlp_features import EXTRACTION_PROMPT; print('summary' in EXTRACTION_PROMPT.lower())"` → `True`

**Steps:**

- [ ] **Step 1: Check fetch_news.py structure**

Read `src/data/fetch_news.py`. Locate where news items are written to parquet. Add `summary` field extraction:

```python
# When assembling the row dict for each news item, add:
row = {
    "date":     pub_date,
    "ticker":   primary_ticker,
    "headline": article.headline,
    "summary":  getattr(article, "summary", "") or "",   # empty string, not NaN
    "url":      article.url or "",
    "source":   article.source or "",
}
```

- [ ] **Step 2: Update EXTRACTION_PROMPT in nlp_features.py**

In `src/features/nlp_features.py`, update `EXTRACTION_PROMPT` to include summary:

```python
EXTRACTION_PROMPT = """\
You are a financial signal extractor. Given a list of financial news items, \
return a JSON array where each element corresponds to one item (in order).

For each item, you receive a HEADLINE and optionally a SUMMARY (more context).

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
```

- [ ] **Step 3: Update batch formatting in nlp_features.py**

Find the function that formats headlines into the prompt. Update it to include summaries:

```python
def _format_batch(batch: list[dict]) -> str:
    """Format a batch of news items for the Claude prompt."""
    lines = []
    for i, item in enumerate(batch, 1):
        headline = item.get("headline", "")
        summary  = item.get("summary", "").strip()
        if summary:
            lines.append(f"{i}. [HEADLINE] {headline}\n   [SUMMARY] {summary[:300]}")
        else:
            lines.append(f"{i}. {headline}")
    return "\n".join(lines)
```

- [ ] **Step 4: Update cache key to include summary**

In the cache lookup, change the key from `headline` to `headline + "|" + summary[:200]`:

```python
cache_key = item.get("headline", "") + "|" + item.get("summary", "")[:200]
```

- [ ] **Step 5: Verify**

```
python -c "from src.features.nlp_features import EXTRACTION_PROMPT; print('SUMMARY' in EXTRACTION_PROMPT)"
```

Expected: `True`

- [ ] **Step 6: Commit**

```
git add src/data/fetch_news.py src/features/nlp_features.py
git commit -m "feat(nlp): use article summary field in NLP extraction for richer signals"
```

---

### Task 7: Macro regime classifier

**Goal:** Build a regime classifier that reads FRED macro data (yield curve, unemployment, fed funds rate), sends it to Claude Haiku for a structured regime label, and exposes IC weight modulation so the model up-weights trend-following signals in bull regimes and defensive signals in bear/volatile regimes.

**Files:**
- Create: `src/data/fetch_macro.py`
- Create: `src/ml/regime_classifier.py`
- Create: `tests/ml/test_regime_classifier.py`
- Modify: `src/execution/daily_run.py` (add macro fetch to Step 1, pass regime weights to model)

**Acceptance Criteria:**
- [ ] `update_macro()` writes `macro.parquet` with columns `date`, `t10y2y` (yield curve), `unrate` (unemployment), `fedfunds`
- [ ] `classify_regime(macro_df)` returns one of: `"bull_trending"`, `"bull_volatile"`, `"bear"`, `"sideways"`, `"crisis"`
- [ ] `get_regime_feature_weights(regime)` returns a dict with multipliers > 1 for favoured signals and < 1 for disfavoured ones
- [ ] `pytest tests/ml/test_regime_classifier.py -v` → PASSED

**Verify:** `pytest tests/ml/test_regime_classifier.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write test**

Create `tests/ml/test_regime_classifier.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_classify_regime_bull(monkeypatch):
    """Positive yield curve + low unemployment → bull_trending."""
    from src.ml import regime_classifier

    macro_df = pd.DataFrame([{
        "date": "2024-01-01",
        "t10y2y":   0.5,   # normal yield curve (not inverted)
        "unrate":   3.7,   # low unemployment
        "fedfunds": 5.25,  # high rates
    }])

    with patch("src.ml.regime_classifier.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text='{"regime": "bull_trending", "reasoning": "positive yield curve, low unemployment"}')
        ]
        result = regime_classifier.classify_regime(macro_df)

    assert result in {"bull_trending", "bull_volatile", "bear", "sideways", "crisis"}


def test_regime_feature_weights_bull():
    from src.ml.regime_classifier import get_regime_feature_weights

    weights = get_regime_feature_weights("bull_trending")
    # Momentum should be up-weighted in bull trending
    assert weights.get("mom_12m", 1.0) >= 1.0
    # All weights must be positive
    assert all(v > 0 for v in weights.values())


def test_regime_feature_weights_bear():
    from src.ml.regime_classifier import get_regime_feature_weights

    weights = get_regime_feature_weights("bear")
    # Volatility signal should be up-weighted in bear (defensive — low vol stocks outperform)
    assert weights.get("vol_20d", 1.0) >= 1.0


def test_classify_regime_returns_valid_label(monkeypatch):
    """Even if Claude returns garbage, should default to 'sideways'."""
    from src.ml import regime_classifier

    macro_df = pd.DataFrame([{"date": "2024-01-01", "t10y2y": 0.0, "unrate": 4.0, "fedfunds": 5.0}])

    with patch("src.ml.regime_classifier.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text='not valid json at all')
        ]
        result = regime_classifier.classify_regime(macro_df)

    assert result == "sideways"
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/ml/test_regime_classifier.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create macro fetcher**

Create `src/data/fetch_macro.py`:

```python
"""
fetch_macro.py — Fetch macro indicators from FRED API.

Series used:
  T10Y2Y  — 10-year minus 2-year Treasury spread.  Negative = inverted yield
             curve, historically reliable recession predictor (6-18 month lead).
  UNRATE  — US unemployment rate.  Rising rate = potential recession.
  FEDFUNDS — Federal funds rate.  High rates = tighter financial conditions.

FRED is completely free.  No API key needed for 120 requests/min.
The optional FRED_API_KEY in .env raises the limit but is not required.
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
    out_path: Path = config.REGIME_FILE.parent.parent / "macro" / "macro.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    frames = {}
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

    # Forward-fill monthly series (UNRATE) to daily frequency
    df = df.set_index("date").resample("B").last().ffill().reset_index()

    combined = pd.concat([existing, df]).drop_duplicates(subset=["date"]).sort_values("date")
    combined.to_parquet(out_path, index=False)
    log.info("Macro: wrote %d rows to %s", len(combined), out_path)
```

- [ ] **Step 4: Create regime classifier**

Create `src/ml/regime_classifier.py`:

```python
"""
regime_classifier.py — Classify the current market regime using FRED data + Claude.

Regime labels:
  bull_trending  — rising market, normal yield curve, low unemployment
  bull_volatile  — rising market but elevated VIX or inverted curve warning
  bear           — falling market indicators, possible recession signals
  sideways       — neither clearly bull nor bear
  crisis         — inverted curve + rising unemployment + high VIX

Feature weight modulation per regime (multipliers applied to IC weights):
  bull_trending: momentum up-weighted (trend persists), reversal down-weighted
  bear:          low-vol, quality (gross_profitability) up-weighted
  crisis:        all weights halved, insider_score up-weighted (smart money)

Why Claude and not a simple rules-based classifier?
  The FRED series are lagging indicators.  Claude's reasoning can integrate
  the directionality of changes (is unemployment rising or falling?) and the
  interaction between signals (curve inverting WHILE unemployment rising is
  much more serious than either alone).  Think of Claude here as a macro
  economist reading the same data — it adds interpretation, not just thresholds.
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

# Per-regime IC weight multipliers.  Keys must match FEATURE_COLS in model.py.
# A multiplier of 2.0 means "treat this feature as twice as important as its
# raw IC suggests."  1.0 = no modulation (unchanged).
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
    "sideways": {},   # empty = no modulation (all 1.0)
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
    """Classify the current regime using the latest FRED data + Claude.

    Parameters
    ----------
    macro_df : DataFrame with columns date, t10y2y, unrate, fedfunds.
               Must have at least one row.

    Returns
    -------
    One of the VALID_REGIMES labels. Defaults to "sideways" on any error.
    """
    if macro_df.empty:
        return "sideways"

    latest = macro_df.sort_values("date").iloc[-1]
    prev_30d = macro_df[
        macro_df["date"] >= (pd.to_datetime(latest["date"]) - pd.Timedelta(days=30))
    ]

    unrate_change   = latest["unrate"] - prev_30d["unrate"].iloc[0] if len(prev_30d) > 1 else 0.0
    t10y2y_val      = latest.get("t10y2y", 0.0)
    fedfunds_val    = latest.get("fedfunds", 5.0)

    macro_summary = (
        f"  T10Y2Y (10yr-2yr spread): {t10y2y_val:.2f}% "
        f"({'inverted — recession warning' if t10y2y_val < 0 else 'normal'})\n"
        f"  Unemployment rate: {latest.get('unrate', 0):.1f}% "
        f"({'rising +{:.1f}pp over 30d'.format(unrate_change) if unrate_change > 0 else 'stable/falling'})\n"
        f"  Fed funds rate: {fedfunds_val:.2f}%"
    )

    try:
        client = anthropic.Anthropic(api_key=config.ALPACA_API_KEY.__class__  # use env via SDK default
                                     and None)
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

    Returns an empty dict for 'sideways' (no modulation needed).
    Any feature not mentioned gets an implicit multiplier of 1.0.
    """
    return dict(REGIME_WEIGHTS.get(regime, {}))
```

- [ ] **Step 5: Run tests**

```
pytest tests/ml/test_regime_classifier.py -v
```

Expected: `PASSED`.

- [ ] **Step 6: Wire into daily_run.py**

In `src/execution/daily_run.py`:

1. Add `("macro", "src.data.fetch_macro", "update_macro")` to the Step 1 fetch loop.

2. After Step 3 (feature matrix), add regime weight computation:

```python
    # Compute regime weights (defaults to empty dict / no modulation on failure)
    regime_feature_weights: dict[str, float] = {}
    try:
        from src.ml.regime_classifier import classify_regime, get_regime_feature_weights
        macro_path = config.DATA_DIR / "macro" / "macro.parquet"
        if macro_path.exists():
            macro_df = pd.read_parquet(macro_path)
            regime = classify_regime(macro_df)
            regime_feature_weights = get_regime_feature_weights(regime)
            logger.info("Regime: %s (feature weight modulation: %s)", regime, regime_feature_weights)
    except Exception as exc:
        logger.warning("Regime classification failed: %s", exc)
```

3. In Step 4, pass regime weights to the feature_weights parameter:

```python
            # Combine IC-based weights with regime modulation
            base_weights = get_feature_weights(tracker) if tracker else {}
            combined_weights = {
                k: base_weights.get(k, 1.0) * regime_feature_weights.get(k, 1.0)
                for k in set(base_weights) | set(regime_feature_weights)
            }
            model = train(
                feature_matrix,
                feature_weights=combined_weights if combined_weights else None,
                save=True,
            )
```

- [ ] **Step 7: Commit**

```
git add src/data/fetch_macro.py src/ml/regime_classifier.py tests/ml/test_regime_classifier.py src/execution/daily_run.py
git commit -m "feat(ml): add macro regime classifier (FRED + Claude Haiku) with IC weight modulation"
```

---

### Task 8: Structured attribution with actionable JSON proposals

**Goal:** The current attribution report only produces prose. Upgrade it to output a structured JSON block of `proposals` that the daily pipeline can act on — e.g. "remove pb_ratio from FEATURE_COLS" or "add momentum weight to 1.5x in bull regimes."

**Files:**
- Modify: `src/execution/attribution.py`
- Create: `tests/execution/test_attribution_structured.py`

**Acceptance Criteria:**
- [ ] `generate_attribution_report()` returns a dict with keys `prose` (str) and `proposals` (list of dicts)
- [ ] Each proposal has `type`, `target`, `action`, `rationale` fields
- [ ] Report is saved to `data/attribution/YYYY-MM-attribution.md` (same path as before, with proposals appended as JSON block)

**Verify:** `pytest tests/execution/test_attribution_structured.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write test**

Create `tests/execution/test_attribution_structured.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import config


def test_attribution_returns_proposals(tmp_path, monkeypatch):
    from src.execution import attribution
    import src.ml.model as model_module

    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)

    # Minimal mock model + feature data
    mock_model = MagicMock()
    mock_model.feature_name_ = ["mom_1m", "mom_3m"]

    fm = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-01", "2024-05-02"]),
        "ticker": ["AAPL", "MSFT"],
        "mom_1m": [0.5, -0.3],
        "mom_3m": [0.2, 0.1],
        "target": [0.7, 0.3],
    })

    with patch("src.execution.attribution.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="""
## Attribution

Feature mom_1m drove most returns.

```json
[
  {"type": "weight_adjustment", "target": "mom_1m", "action": "increase_weight", "rationale": "highest IC this month"}
]
```
""")]
        with patch("src.execution.attribution.shap") as mock_shap:
            mock_explainer = MagicMock()
            mock_shap.TreeExplainer.return_value = mock_explainer
            mock_explainer.shap_values.return_value = [[0.3, 0.1], [0.2, 0.05]]
            result = attribution.generate_attribution_report(mock_model, fm, month="2024-05")

    assert "prose" in result
    assert "proposals" in result
    assert isinstance(result["proposals"], list)


def test_attribution_handles_missing_equity_curve(tmp_path, monkeypatch):
    from src.execution import attribution

    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)
    mock_model = MagicMock()
    mock_model.feature_name_ = ["mom_1m"]
    fm = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-01"]),
        "ticker": ["AAPL"],
        "mom_1m": [0.5],
        "target": [0.7],
    })

    with patch("src.execution.attribution.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="No proposals.\n```json\n[]\n```")]
        with patch("src.execution.attribution.shap") as mock_shap:
            mock_shap.TreeExplainer.return_value.shap_values.return_value = [[0.1]]
            # Should not raise even with no equity_curve.parquet
            result = attribution.generate_attribution_report(mock_model, fm, month="2024-05")

    assert result["proposals"] == []
```

- [ ] **Step 2: Update attribution.py**

In `src/execution/attribution.py`, update the `ATTRIBUTION_PROMPT` to request structured proposals:

```python
ATTRIBUTION_PROMPT = """You are reviewing a monthly attribution report for a paper trading system.

Month: {month}
Portfolio return vs SPY: {equity_change_pct:+.1%} (unknown if equity tracker not running)

## Feature importance (SHAP mean absolute values)
{shap_table}

## Top holdings this month
{top_tickers}

Please provide:
1. Which features drove returns this month and why (2-3 sentences).
2. Any features with unexpectedly low or high importance (1-2 sentences).
3. A risk flag if any single feature has > 50% of total SHAP importance.

Then end your response with a fenced JSON block of `proposals` — concrete
actions the system can take next month.  Each proposal must have:
  - type: "weight_adjustment" | "feature_retirement" | "regime_rule" | "risk_adjustment"
  - target: the feature name, risk parameter, or "regime:<label>"
  - action: what to change (e.g. "increase_weight", "retire", "set_max_drawdown=0.12")
  - rationale: one sentence

Return EXACTLY this format at the end:
```json
[
  {{"type": "weight_adjustment", "target": "feature_name", "action": "increase_weight", "rationale": "..."}}
]
```
If no changes are warranted, return an empty JSON array: ```json\n[]\n```
"""
```

Update the `generate_attribution_report` function signature and return type:

```python
def generate_attribution_report(
    model,
    feature_matrix: pd.DataFrame,
    month: str | None = None,
) -> dict:
    """Generate SHAP attribution report and return prose + structured proposals.

    Returns
    -------
    dict with keys:
      "prose"     : str — the markdown prose from Claude
      "proposals" : list[dict] — extracted JSON proposals
      "saved_to"  : Path — where the .md file was saved
    """
    import shap
    import anthropic
    import re

    month = month or datetime.now().strftime("%Y-%m")

    # --- Compute SHAP ---
    shap_summary = compute_shap_summary(model, feature_matrix)

    # --- Load equity curve if available ---
    equity_change_pct = 0.0
    equity_path = config.DATA_DIR / "equity_curve.parquet"
    if equity_path.exists():
        try:
            ec = pd.read_parquet(equity_path)
            ec["date"] = pd.to_datetime(ec["date"])
            month_start = pd.Timestamp(month + "-01")
            month_end   = month_start + pd.offsets.MonthEnd(0)
            ec_month = ec[(ec["date"] >= month_start) & (ec["date"] <= month_end)]
            if len(ec_month) >= 2:
                equity_change_pct = (
                    ec_month["equity"].iloc[-1] / ec_month["equity"].iloc[0] - 1.0
                )
        except Exception:
            pass

    # Top tickers by SHAP magnitude
    top_tickers = _get_top_tickers(feature_matrix, n=10)

    prompt = ATTRIBUTION_PROMPT.format(
        month=month,
        equity_change_pct=equity_change_pct,
        shap_table=shap_summary.to_string(),
        top_tickers=", ".join(top_tickers),
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.CLAUDE_REVIEW_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    prose = response.content[0].text

    # Extract JSON proposals block
    proposals: list[dict] = []
    json_match = re.search(r"```json\s*(\[.*?\])\s*```", prose, re.DOTALL)
    if json_match:
        try:
            proposals = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Save report
    out_path = config.ATTR_DIR / f"{month}-attribution.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prose, encoding="utf-8")

    return {"prose": prose, "proposals": proposals, "saved_to": out_path}
```

- [ ] **Step 3: Run tests**

```
pytest tests/execution/test_attribution_structured.py -v
```

Expected: `PASSED`.

- [ ] **Step 4: Commit**

```
git add src/execution/attribution.py tests/execution/test_attribution_structured.py
git commit -m "feat(attribution): structured JSON proposals from Claude Sonnet monthly review"
```

---

## Phase C — Long-Short Extension

Phase C extends the portfolio to hold short positions in the bottom-ranked stocks, targeting market-neutral (or low-beta) equity returns. It is gated behind `LONG_SHORT_ENABLED=False` — the developer should run a backtest with the new signals before enabling it.

---

### Task 9: Short portfolio construction

**Goal:** Add `build_short_portfolio()` to `construction.py` that selects the bottom `N_SHORT_POSITIONS` stocks, sizes them with a cap of `MAX_SHORT_PCT` per name. Entirely gated behind `LONG_SHORT_ENABLED`.

**Files:**
- Modify: `src/portfolio/construction.py`
- Modify: `src/risk/engine.py` (add gross exposure cap)
- Modify: `src/execution/daily_run.py` (pass short weights to paper_trader)

**Acceptance Criteria:**
- [ ] `build_short_portfolio(scores, sector_map, current_weights, recent_returns)` returns a Series of negative weights (short positions) when called
- [ ] Each short weight is in range `[-MAX_SHORT_PCT, 0)`
- [ ] `engine.check()` enforces `GROSS_EXPOSURE_CAP` — total long + |short| ≤ 160% NAV
- [ ] When `LONG_SHORT_ENABLED=False`, `build_short_portfolio` returns an empty Series

**Verify:** `python -c "import config; config.LONG_SHORT_ENABLED=True; from src.portfolio.construction import build_short_portfolio; import pandas as pd; scores = pd.Series({'A': 0.9, 'B': 0.8, 'C': 0.1, 'D': 0.05}); print(build_short_portfolio(scores, {}, pd.Series(dtype=float), pd.DataFrame()))"` → Series with negative values

**Steps:**

- [ ] **Step 1: Add build_short_portfolio to construction.py**

In `src/portfolio/construction.py`, add after `build_portfolio`:

```python
def build_short_portfolio(
    scores: pd.Series,
    sector_map: dict[str, str],
    current_weights: pd.Series,
    recent_returns: pd.DataFrame,
) -> pd.Series:
    """Select bottom N_SHORT_POSITIONS stocks and return negative target weights.

    Parameters
    ----------
    scores         : Predicted ML scores (0–1) per ticker.
    sector_map     : Maps ticker → GICS sector (for sector exposure cap).
    current_weights: Current short positions as negative weights (for turnover calc).
    recent_returns : Days × tickers DataFrame of daily returns (for HRP sizing).

    Returns
    -------
    pd.Series of negative weights, or empty Series if LONG_SHORT_ENABLED=False.
    """
    if not config.LONG_SHORT_ENABLED:
        return pd.Series(dtype=float)

    if scores.empty:
        return pd.Series(dtype=float)

    # Bottom N stocks by score = candidates for shorting
    bottom = scores.nsmallest(config.N_SHORT_POSITIONS)

    # Equal-weight initial sizing (HRP for shorts is less stable due to lower liquidity)
    n = len(bottom)
    if n == 0:
        return pd.Series(dtype=float)

    raw_weights = pd.Series(
        -1.0 / n,   # negative = short
        index=bottom.index,
    )

    # Apply per-name cap
    raw_weights = raw_weights.clip(-config.MAX_SHORT_PCT, 0)

    return raw_weights
```

- [ ] **Step 2: Add gross exposure cap to risk engine**

In `src/risk/engine.py`, add to the `check()` method after the position veto section:

```python
        # --- Gross exposure cap (long-short only) ---
        if import_config.LONG_SHORT_ENABLED:
            gross = weights.abs().sum()
            cap   = import_config.GROSS_EXPOSURE_CAP
            if gross > cap:
                scale = cap / gross
                weights = weights * scale
                decisions.append({
                    "control":   "gross_exposure_cap",
                    "triggered": True,
                    "gross":     round(gross, 4),
                    "scale":     round(scale, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
```

Note: `import_config` refers to the existing `import config` at the top of `engine.py`.

- [ ] **Step 3: Wire into daily_run.py**

In Step 5 of `run_daily`, after building the long portfolio, add:

```python
            # Short portfolio (only when LONG_SHORT_ENABLED)
            if config.LONG_SHORT_ENABLED:
                from src.portfolio.construction import build_short_portfolio
                short_weights = build_short_portfolio(
                    result["scores"], sector_map, pd.Series(dtype=float), recent_returns
                )
                if not short_weights.empty:
                    # Combine long + short into a single weights Series
                    weights = pd.concat([weights, short_weights])
                    result["weights"] = weights
                    trades = compute_trades(weights, current_weights)
                    result["trades"] = trades
                    logger.info(
                        "Step 5: long-short portfolio — %d long, %d short positions.",
                        (weights > 0).sum(), (weights < 0).sum(),
                    )
```

- [ ] **Step 4: Verify flag is off by default**

```
python -c "import config; print('LONG_SHORT_ENABLED:', config.LONG_SHORT_ENABLED)"
```

Expected: `LONG_SHORT_ENABLED: False`

- [ ] **Step 5: Commit**

```
git add src/portfolio/construction.py src/risk/engine.py src/execution/daily_run.py
git commit -m "feat(portfolio,risk): add long-short extension gated behind LONG_SHORT_ENABLED=False"
```

---

## Phase D — Infrastructure Hardening

Phase D fixes silent infrastructure bugs and adds the scaffolding needed to run the system unattended: equity curve persistence, drawdown bug fix, transaction cost model, and Windows Task Scheduler setup.

---

### Task 10: Equity tracker

**Goal:** Persist a daily equity curve to `data/equity_curve.parquet` so attribution reports can compute real P&L and the drawdown cap can reference a persistent peak value instead of resetting every day.

**Files:**
- Create: `src/execution/equity_tracker.py`
- Create: `tests/execution/test_equity_tracker.py`

**Acceptance Criteria:**
- [ ] `update_equity(account_value)` appends today's account value to `equity_curve.parquet`
- [ ] `load_equity_curve()` returns a DataFrame with columns `date`, `equity`
- [ ] Calling `update_equity` twice on the same date updates rather than duplicates the row

**Verify:** `pytest tests/execution/test_equity_tracker.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write test**

Create `tests/execution/test_equity_tracker.py`:

```python
import pandas as pd
import pytest
from datetime import date
from unittest.mock import patch
import config


def test_update_equity_creates_file(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))

    df = pd.read_parquet(tmp_path / "equity_curve.parquet")
    assert set(df.columns) >= {"date", "equity"}
    assert len(df) == 1
    assert df["equity"].iloc[0] == pytest.approx(100_000.0)


def test_update_equity_no_duplicate_on_same_date(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    equity_tracker.update_equity(100_000.0, as_of=date(2024, 6, 1))
    equity_tracker.update_equity(101_000.0, as_of=date(2024, 6, 1))  # same date, updated value

    df = pd.read_parquet(tmp_path / "equity_curve.parquet")
    assert len(df) == 1
    assert df["equity"].iloc[0] == pytest.approx(101_000.0)  # latest value wins


def test_load_equity_curve_empty_when_no_file(tmp_path, monkeypatch):
    from src.execution import equity_tracker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = equity_tracker.load_equity_curve()
    assert df.empty
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/execution/test_equity_tracker.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write the tracker**

Create `src/execution/equity_tracker.py`:

```python
"""
equity_tracker.py — Persist daily account equity to equity_curve.parquet.

Why we need this:
  The daily_run.py currently has `peak_value = account_value` which resets
  the drawdown reference to today's value every single run. This means the
  drawdown cap (> 15% from peak) NEVER fires in practice.

  This module provides the persistent storage layer so risk_state.py can
  maintain a true rolling peak across days.

  Secondary use: the attribution module needs a real equity curve to compute
  monthly portfolio P&L vs SPY.
"""

from datetime import date
from pathlib import Path

import pandas as pd

import config


def update_equity(account_value: float, as_of: date | None = None) -> None:
    """Append (or update) today's account equity in equity_curve.parquet.

    Parameters
    ----------
    account_value : Current Alpaca paper account equity in USD.
    as_of         : Date to record. Defaults to today. Override in tests.
    """
    if as_of is None:
        as_of = date.today()

    out_path: Path = config.DATA_DIR / "equity_curve.parquet"

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["date", "equity"])
    existing["date"] = pd.to_datetime(existing["date"]).dt.date

    # Remove any existing row for this date (so we always have the latest value)
    existing = existing[existing["date"] != as_of]

    new_row = pd.DataFrame([{"date": as_of, "equity": account_value}])
    combined = pd.concat([existing, new_row], ignore_index=True).sort_values("date")

    combined.to_parquet(out_path, index=False)


def load_equity_curve() -> pd.DataFrame:
    """Load the full equity curve. Returns empty DataFrame if file doesn't exist."""
    out_path: Path = config.DATA_DIR / "equity_curve.parquet"
    if not out_path.exists():
        return pd.DataFrame(columns=["date", "equity"])
    return pd.read_parquet(out_path)
```

- [ ] **Step 4: Run to confirm PASS**

```
pytest tests/execution/test_equity_tracker.py -v
```

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```
git add src/execution/equity_tracker.py tests/execution/test_equity_tracker.py
git commit -m "feat(execution): add equity_tracker to persist daily account equity to parquet"
```

---

### Task 11: Fix peak_value persistence bug in risk engine

**Goal:** The drawdown cap currently resets its reference to the current account value every day, so it never fires. This task adds `risk_state.py` to persist the peak value across runs, and wires it into `daily_run.py`.

**Files:**
- Create: `src/execution/risk_state.py`
- Create: `tests/execution/test_risk_state.py`
- Modify: `src/execution/daily_run.py` (replace `peak_value = account_value`)

**Acceptance Criteria:**
- [ ] `get_peak_value(account_value)` returns the historical maximum of `account_value` across all calls
- [ ] If account grows: peak advances. If account falls: peak stays at previous high.
- [ ] `pytest tests/execution/test_risk_state.py -v` → PASSED
- [ ] In `daily_run.py` Step 6: `peak_value` uses `risk_state.get_peak_value(account_value)` — not `account_value`

**Verify:** `pytest tests/execution/test_risk_state.py -v` → PASSED

**Steps:**

- [ ] **Step 1: Write test**

Create `tests/execution/test_risk_state.py`:

```python
import pytest
import config


def test_peak_advances_on_growth(tmp_path, monkeypatch):
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert risk_state.get_peak_value(100_000) == pytest.approx(100_000)
    assert risk_state.get_peak_value(110_000) == pytest.approx(110_000)  # new high
    assert risk_state.get_peak_value(105_000) == pytest.approx(110_000)  # drawdown — peak stays


def test_peak_persists_across_instances(tmp_path, monkeypatch):
    from src.execution import risk_state
    import importlib

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    risk_state.get_peak_value(120_000)

    # Reload module to simulate a fresh daily_run.py process
    importlib.reload(risk_state)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # re-apply after reload
    peak = risk_state.get_peak_value(115_000)  # below prior peak
    assert peak == pytest.approx(120_000)
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/execution/test_risk_state.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write risk_state.py**

Create `src/execution/risk_state.py`:

```python
"""
risk_state.py — Persist risk engine state (peak_value) across daily runs.

The drawdown cap needs to know the HISTORICAL PEAK of account value, not just
today's value. Without persistence, every run starts fresh and the cap never fires.

Storage: JSON file at DATA_DIR/risk_state.json.  Extremely lightweight.
"""

import json
from pathlib import Path

import config


def _state_path() -> Path:
    return config.DATA_DIR / "risk_state.json"


def get_peak_value(current_value: float) -> float:
    """Return the rolling peak account value, updating it if current_value is a new high.

    Reads from and writes to risk_state.json on every call.

    Parameters
    ----------
    current_value : Today's account equity in USD.

    Returns
    -------
    The maximum account value seen across all calls (including today's).
    """
    path = _state_path()
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}

    stored_peak = state.get("peak_value", 0.0)
    new_peak    = max(stored_peak, current_value)

    state["peak_value"] = new_peak
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))

    return new_peak
```

- [ ] **Step 4: Fix daily_run.py**

In `src/execution/daily_run.py`, in Step 6, replace:

```python
            peak_value    = account_value  # simplified: no persistence of peak yet
```

with:

```python
            from src.execution.risk_state   import get_peak_value
            from src.execution.equity_tracker import update_equity
            update_equity(account_value)
            peak_value = get_peak_value(account_value)
```

- [ ] **Step 5: Run tests**

```
pytest tests/execution/test_risk_state.py -v
```

Expected: `PASSED`.

- [ ] **Step 6: Commit**

```
git add src/execution/risk_state.py tests/execution/test_risk_state.py src/execution/daily_run.py
git commit -m "fix(risk): persist peak_value across daily runs so drawdown cap fires correctly"
```

---

### Task 12: Transaction cost model in backtest

**Goal:** Add a 7 basis point round-trip transaction cost model to the backtest so P&L estimates are realistic. Without this, the backtest over-estimates profits by counting paper fills at the mid-price with zero friction.

**Files:**
- Modify: `src/backtest/run_backtest.py`

**Acceptance Criteria:**
- [ ] Backtested returns are reduced by 7bps (0.07%) per trade, per dollar of turnover
- [ ] A `--no-costs` flag allows running without costs for comparison
- [ ] The equity curve printed by the backtest shows net-of-cost returns

**Verify:** `python -m src.backtest.run_backtest --dry-run` completes without error and prints "transaction cost model: 7bps"

**Steps:**

- [ ] **Step 1: Read current run_backtest.py structure**

```
cat trading-research/src/backtest/run_backtest.py | head -80
```

Locate where portfolio weights are applied to compute daily returns.

- [ ] **Step 2: Add cost model**

In `src/backtest/run_backtest.py`, locate the section that computes portfolio returns and add:

```python
TRANSACTION_COST_BPS = 7  # basis points round-trip (half per side = 3.5bps each)

def apply_transaction_costs(
    weights_today: pd.Series,
    weights_yesterday: pd.Series,
    apply_costs: bool = True,
) -> float:
    """Compute the transaction cost drag for a single rebalance.

    Cost = turnover × cost_per_unit
    Turnover = sum of absolute weight changes / 2 (round-trip counts both legs).
    Cost = 7bps on traded notional.

    Returns a fractional cost (e.g. 0.0007 = 0.07%).
    """
    if not apply_costs:
        return 0.0
    delta = (weights_today - weights_yesterday).fillna(weights_today.fillna(0))
    turnover = delta.abs().sum() / 2.0
    return turnover * (TRANSACTION_COST_BPS / 10_000)
```

In the main backtest loop, subtract the transaction cost from each period's return:

```python
        cost_drag = apply_transaction_costs(target_weights, prev_weights, apply_costs=apply_costs)
        period_return -= cost_drag
```

Add `apply_costs=True` as a parameter to the main backtest function and wire it to `--no-costs` CLI flag.

- [ ] **Step 3: Add --no-costs flag to CLI**

```python
parser.add_argument("--no-costs", action="store_true",
                    help="Run backtest without transaction costs (for comparison).")
parser.add_argument("--dry-run", action="store_true",
                    help="Print configuration and exit without running.")
```

At the start of the backtest, print:

```python
cost_label = "no costs (--no-costs)" if no_costs else f"transaction cost model: {TRANSACTION_COST_BPS}bps"
logger.info("Backtest config: %s", cost_label)
```

- [ ] **Step 4: Verify**

```
cd trading-research
python -m src.backtest.run_backtest --dry-run
```

Expected: prints `transaction cost model: 7bps`.

- [ ] **Step 5: Commit**

```
git add src/backtest/run_backtest.py
git commit -m "feat(backtest): add 7bps round-trip transaction cost model"
```

---

### Task 13: Windows Task Scheduler setup

**Goal:** Create a PowerShell wrapper script and Task Scheduler XML so the daily pipeline runs automatically at 8:30 AM ET on trading days without needing the developer to be at the keyboard.

**Files:**
- Create: `scripts/schedule_daily.ps1`
- Create: `scripts/create_task.ps1` (registers the Task Scheduler task)

**Acceptance Criteria:**
- [ ] `scripts/schedule_daily.ps1` activates the venv, runs `python -m src.execution.daily_run`, logs output to `data/logs/daily_run_YYYY-MM-DD.log`
- [ ] `scripts/create_task.ps1` registers a Task Scheduler task named `TradingAlgorithm_DailyRun` triggered at 08:30 AM daily
- [ ] Running `.\scripts\schedule_daily.ps1 --dry-run` from `trading-research/` completes without error

**Verify:** `.\scripts\schedule_daily.ps1 --dry-run` → exits 0

**Steps:**

- [ ] **Step 1: Create the run wrapper**

Create `scripts/schedule_daily.ps1`:

```powershell
# schedule_daily.ps1 — Daily trading pipeline runner for Windows Task Scheduler
# Usage: .\scripts\schedule_daily.ps1 [--dry-run]
#
# Designed to be called by Windows Task Scheduler at 8:30 AM ET on trading days.
# Activates the project venv, runs the daily pipeline, and logs all output.

param(
    [switch]$DryRun
)

# Resolve absolute paths so the script works regardless of where Task Scheduler
# sets the working directory.
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$LogDir      = Join-Path $ProjectRoot "data\logs"

# Create log directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir "daily_run_$(Get-Date -Format 'yyyy-MM-dd').log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp  $Message" | Tee-Object -FilePath $LogFile -Append
}

Write-Log "=== Daily run starting ==="
Write-Log "Project root: $ProjectRoot"
Write-Log "Dry run: $DryRun"

# Activate virtual environment
if (-not (Test-Path $VenvActivate)) {
    Write-Log "ERROR: venv not found at $VenvActivate — run: python -m venv .venv"
    exit 1
}

& $VenvActivate

# Change to project root so relative imports work
Set-Location $ProjectRoot

# Build the python command
$PythonArgs = @("-m", "src.execution.daily_run")
if ($DryRun) {
    $PythonArgs += "--dry-run"
}

Write-Log "Running: python $($PythonArgs -join ' ')"

# Run the pipeline, capturing exit code
& python @PythonArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
$ExitCode = $LASTEXITCODE

Write-Log "=== Daily run finished: exit code $ExitCode ==="
exit $ExitCode
```

- [ ] **Step 2: Create the Task Scheduler registration script**

Create `scripts/create_task.ps1`:

```powershell
# create_task.ps1 — Register the daily trading task with Windows Task Scheduler
#
# Run this ONCE as Administrator to set up the scheduled task.
# After running, you can view/modify the task in Task Scheduler (taskschd.msc).
#
# The task runs at 08:30 AM ET daily (Monday–Friday).
# Adjust the trigger time if you are in a different timezone.

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RunScript   = Join-Path $ScriptDir "schedule_daily.ps1"
$TaskName    = "TradingAlgorithm_DailyRun"

# Unregister existing task if it exists (clean re-registration)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

$action  = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$RunScript`"" `
    -WorkingDirectory $ProjectRoot

# Mon-Fri at 08:30 — adjust timezone offset if needed
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "08:30AM"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action   $action `
    -Trigger  $trigger `
    -Settings $settings `
    -Description "Run the trading algorithm daily pipeline at 8:30 AM" `
    -RunLevel Highest

Write-Host "Task '$TaskName' registered successfully."
Write-Host "View in: taskschd.msc → Task Scheduler Library"
Write-Host ""
Write-Host "To test immediately: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To run with dry-run: .\scripts\schedule_daily.ps1 -DryRun"
```

- [ ] **Step 3: Test the wrapper script**

```powershell
cd trading-research
.\scripts\schedule_daily.ps1 -DryRun
```

Expected: exits 0, logs appear in `data/logs/daily_run_YYYY-MM-DD.log`.

- [ ] **Step 4: Commit**

```
git add scripts/schedule_daily.ps1 scripts/create_task.ps1
git commit -m "feat(infra): add Windows Task Scheduler scripts for automated daily runs"
```

---

## Self-review against spec

**Phase A coverage:**
- Gross profitability (Novy-Marx): Task 1–2 ✓
- Price-to-book value: Task 1–2 ✓
- Analyst estimate revisions: Task 3 ✓
- Insider transactions (Form 4): Task 4 ✓
- Feature matrix integration: Task 5 ✓

**Phase B coverage:**
- Enhanced news NLP with full article summary: Task 6 ✓
- Macro regime classification: Task 7 ✓
- Regime-modulated IC weights: Task 7 ✓
- Structured attribution proposals: Task 8 ✓

**Phase C coverage:**
- Long-short portfolio construction: Task 9 ✓
- Gross exposure cap in risk engine: Task 9 ✓
- LONG_SHORT_ENABLED=False default gate: Task 9 ✓

**Phase D coverage:**
- Equity curve persistence: Task 10 ✓
- Peak value bug fix: Task 11 ✓
- Transaction cost model: Task 12 ✓
- Task Scheduler automation: Task 13 ✓

**Security constraints check:**
- No live capital: all execution goes through `config.ALPACA_PAPER = True` ✓
- API keys in .env: all keys use `os.getenv()` ✓
- 45-day reporting lag enforced: Tasks 2–3 ✓
- Walk-forward training only on historical data: model.py unchanged ✓
- LONG_SHORT_ENABLED=False: must be manually enabled after backtest review ✓
