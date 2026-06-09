# Multi-Factor ML Trading System — Design Spec
**Date:** 2026-06-09  
**Status:** Approved for implementation  
**Strategy class:** Cross-sectional equity long-only, daily rebalance (weekly portfolio changes)  
**Universe:** S&P 500  
**Execution:** Alpaca paper trading only — no live capital, ever

---

## 1. What We're Building

A quantitative equity trading system that ranks all S&P 500 stocks daily using a combination of price momentum, mean reversion, NLP-extracted news sentiment, and earnings surprises. A LightGBM model learns the optimal combination of these signals to predict which stocks will outperform over the next 20 trading days. Portfolio construction uses Hierarchical Risk Parity (HRP) to allocate across the top 25 picks in a genuinely diversified way.

**This is not a parameter-optimisation system.** Walk-forward is used as a validation method, not a strategy. Every signal has an independent economic justification — not just backtest fit.

**Realistic target:** Sharpe ratio 0.5–0.8, modest consistent outperformance vs SPY. Not 50% annual returns.

---

## 2. Architecture Overview

Five layers, each with one job, communicating through clean DataFrame interfaces:

```
[Data Ingestion] → [Feature Engineering] → [ML Model] → [Portfolio Construction] → [Risk Engine] → [Execution]
```

The risk engine is the final gate before any order reaches Alpaca. It can veto anything.

---

## 3. Data Pipeline

All data is free. Two sources use the existing Alpaca API key.

| Source | What | Frequency | Stored as |
|---|---|---|---|
| Alpaca Historical | OHLCV daily bars, 500 symbols, 5yr, adjusted | Daily top-up | `data/prices/<TICKER>.parquet` |
| Alpaca News API | Headlines + snippets per ticker, 90-day rolling | Daily (overnight) | `data/news/YYYY-MM-DD.parquet` |
| FMP API | Quarterly EPS actuals vs analyst estimates | Quarterly refresh | `data/earnings/earnings.parquet` |
| yfinance | ^VIX daily close | Daily top-up | `data/vix.parquet` |
| Wikipedia scrape | S&P 500 constituents + GICS sectors | Quarterly | `data/sp500_tickers.csv` |

**Why parquet over CSV:** 500 symbols × 5 years × daily bars makes CSV reads slow. Parquet is ~10× faster to read and ~5× smaller. `pd.read_parquet()` is a drop-in replacement for `pd.read_csv()`.

New data modules:
```
src/data/
  fetch_prices.py       ← replaces fetch_data.py (500 symbols, parquet output)
  fetch_news.py         ← Alpaca News API
  fetch_earnings.py     ← FMP API (250 calls/day free tier, quarterly cadence)
  fetch_vix.py          ← yfinance, no auth required
  sp500_universe.py     ← Wikipedia scraper + local cache
```

---

## 4. Feature Engineering

Six signals computed daily for all 500 stocks. All z-scored cross-sectionally before entering the model.

### Price signals (from Alpaca OHLCV)

| Feature | Computation | Why |
|---|---|---|
| `mom_1m` | 21-day return, skip last 5 days | Short-term momentum (Jegadeesh-Titman) |
| `mom_3m` | 63-day return, skip last 5 days | Medium-term momentum |
| `mom_12m` | 252-day return, skip last 21 days | Long-term momentum |
| `reversal_1w` | Negative of 5-day return | Short-term mean reversion |
| `vol_20d` | Std dev of daily returns over 20 days | Low-vol anomaly + used by HRP |

**Why skip periods?** Including the most recent days in a momentum calculation mixes a momentum signal with a mean-reversion signal, contaminating both. Separating them gives the ML model cleaner inputs.

### NLP signal (from Alpaca News via Claude API)

**Feature:** `news_score` — daily sentiment score per ticker, aggregated from last 3 days of headlines with recency weighting (today's headlines weighted 2× yesterday's).

Also derived: `supply_chain_score` — sentiment propagated to tickers identified as suppliers/customers by Claude's extraction (e.g. Apple earnings → TSMC uplift).

**Pipeline:**
1. Fetch overnight headlines from Alpaca News API
2. Send to Claude Haiku in batches of 20 with a structured extraction prompt
3. Claude returns JSON per headline: `event_type`, `primary_ticker`, `affected_tickers` (with direction + relationship), `magnitude`, `time_horizon`, `confidence`
4. Aggregate to one score per ticker per day

**Prompt template (simplified):**
```
Given this financial headline, extract a structured signal.
Return JSON: {event_type, primary_ticker, affected_tickers: [{ticker, direction, reason}],
magnitude, direction, time_horizon, confidence}
```

**Cost:** ~500 headlines/day × ~200 tokens = ~100k tokens/day ≈ $0.30/day at Haiku 4.5 pricing. Headlines already processed are cached and skipped on re-runs.

### Earnings signal (from FMP)

**Feature:** `earnings_surprise` — Standardised Unexpected Earnings (SUE):
```
SUE = (actual_EPS - estimated_EPS) / std(last_8_surprises)
```
Standardising prevents high-EPS-volatility stocks from dominating. Signal decays linearly over 60 days post-announcement — last quarter's beat still carries weight, just reduced.

### Cross-sectional normalisation

Every feature is z-scored within the S&P 500 on each day before entering the model. This removes market-wide moves and makes the model rank stocks against each other, not against their own history. It also makes the model stable across different market regimes.

### Historical data availability note

Price and earnings history: 5 years available. Alpaca News API historical coverage: approximately 2–3 years on free tier. For backtest dates where news data is unavailable, `news_score` and `supply_chain_score` will be NaN — LightGBM handles missing values natively and will predict using only the available features. This is correct behaviour, not a bug. The backtest should therefore show two periods: pre-news (price + earnings factors only) and post-news (all six factors), which is also a useful validation — the news signal's marginal contribution becomes visible.

---

## 5. ML Model

### Algorithm: LightGBM (gradient boosted trees)

LightGBM outperforms neural networks on tabular financial data — well-established in academic and industry literature as of 2026. It handles missing values natively (not every stock has earnings data every day), trains in seconds on our feature set, and provides built-in feature importance for interpretability.

### Target variable

**20-day forward cross-sectional return rank** (0–1 percentile within S&P 500). Ranks are more stationary than raw returns and make the model learn relative performance, which is what portfolio construction needs.

### Walk-forward training

- **Training window:** rolling 2 years (504 trading days)
- **Test window:** 1 month (21 days)
- **Step:** 1 month
- **With 5 years of data:** ~3 years of out-of-sample backtest predictions, covering the 2022 bear market, 2023 recovery, and 2024–25 AI bull run

Monthly retrain: first trading day of each month, retrain on latest 2-year window. Model artefacts saved to `data/models/lgbm_YYYY-MM.pkl`.

### Dynamic IC weighting

Information Coefficient (IC) = Spearman correlation between a feature's values and actual 20-day forward returns. Computed on a rolling 60-day window for each feature.

Before each monthly retrain, IC values are computed and passed as `feature_weight` inputs to LightGBM. Features that have been predictive in the recent market environment get upweighted; features that have gone stale get downweighted. This allows the model to adapt automatically when factor regimes shift — e.g. momentum going negative in a bear market while mean reversion strengthens.

IC tracking module: `src/ml/ic_tracker.py`

### Output

One `predicted_score` (0–1) per stock per day. Top 25 by score pass to portfolio construction.

---

## 6. Portfolio Construction

### Stock selection

Top 25 stocks by LightGBM `predicted_score`, filtered to remove any stock with less than 60 days of price history (to ensure reliable covariance estimates).

### Hierarchical Risk Parity (HRP) sizing

HRP (Lopez de Prado, 2016) assigns weights based on a hierarchical clustering of the correlation structure of the selected stocks. Stocks that move together are treated as a cluster and collectively receive the same weight allocation as a single uncorrelated stock.

**Four steps:**
1. Compute 60-day rolling correlation matrix of the 25 selected stocks
2. Convert correlations to distances: `d = sqrt(0.5 × (1 - corr))`
3. Hierarchical clustering (Ward linkage) — groups correlated stocks into clusters
4. Recursive bisection — allocate inversely proportional to each cluster's variance, recurse within each cluster until individual stock weights are assigned

Implementation: `src/portfolio/hrp.py`

### Sector constraint

After HRP: if any GICS sector exceeds 30% of portfolio weight, scale those positions down proportionally and redistribute the freed weight to other sectors pro-rata. Prevents concentrated sector bets even when ML strongly favours one industry.

### Rebalancing

Weekly (every Monday before market open). Turnover cap: 40% of portfolio value per rebalance. If the optimal portfolio requires more turnover than this, only the highest-priority trades (largest weight changes) are executed first.

---

## 6b. Intraday Signal Layer (Multi-Timeframe)

The daily model answers "which stocks?" The intraday layer answers "when exactly?" and "is there a short-term catalyst worth acting on now?"

Two components, both using 30-minute bars from Alpaca (same API key, `TimeFrame.Minute` aggregated to 30-min).

**Data storage:** `data/intraday/<TICKER>/YYYY-MM-DD.parquet` — one file per symbol per day, fetched each morning and retained for 5 trading days then purged.

### Component 1: Entry Timing Optimiser

Applies to all daily-model-selected stocks. Instead of blindly buying at 9:30 AM on Monday, the system waits up to 90 minutes for intraday confirmation before executing.

Signals checked (first 30–90 minutes of trading):
- **VWAP**: Is the stock above its VWAP? Below VWAP on a bullish daily signal = wait.
- **Opening Range Breakout (ORB)**: First 30-minute bar high/low. Breakout above = confirm entry.
- **Volume confirmation**: Volume > 120% of 20-day average 30-min volume at same time of day = stronger signal.

Logic: if all three confirm → execute full target weight. If two of three → execute 50% now, rest at end of day. If zero/one confirm → execute at end of day regardless (we still want the daily position, just not at an unfavourable intraday price).

### Component 2: Event-Driven Intraday Trades

Triggered independently of the daily model. When Claude's NLP extraction flags a headline with `confidence > 0.85` AND `magnitude = large` AND `time_horizon = intraday`, a short-term trade is considered.

Entry: confirmed only if 30-minute intraday momentum corroborates direction (stock moving in predicted direction with above-average volume).

Position size: max 2% of portfolio per event-driven trade. Separate allocation bucket — does not consume the daily portfolio's weight budget.

Hold: 30 minutes to end of trading day. All event-driven positions close at 3:50 PM ET unless explicitly rolled to the daily portfolio (requires the daily model to also rank the stock in its top 25).

Implementation: `src/execution/intraday_signal.py`

---

## 6c. Self-Improvement Framework

Three mechanisms, all safe (validated on held-out data before deployment, no retraining on live results).

### Automated Hyperparameter Tuning (Optuna)

Monthly, before the LightGBM retrain step. Optuna runs 50 trials optimising hyperparameters (`num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, `bagging_fraction`) using time-series cross-validation on the training window. Best parameters are validated on a held-out slice before being written to `data/models/params_YYYY-MM.json`. If the tuned parameters perform worse than the previous month's params on held-out data, the previous params are retained.

Implementation: `src/ml/tuner.py`

### Feature IC Retirement Flagging

If any feature's 60-day rolling IC falls below 0.01 for three consecutive monthly cycles, it is flagged in the daily report with a warning: `"[IC WARNING] feature X has been near-zero for 3 months — consider reviewing."` The feature is never automatically removed — that decision stays with the developer. This prevents silently carrying dead weight while avoiding accidental removal of an economically justified signal that is temporarily dormant.

Implementation: extend `src/ml/ic_tracker.py`

### Trade Attribution Loop (Monthly, Claude-reviewed)

After each month closes, a post-mortem runs across all closed positions:

1. For each closed position, compute actual 20-day return vs model's predicted rank
2. Decompose each pick using **SHAP values** — which features drove the LightGBM prediction? (SHAP = a standard ML interpretability method that assigns each feature a contribution score for a specific prediction)
3. Aggregate: which features have been systematically over-predicting? Which have been under-predicting?
4. Claude reviews the SHAP attribution report and writes a brief structured finding: any systematic biases, any factor that's been consistently wrong

Output: `data/attribution/YYYY-MM-attribution.md` — readable post-mortem that builds up a monthly record of the system's learning history.

Implementation: `src/execution/attribution.py`

---

## 7. Risk Engine

Extends the existing `src/risk/engine.py`. The risk engine is the final gate — it checks every proposed order before it reaches Alpaca. Orders that fail any check are vetoed and logged.

| Control | Threshold | Action |
|---|---|---|
| Position limit | Max 8% per stock | Hard veto |
| Sector exposure | Max 30% per GICS sector | Proportional scale-down |
| Daily loss kill switch | Portfolio down >2% intraday | Exit all positions |
| Drawdown cap | Portfolio DD >15% from peak | Scale all positions to 50% |
| VIX regime filter | VIX >30 | Halve all position sizes |
| Turnover cap | >40% of portfolio per rebalance | Execute only top-priority trades |

All decisions — approved and vetoed — are written to `data/decisions.jsonl` with timestamp, reasoning, and signal values at time of decision.

---

## 8. Daily Execution Loop

Orchestrated by `src/execution/daily_run.py`. Runs each morning before market open (~8:30 AM ET). Total runtime: ~6–7 minutes daily, ~12 minutes on retrain days (first of month).

```
Step 1 — Fetch data (~2 min)
  Top-up price bars, overnight news, VIX, check for new earnings

Step 2 — Claude NLP extraction (~3 min, ~$0.30)
  Batch headlines → Claude Haiku → JSON signals → cache processed headlines

Step 3 — Compute features + IC update (~30 sec)
  Price features, merge news + earnings scores, recompute 60-day rolling IC,
  cross-sectional z-score all features

Step 4 — ML inference, monthly retrain on 1st of month (~30 sec / 5 min)
  Run LightGBM → predicted_score for 500 stocks

Step 5 — Portfolio construction, Mondays only (~10 sec)
  Select top 25 → HRP → sector cap → diff vs current holdings → trade list

Step 6 — Risk engine veto + execute (~5 sec)
  VIX, drawdown, position limits → submit approved orders to Alpaca paper API

Step 7 — Daily report (printed + saved)
  P&L vs SPY, current positions, factor attribution (which signal drove each pick)
```

---

## 9. Full Directory Layout

```
trading-research/
├── config.py                        ← add SP500, FMP_API_KEY, Claude model settings
├── requirements.txt                 ← add lightgbm, scipy, scikit-learn, yfinance, pyarrow, optuna, shap
├── .env                             ← add FMP_API_KEY (Anthropic key already present)
├── data/
│   ├── prices/                      ← AAPL.parquet … (500 files, daily bars)
│   ├── intraday/                    ← AAPL/2026-06-09.parquet … (30-min bars, 5-day rolling window)
│   ├── news/                        ← 2026-06-09.parquet … (one per trading day)
│   ├── earnings/earnings.parquet
│   ├── features/                    ← cached daily feature matrices
│   ├── models/
│   │   ├── lgbm_YYYY-MM.pkl         ← monthly retrain artefacts
│   │   └── params_YYYY-MM.json      ← Optuna-tuned hyperparameters per month
│   ├── attribution/                 ← YYYY-MM-attribution.md (monthly post-mortems)
│   ├── sp500_tickers.csv
│   └── vix.parquet
├── src/
│   ├── data/
│   │   ├── fetch_prices.py          ← replaces fetch_data.py
│   │   ├── fetch_news.py
│   │   ├── fetch_earnings.py
│   │   ├── fetch_vix.py
│   │   ├── fetch_intraday.py        ← 30-min bars from Alpaca
│   │   └── sp500_universe.py
│   ├── features/
│   │   ├── price_features.py
│   │   ├── nlp_features.py          ← Claude extraction pipeline
│   │   └── earnings_features.py
│   ├── ml/
│   │   ├── model.py                 ← LightGBM train + inference
│   │   ├── ic_tracker.py           ← rolling IC + retirement flagging
│   │   └── tuner.py                ← Optuna hyperparameter search
│   ├── portfolio/
│   │   ├── hrp.py
│   │   └── construction.py
│   ├── risk/
│   │   └── engine.py                ← extend existing
│   ├── backtest/
│   │   ├── run_backtest.py          ← extend existing
│   │   └── walk_forward.py          ← extend existing
│   └── execution/
│       ├── daily_run.py             ← main orchestrator
│       ├── paper_trader.py
│       ├── intraday_signal.py       ← entry timing + event-driven intraday trades
│       └── attribution.py          ← monthly SHAP post-mortem + Claude review
└── docs/
    └── superpowers/specs/
        └── 2026-06-09-multi-factor-ml-trading-system-design.md  ← this file
```

---

## 10. Technology Stack

| Component | Library | Reason |
|---|---|---|
| Data frames | pandas 2.0+ | Already in use |
| Parquet I/O | pyarrow | Fast columnar storage |
| ML model | lightgbm | Best-in-class for tabular data |
| HRP / linear algebra | scipy, numpy | Clustering + matrix operations |
| Backtesting | vectorbt | Already in use |
| NLP | Claude Haiku 4.5 via anthropic SDK | Structured event extraction |
| Hyperparameter tuning | optuna | Bayesian optimisation, time-series CV |
| Feature interpretability | shap | SHAP values for monthly attribution post-mortem |
| Price data | alpaca-py | Already in use |
| News data | alpaca-py | Same API key |
| Earnings data | requests (FMP REST API) | Simple HTTP, no SDK needed |
| VIX data | yfinance | Free, no auth |
| Execution | alpaca-py | Already in use |

---

## 11. What We're Not Building (Scope Boundaries)

- No live capital. Paper trading only.
- No intraday / minute-bar data. Daily bars only.
- No options or derivatives.
- No short positions (long-only portfolio).
- No social media / Reddit sentiment (not in scope for v1 — addable later).
- No automatic CLAUDE.md updates (user retains full control of project direction).
