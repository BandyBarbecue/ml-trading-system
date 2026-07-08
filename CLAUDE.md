# CLAUDE.md — Multi-Factor ML Trading System

> This file gives Claude Code persistent context for this project.
> Read it at the start of every session.

## What this project is

A **professional-grade quantitative trading research system** built around a multi-factor
ML model, LLM-powered news analysis, and Hierarchical Risk Parity portfolio construction.
Universe: S&P 500. Execution: Alpaca paper trading only — no live capital, ever.

Full design spec: `docs/superpowers/specs/2026-06-09-multi-factor-ml-trading-system-design.md`

The system has two timeframe layers:
- **Daily layer** — LightGBM ranks all 500 S&P stocks by predicted 20-day forward return,
  selects top 25, sizes them with HRP. Rebalances weekly.
- **Intraday layer** — 30-minute bars. Optimises entry timing for daily picks (VWAP/ORB
  confirmation). Also fires event-driven trades when Claude detects high-confidence news
  catalysts (confidence >0.85, magnitude large).

The system is self-improving in three safe ways:
- Monthly LightGBM retraining on a rolling 2-year window (walk-forward, never on live results)
- Dynamic IC weighting — factor weights adapt as market regimes shift
- Optuna hyperparameter tuning, validated on held-out data before deployment
- Monthly SHAP attribution post-mortem reviewed by Claude

## Who the developer is

- First-year Mechanical Engineering student, University of Edinburgh, currently on
  summer holiday (June–August 2026) with substantial time available.
- Python level: capable beginner (GCSE background, but learning fast through this project).
- Wants to **understand every line**. When writing non-obvious code, explain the *why*
  in a short inline comment or in chat. No black boxes — every component should be
  explainable to a smart non-expert.
- Use analogies when explaining new concepts (e.g. "LightGBM is like 100 junior analysts
  each specialising in different patterns, combining votes").

## The rules we don't break

1. **Risk before returns.** The risk engine is the final gate before any order reaches
   Alpaca. It can veto anything. Position limits, kill switch, drawdown cap, VIX filter
   are non-negotiable.
2. **No data leakage, ever.** No look-ahead bias. Walk-forward train/test boundaries must
   be explicit in every backtest. When writing ML or backtest code, always state clearly
   where the boundary is.
3. **No live capital.** Paper trading only. Never in this repo.
4. **Self-improvement must be validated.** Retraining only on historical data with
   held-out validation. Never retrain on live paper-trading results — that amplifies
   overfitting from simulated fills.
5. **Secrets stay in `.env`.** Never hardcode API keys. `.env` is gitignored.
6. **Each module has one job.** Data layer → features → ML → portfolio → risk → execution.
   Modules talk through DataFrames. None knows the internals of another.
7. **Survivorship bias — backfill half FIXED, price-history half remains.**
   Point-in-time S&P 500 membership is now built by `src/data/sp500_history.py`
   (Wikipedia's "Selected changes" table, cached in `data/sp500_changes.parquet` —
   run `python -m src.data.sp500_history` to build/refresh it). The backfill bias —
   treating recent index joiners as if they were members throughout history — is
   fixed: feature ranking (`feature_matrix.py`, `use_pit_universe=True`) and the
   backtest's candidate universe (`run_backtest.py`, `membership_mask=`) only
   consider actual members on each date. **Residual limitation:** price history for
   delisted/acquired ex-members is often unavailable from free feeds, so those names
   may still be missing from the data even though membership is now correct —
   backtests remain somewhat upward-biased. `sp500_history.warn_missing_price_history()`
   logs exactly which ex-member tickers have no price parquet in `data/prices/`.

## Architecture

```
trading-research/
├── config.py                    ← SYMBOLS (S&P 500), paths, API settings, model params
├── requirements.txt
├── .env                         ← ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY,
│                                   FMP_API_KEY
├── data/
│   ├── prices/                  ← <TICKER>.parquet (500 files, daily OHLCV, 5yr)
│   ├── intraday/                ← <TICKER>/YYYY-MM-DD.parquet (30-min bars, 5-day window)
│   ├── news/                    ← YYYY-MM-DD.parquet (Claude-extracted signals per day)
│   ├── earnings/                ← earnings.parquet (EPS actuals vs estimates, all tickers)
│   ├── features/                ← cached daily feature matrices
│   ├── models/                  ← lgbm_YYYY-MM.pkl, params_YYYY-MM.json
│   ├── attribution/             ← YYYY-MM-attribution.md (monthly SHAP post-mortems)
│   ├── sp500_tickers.csv        ← constituents + GICS sectors (updated quarterly)
│   └── vix.parquet
├── src/
│   ├── data/
│   │   ├── fetch_prices.py      ← 500-symbol daily bars → parquet
│   │   ├── fetch_news.py        ← Alpaca News API → daily parquet
│   │   ├── fetch_earnings.py    ← FMP API → earnings.parquet
│   │   ├── fetch_vix.py         ← yfinance ^VIX
│   │   ├── fetch_intraday.py    ← Alpaca 30-min bars
│   │   └── sp500_universe.py    ← Wikipedia scraper + cache
│   ├── features/
│   │   ├── price_features.py    ← mom_1m/3m/12m, reversal_1w, vol_20d
│   │   ├── nlp_features.py      ← Claude Haiku extraction → news_score, supply_chain_score
│   │   └── earnings_features.py ← SUE (Standardised Unexpected Earnings)
│   ├── ml/
│   │   ├── model.py             ← LightGBM train + inference, walk-forward
│   │   ├── ic_tracker.py       ← rolling IC per feature, retirement flagging
│   │   └── tuner.py            ← Optuna monthly hyperparameter search
│   ├── portfolio/
│   │   ├── hrp.py               ← Hierarchical Risk Parity (Lopez de Prado 2016)
│   │   └── construction.py      ← select top 25, sector cap, turnover limit
│   ├── risk/
│   │   └── engine.py            ← position limits, kill switch, drawdown cap, VIX filter
│   ├── backtest/
│   │   ├── run_backtest.py      ← full backtest with all signals
│   │   └── walk_forward.py      ← walk-forward validation harness
│   └── execution/
│       ├── daily_run.py         ← main orchestrator (runs 8:30 AM ET daily)
│       ├── paper_trader.py      ← Alpaca order submission
│       ├── intraday_signal.py   ← entry timing + event-driven intraday trades
│       └── attribution.py      ← monthly SHAP post-mortem + Claude review
└── docs/
    └── superpowers/specs/       ← design specs
```

## Daily execution sequence (8:30 AM ET)

```
1. Fetch: price top-up, overnight news, VIX, earnings check       (~2 min)
2. Claude NLP: batch headlines → structured JSON signals           (~3 min, ~$0.30/day)
3. Features + IC update: price features, merge news/earnings,
   recompute 60-day rolling IC, cross-sectional z-score           (~30 sec)
4. ML inference (daily) / retrain (1st of month)                  (~30 sec / 5 min)
5. Portfolio construction (Mondays only): top 25 → HRP →
   sector cap → trade list                                        (~10 sec)
6. Intraday signal: entry timing / event-driven trades            (continuous until 3:50 PM)
7. Risk engine veto → submit to Alpaca paper API                  (~5 sec)
8. Daily report: P&L vs SPY, positions, factor attribution        (printed + saved)
```

## Key signals

| Feature | What it captures |
|---|---|
| `mom_1m / 3m / 12m` | Price momentum (skip-period adjusted) |
| `reversal_1w` | Short-term mean reversion |
| `vol_20d` | Realised volatility (also used by HRP) |
| `news_score` | Claude-extracted headline sentiment, 3-day decayed |
| `supply_chain_score` | Sentiment propagated to supplier/customer tickers |
| `earnings_surprise` | Standardised Unexpected Earnings (SUE), 60-day decay |

All features are cross-sectionally z-scored within the S&P 500 daily before entering
the model.

## Risk controls

| Control | Threshold | Action |
|---|---|---|
| Position limit | >8% per stock | Hard veto |
| Sector exposure | >30% per GICS sector | Scale down |
| Daily kill switch | Portfolio down >2% | Exit all |
| Drawdown cap | DD >15% from peak | Scale to 50% |
| VIX regime filter | VIX >30 | Halve all sizes |
| Turnover cap | >40% per rebalance | Priority-order only |

## Self-improvement mechanisms

- **Monthly retrain** — LightGBM on rolling 2-year window (walk-forward, not on live results)
- **Dynamic IC weighting** — Spearman IC per feature on 60-day rolling window → `feature_weight` in LightGBM
- **Optuna tuning** — 50-trial Bayesian search monthly, validated on held-out slice before deployment
- **IC retirement flagging** — IC < 0.01 for 3 consecutive months → warning in daily report
- **SHAP attribution** — monthly post-mortem on closed positions, Claude reviews for systematic bias

## Conventions

- Python 3.11+, `venv` for isolation.
- Type hints on all function signatures. Docstrings on anything non-trivial.
- `config.py` is the single source of truth for symbols, paths, model settings.
- Prefer clarity over cleverness.
- SDK: `alpaca-py` (OOP SDK — request objects, not the old REST wrapper).
- Claude API: `anthropic` SDK, use `claude-haiku-4-5-20251001` for NLP extraction
  (speed + cost), `claude-sonnet-4-6` for attribution review (quality).

## Current state (as of 2026-06-09)

Design spec approved and written. Implementation plan being created.
Previous codebase (MA crossover, old feedback loop, old walk-forward) is retained
in `src/` but will be superseded module by module as the new system is built.
