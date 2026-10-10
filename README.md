# Multi-Factor ML Trading System

A quantitative equity research and **paper-trading** system for the S&P 500. Every trading day it:

- gathers market, fundamental and news data;
- turns the news into structured signals with an LLM;
- ranks all ~500 stocks with a LightGBM model trained walk-forward;
- builds a diversified portfolio with Hierarchical Risk Parity;
- passes every order through a risk engine that can veto anything.

It runs unattended as a scheduled job on [Modal](https://modal.com).

> **Paper trading only.** Orders go to Alpaca's paper-trading API. No real capital is used, and that is a hard rule of the project. The realistic target is a Sharpe ratio of about 0.5–0.8, not a money machine.

## Pipeline

```
[Data] → [Features] → [ML ranking] → [Portfolio] → [Risk engine] → [Execution]
```

| Stage | What happens |
|---|---|
| **Data** | Alpaca (prices, news, 30-min bars), yfinance (VIX, earnings, fundamentals), SEC EDGAR (insider Form 4s), FRED (macro), Wikipedia (point-in-time S&P 500 membership). All stored as parquet. |
| **Features** | Momentum (1/3/12-month, skip-period adjusted), 1-week reversal, 20-day volatility, earnings surprise (SUE, decaying), gross profitability, P/B, analyst revision momentum, insider score, and **news sentiment and supply-chain spillover signals extracted from headlines by Claude Haiku** as structured JSON. Every feature is cross-sectionally z-scored. |
| **ML** | LightGBM predicts each stock's 20-day forward return rank. It is retrained monthly on a rolling 2-year window using walk-forward validation with an embargo. Factor weights adapt through a rolling Spearman IC (information coefficient), and a regime classifier adjusts them. |
| **Portfolio** | The top 25 stocks are sized with Hierarchical Risk Parity (López de Prado, 2016), then capped by sector and turnover. Rebalanced weekly. |
| **Risk engine** | 8% per-position limit, 30% sector cap, 2% daily kill switch, 15% drawdown cap (scales the book to 50%), VIX > 30 halves all sizes. Caps leave the residual in cash; they never re-normalise back up. |
| **Execution** | Orders go to Alpaca paper with **idempotent order IDs**, so a restarted run can't double-fill. |
| **Feedback** | Monthly SHAP attribution, which Claude Sonnet reviews and returns structured proposals for. Weekly reports reconcile paper trading against the backtest. A backtest with a 7 bps cost model. |

## Engineering highlights

These are some of the problems found and fixed while building and running it:

- **Look-ahead bias controls.**
  - An embargo on the forward-return target in walk-forward splits.
  - Event-availability lags: after-close earnings, insider filings (+2 business days) and after-4pm news all only affect the *next* day.
  - Point-in-time index membership, which removes the bias of including stocks that only joined the index later.
- **Risk caps that silently did nothing.** An audit found that limits were applied and then the weights were re-normalised back to 100%, which undid them. Fixed, and the tests that had asserted the broken behaviour were replaced with invariant tests ("the cap always holds").
- **Dead integrations hidden by "graceful degradation."** Three data sources had never worked: a wrong FRED domain, a retired earnings endpoint, and a regime classifier stuck on its default. Because each failure was caught and logged only as a warning, none of them looked broken. All were fixed, and the lesson was recorded: a warning that fires on every run is a dead feature.
- **Crash-safe LLM spend.** The headline-extraction cache is saved after every batch and caches "no signal" results. Paid extraction is capped to a recent window, and a resumable backfill script defaults to a dry run.
- **Cloud migration (Windows Task Scheduler → Modal).** Runs had been dying mid-pipeline. The final log line came 629 ms before Windows logged the laptop going to sleep, which proved the code wasn't at fault. The migration was done in stages and had to handle:
  - timezone data missing from slim containers;
  - import-time dates going stale in reused containers;
  - Modal restarting a preempted container regardless of its retry setting.
- **Kill switch false positive.** The kill switch once compared today's equity with a 28-day-old baseline and liquidated the paper book on a "−2.06% day". The true one-day move was −0.62%. Fixed to use the broker's own previous close and to reject stale baselines. When data is unreliable it now **fails safe** by not acting.

## Tests

**352 pytest tests** cover features, the model, portfolio construction, the risk engine, backtesting, execution and attribution, including invariant tests for every risk cap.

```bash
python -m pytest -q
```

## Running locally

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
copy .env.example .env      # Alpaca paper keys, Anthropic key, etc.
python scripts/bootstrap.py # initial data download
python -m src.execution.daily_run --dry-run
```

`modal_app.py` defines the cloud deployment. It includes the daily cron, dry-run, validation and forced-rebalance entry points. Market data, models and state live in a Modal Volume and are not stored in this repo.

## How it was built

I built this with **Claude Code** over summer 2026, starting from a simple moving-average-crossover script. The work followed a spec-driven process:

1. Brainstorming the architecture with browser mockups.
2. A written design spec that I approved.
3. Detailed implementation plans (in [`docs/superpowers/`](docs/superpowers/)).
4. Test-first implementation.

That was followed by a full-system audit, production incident fixes, and the cloud migration. [`CLAUDE.md`](CLAUDE.md) holds the project's working rules.

I directed the project: I set the goals, constraints and risk rules, made the design and migration decisions, and set up and ran the infrastructure. Throughout, I worked to understand the methods: walk-forward validation, IC, HRP and look-ahead bias. Claude Code wrote most of the implementation.
