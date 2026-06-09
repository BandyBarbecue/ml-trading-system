# Systematic Trading Research Framework

A learning project: a quant trading **research and risk system**, not a money machine.
See `CLAUDE.md` for the full context, rules, and roadmap.

## Quick start (Phase 0)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install Phase 0 dependencies
pip install -r requirements.txt

# 3. Add your Alpaca paper-trading keys
cp .env.example .env             # then edit .env and paste your keys

# 4. Pull one year of daily data
python -m src.data.fetch_data
```

If step 4 prints row counts and saves CSVs into `data/`, Phase 0 is done.

## Phases
- **0 — Foundations (now):** data ingestion.
- **1 — Dumb strategy:** moving-average crossover, backtested.
- **2 — Risk engine:** hard limits, kill switch.
- **3 — Human-feedback loop:** approve / reject / override, logged.
- **4 — Bounded learning:** walk-forward optimisation with out-of-sample holdout.
