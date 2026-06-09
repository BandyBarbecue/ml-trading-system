"""Decision logger: appends every human judgement to data/decisions.jsonl.

JSON Lines format — one JSON object per line. This makes the file easy to
append to (no need to parse the whole file first) and easy to load later:

    import pandas as pd
    df = pd.read_json("data/decisions.jsonl", lines=True)

Each record captures everything needed to reconstruct *why* you made the
decision: the signal, the market context, the risk engine verdict, and your
choice. This is the raw material for Phase 4.
"""

import json
from datetime import datetime
from pathlib import Path


def log_decision(
    log_path: Path,
    *,
    symbol: str,
    signal_type: str,          # "buy" or "sell"
    signal_date: str,          # ISO date string, e.g. "2025-10-15"
    price: float,
    short_ma: float,
    long_ma: float,
    portfolio_value: float,
    risk_approved: bool,
    risk_reason: str,
    user_decision: str,        # "approve", "reject", or "override"
    override_price: float | None = None,
    notes: str = "",
) -> None:
    """Append one decision record to the log file."""
    record = {
        "logged_at": datetime.now().isoformat(),
        "symbol": symbol,
        "signal_type": signal_type,
        "signal_date": signal_date,
        "price": round(price, 4),
        "short_ma": round(short_ma, 4),
        "long_ma": round(long_ma, 4),
        "portfolio_value": round(portfolio_value, 2),
        "risk_approved": risk_approved,
        "risk_reason": risk_reason,
        "user_decision": user_decision,
        "override_price": override_price,
        "notes": notes,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
