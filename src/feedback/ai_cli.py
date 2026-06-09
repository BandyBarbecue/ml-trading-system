"""AI-driven feedback loop: Claude reviews and decides on every MA crossover signal.

Automated mirror of src/feedback/cli.py. Replaces the human prompt with a call
to the Anthropic API — Claude sees each signal card (price, MAs, portfolio state,
risk verdict) and returns a structured approve/reject decision with written
reasoning.

Run from the project root with:

    python -m src.feedback.ai_cli

All decisions are written to data/decisions.jsonl in the same format as the
human feedback loop, so both can be analysed together in Phase 4.

Requires:
  - ANTHROPIC_API_KEY in .env
  - ALPACA_API_KEY and ALPACA_SECRET_KEY in .env (for historical data)
"""

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.strategies.ma_crossover import ma_crossover_signals
from src.risk.engine import RiskEngine
from src.feedback.logger import log_decision
from src.feedback.ai_reviewer import review_signal

INIT_CASH = 10_000
SHARES_PER_TRADE = 10
LOG_PATH = config.DATA_DIR / "decisions.jsonl"


def run_symbol(symbol: str) -> None:
    """Run Claude's decision loop for all MA crossover signals on a single symbol."""
    csv_path = config.DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        print(f"No data for {symbol}. Run `python -m src.data.fetch_data` first.")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close = df["close"]
    short_ma = close.rolling(20).mean()
    long_ma = close.rolling(50).mean()

    entries, exits = ma_crossover_signals(close)

    risk = RiskEngine(starting_cash=INIT_CASH)
    cash = float(INIT_CASH)
    shares_held = 0
    prev_date = None
    decisions_made = 0

    for date, price in close.items():
        portfolio_value = cash + shares_held * price
        new_day = (prev_date is None) or (date.date() != prev_date)
        risk.update(portfolio_value, new_day=new_day)
        prev_date = date.date()

        s_ma = short_ma.get(date)
        l_ma = long_ma.get(date)

        # Skip bars where MAs aren't ready yet (first 50 days).
        if pd.isna(s_ma) or pd.isna(l_ma):
            continue

        signal_type = None
        trade_value = SHARES_PER_TRADE * price

        if entries.get(date, False) and shares_held == 0:
            signal_type = "buy"
        elif exits.get(date, False) and shares_held > 0:
            signal_type = "sell"

        if signal_type is None:
            continue

        risk_approved, risk_reason = risk.approve_trade(trade_value, portfolio_value)

        print(f"\n  Reviewing signal: {signal_type.upper()} {symbol} on {date.date()}")
        print(f"  Price: ${price:.2f}  |  Short MA: ${s_ma:.2f}  |  Long MA: ${l_ma:.2f}")
        print(f"  Risk: {'APPROVED' if risk_approved else 'VETOED'} — {risk_reason}")
        print("  Asking Claude...", end="", flush=True)

        decision, reasoning = review_signal(
            symbol=symbol,
            signal_type=signal_type,
            signal_date=str(date.date()),
            price=price,
            short_ma=s_ma,
            long_ma=l_ma,
            portfolio_value=portfolio_value,
            risk_approved=risk_approved,
            risk_reason=risk_reason,
        )

        print(f" {decision.upper()}")
        print(f"  Reasoning: {reasoning}")

        log_decision(
            LOG_PATH,
            symbol=symbol,
            signal_type=signal_type,
            signal_date=str(date.date()),
            price=price,
            short_ma=s_ma,
            long_ma=l_ma,
            portfolio_value=portfolio_value,
            risk_approved=risk_approved,
            risk_reason=risk_reason,
            user_decision=decision,
            notes=f"[AI] {reasoning}",
        )
        decisions_made += 1

        # Execute the trade if approved (so portfolio state stays accurate).
        if decision == "approve" and risk_approved:
            if signal_type == "buy":
                shares_held = SHARES_PER_TRADE
                cash -= trade_value
            elif signal_type == "sell":
                cash += shares_held * price
                shares_held = 0

    print(f"\n  Done. {decisions_made} AI decision(s) logged to {LOG_PATH.relative_to(config.PROJECT_ROOT)}")


def main() -> None:
    print("\n  AI-Feedback Loop — Phase 3 (Automated)")
    print("  Claude reviews each signal. Decisions saved to data/decisions.jsonl\n")

    print("  Available symbols:", ", ".join(config.SYMBOLS))
    print("  Enter a symbol, or 'ALL' to process every symbol in the universe.")
    raw = input("  Which symbol? ").strip().upper()

    if raw == "ALL":
        for symbol in config.SYMBOLS:
            print(f"\n{'='*60}")
            print(f"  Processing {symbol}")
            print(f"{'='*60}")
            run_symbol(symbol)
    elif raw in config.SYMBOLS:
        run_symbol(raw)
    else:
        print(f"  '{raw}' not in universe. Choose from {config.SYMBOLS} or 'ALL'")
        sys.exit(1)


if __name__ == "__main__":
    main()
