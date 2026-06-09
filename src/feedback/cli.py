"""Phase 3: human-feedback CLI.

Run from the project root with:

    python -m src.feedback.cli

For each MA crossover signal the system surfaces:
  - The signal context (price, short MA, long MA, what the trend looks like)
  - The risk engine's verdict
  - Your four options: Approve / Reject / Override / Skip (don't log)

Every approved/rejected/overridden decision is written to data/decisions.jsonl.
Skipped signals are not logged.

'Override' lets you set a different price — useful for noting "I'd only buy
this at $X", which is valuable data even if the trade doesn't happen.
"""

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.strategies.ma_crossover import ma_crossover_signals
from src.risk.engine import RiskEngine
from src.feedback.logger import log_decision

INIT_CASH = 10_000
SHARES_PER_TRADE = 10
LOG_PATH = config.DATA_DIR / "decisions.jsonl"


def prompt_decision(
    symbol: str,
    signal_type: str,
    date: str,
    price: float,
    short_ma: float,
    long_ma: float,
    portfolio_value: float,
    risk_approved: bool,
    risk_reason: str,
) -> tuple[str, float | None, str]:
    """Show the signal card and collect the user's decision.

    Returns:
        (decision, override_price, notes)
        decision is one of: "approve", "reject", "override", "skip"
    """
    trend = "↑ short MA above long" if short_ma > long_ma else "↓ short MA below long"
    risk_icon = "✓" if risk_approved else "✗"

    print(f"\n{'='*60}")
    print(f"  SIGNAL: {signal_type.upper()}  {symbol}  {date}")
    print(f"{'='*60}")
    print(f"  Price:        ${price:.2f}")
    print(f"  Short MA:     ${short_ma:.2f}  (20-day)   {trend}")
    print(f"  Long MA:      ${long_ma:.2f}  (50-day)")
    print(f"  Portfolio:    ${portfolio_value:,.2f}")
    print(f"  Risk engine:  {risk_icon} {risk_reason}")
    print()
    print("  [A] Approve    [R] Reject    [O] Override price    [S] Skip (no log)")
    print()

    while True:
        raw = input("  Your decision: ").strip().upper()
        if raw in ("A", "APPROVE"):
            notes = input("  Notes (optional, press Enter to skip): ").strip()
            return "approve", None, notes
        elif raw in ("R", "REJECT"):
            notes = input("  Notes (optional, press Enter to skip): ").strip()
            return "reject", None, notes
        elif raw in ("O", "OVERRIDE"):
            while True:
                try:
                    override_price = float(input("  Override price $: ").strip())
                    break
                except ValueError:
                    print("  Please enter a number.")
            notes = input("  Notes (optional, press Enter to skip): ").strip()
            return "override", override_price, notes
        elif raw in ("S", "SKIP"):
            return "skip", None, ""
        else:
            print("  Please enter A, R, O, or S.")


def run_symbol(symbol: str) -> None:
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

        decision, override_price, notes = prompt_decision(
            symbol=symbol,
            signal_type=signal_type,
            date=str(date.date()),
            price=price,
            short_ma=s_ma,
            long_ma=l_ma,
            portfolio_value=portfolio_value,
            risk_approved=risk_approved,
            risk_reason=risk_reason,
        )

        if decision == "skip":
            print("  (skipped — not logged)")
            continue

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
            override_price=override_price,
            notes=notes,
        )
        decisions_made += 1
        print(f"  Logged. ({decisions_made} decision{'s' if decisions_made != 1 else ''} recorded this session)")

        # Execute the trade if approved (so portfolio state stays accurate).
        if decision == "approve" and risk_approved:
            if signal_type == "buy":
                shares_held = SHARES_PER_TRADE
                cash -= trade_value
            elif signal_type == "sell":
                cash += shares_held * price
                shares_held = 0

    print(f"\n  Done. {decisions_made} decision(s) logged to {LOG_PATH.relative_to(config.PROJECT_ROOT)}")


def main() -> None:
    print("\n  Human-Feedback Loop — Phase 3")
    print("  Decisions are saved to data/decisions.jsonl\n")

    print("  Available symbols:", ", ".join(config.SYMBOLS))
    symbol = input("  Which symbol? ").strip().upper()

    if symbol not in config.SYMBOLS:
        print(f"  '{symbol}' not in universe. Choose from {config.SYMBOLS}")
        sys.exit(1)

    run_symbol(symbol)


if __name__ == "__main__":
    main()
