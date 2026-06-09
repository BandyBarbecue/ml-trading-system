"""Demo: show the risk engine approving and vetoing trades on MSFT.

Run from the project root with:

    python -m src.risk.demo

We replay the MSFT backtest day by day. At each MA crossover signal, we ask
the risk engine whether the trade is allowed. The engine tracks portfolio value
as trades execute and vetoes anything that breaches a limit.

To see the kill switches fire, we use tighter limits than the defaults:
  - Max daily loss:  3%  (tighter than the real 5% to force a veto in demo data)
  - Max drawdown:    8%  (tighter than the real 10%)
"""

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.strategies.ma_crossover import ma_crossover_signals
from src.risk.engine import RiskEngine

INIT_CASH = 10_000
SHARES_PER_TRADE = 10  # fixed lot size to keep the demo simple


def main() -> None:
    csv_path = config.DATA_DIR / "MSFT.csv"
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close = df["close"]

    entries, exits = ma_crossover_signals(close)

    # Tighter limits so the demo data triggers at least one veto.
    risk = RiskEngine(
        starting_cash=INIT_CASH,
        max_position_pct=0.25,
        max_daily_loss_pct=0.03,   # 3% daily loss → stop
        max_drawdown_pct=0.08,     # 8% drawdown → stop
    )

    cash = float(INIT_CASH)
    shares_held = 0
    prev_date = None

    print(f"{'='*60}")
    print(f"  Risk Engine Demo — MSFT")
    print(f"  Capital: ${INIT_CASH:,}  |  Lot size: {SHARES_PER_TRADE} shares")
    print(f"  Daily loss limit: {risk.max_daily_loss_pct:.0%}  |  "
          f"Drawdown cap: {risk.max_drawdown_pct:.0%}  |  "
          f"Max position: {risk.max_position_pct:.0%}")
    print(f"{'='*60}\n")

    for date, price in close.items():
        portfolio_value = cash + shares_held * price

        # Tell the engine a new day has started so its daily counter resets.
        new_day = (prev_date is None) or (date.date() != prev_date)
        risk.update(portfolio_value, new_day=new_day)
        prev_date = date.date()

        # --- Buy signal ---
        if entries.get(date, False) and shares_held == 0:
            trade_value = SHARES_PER_TRADE * price
            approved, reason = risk.approve_trade(trade_value, portfolio_value)
            if approved:
                shares_held = SHARES_PER_TRADE
                cash -= trade_value
                print(f"  ✓ BUY   {date.date()}  {SHARES_PER_TRADE} shares @ ${price:.2f}  "
                      f"(portfolio: ${portfolio_value:,.2f})")
            else:
                print(f"  ✗ BUY VETOED  {date.date()} @ ${price:.2f}  → {reason}")

        # --- Sell signal ---
        elif exits.get(date, False) and shares_held > 0:
            trade_value = shares_held * price
            approved, reason = risk.approve_trade(trade_value, portfolio_value)
            if approved:
                cash += trade_value
                print(f"  ✓ SELL  {date.date()}  {shares_held} shares @ ${price:.2f}  "
                      f"(portfolio: ${portfolio_value:,.2f})")
                shares_held = 0
            else:
                print(f"  ✗ SELL VETOED  {date.date()} @ ${price:.2f}  → {reason}")

    # Final portfolio value.
    final_value = cash + shares_held * close.iloc[-1]
    total_return = (final_value - INIT_CASH) / INIT_CASH * 100
    print(f"\n{'='*60}")
    print(f"  Final portfolio value: ${final_value:,.2f}")
    print(f"  Total return:          {total_return:+.2f}%")
    print(f"  Peak drawdown:         {risk.current_drawdown:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
