"""Risk engine: enforces hard trading limits independent of any strategy.

Think of this like a bouncer at a club. The strategy says "I want to make this
trade." The risk engine checks the rules and either waves it through or says no.
The strategy has no say in the final decision — the engine's veto is absolute.

Limits modelled on FTMO prop-firm rules (a standard benchmark for serious traders):
  - Max position size:  no single trade > 25% of portfolio value
  - Daily loss limit:   kill switch fires if down 5% on the day
  - Max drawdown cap:   kill switch fires if down 10% from the all-time portfolio peak
"""

from dataclasses import dataclass, field


@dataclass
class RiskEngine:
    """Stateful risk gate. Instantiate once per backtest run (or per trading session).

    Attributes:
        starting_cash:        The portfolio value at the start of the session.
        max_position_pct:     Largest fraction of portfolio allowed in one trade (0–1).
        max_daily_loss_pct:   Daily loss that triggers the kill switch (0–1).
        max_drawdown_pct:     Drawdown from peak that triggers the kill switch (0–1).
    """
    starting_cash: float
    max_position_pct: float = 0.25   # 25% of portfolio per trade
    max_daily_loss_pct: float = 0.05  # 5% daily loss → stop
    max_drawdown_pct: float = 0.10    # 10% from peak → stop

    # Internal state — don't set these manually.
    _peak_value: float = field(init=False)
    _day_start_value: float = field(init=False)

    def __post_init__(self) -> None:
        self._peak_value = self.starting_cash
        self._day_start_value = self.starting_cash

    def approve_trade(self, trade_value: float, portfolio_value: float) -> tuple[bool, str]:
        """Decide whether a proposed trade is allowed.

        Args:
            trade_value:     The dollar amount of the proposed trade (e.g. price × shares).
            portfolio_value: Current total portfolio value (cash + open positions).

        Returns:
            (True, "approved") if all limits pass.
            (False, <reason>) if any limit is breached — reason explains which one.
        """
        # --- 1. Position size ---
        position_pct = trade_value / portfolio_value
        if position_pct > self.max_position_pct:
            return False, (
                f"POSITION SIZE: trade is {position_pct:.1%} of portfolio "
                f"(limit {self.max_position_pct:.1%})"
            )

        # --- 2. Daily loss kill switch ---
        # How much have we lost *today* as a fraction of where we started the day?
        daily_loss_pct = (self._day_start_value - portfolio_value) / self._day_start_value
        if daily_loss_pct >= self.max_daily_loss_pct:
            return False, (
                f"DAILY LOSS KILL SWITCH: down {daily_loss_pct:.1%} today "
                f"(limit {self.max_daily_loss_pct:.1%})"
            )

        # --- 3. Max drawdown cap ---
        # How far are we below the highest value the portfolio has ever reached?
        drawdown_pct = (self._peak_value - portfolio_value) / self._peak_value
        if drawdown_pct >= self.max_drawdown_pct:
            return False, (
                f"MAX DRAWDOWN: down {drawdown_pct:.1%} from peak "
                f"(limit {self.max_drawdown_pct:.1%})"
            )

        return True, "approved"

    def update(self, portfolio_value: float, new_day: bool = False) -> None:
        """Update internal state. Call this at the end of each bar.

        Args:
            portfolio_value: Current portfolio value after the bar closes.
            new_day:         Set True on the first bar of a new trading day
                             so the daily loss counter resets.
        """
        if new_day:
            self._day_start_value = portfolio_value

        # Peak only ever moves up — it never resets.
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak as a fraction (e.g. 0.08 = 8%)."""
        return (self._peak_value - self._day_start_value) / self._peak_value
