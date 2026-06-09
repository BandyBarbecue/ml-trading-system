"""Risk engine: enforces hard trading limits independent of any strategy.

Think of this like a bouncer at a club. The strategy says "I want to hold
these weights." The risk engine checks a set of rules in order and either
passes the weights through unchanged, scales them down, or zeros everything
out. The strategy has no say — the engine's veto is final.

Controls applied in order
--------------------------
1. Kill switch      — daily loss too severe → zero all weights.
2. Drawdown cap     — portfolio too far below its peak → halve all weights.
3. VIX regime       — market fear is extreme → halve all weights.
4. Position veto    — any single weight over the per-stock limit → clip it.
5. Re-normalise     — make surviving weights sum back to 1.0.

Every triggered control is logged as a JSON line to decisions.jsonl.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make sure the project root is importable regardless of where pytest is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config


class RiskEngine:
    """Stateful risk gate for a single backtest run or paper-trading session.

    Parameters
    ----------
    log_path:
        Where to write decision logs (JSON lines).  Defaults to
        ``config.DATA_DIR / "decisions.jsonl"``.

    Usage
    -----
    At the start of each trading day call ``check()`` with the strategy's
    proposed weights.  After the day closes call ``update_peak()`` so the
    drawdown tracker advances correctly.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path: Path = (
            log_path if log_path is not None else config.DATA_DIR / "decisions.jsonl"
        )
        # Ensure the parent directory exists so the first write never fails.
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Peak portfolio value used for drawdown calculation.
        # Starts at 1.0 (i.e. we track fractions, not raw dollars — the
        # caller always passes in the actual current and peak dollar values).
        self._peak_value: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        proposed_weights: pd.Series,
        current_value: float,
        peak_value: float,
        vix: float,
        daily_pnl_pct: float,
    ) -> tuple[pd.Series, list[dict]]:
        """Apply all risk controls and return adjusted weights plus a log.

        Parameters
        ----------
        proposed_weights:
            Target portfolio weights indexed by ticker.  Should sum to 1.0.
        current_value:
            Current total portfolio value in dollars.
        peak_value:
            Highest portfolio value seen so far in dollars.
        vix:
            Current VIX level (e.g. 18.5 or 35.0).
        daily_pnl_pct:
            Today's P&L as a fraction of yesterday's close value
            (e.g. -0.025 means down 2.5% on the day).

        Returns
        -------
        adjusted_weights:
            pd.Series of weights after all risk adjustments.
        decisions:
            List of dicts describing each control that fired.  Each dict
            is also appended as a JSON line to ``log_path``.
        """
        # Work on a copy — never mutate the caller's Series.
        weights = proposed_weights.copy().astype(float)
        decisions: list[dict] = []
        kill_switch_fired = False

        # ----------------------------------------------------------
        # 1. Kill switch — daily loss too large → zero everything out
        # ----------------------------------------------------------
        # daily_pnl_pct is negative when we're down on the day.
        # The limit is stored as a positive fraction (e.g. 0.02 = 2%).
        if daily_pnl_pct <= -config.DAILY_LOSS_LIMIT:
            weights[:] = 0.0
            kill_switch_fired = True
            decisions.append(self._log({
                "action": "kill_switch",
                "daily_pnl_pct": daily_pnl_pct,
                "limit": config.DAILY_LOSS_LIMIT,
            }))

        # ----------------------------------------------------------
        # 2. Drawdown cap — portfolio too far below its peak
        # ----------------------------------------------------------
        # Drawdown is positive when we are below peak.
        if not kill_switch_fired:
            drawdown = (peak_value - current_value) / peak_value
            if drawdown >= config.MAX_DRAWDOWN:
                weights *= config.VIX_SCALE_FACTOR
                decisions.append(self._log({
                    "action": "drawdown_scale",
                    "drawdown": drawdown,
                    "scale_factor": config.VIX_SCALE_FACTOR,
                }))

        # ----------------------------------------------------------
        # 3. VIX regime filter — market is too fearful
        # ----------------------------------------------------------
        if not kill_switch_fired:
            if vix > config.VIX_REGIME_THRESHOLD:
                weights *= config.VIX_SCALE_FACTOR
                decisions.append(self._log({
                    "action": "vix_scale",
                    "vix": vix,
                    "threshold": config.VIX_REGIME_THRESHOLD,
                    "scale_factor": config.VIX_SCALE_FACTOR,
                }))

        # ----------------------------------------------------------
        # 4. Position limit veto — clip any over-weight single stock
        # ----------------------------------------------------------
        if not kill_switch_fired:
            for ticker in weights.index:
                original = float(weights[ticker])
                if original > config.MAX_POSITION_PCT:
                    weights[ticker] = config.MAX_POSITION_PCT
                    decisions.append(self._log({
                        "action": "position_veto",
                        "ticker": ticker,
                        "original": original,
                        "clipped_to": config.MAX_POSITION_PCT,
                    }))

        # ----------------------------------------------------------
        # 5. Re-normalise — surviving weights should sum to 1.0
        #    (skip if kill switch fired — all zeros, nothing to normalise)
        # ----------------------------------------------------------
        if not kill_switch_fired:
            total = weights.sum()
            if total > 0.0:
                weights = weights / total

        return weights, decisions

    def update_peak(self, current_value: float) -> None:
        """Advance the internal peak tracker.

        Call this once per trading day after the close so that the
        drawdown calculation stays current.

        Parameters
        ----------
        current_value:
            Today's closing portfolio value in dollars.
        """
        if current_value > self._peak_value:
            self._peak_value = current_value

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, payload: dict) -> dict:
        """Stamp payload with a UTC timestamp, write it to the log file,
        and return it so it can also be collected in the decisions list."""
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        return payload
