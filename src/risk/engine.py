"""Risk engine: enforces hard trading limits independent of any strategy.

Think of this like a bouncer at a club. The strategy says "I want to hold
these weights." The risk engine checks a set of rules in order and either
passes the weights through unchanged, scales them down, or zeros everything
out. The strategy has no say — the engine's veto is final.

Controls applied in order
--------------------------
1. Kill switch      — daily loss too severe → zero all weights.
2. Normalise        — proposed weights are scaled to sum to 1.0 (long-only
                       only — see the note beside the guard below) so every
                       control after this point works off a consistent budget.
3. Position cap     — no single weight may exceed MAX_POSITION_PCT. Excess is
                       redistributed to tickers still under the cap, iterating
                       until nothing is left to redistribute (or a safety
                       iteration limit), then a final hard clip.
4. Drawdown cap     — portfolio too far below its peak → scale ALL weights by
                       DRAWDOWN_SCALE_FACTOR.
5. VIX regime       — market fear is extreme → scale ALL weights by
                       VIX_SCALE_FACTOR.
6. Gross exposure cap (long-short only) — cap long+|short| exposure.

Critically, steps 4 and 5 are applied LAST and are NEVER renormalised back to
sum to 1.0 afterwards. A sum < 1.0 after de-risking means the remainder sits
in cash — that is the entire point of a drawdown/VIX scale-down. An earlier
version of this engine renormalised at the very end, which silently undid
every scale-down and clip above it (proof: 20% drawdown + VIX 35 + an
oversized position produced output weights identical to the input). Don't
reintroduce a final renormalisation.

Every triggered control is logged as a JSON line to decisions.jsonl, all
using the "action" key (previously the gross-exposure-cap entry used
"control"/"triggered" instead, which made the log schema inconsistent).
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

        if not kill_switch_fired:
            # ------------------------------------------------------
            # 2. Normalise proposed weights to sum to 1.0 — FIRST, before
            #    any cap or scale, so those controls act on a consistent
            #    1.0 budget rather than whatever the strategy handed us.
            #
            #    Skipped for long-short portfolios. In a mixed long/short
            #    book, weights.sum() equals (long exposure - short exposure)
            #    which is the *net* exposure, not the gross. Dividing by a
            #    net < 1 inflates every absolute weight and can push gross
            #    exposure above GROSS_EXPOSURE_CAP, defeating step 6.
            #
            #    Example: 120% long + 30% short → sum = 0.90, gross = 1.50.
            #    Dividing by 0.90 → gross becomes 1.667, exceeding a 1.60 cap.
            #
            #    The gross-exposure cap in step 6 already handles scaling for
            #    long-short portfolios, so this step is skipped for them.
            # ------------------------------------------------------
            if not config.LONG_SHORT_ENABLED:
                total = float(weights.sum())
                if total > 0.0:
                    weights = weights / total

            # ------------------------------------------------------
            # 3. Position limit veto — clip any over-weight single stock,
            #    redistributing the clipped excess to tickers still under
            #    the cap (iterating so redistribution itself can't create a
            #    new breach), then a final hard safety clip.
            # ------------------------------------------------------
            weights = self._apply_position_cap(weights, decisions)

            # ------------------------------------------------------
            # 4. Drawdown cap — portfolio too far below its peak.
            #    Drawdown is positive when we are below peak.
            #    Multiplicative scale, no renormalisation after — the
            #    shortfall is intentionally left un-invested (held as cash).
            # ------------------------------------------------------
            drawdown = (peak_value - current_value) / peak_value
            if drawdown >= config.MAX_DRAWDOWN:
                weights = weights * config.DRAWDOWN_SCALE_FACTOR
                decisions.append(self._log({
                    "action": "drawdown_scale",
                    "drawdown": drawdown,
                    "scale_factor": config.DRAWDOWN_SCALE_FACTOR,
                }))

            # ------------------------------------------------------
            # 5. VIX regime filter — market is too fearful.
            #    Same rule: multiplicative, no renormalisation after.
            # ------------------------------------------------------
            if vix > config.VIX_REGIME_THRESHOLD:
                weights = weights * config.VIX_SCALE_FACTOR
                decisions.append(self._log({
                    "action": "vix_scale",
                    "vix": vix,
                    "threshold": config.VIX_REGIME_THRESHOLD,
                    "scale_factor": config.VIX_SCALE_FACTOR,
                }))

            # ------------------------------------------------------
            # 6. Gross exposure cap (only when long-short is enabled)
            #    Gross exposure = sum of absolute weights (long + |short|).
            #    If the combined exposure would exceed GROSS_EXPOSURE_CAP,
            #    scale every weight down proportionally so the cap is met
            #    exactly. No renormalisation follows this either.
            # ------------------------------------------------------
            if config.LONG_SHORT_ENABLED:
                gross = float(weights.abs().sum())
                cap   = config.GROSS_EXPOSURE_CAP
                if gross > cap:
                    scale = cap / gross
                    weights = (weights * scale).round(6)
                    decisions.append(self._log({
                        "action":  "gross_exposure_cap",
                        "gross":   round(gross, 4),
                        "scale":   round(scale, 4),
                        "cap":     cap,
                    }))

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

    def _apply_position_cap(self, weights: pd.Series, decisions: list[dict]) -> pd.Series:
        """Clip any weight over MAX_POSITION_PCT and redistribute the excess.

        Analogy: think of the cap as a glass that can only hold so much
        water (8%). Whatever overflows one glass gets poured into glasses
        that still have room, not just thrown away — but a glass that's
        already full can't take on more, so we keep pouring in rounds until
        either everything fits or every glass is completely full (at which
        point the remainder has nowhere to go and is simply left unassigned
        — that money is later left in cash rather than force-fit anywhere).

        A single pass isn't enough: redistributing excess to tickers under
        the cap can itself push one of them over, so we iterate until
        nothing is left over the cap (or a safety limit is hit), then apply
        one final hard clip so the invariant "nothing exceeds the cap" holds
        even if float drift or the iteration limit left a sliver over.
        """
        w = weights.copy()
        cap = config.MAX_POSITION_PCT
        # Tiny tolerance: the normalise-to-1.0 division a few lines up in
        # check() can turn a weight that was exactly `cap` into something
        # like cap + 4e-17 through ordinary float rounding. Without this
        # slack that harmless dust would trip a veto that shouldn't fire.
        tol = 1e-9

        for _ in range(20):
            over_mask = w > cap + tol
            if not over_mask.any():
                break

            # Log one decision per ticker the moment it's found over cap.
            # A ticker set to exactly `cap` can never re-enter over_mask in a
            # later round (redistribution only adds to tickers still under
            # cap), so this can't double-log the same ticker.
            for ticker in w.index[over_mask]:
                decisions.append(self._log({
                    "action": "position_veto",
                    "ticker": ticker,
                    "original": float(w[ticker]),
                    "clipped_to": cap,
                }))

            excess = float((w[over_mask] - cap).sum())
            w[over_mask] = cap

            under_mask = w < cap
            under_total = float(w[under_mask].sum())
            if under_total > 0:
                # Spread the excess proportionally to current weight among
                # tickers still under the cap.
                w[under_mask] += excess * (w[under_mask] / under_total)
            else:
                # Every ticker is already at the cap — nowhere to put the
                # excess. It's simply not allocated (effectively cash).
                break

        # Final hard safety clip: guards against float drift or hitting the
        # iteration limit while something is still (barely) over the cap.
        w[w > cap] = cap
        return w

    def _log(self, payload: dict) -> dict:
        """Stamp payload with a UTC timestamp, write it to the log file,
        and return it so it can also be collected in the decisions list."""
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        return payload
