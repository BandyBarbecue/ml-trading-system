"""Tests for src/risk/engine.py

Each test checks one or more risk controls in isolation, plus one combined case.

Analogy: think of each test as checking one specific rule the bouncer enforces —
e.g. "does the bouncer actually turn away someone who is too drunk?" — without
worrying about the other rules at the same time.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so `import config` works.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.risk.engine import RiskEngine
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights(values: dict[str, float]) -> pd.Series:
    """Convenience: build a weight Series from a dict."""
    return pd.Series(values, dtype=float)


def _safe_inputs() -> dict:
    """Inputs that trigger no risk controls (all conditions green)."""
    return {
        "current_value": 100_000.0,
        "peak_value":    100_000.0,
        "vix":           20.0,           # below 30 threshold
        "daily_pnl_pct": 0.0,            # no loss today
    }


def _small_weights() -> pd.Series:
    """Equal weights summing to 1.0, all well below MAX_POSITION_PCT (0.08).
    Use these wherever the position veto should NOT fire.
    25 tickers × 0.04 = 1.0 total; each weight is half the 0.08 limit."""
    tickers = [f"T{i:02d}" for i in range(25)]
    return pd.Series({t: 0.04 for t in tickers}, dtype=float)


def _weights_with_fillers(named: dict[str, float], n_fillers: int = 20) -> pd.Series:
    """Build a weights Series where `named` tickers get the given (already
    under-cap) weights, and the rest of the budget is spread across small
    filler tickers so the whole thing sums to exactly 1.0 without any
    ticker exceeding MAX_POSITION_PCT.

    Since the risk engine now applies the position cap BEFORE the
    drawdown/VIX scaling (see engine.py), tests that want to isolate
    drawdown/VIX behaviour need weights that never trip the position cap in
    the first place — otherwise the position_veto control would also fire
    and the test would be checking two controls at once.
    """
    remainder = 1.0 - sum(named.values())
    filler_weight = remainder / n_fillers
    assert 0 <= filler_weight < config.MAX_POSITION_PCT, (
        "fillers must stay under the cap for this helper to isolate drawdown/VIX behaviour"
    )
    data = dict(named)
    data.update({f"FILL{i:02d}": filler_weight for i in range(n_fillers)})
    return pd.Series(data, dtype=float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKillSwitch:
    """Daily loss >= 2% fires the kill switch → all weights become 0."""

    def test_kill_switch_zeros_all_weights(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})
        inputs = _safe_inputs()
        inputs["daily_pnl_pct"] = -0.03   # down 3% — over the 2% limit

        adjusted, decisions = engine.check(weights, **inputs)

        assert (adjusted == 0.0).all(), "All weights must be 0 after kill switch"
        assert len(decisions) == 1
        assert decisions[0]["action"] == "kill_switch"

    def test_kill_switch_logs_to_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "decisions.jsonl"
        engine = RiskEngine(log_path=log_file)
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})
        inputs = _safe_inputs()
        inputs["daily_pnl_pct"] = -0.03

        engine.check(weights, **inputs)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["action"] == "kill_switch"
        assert "timestamp" in record

    def test_kill_switch_does_not_mutate_input(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})
        original = weights.copy()
        inputs = _safe_inputs()
        inputs["daily_pnl_pct"] = -0.03

        engine.check(weights, **inputs)

        pd.testing.assert_series_equal(weights, original)

    def test_exactly_at_limit_triggers_kill_switch(self, tmp_path: Path) -> None:
        """Boundary: daily_pnl_pct == -DAILY_LOSS_LIMIT should fire."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 1.0})
        inputs = _safe_inputs()
        inputs["daily_pnl_pct"] = -config.DAILY_LOSS_LIMIT   # exactly -0.02

        adjusted, decisions = engine.check(weights, **inputs)

        assert (adjusted == 0.0).all()
        assert decisions[0]["action"] == "kill_switch"

    def test_just_below_limit_does_not_trigger(self, tmp_path: Path) -> None:
        """daily_pnl_pct = -0.019 (< 0.02 limit) should NOT fire kill switch."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 1.0})
        inputs = _safe_inputs()
        inputs["daily_pnl_pct"] = -0.019   # just below the threshold

        _, decisions = engine.check(weights, **inputs)

        actions = [d["action"] for d in decisions]
        assert "kill_switch" not in actions


class TestDrawdownCap:
    """Drawdown >= 15% scales all weights by DRAWDOWN_SCALE_FACTOR (0.50) —
    and, crucially, does NOT renormalise back to 1.0 afterwards. The
    shortfall is meant to sit in cash; that's the whole point of de-risking."""

    def test_drawdown_scales_weights(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # AAPL/MSFT stay under the position cap (0.06, 0.04) so this test
        # isolates the drawdown control from the position cap.
        weights = _weights_with_fillers({"AAPL": 0.06, "MSFT": 0.04})
        inputs = _safe_inputs()
        # drawdown = (120_000 - 96_000) / 120_000 = 0.20 (20% > 15% threshold)
        inputs["peak_value"]    = 120_000.0
        inputs["current_value"] = 96_000.0

        adjusted, decisions = engine.check(weights, **inputs)

        actions = [d["action"] for d in decisions]
        assert "drawdown_scale" in actions
        assert "position_veto" not in actions
        # No renormalisation: output sum should equal DRAWDOWN_SCALE_FACTOR,
        # not 1.0. Relative ratios between tickers are preserved.
        assert abs(adjusted.sum() - config.DRAWDOWN_SCALE_FACTOR) < 1e-9
        assert abs(adjusted["AAPL"] - 0.06 * config.DRAWDOWN_SCALE_FACTOR) < 1e-9
        assert abs(adjusted["MSFT"] - 0.04 * config.DRAWDOWN_SCALE_FACTOR) < 1e-9

    def test_drawdown_decision_logged(self, tmp_path: Path) -> None:
        log_file = tmp_path / "decisions.jsonl"
        engine = RiskEngine(log_path=log_file)
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})
        inputs = _safe_inputs()
        inputs["peak_value"]    = 120_000.0
        inputs["current_value"] = 96_000.0   # 20% drawdown

        engine.check(weights, **inputs)

        records = [json.loads(decision) for decision in log_file.read_text().strip().splitlines()]
        actions = [r["action"] for r in records]
        assert "drawdown_scale" in actions

    def test_drawdown_at_exact_threshold(self, tmp_path: Path) -> None:
        """Boundary: drawdown == MAX_DRAWDOWN (0.15) should trigger."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 1.0})
        inputs = _safe_inputs()
        # drawdown = 0.15 exactly
        inputs["peak_value"]    = 100_000.0
        inputs["current_value"] = 85_000.0

        _, decisions = engine.check(weights, **inputs)

        assert any(d["action"] == "drawdown_scale" for d in decisions)


class TestVixScale:
    """VIX > 30 scales all weights by VIX_SCALE_FACTOR (0.50), with no
    renormalisation afterwards — the scaled-down remainder is cash."""

    def test_high_vix_scales_weights(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # AAPL/MSFT stay under the position cap so this test isolates the
        # VIX control from the position cap.
        weights = _weights_with_fillers({"AAPL": 0.05, "MSFT": 0.05})
        inputs = _safe_inputs()
        inputs["vix"] = 35.0   # above 30 threshold

        adjusted, decisions = engine.check(weights, **inputs)

        actions = [d["action"] for d in decisions]
        assert "vix_scale" in actions
        assert "position_veto" not in actions
        # No renormalisation: sum should equal VIX_SCALE_FACTOR, not 1.0.
        assert abs(adjusted.sum() - config.VIX_SCALE_FACTOR) < 1e-9
        assert abs(adjusted["AAPL"] - 0.05 * config.VIX_SCALE_FACTOR) < 1e-9
        assert abs(adjusted["MSFT"] - 0.05 * config.VIX_SCALE_FACTOR) < 1e-9

    def test_vix_exactly_at_threshold_does_not_trigger(self, tmp_path: Path) -> None:
        """VIX == 30 is NOT strictly greater than 30 → should not fire."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})
        inputs = _safe_inputs()
        inputs["vix"] = 30.0   # exactly at threshold

        _, decisions = engine.check(weights, **inputs)

        actions = [d["action"] for d in decisions]
        assert "vix_scale" not in actions

    def test_vix_just_above_threshold_triggers(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 1.0})
        inputs = _safe_inputs()
        inputs["vix"] = 30.01

        _, decisions = engine.check(weights, **inputs)

        assert any(d["action"] == "vix_scale" for d in decisions)


class TestPositionVeto:
    """Any weight > MAX_POSITION_PCT (0.08) is clipped to 0.08."""

    def test_oversized_position_clipped(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # AAPL weight 0.12 is over the 0.08 limit
        weights = _weights({"AAPL": 0.12, "MSFT": 0.50, "JNJ": 0.38})
        inputs = _safe_inputs()

        adjusted, decisions = engine.check(weights, **inputs)

        assert any(d["action"] == "position_veto" for d in decisions)
        veto = next(d for d in decisions if d["action"] == "position_veto")
        assert veto["ticker"] == "AAPL"
        assert abs(veto["original"] - 0.12) < 1e-9
        # After clip + renorm, AAPL's share should be <= MAX_POSITION_PCT
        # (renorm may push it slightly over, but the clip happened first)
        assert veto["clipped_to"] == config.MAX_POSITION_PCT

    def test_multiple_vetoes_logged(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # AAPL and MSFT both over the 0.08 limit; 30 filler tickers are
        # small enough (0.02 each) that even after absorbing their share of
        # the redistributed excess they stay comfortably under the cap.
        # (With only a couple of large "safe" recipients, redistribution can
        # itself push them over the cap and trigger a second-round veto —
        # that's correct engine behaviour, but it would muddy this test, so
        # we spread the excess across many small recipients instead.)
        fillers = {f"T{i:02d}": 0.02 for i in range(30)}  # 30 * 0.02 = 0.60
        weights = _weights({"AAPL": 0.20, "MSFT": 0.20, **fillers})
        assert abs(weights.sum() - 1.0) < 1e-9
        inputs = _safe_inputs()

        _, decisions = engine.check(weights, **inputs)

        veto_actions = [d for d in decisions if d["action"] == "position_veto"]
        assert len(veto_actions) == 2
        assert {v["ticker"] for v in veto_actions} == {"AAPL", "MSFT"}

    def test_weight_exactly_at_limit_not_vetoed(self, tmp_path: Path) -> None:
        """Weight exactly equal to MAX_POSITION_PCT (0.08) should pass through."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # All weights at or below 0.08 — none should be vetoed.
        # 12 × 0.08 = 0.96, plus two at 0.02 each = 1.0 total.
        weights = pd.Series({f"T{i:02d}": 0.08 for i in range(12)}, dtype=float)
        weights["T12"] = 0.02
        weights["T13"] = 0.02
        inputs = _safe_inputs()

        _, decisions = engine.check(weights, **inputs)

        actions = [d["action"] for d in decisions]
        assert "position_veto" not in actions


class TestNoAction:
    """All conditions safe → weights unchanged, no decisions logged."""

    def test_no_action_when_all_safe(self, tmp_path: Path) -> None:
        log_file = tmp_path / "decisions.jsonl"
        engine = RiskEngine(log_path=log_file)
        # Use weights that are all well below MAX_POSITION_PCT (0.08).
        weights = _small_weights()
        inputs = _safe_inputs()

        adjusted, decisions = engine.check(weights, **inputs)

        assert decisions == [], "No risk controls should have fired"
        assert not log_file.exists() or log_file.read_text() == ""
        # Weights should be unchanged (renorm of weights that already sum to 1
        # leaves them identical).
        pd.testing.assert_series_equal(adjusted, weights)


class TestCombinedVixAndDrawdown:
    """Both VIX>30 AND drawdown>15% → weights scaled by 0.25 (0.5 × 0.5)."""

    def test_combined_scaling_logs_both_decisions(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})

        adjusted, decisions = engine.check(
            proposed_weights=weights,
            current_value=80_000.0,   # drawdown = (100k - 80k) / 100k = 20% > 15%
            peak_value=100_000.0,
            vix=35.0,                 # above 30 threshold
            daily_pnl_pct=0.0,        # no daily loss
        )

        actions = [d["action"] for d in decisions]
        assert "drawdown_scale" in actions, "Drawdown scale must be logged"
        assert "vix_scale" in actions, "VIX scale must be logged"

    def test_combined_scaling_order(self, tmp_path: Path) -> None:
        """Drawdown scale fires first, then VIX scale — order matters."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})

        _, decisions = engine.check(
            proposed_weights=weights,
            current_value=80_000.0,
            peak_value=100_000.0,
            vix=35.0,
            daily_pnl_pct=0.0,
        )

        actions = [d["action"] for d in decisions]
        drawdown_idx = actions.index("drawdown_scale")
        vix_idx = actions.index("vix_scale")
        assert drawdown_idx < vix_idx, "drawdown_scale must be logged before vix_scale"

    def test_combined_net_scale_is_quarter(self, tmp_path: Path) -> None:
        """0.5 x 0.5 = 0.25 effective scale, and NO renormalisation happens
        afterwards — this is the proven-bug scenario from the audit: with the
        old code, renormalising here undid both scale-downs and returned the
        original weights unchanged. The fix must leave 75% of the portfolio
        in cash."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # Equal weights, kept under the position cap so this test isolates
        # drawdown+VIX scaling from the position cap.
        weights = _weights_with_fillers({"AAPL": 0.05, "MSFT": 0.05})

        adjusted, decisions = engine.check(
            proposed_weights=weights,
            current_value=80_000.0,
            peak_value=100_000.0,
            vix=35.0,
            daily_pnl_pct=0.0,
        )

        actions = [d["action"] for d in decisions]
        assert "position_veto" not in actions
        # DRAWDOWN_SCALE_FACTOR x VIX_SCALE_FACTOR = 0.5 x 0.5 = 0.25 combined scale.
        expected_each = 0.05 * config.DRAWDOWN_SCALE_FACTOR * config.VIX_SCALE_FACTOR
        assert abs(adjusted.sum() - 0.25) < 1e-9, "Sum must be 0.25, NOT renormalised to 1.0"
        assert abs(adjusted["AAPL"] - expected_each) < 1e-9
        assert abs(adjusted["MSFT"] - expected_each) < 1e-9


class TestUpdatePeak:
    """update_peak() advances internal peak tracker correctly."""

    def test_peak_advances_on_new_high(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        engine.update_peak(110_000.0)
        assert engine._peak_value == 110_000.0

    def test_peak_does_not_retreat(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        engine.update_peak(110_000.0)
        engine.update_peak(95_000.0)   # portfolio dropped — peak must stay
        assert engine._peak_value == 110_000.0


class TestPositionCapRedistribution:
    """Position-cap excess is redistributed to tickers that still have room
    under the cap. When there IS room, total portfolio value is conserved
    (sum stays 1.0). When there is NO room (everyone is already at the cap),
    the excess is simply left unallocated as cash — it is NOT forced back in
    via a final renormalisation (that was the bug: renormalising here would
    silently push weights back over the cap, defeating the whole control)."""

    def test_excess_redistributed_when_room_available(self, tmp_path: Path) -> None:
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        # One oversized ticker (BIG=0.16) plus 24 small ones with plenty of
        # headroom under the 0.08 cap to absorb BIG's excess.
        weights = _weights({f"T{i:02d}": 0.035 for i in range(24)})
        weights["BIG"] = 1.0 - float(weights.sum())  # tops up to exactly 1.0 -> 0.16
        inputs = _safe_inputs()

        adjusted, decisions = engine.check(weights, **inputs)

        assert any(d["action"] == "position_veto" for d in decisions)
        assert (adjusted <= config.MAX_POSITION_PCT + 1e-9).all()
        # There was room to redistribute BIG's excess, so total value is
        # conserved -- nothing needs to be parked in cash for this case.
        assert abs(adjusted.sum() - 1.0) < 1e-9

    def test_excess_left_as_cash_when_no_room(self, tmp_path: Path) -> None:
        """All three tickers end up at the cap with nowhere to redistribute
        to, so the clipped total legitimately sums to less than 1.0. That
        shortfall is cash, and must NOT be re-inflated by renormalising."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"A": 0.15, "B": 0.15, "C": 0.70})
        inputs = _safe_inputs()

        adjusted, _ = engine.check(weights, **inputs)

        assert (adjusted <= config.MAX_POSITION_PCT + 1e-9).all()
        assert abs(adjusted.sum() - 3 * config.MAX_POSITION_PCT) < 1e-9

    def test_kill_switch_keeps_zeros(self, tmp_path: Path) -> None:
        """Kill switch must NOT renormalise (zeros must stay zero)."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.5})
        inputs = _safe_inputs()
        inputs["daily_pnl_pct"] = -0.05

        adjusted, _ = engine.check(weights, **inputs)

        assert (adjusted == 0.0).all()
        assert adjusted.sum() == 0.0


class TestInvariants:
    """Cross-cutting invariants the redesigned check() must hold in every
    scenario -- these are the properties the old renormalise-at-the-end bug
    violated (see the module docstring for the proof case)."""

    def test_i_max_weight_never_exceeds_cap(self, tmp_path: Path) -> None:
        """No matter what combination of controls fires, no output weight
        may exceed MAX_POSITION_PCT."""
        scenarios = [
            # (weights, current_value, peak_value, vix, daily_pnl_pct)
            (_weights({"AAPL": 0.5, "MSFT": 0.3, "KO": 0.2}), 100_000, 100_000, 20.0, 0.0),
            (_weights({"AAPL": 0.5, "MSFT": 0.3, "KO": 0.2}), 80_000, 100_000, 35.0, 0.0),
            (_weights({"AAPL": 1.0}), 96_000, 120_000, 35.0, -0.019),
            (_small_weights(), 100_000, 100_000, 20.0, 0.0),
            (_weights({"A": 0.15, "B": 0.15, "C": 0.70}), 100_000, 100_000, 20.0, 0.0),
        ]
        for weights, current_value, peak_value, vix, daily_pnl_pct in scenarios:
            engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
            adjusted, _ = engine.check(
                proposed_weights=weights,
                current_value=current_value,
                peak_value=peak_value,
                vix=vix,
                daily_pnl_pct=daily_pnl_pct,
            )
            assert (adjusted <= config.MAX_POSITION_PCT + 1e-9).all(), (
                f"Weight exceeded cap for scenario vix={vix}, dd_input=({peak_value},{current_value}): "
                f"{adjusted[adjusted > config.MAX_POSITION_PCT]}"
            )

    def test_ii_drawdown_alone_sum_equals_scale_factor(self, tmp_path: Path) -> None:
        """When only the drawdown control fires, sum(output) ~= DRAWDOWN_SCALE_FACTOR."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _small_weights()  # already under cap, sums to 1.0
        adjusted, decisions = engine.check(
            proposed_weights=weights,
            current_value=96_000.0,
            peak_value=120_000.0,   # 20% drawdown > 15% threshold
            vix=20.0,               # below VIX threshold
            daily_pnl_pct=0.0,
        )
        actions = [d["action"] for d in decisions]
        assert "drawdown_scale" in actions
        assert "vix_scale" not in actions
        assert abs(adjusted.sum() - config.DRAWDOWN_SCALE_FACTOR) < 1e-9

    def test_iii_drawdown_and_vix_sum_equals_quarter(self, tmp_path: Path) -> None:
        """Both controls firing together -> sum(output) ~= 0.25 (0.5 x 0.5)."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _small_weights()
        adjusted, decisions = engine.check(
            proposed_weights=weights,
            current_value=96_000.0,
            peak_value=120_000.0,   # 20% drawdown
            vix=35.0,               # above threshold
            daily_pnl_pct=0.0,
        )
        actions = [d["action"] for d in decisions]
        assert "drawdown_scale" in actions
        assert "vix_scale" in actions
        assert abs(adjusted.sum() - 0.25) < 1e-9

    def test_iv_repro_benign_conditions_still_caps_oversized_weights(self, tmp_path: Path) -> None:
        """The audit's repro case: {AAPL:.5, MSFT:.3, KO:.2} under otherwise
        benign conditions -- every output weight must be <= MAX_POSITION_PCT
        (0.08), not left untouched as the old renormalise-at-the-end bug did."""
        engine = RiskEngine(log_path=tmp_path / "decisions.jsonl")
        weights = _weights({"AAPL": 0.5, "MSFT": 0.3, "KO": 0.2})
        inputs = _safe_inputs()  # benign: vix=20, no drawdown, no daily loss

        adjusted, _ = engine.check(weights, **inputs)

        assert (adjusted <= config.MAX_POSITION_PCT + 1e-9).all(), (
            f"Every weight must be <= {config.MAX_POSITION_PCT}: {adjusted}"
        )
