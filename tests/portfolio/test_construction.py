"""
Tests for src/portfolio/construction.py
========================================
Each test is deliberately narrow so failures are easy to diagnose.
"""

import numpy as np
import pandas as pd
import pytest
import config
from src.portfolio.construction import build_portfolio, compute_trades


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_returns(n_tickers: int = 25, n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily returns suitable for HRP.

    We need at least as many rows as tickers for a well-conditioned covariance
    matrix, so default to 100 days for 25 tickers.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    data = rng.standard_normal((n_days, n_tickers))
    return pd.DataFrame(data, columns=tickers)


def make_scores(returns: pd.DataFrame) -> pd.Series:
    """Assign scores 1..N (highest score = first ticker)."""
    n = len(returns.columns)
    return pd.Series(
        range(n, 0, -1),  # T00 gets score N, T24 gets score 1
        index=returns.columns,
        name="score",
        dtype=float,
    )


def make_sector_map(returns: pd.DataFrame, sector: str = "Tech") -> dict[str, str]:
    """Map every ticker in `returns` to a single sector (for cap tests)."""
    return {t: sector for t in returns.columns}


# ---------------------------------------------------------------------------
# test_top_n_selected
# ---------------------------------------------------------------------------

def test_top_n_selected():
    """With 50 scored tickers, only the top 25 appear in the output."""
    n_total = 50
    rng = np.random.default_rng(0)
    tickers = [f"T{i:02d}" for i in range(n_total)]
    returns = pd.DataFrame(
        rng.standard_normal((100, n_total)),
        columns=tickers,
    )
    # Score tickers so T00 is best … T49 is worst
    scores = pd.Series(
        range(n_total, 0, -1),
        index=tickers,
        dtype=float,
    )
    sector_map = {t: "Tech" for t in tickers}

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    # Only top-25 tickers should have non-zero weight
    assert len(target) == config.N_POSITIONS
    # The selected tickers must be the top-25 by score
    expected_tickers = set(scores.nlargest(config.N_POSITIONS).index)
    assert set(target.index) == expected_tickers


# ---------------------------------------------------------------------------
# test_sector_cap
# ---------------------------------------------------------------------------

def test_sector_cap():
    """All tickers in one sector: no individual sector weight exceeds MAX_SECTOR_PCT."""
    returns = make_returns()  # 25 tickers, 100 days
    scores = make_scores(returns)
    # Put every ticker in the same sector to guarantee the cap will trigger
    sector_map = {t: "SingleSector" for t in returns.columns}

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    # With 25 equally-eligible tickers in one sector, each must be <= MAX_SECTOR_PCT
    assert (target <= config.MAX_SECTOR_PCT + 1e-9).all(), (
        f"Some positions exceed MAX_SECTOR_PCT={config.MAX_SECTOR_PCT}:\n{target[target > config.MAX_SECTOR_PCT]}"
    )


# ---------------------------------------------------------------------------
# test_position_cap
# ---------------------------------------------------------------------------

def test_position_cap(monkeypatch):
    """Even if HRP naturally assigns >8% to one ticker, it gets clipped."""
    returns = make_returns()
    scores = make_scores(returns)
    # Different sectors so the sector cap doesn't obscure the position cap test
    sector_map = {t: f"Sector{i}" for i, t in enumerate(returns.columns)}

    # Force HRP to assign a huge weight to the first ticker by making its
    # variance tiny (almost flat) and all others very volatile.
    # We do this by patching hrp_weights inside construction.
    import src.portfolio.construction as cons

    original_hrp = cons.hrp_weights

    def biased_hrp(ret: pd.DataFrame) -> pd.Series:
        """Return a weight series with the first ticker given 50% weight."""
        tickers = ret.columns.tolist()
        w = pd.Series(1.0 / len(tickers), index=tickers)
        w.iloc[0] = 0.50
        remainder = 0.50 / (len(tickers) - 1)
        w.iloc[1:] = remainder
        return w / w.sum()

    monkeypatch.setattr(cons, "hrp_weights", biased_hrp)

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    assert (target <= config.MAX_POSITION_PCT + 1e-9).all(), (
        f"Some positions exceed MAX_POSITION_PCT={config.MAX_POSITION_PCT}:\n{target[target > config.MAX_POSITION_PCT]}"
    )


# ---------------------------------------------------------------------------
# test_weights_sum_to_one
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    """Output weights must sum to exactly 1.0 (within floating-point tolerance)."""
    returns = make_returns()
    scores = make_scores(returns)
    sector_map = make_sector_map(returns)

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    assert abs(target.sum() - 1.0) < 1e-9, f"Weights sum to {target.sum():.10f}, not 1.0"


# ---------------------------------------------------------------------------
# test_turnover_cap
# ---------------------------------------------------------------------------

def test_turnover_cap():
    """When the unconstrained target would exceed 40% turnover, cap is enforced."""
    returns = make_returns()
    scores = make_scores(returns)
    sector_map = {t: f"Sector{i}" for i, t in enumerate(returns.columns)}

    # First build a portfolio with no existing positions (first day)
    target_first = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    # Now simulate a large shock: shuffle scores so the new ranking is completely
    # different, which would normally force >40% turnover
    shuffled_scores = scores.iloc[::-1].values  # reverse: T24 is now best
    new_scores = pd.Series(shuffled_scores, index=scores.index, dtype=float)

    target_second = build_portfolio(
        scores=new_scores,
        sector_map=sector_map,
        current_weights=target_first,
        prices=returns,
    )

    # Compute actual two-way turnover
    all_tickers = target_first.index.union(target_second.index)
    t1 = target_first.reindex(all_tickers, fill_value=0.0)
    t2 = target_second.reindex(all_tickers, fill_value=0.0)
    actual_turnover = (t2 - t1).abs().sum() / 2.0

    assert actual_turnover <= config.TURNOVER_CAP + 1e-9, (
        f"Turnover {actual_turnover:.4f} exceeds cap {config.TURNOVER_CAP}"
    )
    # Sanity: weights still sum to 1
    assert abs(target_second.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# test_compute_trades
# ---------------------------------------------------------------------------

def test_compute_trades():
    """compute_trades returns the correct signed delta per ticker."""
    target = pd.Series({"AAPL": 0.10, "MSFT": 0.20, "JNJ": 0.05})
    current = pd.Series({"AAPL": 0.08, "MSFT": 0.25})  # JNJ is a new buy, MSFT is a sell

    trades = compute_trades(target, current)

    # AAPL: 0.10 - 0.08 = +0.02 (buy)
    assert abs(trades["AAPL"] - 0.02) < 1e-9
    # MSFT: 0.20 - 0.25 = -0.05 (sell)
    assert abs(trades["MSFT"] - (-0.05)) < 1e-9
    # JNJ: 0.05 - 0.00 = +0.05 (new buy, was not in current)
    assert abs(trades["JNJ"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Invariant tests on adversarial inputs
# ---------------------------------------------------------------------------
# These target the "cap-then-renormalise" defect class: each control used to
# be undone by a later renormalisation (e.g. a 60% tech sector "capped" at
# 30% came out at 42.9%). The invariants below must hold on inputs built to
# make each cap bind hard.

def _turnover(a: pd.Series, b: pd.Series) -> float:
    """Two-way turnover between two weight Series (union-aligned)."""
    idx = a.index.union(b.index)
    return float(
        (a.reindex(idx, fill_value=0.0) - b.reindex(idx, fill_value=0.0)).abs().sum()
    ) / 2.0


def test_invariant_sector_cap_holds_under_concentration(monkeypatch):
    """Adversarial: HRP hands 60% of the book to one 5-ticker sector.
    After construction, EVERY sector must be <= MAX_SECTOR_PCT + 1e-9 —
    the proven bug produced 42.9% for a '30% capped' sector."""
    returns = make_returns()  # 25 tickers
    scores = make_scores(returns)
    tickers = returns.columns.tolist()
    # 5 sectors x 5 tickers: feasible under a 30% cap (5 x 30% = 150% >= 100%)
    sector_map = {t: f"Sector{i // 5}" for i, t in enumerate(tickers)}

    import src.portfolio.construction as cons

    def concentrated_hrp(ret: pd.DataFrame) -> pd.Series:
        """60% into Sector0's five tickers, 40% spread across the rest."""
        cols = ret.columns.tolist()
        w = pd.Series(0.0, index=cols)
        sector0 = [t for t in cols if sector_map[t] == "Sector0"]
        others = [t for t in cols if t not in sector0]
        w[sector0] = 0.60 / len(sector0)
        w[others] = 0.40 / len(others)
        return w

    monkeypatch.setattr(cons, "hrp_weights", concentrated_hrp)

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    for sector in set(sector_map.values()):
        sector_total = target[[t for t in target.index if sector_map[t] == sector]].sum()
        assert sector_total <= config.MAX_SECTOR_PCT + 1e-9, (
            f"{sector} = {sector_total:.6f} exceeds cap {config.MAX_SECTOR_PCT}"
        )
    # Every individual position must also respect its cap
    assert (target <= config.MAX_POSITION_PCT + 1e-9).all()


def test_invariant_position_cap_holds_with_extreme_concentration(monkeypatch):
    """Adversarial: HRP gives 80% to a single ticker. Output must never
    have any position above MAX_POSITION_PCT + 1e-9."""
    returns = make_returns()
    scores = make_scores(returns)
    sector_map = {t: f"Sector{i}" for i, t in enumerate(returns.columns)}

    import src.portfolio.construction as cons

    def extreme_hrp(ret: pd.DataFrame) -> pd.Series:
        cols = ret.columns.tolist()
        w = pd.Series(0.20 / (len(cols) - 1), index=cols)
        w.iloc[0] = 0.80
        return w / w.sum()

    monkeypatch.setattr(cons, "hrp_weights", extreme_hrp)

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=pd.Series(dtype=float),
        prices=returns,
    )

    assert (target <= config.MAX_POSITION_PCT + 1e-9).all(), (
        f"Positions above cap:\n{target[target > config.MAX_POSITION_PCT]}"
    )
    # Shortfall (if any) must be cash, never leverage
    assert target.sum() <= 1.0 + 1e-9


def test_invariant_turnover_cap_holds_on_full_portfolio_swap():
    """Adversarial: current book holds 25 tickers the new target doesn't
    want at all (a full swap = ~100% unconstrained turnover). The result
    must keep turnover <= TURNOVER_CAP + 1e-9 — the old renormalise-at-the-
    end could quietly break the budget and move rejected tickers."""
    n_total = 50
    rng = np.random.default_rng(7)
    tickers = [f"T{i:02d}" for i in range(n_total)]
    returns = pd.DataFrame(rng.standard_normal((120, n_total)), columns=tickers)
    # New scores favour T00..T24; current book holds T25..T49 equally.
    scores = pd.Series(range(n_total, 0, -1), index=tickers, dtype=float)
    sector_map = {t: f"Sector{i % 10}" for i, t in enumerate(tickers)}
    current = pd.Series(1.0 / 25, index=tickers[25:], dtype=float)

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=current,
        prices=returns,
    )

    actual_turnover = _turnover(target, current)
    assert actual_turnover <= config.TURNOVER_CAP + 1e-9, (
        f"Turnover {actual_turnover:.4f} exceeds cap {config.TURNOVER_CAP}"
    )
    # Weights must never exceed 1.0 in total (shortfall is cash, never leverage)
    assert target.sum() <= 1.0 + 1e-9


def test_invariant_all_caps_hold_together(monkeypatch):
    """Adversarial combo: concentrated HRP + hostile current book. All three
    invariants (sector, position, turnover) must hold simultaneously."""
    n_total = 40
    rng = np.random.default_rng(11)
    tickers = [f"T{i:02d}" for i in range(n_total)]
    returns = pd.DataFrame(rng.standard_normal((120, n_total)), columns=tickers)
    scores = pd.Series(range(n_total, 0, -1), index=tickers, dtype=float)
    # 8 sectors of 5 — feasible under the 30% sector cap.
    sector_map = {t: f"Sector{i // 5}" for i, t in enumerate(tickers)}
    # Current book: everything parked in the 15 lowest-score tickers.
    current = pd.Series(1.0 / 15, index=tickers[-15:], dtype=float)

    import src.portfolio.construction as cons

    def hostile_hrp(ret: pd.DataFrame) -> pd.Series:
        cols = ret.columns.tolist()
        w = pd.Series(0.30 / (len(cols) - 2), index=cols)
        w.iloc[0] = 0.40   # way over the 8% position cap
        w.iloc[1] = 0.30   # also over
        return w / w.sum()

    monkeypatch.setattr(cons, "hrp_weights", hostile_hrp)

    target = build_portfolio(
        scores=scores,
        sector_map=sector_map,
        current_weights=current,
        prices=returns,
    )

    # Position cap
    assert (target <= config.MAX_POSITION_PCT + 1e-9).all()
    # Sector cap
    for sector in set(sector_map.values()):
        sector_total = target[[t for t in target.index if sector_map[t] == sector]].sum()
        assert sector_total <= config.MAX_SECTOR_PCT + 1e-9
    # Turnover cap
    assert _turnover(target, current) <= config.TURNOVER_CAP + 1e-9
    # Never levered
    assert target.sum() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# test_build_short_portfolio
# ---------------------------------------------------------------------------

def test_build_short_portfolio_returns_empty_when_disabled(monkeypatch):
    """build_short_portfolio returns an empty Series when LONG_SHORT_ENABLED=False."""
    monkeypatch.setattr(config, "LONG_SHORT_ENABLED", False)
    from src.portfolio.construction import build_short_portfolio
    scores = pd.Series({"A": 0.9, "B": 0.1, "C": 0.05})
    result = build_short_portfolio(scores, {}, pd.Series(dtype=float), pd.DataFrame())
    assert result.empty


def test_build_short_portfolio_returns_negative_weights_when_enabled(monkeypatch):
    """build_short_portfolio returns negative weights capped at MAX_SHORT_PCT when enabled."""
    monkeypatch.setattr(config, "LONG_SHORT_ENABLED", True)
    monkeypatch.setattr(config, "N_SHORT_POSITIONS", 2)
    monkeypatch.setattr(config, "MAX_SHORT_PCT", 0.03)
    from src.portfolio.construction import build_short_portfolio
    scores = pd.Series({"A": 0.9, "B": 0.8, "C": 0.1, "D": 0.05})
    result = build_short_portfolio(scores, {}, pd.Series(dtype=float), pd.DataFrame())
    assert not result.empty
    assert (result < 0).all()
    assert (result >= -config.MAX_SHORT_PCT).all()
