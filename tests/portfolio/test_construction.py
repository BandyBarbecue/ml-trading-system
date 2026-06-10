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
