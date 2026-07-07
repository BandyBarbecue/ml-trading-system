"""Tests for src/execution/intraday_signal.py"""

import math
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal intraday bars DataFrame for testing."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000.0] * n
    return pd.DataFrame(
        {
            "open":   closes,   # not used by signals, but keeps schema consistent
            "high":   closes,
            "low":    closes,
            "close":  closes,
            "volume": volumes,
            "vwap":   closes,   # placeholder; compute_vwap ignores this column
        },
        index=pd.date_range("2024-01-15 09:30", periods=n, freq="30min", tz="US/Eastern"),
    )


# ---------------------------------------------------------------------------
# compute_vwap tests
# ---------------------------------------------------------------------------

def test_vwap_calculation():
    """Known bars → correct VWAP value."""
    from src.execution.intraday_signal import compute_vwap

    # Bar 1: close=100, volume=1000  → weighted contribution = 100_000
    # Bar 2: close=110, volume=2000  → weighted contribution = 220_000
    # Bar 3: close=90,  volume=1000  → weighted contribution =  90_000
    # VWAP = (100_000 + 220_000 + 90_000) / (1000 + 2000 + 1000) = 410_000 / 4000 = 102.5
    bars = _make_bars(closes=[100.0, 110.0, 90.0], volumes=[1000.0, 2000.0, 1000.0])
    result = compute_vwap(bars)
    assert abs(result - 102.5) < 1e-6


def test_vwap_empty_bars():
    """Empty DataFrame → NaN (no volume to weight by)."""
    from src.execution.intraday_signal import compute_vwap

    bars = _make_bars([])
    result = compute_vwap(bars)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# get_entry_signal tests
# ---------------------------------------------------------------------------

def test_entry_signal_open_bar():
    """Bar index 0 → always 'full' (enter at the open)."""
    from src.execution.intraday_signal import get_entry_signal

    bars = _make_bars(closes=[100.0, 99.0, 98.0])
    assert get_entry_signal(bars, current_bar_idx=0) == "full"


def test_entry_signal_above_vwap():
    """Current close above VWAP → 'full' (momentum confirmed)."""
    from src.execution.intraday_signal import get_entry_signal

    # Bar 0: close=100, vol=1000  → VWAP contrib = 100_000
    # Bar 1: close=90,  vol=1000  → VWAP contrib =  90_000
    # VWAP = 190_000 / 2000 = 95.0
    # Bar 1 close = 90 ... wait, need bar 1 close > VWAP.
    # Let's set bar 1 close to 110 (above VWAP of 105):
    # Bar 0: close=100, vol=1000 → 100_000
    # Bar 1: close=110, vol=1000 → 110_000
    # VWAP = 210_000 / 2000 = 105.0; bar 1 close = 110 > 105 → full
    bars = _make_bars(closes=[100.0, 110.0], volumes=[1000.0, 1000.0])
    assert get_entry_signal(bars, current_bar_idx=1) == "full"


def test_entry_signal_below_vwap_early():
    """Close < VWAP, idx < ENTRY_WINDOW_BARS → 'half' (partial entry)."""
    from src.execution.intraday_signal import get_entry_signal
    import config

    # Make sure idx=1 is within the entry window (ENTRY_WINDOW_BARS=3 by default)
    assert 1 < config.ENTRY_WINDOW_BARS, "Test assumption: idx=1 must be inside the window"

    # Bar 0: close=110, vol=1000 → 110_000
    # Bar 1: close=90,  vol=1000 →  90_000
    # VWAP = 200_000 / 2000 = 100.0; bar 1 close = 90 < 100 → half
    bars = _make_bars(closes=[110.0, 90.0], volumes=[1000.0, 1000.0])
    assert get_entry_signal(bars, current_bar_idx=1) == "half"


def test_entry_signal_past_window():
    """idx >= ENTRY_WINDOW_BARS → 'eod' regardless of price."""
    from src.execution.intraday_signal import get_entry_signal
    import config

    window = config.ENTRY_WINDOW_BARS  # 3 by default
    # Create enough bars; make all closes the same so VWAP == close
    # (to avoid accidentally hitting the "above VWAP → full" branch)
    n = window + 2
    # To guarantee close < VWAP at index `window`: make bar 0 close very high
    closes = [200.0] + [80.0] * (n - 1)   # VWAP is pulled high by bar 0
    bars = _make_bars(closes=closes)

    result = get_entry_signal(bars, current_bar_idx=window)
    assert result == "eod"


# ---------------------------------------------------------------------------
# get_event_signal tests
# ---------------------------------------------------------------------------

def _make_signal(ticker, confidence, magnitude, direction="long"):
    return {
        "ticker":     ticker,
        "confidence": confidence,
        "magnitude":  magnitude,
        "direction":  direction,
    }


def test_event_signal_high_confidence():
    """confidence=0.90, magnitude='large' → returns trade dict."""
    from src.execution.intraday_signal import get_event_signal

    signals = [_make_signal("AAPL", confidence=0.90, magnitude="large", direction="long")]
    result = get_event_signal(signals, "AAPL")

    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["direction"] == "long"
    assert result["size"] > 0


def test_event_signal_low_confidence():
    """confidence=0.70 < 0.85 threshold → returns None."""
    from src.execution.intraday_signal import get_event_signal

    signals = [_make_signal("AAPL", confidence=0.70, magnitude="large")]
    result = get_event_signal(signals, "AAPL")
    assert result is None


def test_event_signal_not_large():
    """magnitude='medium' → returns None even if confidence is high."""
    from src.execution.intraday_signal import get_event_signal

    signals = [_make_signal("AAPL", confidence=0.90, magnitude="medium")]
    result = get_event_signal(signals, "AAPL")
    assert result is None


def test_event_signal_wrong_ticker():
    """Event for AAPL but querying MSFT → returns None."""
    from src.execution.intraday_signal import get_event_signal

    signals = [_make_signal("AAPL", confidence=0.90, magnitude="large")]
    result = get_event_signal(signals, "MSFT")
    assert result is None


def test_event_signal_direction_normalisation():
    """'bullish' → 'long', 'bearish' → 'short'."""
    from src.execution.intraday_signal import get_event_signal

    bullish = [_make_signal("AAPL", 0.90, "large", direction="bullish")]
    assert get_event_signal(bullish, "AAPL")["direction"] == "long"

    bearish = [_make_signal("AAPL", 0.90, "large", direction="bearish")]
    assert get_event_signal(bearish, "AAPL")["direction"] == "short"


def test_event_signal_picks_highest_confidence():
    """Multiple qualifying events → returns the highest-confidence one."""
    from src.execution.intraday_signal import get_event_signal

    signals = [
        _make_signal("AAPL", 0.86, "large", direction="long"),
        _make_signal("AAPL", 0.95, "large", direction="short"),  # best
        _make_signal("AAPL", 0.88, "large", direction="long"),
    ]
    result = get_event_signal(signals, "AAPL")
    assert result is not None
    assert result["direction"] == "short"  # from the 0.95-confidence event
