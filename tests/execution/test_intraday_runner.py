"""Tests for src/execution/intraday_runner.py

Each test focuses on a single behaviour and uses mocks / monkeypatching so no
Alpaca API calls or file I/O ever happens during CI.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et(hour: int, minute: int) -> datetime:
    """Build a timezone-aware ET datetime for today at (hour, minute)."""
    return datetime.now(_ET).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


# ---------------------------------------------------------------------------
# _is_market_hours
# ---------------------------------------------------------------------------

def test_is_market_hours_open():
    """09:35 ET is inside market hours → True."""
    from src.execution.intraday_runner import _is_market_hours

    assert _is_market_hours(_et(9, 35)) is True


def test_is_market_hours_closed():
    """16:00 ET is after market close → False."""
    from src.execution.intraday_runner import _is_market_hours

    assert _is_market_hours(_et(16, 0)) is False


def test_is_market_hours_before_open():
    """08:00 ET is before market open → False."""
    from src.execution.intraday_runner import _is_market_hours

    assert _is_market_hours(_et(8, 0)) is False


def test_is_market_hours_at_open():
    """09:30 ET is the opening boundary → True (inclusive)."""
    from src.execution.intraday_runner import _is_market_hours

    assert _is_market_hours(_et(9, 30)) is True


def test_is_market_hours_at_close():
    """15:50 ET is the closing boundary → False (exclusive upper bound)."""
    from src.execution.intraday_runner import _is_market_hours

    assert _is_market_hours(_et(15, 50)) is False


# ---------------------------------------------------------------------------
# _next_bar_time
# ---------------------------------------------------------------------------

def test_next_bar_time_mid_hour():
    """10:47 → 11:00 (round up to next 30-min boundary)."""
    from src.execution.intraday_runner import _next_bar_time

    now  = _et(10, 47)
    nxt  = _next_bar_time(now)

    assert nxt.hour   == 11
    assert nxt.minute == 0
    assert nxt.second == 0


def test_next_bar_time_on_boundary():
    """11:00 (exactly on boundary) → 11:30 (advance to next)."""
    from src.execution.intraday_runner import _next_bar_time

    now = _et(11, 0)
    nxt = _next_bar_time(now)

    assert nxt.hour   == 11
    assert nxt.minute == 30


def test_next_bar_time_half_hour_boundary():
    """11:30 (exactly on half-hour boundary) → 12:00."""
    from src.execution.intraday_runner import _next_bar_time

    # 11:30 → minutes=30 → next_minutes = 30 * (1+1) = 60 → wrap to 12:00
    now = _et(11, 30)
    nxt = _next_bar_time(now)

    assert nxt.hour   == 12
    assert nxt.minute == 0


def test_next_bar_time_just_past_half():
    """11:45 → 12:00 (rounds up past :30 within same hour to next hour)."""
    from src.execution.intraday_runner import _next_bar_time

    now = _et(11, 45)
    nxt = _next_bar_time(now)

    assert nxt.hour   == 12
    assert nxt.minute == 0


def test_next_bar_time_before_half():
    """10:15 → 10:30."""
    from src.execution.intraday_runner import _next_bar_time

    now = _et(10, 15)
    nxt = _next_bar_time(now)

    assert nxt.hour   == 10
    assert nxt.minute == 30


# ---------------------------------------------------------------------------
# run_bar
# ---------------------------------------------------------------------------

def _make_bars(n: int = 3) -> pd.DataFrame:
    """Minimal DataFrame matching fetch_intraday's output schema."""
    closes = [100.0] * n
    return pd.DataFrame(
        {
            "open":   closes,
            "high":   closes,
            "low":    closes,
            "close":  closes,
            "volume": [1_000_000.0] * n,
            "vwap":   closes,
        },
        index=pd.date_range(
            "2024-01-15 09:30", periods=n, freq="30min", tz="US/Eastern"
        ),
    )


def test_run_bar_dry_run_returns_empty():
    """dry_run=True → no orders submitted and return value is []."""
    _fake_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=_ET)

    with (
        patch("src.execution.intraday_runner._now_et", return_value=_fake_now),
        patch("src.data.fetch_intraday.fetch_intraday", return_value=_make_bars()),
        patch("src.data.fetch_intraday._make_client", return_value=MagicMock()),
        patch("src.execution.paper_trader._make_trading_client", return_value=MagicMock()),
        patch("src.execution.paper_trader.get_current_positions",
              return_value=pd.Series({"AAPL": 0.5})),
        patch("src.execution.paper_trader.get_account_value", return_value=100_000.0),
        patch("src.execution.paper_trader.submit_rebalance") as mock_submit,
    ):
        from src.execution.intraday_runner import run_bar

        result = run_bar(
            bar_idx      = 1,
            held_tickers = ["AAPL"],
            news_signals = [],
            dry_run      = True,
        )

    assert result == []
    mock_submit.assert_not_called()


def test_run_bar_calls_fetch_intraday():
    """fetch_intraday must be called once for each held ticker."""
    mock_bars  = _make_bars(3)
    mock_trade = MagicMock()
    mock_data  = MagicMock()
    _fake_now  = datetime(2024, 1, 15, 10, 0, 0, tzinfo=_ET)

    with (
        patch("src.execution.intraday_runner._now_et", return_value=_fake_now),
        patch(
            "src.data.fetch_intraday._make_client",
            return_value=mock_data,
        ),
        patch(
            "src.data.fetch_intraday.fetch_intraday",
            return_value=mock_bars,
        ) as mock_fetch,
        patch(
            "src.execution.intraday_signal.get_entry_signal",
            return_value="full",
        ),
        patch(
            "src.execution.intraday_signal.get_event_signal",
            return_value=None,
        ),
        patch(
            "src.execution.paper_trader._make_trading_client",
            return_value=mock_trade,
        ),
        patch(
            "src.execution.paper_trader.get_current_positions",
            return_value=pd.Series({"AAPL": 0.33, "MSFT": 0.33}),
        ),
        patch(
            "src.execution.paper_trader.get_account_value",
            return_value=100_000.0,
        ),
        patch(
            "src.execution.paper_trader.submit_rebalance",
            return_value=[],
        ),
    ):
        from src.execution.intraday_runner import run_bar

        run_bar(
            bar_idx      = 0,
            held_tickers = ["AAPL", "MSFT"],
            news_signals = [],
            dry_run      = False,
        )

    # One call per ticker
    assert mock_fetch.call_count == 2
    called_tickers = {call.args[1] for call in mock_fetch.call_args_list}
    assert called_tickers == {"AAPL", "MSFT"}


def test_run_bar_empty_tickers_returns_empty():
    """If held_tickers is empty, run_bar returns [] immediately."""
    from src.execution.intraday_runner import run_bar

    result = run_bar(bar_idx=0, held_tickers=[], news_signals=[], dry_run=False)
    assert result == []


# ---------------------------------------------------------------------------
# run_intraday — loop stop behaviour
# ---------------------------------------------------------------------------

def test_run_intraday_stops_at_close(monkeypatch):
    """_is_market_hours returns True twice then False → run_bar called exactly twice."""
    import src.execution.intraday_runner as runner_mod

    # A fixed real ET datetime — used wherever _now_et() is called inside
    # run_intraday so that strftime() and timedelta arithmetic work normally.
    _fake_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=_ET)

    # Sequence: True, True, False → loop body runs twice, then exits.
    call_count = {"n": 0}

    def fake_market_hours(_now):
        call_count["n"] += 1
        # Return True for the first two calls, False on the third.
        return call_count["n"] <= 2

    # Track how many times run_bar is called.
    run_bar_calls: list[int] = []

    def fake_run_bar(bar_idx, held_tickers, news_signals, dry_run=False):
        run_bar_calls.append(bar_idx)
        return []

    monkeypatch.setattr(runner_mod, "_is_market_hours", fake_market_hours)
    monkeypatch.setattr(runner_mod, "run_bar", fake_run_bar)
    # _now_et() returns the fixed datetime so strftime / arithmetic don't fail.
    monkeypatch.setattr(runner_mod, "_now_et", lambda: _fake_now)
    monkeypatch.setattr("time.sleep", lambda _: None)  # don't actually sleep

    with (
        patch(
            "src.execution.paper_trader._make_trading_client",
            return_value=MagicMock(),
        ),
        patch(
            "src.execution.paper_trader.get_current_positions",
            return_value=pd.Series(dtype=float),
        ),
        patch(
            "src.features.nlp_features.extract_signals",
            return_value=[],
        ),
        # _next_bar_time must return a real datetime so subtraction works.
        patch(
            "src.execution.intraday_runner._next_bar_time",
            return_value=datetime(2024, 1, 15, 10, 30, 0, tzinfo=_ET),
        ),
    ):
        runner_mod.run_intraday(dry_run=True, date="2024-01-15")

    assert len(run_bar_calls) == 2, (
        f"Expected run_bar called 2 times, got {len(run_bar_calls)}"
    )
    assert run_bar_calls == [0, 1], f"Expected bar indices [0, 1], got {run_bar_calls}"
