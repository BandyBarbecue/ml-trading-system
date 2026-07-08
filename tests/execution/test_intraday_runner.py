"""Tests for src/execution/intraday_runner.py

Each test focuses on a single behaviour and uses mocks / monkeypatching so no
Alpaca API calls or file I/O ever happens during CI.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import config

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
# _is_final_bar
# ---------------------------------------------------------------------------

def test_is_final_bar_true_when_next_boundary_past_close():
    """15:30 → next boundary is 16:00, which is >= close (15:50) → final bar."""
    from src.execution.intraday_runner import _is_final_bar

    assert _is_final_bar(_et(15, 30)) is True


def test_is_final_bar_false_mid_day():
    """11:00 → next boundary is 11:30, well before close → not final."""
    from src.execution.intraday_runner import _is_final_bar

    assert _is_final_bar(_et(11, 0)) is False


# ---------------------------------------------------------------------------
# _load_target_weights  (#15b)
# ---------------------------------------------------------------------------

def test_load_target_weights_missing_file_falls_back(tmp_path, monkeypatch):
    from src.execution import intraday_runner

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert intraday_runner._load_target_weights() == {}


def test_load_target_weights_corrupt_file_falls_back(tmp_path, monkeypatch):
    from src.execution import intraday_runner

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "target_weights.json").write_text("{not valid json")

    assert intraday_runner._load_target_weights() == {}


def test_load_target_weights_stale_file_falls_back(tmp_path, monkeypatch):
    from src.execution import intraday_runner

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    old_date = "2024-01-01"
    (tmp_path / "target_weights.json").write_text(
        json.dumps({"date": old_date, "weights": {"AAPL": 0.08}})
    )

    now = datetime(2024, 2, 1, tzinfo=_ET)  # 31 days later — past the 7-day threshold
    assert intraday_runner._load_target_weights(now=now) == {}


def test_load_target_weights_fresh_file_used(tmp_path, monkeypatch):
    from src.execution import intraday_runner

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    recent_date = "2024-01-01"
    (tmp_path / "target_weights.json").write_text(
        json.dumps({"date": recent_date, "weights": {"AAPL": 0.08, "MSFT": 0.03}})
    )

    now = datetime(2024, 1, 3, tzinfo=_ET)  # 2 days later — within the threshold
    result = intraday_runner._load_target_weights(now=now)
    assert result == {"AAPL": 0.08, "MSFT": 0.03}


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
        patch("src.execution.intraday_runner._load_target_weights", return_value={}),
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
        patch("src.execution.intraday_runner._load_target_weights", return_value={}),
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
# "eod" signal behaviour  (#15a)
# ---------------------------------------------------------------------------

def test_run_bar_eod_holds_position_mid_day():
    """'eod' on a non-final bar must NOT trade — holding the position avoids
    a sell/rebuy whipsaw every 30 minutes (the bug being fixed)."""
    mock_bars = _make_bars(5)
    _fake_now = datetime(2024, 1, 15, 11, 0, 0, tzinfo=_ET)

    with (
        patch("src.execution.intraday_runner._now_et", return_value=_fake_now),
        patch("src.data.fetch_intraday._make_client", return_value=MagicMock()),
        patch("src.data.fetch_intraday.fetch_intraday", return_value=mock_bars),
        patch("src.execution.intraday_signal.get_entry_signal", return_value="eod"),
        patch("src.execution.intraday_signal.get_event_signal", return_value=None),
        patch("src.execution.intraday_runner._load_target_weights", return_value={}),
        patch("src.execution.paper_trader._make_trading_client", return_value=MagicMock()),
        patch("src.execution.paper_trader.get_current_positions",
              return_value=pd.Series({"AAPL": 0.5})),
        patch("src.execution.paper_trader.get_account_value", return_value=100_000.0),
        patch("src.execution.paper_trader.submit_rebalance") as mock_submit,
    ):
        from src.execution.intraday_runner import run_bar

        result = run_bar(
            bar_idx=2, held_tickers=["AAPL"], news_signals=[],
            dry_run=False, is_final_bar=False,
        )

    assert result == []
    mock_submit.assert_not_called()


def test_run_bar_eod_liquidates_on_final_bar():
    """'eod' on the FINAL bar of the day DOES liquidate (target weight 0)."""
    mock_bars = _make_bars(5)
    _fake_now = datetime(2024, 1, 15, 15, 30, 0, tzinfo=_ET)

    with (
        patch("src.execution.intraday_runner._now_et", return_value=_fake_now),
        patch("src.data.fetch_intraday._make_client", return_value=MagicMock()),
        patch("src.data.fetch_intraday.fetch_intraday", return_value=mock_bars),
        patch("src.execution.intraday_signal.get_entry_signal", return_value="eod"),
        patch("src.execution.intraday_signal.get_event_signal", return_value=None),
        patch("src.execution.intraday_runner._load_target_weights", return_value={}),
        patch("src.execution.paper_trader._make_trading_client", return_value=MagicMock()),
        patch("src.execution.paper_trader.get_current_positions",
              return_value=pd.Series({"AAPL": 0.5})),
        patch("src.execution.paper_trader.get_account_value", return_value=100_000.0),
        patch("src.execution.paper_trader.submit_rebalance", return_value=[
            {"ticker": "AAPL", "side": "sell", "shares": 1, "status": "accepted"}
        ]) as mock_submit,
    ):
        from src.execution.intraday_runner import run_bar

        run_bar(
            bar_idx=5, held_tickers=["AAPL"], news_signals=[],
            dry_run=False, is_final_bar=True,
        )

    mock_submit.assert_called_once()
    delta_series = mock_submit.call_args[0][1]
    assert delta_series["AAPL"] == pytest.approx(-0.5)  # 0.5 held → 0.0 target


# ---------------------------------------------------------------------------
# HRP target-weight sizing  (#15b)
# ---------------------------------------------------------------------------

def test_run_bar_uses_hrp_target_weights():
    """'full' signal must scale toward the HRP target weight from
    target_weights.json, not naive equal-weight (1/N)."""
    mock_bars = _make_bars(3)
    _fake_now = datetime(2024, 1, 15, 9, 30, 0, tzinfo=_ET)

    with (
        patch("src.execution.intraday_runner._now_et", return_value=_fake_now),
        patch("src.data.fetch_intraday._make_client", return_value=MagicMock()),
        patch("src.data.fetch_intraday.fetch_intraday", return_value=mock_bars),
        patch("src.execution.intraday_signal.get_entry_signal", return_value="full"),
        patch("src.execution.intraday_signal.get_event_signal", return_value=None),
        patch("src.execution.intraday_runner._load_target_weights",
              return_value={"AAPL": 0.08, "MSFT": 0.03}),
        patch("src.execution.paper_trader._make_trading_client", return_value=MagicMock()),
        patch("src.execution.paper_trader.get_current_positions",
              return_value=pd.Series(dtype=float)),
        patch("src.execution.paper_trader.get_account_value", return_value=100_000.0),
        patch("src.execution.paper_trader.submit_rebalance", return_value=[]) as mock_submit,
    ):
        from src.execution.intraday_runner import run_bar

        run_bar(bar_idx=0, held_tickers=["AAPL", "MSFT"], news_signals=[], dry_run=False)

    delta_series = mock_submit.call_args[0][1]
    # Naive equal-weight would have given each ticker 0.5 (1/2 held tickers).
    # Instead each delta must match its own HRP weight.
    assert delta_series["AAPL"] == pytest.approx(0.08)
    assert delta_series["MSFT"] == pytest.approx(0.03)


def test_run_bar_falls_back_to_equal_weight_when_ticker_missing_from_hrp():
    """A ticker not present in target_weights.json falls back to equal-weight
    (1/N) rather than crashing or silently dropping the ticker."""
    mock_bars = _make_bars(3)
    _fake_now = datetime(2024, 1, 15, 9, 30, 0, tzinfo=_ET)

    with (
        patch("src.execution.intraday_runner._now_et", return_value=_fake_now),
        patch("src.data.fetch_intraday._make_client", return_value=MagicMock()),
        patch("src.data.fetch_intraday.fetch_intraday", return_value=mock_bars),
        patch("src.execution.intraday_signal.get_entry_signal", return_value="full"),
        patch("src.execution.intraday_signal.get_event_signal", return_value=None),
        patch("src.execution.intraday_runner._load_target_weights", return_value={}),
        patch("src.execution.paper_trader._make_trading_client", return_value=MagicMock()),
        patch("src.execution.paper_trader.get_current_positions",
              return_value=pd.Series(dtype=float)),
        patch("src.execution.paper_trader.get_account_value", return_value=100_000.0),
        patch("src.execution.paper_trader.submit_rebalance", return_value=[]) as mock_submit,
    ):
        from src.execution.intraday_runner import run_bar

        run_bar(bar_idx=0, held_tickers=["AAPL", "MSFT"], news_signals=[], dry_run=False)

    delta_series = mock_submit.call_args[0][1]
    assert delta_series["AAPL"] == pytest.approx(0.5)
    assert delta_series["MSFT"] == pytest.approx(0.5)


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

    def fake_run_bar(bar_idx, held_tickers, news_signals, dry_run=False, is_final_bar=False):
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
        # New restart-safety guards (#14/#18c) — keep this test isolated
        # from real risk_state.json / run.lock on disk.
        patch("src.execution.risk_state.is_halted", return_value=False),
        patch("src.execution.paper_trader.acquire_run_lock", return_value=True),
        patch("src.execution.paper_trader.release_run_lock"),
        # 2024-01-15 is MLK Day (a real NYSE holiday) — force the calendar
        # gate open so this test still exercises the loop-stop behaviour.
        patch("src.data.market_calendar.is_trading_day", return_value=True),
    ):
        runner_mod.run_intraday(dry_run=True, date="2024-01-15")

    assert len(run_bar_calls) == 2, (
        f"Expected run_bar called 2 times, got {len(run_bar_calls)}"
    )
    assert run_bar_calls == [0, 1], f"Expected bar indices [0, 1], got {run_bar_calls}"


# ---------------------------------------------------------------------------
# Restart-safety guards on run_intraday  (#14 / #18c)
# ---------------------------------------------------------------------------

def test_run_intraday_halted_skips_loop_entirely(monkeypatch):
    """If today is already halted (kill switch fired earlier), run_intraday
    must not process any bars at all."""
    import src.execution.intraday_runner as runner_mod

    called = {"n": 0}

    def fake_run_bar(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(runner_mod, "run_bar", fake_run_bar)
    # If the halt check didn't short-circuit, this would loop forever.
    monkeypatch.setattr(runner_mod, "_is_market_hours", lambda now: True)

    with (
        patch("src.execution.risk_state.is_halted", return_value=True),
        patch("src.execution.paper_trader.acquire_run_lock") as mock_acquire,
    ):
        runner_mod.run_intraday(dry_run=True, date="2024-01-15")
        mock_acquire.assert_not_called()

    assert called["n"] == 0


def test_run_intraday_refuses_to_start_while_lock_held(monkeypatch):
    """If run.lock is already held, run_intraday must not process any bars."""
    import src.execution.intraday_runner as runner_mod

    called = {"n": 0}

    def fake_run_bar(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(runner_mod, "run_bar", fake_run_bar)
    monkeypatch.setattr(runner_mod, "_is_market_hours", lambda now: True)

    with (
        patch("src.execution.risk_state.is_halted", return_value=False),
        # 2024-01-15 is MLK Day — force the calendar gate open so the test
        # genuinely reaches (and is stopped by) the lock check.
        patch("src.data.market_calendar.is_trading_day", return_value=True),
        patch("src.execution.paper_trader.acquire_run_lock", return_value=False),
    ):
        runner_mod.run_intraday(dry_run=True, date="2024-01-15")

    assert called["n"] == 0


# ---------------------------------------------------------------------------
# NYSE calendar gate + early-close handling  (#20)
# ---------------------------------------------------------------------------

def test_run_intraday_exits_on_non_trading_day(monkeypatch):
    """On a weekend/holiday, run_intraday must exit before acquiring the run
    lock or processing any bars."""
    import src.execution.intraday_runner as runner_mod

    called = {"n": 0}

    def fake_run_bar(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(runner_mod, "run_bar", fake_run_bar)
    # If the calendar gate didn't short-circuit, this would loop forever.
    monkeypatch.setattr(runner_mod, "_is_market_hours", lambda now: True)

    with (
        patch("src.execution.risk_state.is_halted", return_value=False),
        patch("src.data.market_calendar.is_trading_day", return_value=False),
        patch("src.execution.paper_trader.acquire_run_lock") as mock_acquire,
    ):
        runner_mod.run_intraday(dry_run=True, date="2024-01-15")
        mock_acquire.assert_not_called()

    assert called["n"] == 0


def test_effective_close_uses_early_close(monkeypatch):
    """On a 13:00 ET half-day, the effective 'stop trading' time must be
    12:50 (10 min before the early bell), not the hardcoded 15:50."""
    from datetime import time as dt_time
    from src.execution.intraday_runner import _effective_close_hm

    with patch("src.data.market_calendar.market_close_et",
               return_value=dt_time(13, 0)):
        assert _effective_close_hm(_et(11, 0)) == (12, 50)


def test_is_market_hours_respects_early_close():
    """At 13:30 ET on a half-day the market has already closed — the loop
    must see 'doors shut' even though 13:30 < the normal 15:50 close."""
    from datetime import time as dt_time
    from src.execution.intraday_runner import _is_market_hours

    with patch("src.data.market_calendar.market_close_et",
               return_value=dt_time(13, 0)):
        assert _is_market_hours(_et(13, 30)) is False   # after early close
        assert _is_market_hours(_et(11, 0)) is True     # mid-morning still open


def test_is_final_bar_respects_early_close():
    """At 12:30 on a half-day the next boundary (13:00) is past the 12:50
    effective close → this IS the final bar and 'eod' should liquidate."""
    from datetime import time as dt_time
    from src.execution.intraday_runner import _is_final_bar

    with patch("src.data.market_calendar.market_close_et",
               return_value=dt_time(13, 0)):
        assert _is_final_bar(_et(12, 30)) is True
        assert _is_final_bar(_et(11, 0)) is False


def test_effective_close_falls_back_when_calendar_fails():
    """If the calendar lookup raises, the runner must fall back to
    config.INTRADAY_CLOSE_TIME rather than crash mid-session."""
    from src.execution.intraday_runner import _effective_close_hm

    expected = tuple(map(int, config.INTRADAY_CLOSE_TIME.split(":")))

    with patch("src.data.market_calendar.market_close_et",
               side_effect=RuntimeError("calendar exploded")):
        assert _effective_close_hm(_et(11, 0)) == expected


def test_effective_close_falls_back_on_non_trading_day():
    """market_close_et returns None on a non-trading day — fall back to the
    config value (the run_intraday gate exits earlier anyway; this just
    keeps the helper total)."""
    from src.execution.intraday_runner import _effective_close_hm

    expected = tuple(map(int, config.INTRADAY_CLOSE_TIME.split(":")))

    with patch("src.data.market_calendar.market_close_et", return_value=None):
        assert _effective_close_hm(_et(11, 0)) == expected
