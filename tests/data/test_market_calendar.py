"""Tests for src/data/market_calendar.py

These use REAL calendar lookups (no mocking) against fixed, known dates —
NYSE holidays and early closes are published years in advance, so these
assertions are deterministic and never touch the network.
"""

from __future__ import annotations

from datetime import date, datetime, time

from src.data.market_calendar import is_trading_day, market_close_et


# ---------------------------------------------------------------------------
# is_trading_day
# ---------------------------------------------------------------------------

def test_holiday_is_not_trading_day():
    """2026-07-03 is Independence Day observed (4th falls on a Saturday)."""
    assert is_trading_day("2026-07-03") is False


def test_normal_weekday_is_trading_day():
    """2026-07-06 is an ordinary Monday — market open."""
    assert is_trading_day("2026-07-06") is True


def test_saturday_is_not_trading_day():
    """2026-07-04 is a Saturday — market always closed on weekends."""
    assert is_trading_day("2026-07-04") is False


def test_accepts_date_and_datetime_objects():
    """The gate is called with date objects from the runners, and strings
    from tests/CLI — all three input types must agree."""
    assert is_trading_day(date(2026, 7, 3)) is False
    assert is_trading_day(datetime(2026, 7, 6, 9, 30)) is True


# ---------------------------------------------------------------------------
# market_close_et
# ---------------------------------------------------------------------------

def test_half_day_closes_at_1pm():
    """2026-11-27 is the day after Thanksgiving — NYSE closes early at 13:00 ET."""
    assert market_close_et("2026-11-27") == time(13, 0)


def test_normal_day_closes_at_4pm():
    assert market_close_et("2026-07-06") == time(16, 0)


def test_non_trading_day_returns_none():
    """Weekends and holidays have no close time at all."""
    assert market_close_et("2026-07-04") is None   # Saturday
    assert market_close_et("2026-07-03") is None   # holiday
