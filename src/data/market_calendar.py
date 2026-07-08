"""market_calendar.py — NYSE trading-calendar gate.

Answers two questions every runner needs before doing anything else:
  1. Is today a trading day at all (not a weekend, not a holiday)?
  2. What time does the market actually close today (16:00 ET normally,
     but 13:00 ET on the handful of early-close days like the day after
     Thanksgiving)?

Why an offline calendar (pandas-market-calendars) instead of asking Alpaca's
API "is the market open today?": this is a *safety gate* that every daily
and intraday run checks first, before acquiring locks or touching the
network. A calendar computed from a fixed rule table (XNYS holidays/early
closes) is deterministic and available even if the broker API is down,
rate-limited, or slow — the one thing we never want is a network hiccup
making the gate itself unreliable. NYSE holidays are known years in advance,
so there's no accuracy cost to computing this offline.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

_ET = ZoneInfo("America/New_York")

# Built once, on first use, and reused for the life of the process — building
# a Calendar object re-parses NYSE's holiday rule tables, which is wasted
# work if every call to is_trading_day() / market_close_et() rebuilds it.
_calendar = None


def _get_calendar():
    """Lazily construct and cache the XNYS (NYSE) calendar singleton."""
    global _calendar
    if _calendar is None:
        _calendar = mcal.get_calendar("XNYS")
    return _calendar


def _to_date(d: date | datetime | str) -> date:
    """Normalise date | datetime | 'YYYY-MM-DD' string into a plain date."""
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def is_trading_day(d: date | datetime | str) -> bool:
    """True if the NYSE is open for regular trading on date *d*.

    False for weekends and NYSE holidays (e.g. Independence Day observed).
    """
    day = _to_date(d)
    # valid_days returns only the dates the exchange is open, so a simple
    # membership check answers "is this a trading day?" directly.
    valid = _get_calendar().valid_days(start_date=day, end_date=day)
    return len(valid) > 0


def market_close_et(d: date | datetime | str) -> time | None:
    """Return the actual NYSE close time (in America/New_York) for date *d*.

    Returns time(16, 0) on a normal day, time(13, 0) on a half-day (e.g. the
    day after Thanksgiving), and None if *d* is not a trading day at all.
    """
    day = _to_date(d)
    schedule = _get_calendar().schedule(start_date=day, end_date=day)
    if schedule.empty:
        return None

    # schedule's market_close column is UTC-localized; convert to ET before
    # reading off the wall-clock time, otherwise a half-day would come back
    # as 18:00 (UTC) instead of 13:00 (ET) and look identical to a full day.
    close_utc = schedule["market_close"].iloc[0]
    return close_utc.tz_convert(_ET).time()
