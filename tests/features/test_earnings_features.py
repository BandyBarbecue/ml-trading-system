"""
Tests for earnings_features.py — SUE computation with 60-day linear decay.

Why SUE (Standardised Unexpected Earnings)?
  It's one of finance's most persistent anomalies: stocks that beat earnings
  expectations keep drifting upward for weeks (post-earnings announcement drift).
  Standardising by the std of recent surprises makes the signal comparable
  across companies with different earnings volatility.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.earnings_features import compute_sue, compute_earnings_features

# Minimal earnings fixture: AAPL has 4 quarters of data (enough history),
# MSFT has 3 quarters, and we also test a sparse case with only 1 record.
EARNINGS = pd.DataFrame({
    "ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
    "date": pd.to_datetime([
        "2023-02-01", "2022-11-01", "2022-08-01", "2022-05-01",
        "2023-01-25", "2022-10-25", "2022-07-25",
    ]),
    "actual_eps":    [2.18, 1.29, 1.20, 1.52,  2.32, 2.35, 2.23],
    "estimated_eps": [2.10, 1.22, 1.15, 1.45,  2.23, 2.29, 2.19],
    "surprise_pct":  [0.038, 0.057, 0.043, 0.048, 0.040, 0.026, 0.018],
})


def test_sue_recent_beat_positive():
    """After a positive earnings beat, SUE for that ticker should be > 0."""
    dates = pd.bdate_range("2023-02-01", "2023-02-28")
    sue = compute_sue(EARNINGS, dates)

    assert "AAPL" in sue.columns
    # Feb 2 is the first *trading* date after the Feb 1 announcement,
    # so AAPL's latest surprise (0.038) should produce a positive SUE
    assert sue.loc["2023-02-02", "AAPL"] > 0


def test_sue_decays_over_time():
    """SUE signal should decay toward zero as days pass since the announcement.

    Note on calendar vs business-day math:
      DECAY_DAYS = 60 calendar days.  dates[59] is the 60th *business* day
      from Feb 1, which is roughly 84 calendar days away — past the decay window,
      so it would return NaN.  We use dates[30] (≈ 42 calendar days) instead,
      which is still well within the 60-day window but clearly smaller than day 1.
      This preserves the test intent: the signal decays monotonically.
    """
    dates = pd.bdate_range("2023-02-01", "2023-05-31")
    sue = compute_sue(EARNINGS, dates)

    val_day1 = sue.loc["2023-02-02", "AAPL"]   # 1 business day after announcement
    val_day30 = sue.loc[dates[30], "AAPL"]     # ~30 business days ≈ 42 calendar days

    # Both should be non-NaN and day-1 value should be larger than day-30
    assert not np.isnan(val_day1), "day-1 SUE should not be NaN"
    assert not np.isnan(val_day30), "day-30 SUE should not be NaN (still within 60-day window)"
    assert val_day1 > val_day30, "SUE should decay: day-1 > day-30"


def test_sue_zero_after_decay_window():
    """After 60+ calendar days, the signal should be NaN (fully decayed)."""
    # dates[59] is the 60th business day ≈ 84 calendar days — past the window
    dates = pd.bdate_range("2023-02-01", "2023-05-31")
    sue = compute_sue(EARNINGS, dates)

    val_day60biz = sue.loc[dates[59], "AAPL"]
    assert np.isnan(val_day60biz), (
        "SUE should be NaN after 60 calendar days (>= DECAY_DAYS)"
    )


def test_sue_nan_when_insufficient_history():
    """Fewer than MIN_HISTORY (3) past surprises → NaN, not a spurious number."""
    # Only 1 earnings record — can't compute a meaningful std
    sparse = EARNINGS[EARNINGS["ticker"] == "MSFT"].head(1).copy()
    dates = pd.bdate_range("2023-01-26", "2023-01-31")
    sue = compute_sue(sparse, dates)

    assert "MSFT" in sue.columns
    assert sue["MSFT"].isna().all(), (
        "SUE should be NaN when there are fewer than MIN_HISTORY past surprises"
    )


def test_sue_excluded_on_announcement_day():
    """SUE must not leak the announcement-day report itself (AMC reporting).

    Most S&P 500 companies report After Market Close, so day T's close has
    already happened before the earnings news is public. The SUE feature on
    day T (the announcement date) must therefore be computed WITHOUT that
    day's own report; it should only become available starting day T+1.
    """
    dates = pd.bdate_range("2023-02-01", "2023-02-03")
    sue = compute_sue(EARNINGS, dates)

    # On the announcement day itself, the Feb-1 report must not be visible
    # yet. Falling back to the prior report (2022-11-01, ~92 days earlier,
    # past the 60-day decay window) means the value should be NaN.
    val_announce_day = sue.loc["2023-02-01", "AAPL"]
    assert np.isnan(val_announce_day), (
        "SUE on the announcement day itself must not use that day's own report"
    )

    # The very next trading day, the Feb-1 report becomes visible.
    val_next_day = sue.loc["2023-02-02", "AAPL"]
    assert not np.isnan(val_next_day), (
        "SUE should be available starting the trading day after the announcement"
    )


def _earnings_with_report_time(report_time: str) -> pd.DataFrame:
    """Copy of the AAPL fixture rows with a report_time column added.

    Only the *latest* report (2023-02-01) gets the flag under test; the
    older history rows are marked "unknown" — their timing doesn't matter
    because they're all strictly in the past for the dates we evaluate.
    """
    df = EARNINGS[EARNINGS["ticker"] == "AAPL"].copy()
    df["report_time"] = "unknown"
    df.loc[df["date"] == pd.Timestamp("2023-02-01"), "report_time"] = report_time
    return df


def test_sue_bmo_counts_on_announcement_day():
    """A BMO report is public ~7h before day T's close → full-strength day-0 signal."""
    dates = pd.bdate_range("2023-02-01", "2023-02-03")
    sue = compute_sue(_earnings_with_report_time("bmo"), dates)

    val_day0 = sue.loc["2023-02-01", "AAPL"]
    val_day1 = sue.loc["2023-02-02", "AAPL"]

    assert not np.isnan(val_day0), "BMO report must be visible on the announcement day"
    assert val_day0 > 0

    # Day 0 is full strength (decay factor 1.0); day 1 has decayed by 1/60.
    # So day0 * (1 - 1/60) should equal day1 exactly.
    from src.features.earnings_features import DECAY_DAYS
    assert val_day1 == pytest.approx(val_day0 * (1 - 1 / DECAY_DAYS))


def test_sue_amc_excluded_day0_included_day1():
    """An AMC report lands ~30min after day T's close → visible only from T+1."""
    dates = pd.bdate_range("2023-02-01", "2023-02-03")
    sue = compute_sue(_earnings_with_report_time("amc"), dates)

    # Day 0: the report isn't public yet; falling back to the previous report
    # (2022-11-01, past the 60-day decay window) yields NaN.
    assert np.isnan(sue.loc["2023-02-01", "AAPL"]), (
        "AMC report must NOT be visible on the announcement day"
    )

    # Day 1: visible, decayed by one calendar day (days_since == 1).
    val_day1 = sue.loc["2023-02-02", "AAPL"]
    assert not np.isnan(val_day1)
    assert val_day1 > 0


def test_sue_unknown_behaves_like_amc():
    """When timing is unknown, assume the worst (AMC) — no day-0 inclusion."""
    dates = pd.bdate_range("2023-02-01", "2023-02-03")
    sue = compute_sue(_earnings_with_report_time("unknown"), dates)

    assert np.isnan(sue.loc["2023-02-01", "AAPL"])
    assert not np.isnan(sue.loc["2023-02-02", "AAPL"])


def test_sue_legacy_frame_without_report_time_column():
    """Old parquet files predate report_time — must degrade to conservative, not crash.

    The module-level EARNINGS fixture deliberately has no report_time column,
    so this exercises the missing-column code path end to end.
    """
    assert "report_time" not in EARNINGS.columns  # guard: fixture stays legacy

    dates = pd.bdate_range("2023-02-01", "2023-02-03")
    sue = compute_sue(EARNINGS, dates)  # must not raise

    # Conservative behaviour == AMC: excluded day 0, included day 1.
    assert np.isnan(sue.loc["2023-02-01", "AAPL"])
    assert not np.isnan(sue.loc["2023-02-02", "AAPL"])

    # compute_sue must not have mutated the caller's DataFrame.
    assert "report_time" not in EARNINGS.columns


def test_sue_columns_match_tickers():
    """Output DataFrame columns should exactly match tickers present in input."""
    dates = pd.bdate_range("2023-02-01", "2023-02-10")
    sue = compute_sue(EARNINGS, dates)

    expected_tickers = set(EARNINGS["ticker"].unique())
    assert set(sue.columns) == expected_tickers


def test_sue_index_matches_dates():
    """Output DataFrame index should match the requested date range exactly."""
    dates = pd.bdate_range("2023-02-01", "2023-02-10")
    sue = compute_sue(EARNINGS, dates)

    pd.testing.assert_index_equal(sue.index, dates)


def test_sue_empty_earnings():
    """An empty earnings DataFrame should return an empty DataFrame (no crash)."""
    empty = pd.DataFrame(columns=["ticker", "date", "actual_eps", "estimated_eps", "surprise_pct"])
    dates = pd.bdate_range("2023-02-01", "2023-02-10")
    sue = compute_sue(empty, dates)

    assert sue.empty or sue.shape[1] == 0, "Empty earnings should yield empty result"
