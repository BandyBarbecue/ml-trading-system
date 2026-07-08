"""
Tests for sp500_history.py — point-in-time S&P 500 membership.

reconstruct_membership() is a PURE function, so we can feed it tiny synthetic
event logs and assert exact member sets — no network, no disk. The Wikipedia
parser is tested with a mocked pd.read_html returning a synthetic frame with
the same MultiIndex header shape the real page produces.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data import sp500_history
from src.data.sp500_history import (
    fetch_membership_changes,
    get_membership_mask,
    reconstruct_membership,
)


def _changes(rows: list[tuple]) -> pd.DataFrame:
    """Build a long-format change log from (date, added, removed) tuples."""
    df = pd.DataFrame(rows, columns=["date", "added", "removed"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# reconstruct_membership — the pure rewind logic
# ---------------------------------------------------------------------------

class TestReconstructMembership:
    def test_pure_addition_is_rewound(self):
        """NEW joined on 2025-06-02; before that date it was NOT a member."""
        changes = _changes([("2025-06-02", "NEW", None)])
        current = ["AAA", "NEW"]

        members = reconstruct_membership(current, changes, as_of="2025-01-15")
        assert members == {"AAA"}

    def test_pure_removal_is_rewound(self):
        """OLD left on 2025-06-02; before that date it WAS a member."""
        changes = _changes([("2025-06-02", None, "OLD")])
        current = ["AAA"]

        members = reconstruct_membership(current, changes, as_of="2025-01-15")
        assert members == {"AAA", "OLD"}

    def test_same_row_substitution(self):
        """A single row swapping OLD → NEW: rewinding restores OLD, evicts NEW."""
        changes = _changes([("2025-06-02", "NEW", "OLD")])
        current = ["AAA", "NEW"]

        members = reconstruct_membership(current, changes, as_of="2025-01-15")
        assert members == {"AAA", "OLD"}

    def test_boundary_change_on_as_of_already_reflected(self):
        """A change dated exactly ON as_of took effect that day — it must NOT
        be rewound. Only changes with date STRICTLY greater than as_of are."""
        changes = _changes([("2025-06-02", "NEW", "OLD")])
        current = ["AAA", "NEW"]

        # Exactly on the event date: the swap already happened.
        on_boundary = reconstruct_membership(current, changes, as_of="2025-06-02")
        assert on_boundary == {"AAA", "NEW"}

        # One day earlier: the swap is rewound.
        day_before = reconstruct_membership(current, changes, as_of="2025-06-01")
        assert day_before == {"AAA", "OLD"}

    def test_multiple_events_rewind_in_order(self):
        """Two sequential swaps rewind newest-first, like playing a tape backwards."""
        changes = _changes([
            ("2025-03-03", "BBB", "AAA"),   # AAA out, BBB in
            ("2025-06-02", "CCC", "BBB"),   # BBB out, CCC in
        ])
        current = ["CCC", "ZZZ"]

        # Before everything: neither swap has happened yet.
        assert reconstruct_membership(current, changes, as_of="2025-01-02") == {"AAA", "ZZZ"}
        # Between the two swaps: only the first has happened.
        assert reconstruct_membership(current, changes, as_of="2025-04-15") == {"BBB", "ZZZ"}

    def test_empty_changes_returns_current(self):
        changes = pd.DataFrame(columns=["date", "added", "removed"])
        assert reconstruct_membership(["AAA", "BBB"], changes, as_of="2020-01-01") == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# Wikipedia parser — mocked pd.read_html with a MultiIndex-header frame
# ---------------------------------------------------------------------------

# Wikipedia's constituents table (single-level columns) — must be SKIPPED by
# the table finder, which looks for the MultiIndex changes table.
MOCK_CONSTITUENTS = pd.DataFrame({
    "Symbol": ["AAA", "BBB"],
    "Security": ["Aaa Co", "Bbb Co"],
    "GICS Sector": ["Tech", "Tech"],
    "GICS Sub-Industry": ["X", "Y"],
})


def _mock_changes_table() -> pd.DataFrame:
    """Synthetic frame matching pd.read_html's shape for the changes table."""
    cols = pd.MultiIndex.from_tuples([
        # The real page titles this column "Effective Date" (older revisions
        # used plain "Date") — the parser matches by substring, so test the
        # harder variant.
        ("Effective Date", "Effective Date"),
        ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"),
        ("Reason", "Reason"),
    ])
    data = [
        # Normal substitution row.
        ["June 2, 2025", "NEW", "New Co", "OLD", "Old Co", "S&P change"],
        # Add-only row (empty Removed cells) + dot ticker to normalise.
        ["March 3, 2025", "XY.Z", "Dotty Co", "", "", "Market cap change"],
        # Unparseable date — must be skipped and logged, not crash.
        ["not a real date", "BAD", "Bad Co", "WORSE", "Worse Co", "corrupt row"],
        # Remove-only row (NaN Added cells).
        ["January 6, 2025", np.nan, np.nan, "GONE", "Gone Co", "acquired"],
    ]
    return pd.DataFrame(data, columns=cols)


class TestFetchMembershipChanges:
    def _fetch(self, tmp_path, monkeypatch) -> pd.DataFrame:
        monkeypatch.setattr(sp500_history, "CHANGES_FILE", tmp_path / "changes.parquet")
        mock_response = MagicMock()
        mock_response.text = "<html>ignored — read_html is mocked</html>"
        with patch("src.data.sp500_history.requests.get", return_value=mock_response), \
             patch("src.data.sp500_history.pd.read_html",
                   return_value=[MOCK_CONSTITUENTS, _mock_changes_table()]):
            return fetch_membership_changes(save=True)

    def test_parses_valid_rows_and_skips_bad_dates(self, tmp_path, monkeypatch):
        changes = self._fetch(tmp_path, monkeypatch)
        # 4 raw rows, 1 has an unparseable date → 3 events survive.
        assert len(changes) == 3
        assert "BAD" not in set(changes["added"].dropna())
        # Sorted oldest-first.
        assert changes["date"].is_monotonic_increasing

    def test_ticker_normalisation_and_empty_cells(self, tmp_path, monkeypatch):
        changes = self._fetch(tmp_path, monkeypatch)
        add_only = changes[changes["date"] == pd.Timestamp("2025-03-03")].iloc[0]
        # Dot → hyphen, consistent with sp500_universe (BRK.B → BRK-B).
        assert add_only["added"] == "XY-Z"
        # Empty Removed cell → None (stored as null, not "").
        assert add_only["removed"] is None or pd.isna(add_only["removed"])

        remove_only = changes[changes["date"] == pd.Timestamp("2025-01-06")].iloc[0]
        assert remove_only["added"] is None or pd.isna(remove_only["added"])
        assert remove_only["removed"] == "GONE"

    def test_parquet_written(self, tmp_path, monkeypatch):
        self._fetch(tmp_path, monkeypatch)
        saved = pd.read_parquet(tmp_path / "changes.parquet")
        assert list(saved.columns) == ["date", "added", "removed"]
        assert len(saved) == 3


# ---------------------------------------------------------------------------
# get_membership_mask — correctness across a mid-window change
# ---------------------------------------------------------------------------

class TestMembershipMask:
    def test_mask_flips_on_event_date(self):
        """One swap in the middle of the window: NEW replaces OLD on 2025-02-03.
        Rows before that date must show OLD in / NEW out; on and after, the
        opposite. AAA (untouched) is a member throughout."""
        dates = pd.bdate_range("2025-01-02", periods=40)  # ends ~2025-02-26
        event_date = pd.Timestamp("2025-02-03")           # a Monday inside the window
        changes = _changes([(event_date, "NEW", "OLD")])
        current = ["AAA", "NEW"]  # today's members (after the swap)

        mask = get_membership_mask(
            dates, ["AAA", "NEW", "OLD"], changes=changes, current=current
        )

        assert mask.shape == (40, 3)
        assert mask.dtypes.eq(bool).all()

        before = mask.loc[mask.index < event_date]
        on_and_after = mask.loc[mask.index >= event_date]
        assert len(before) > 0 and len(on_and_after) > 0

        # Before the swap: OLD is a member, NEW is not.
        assert before["OLD"].all()
        assert not before["NEW"].any()
        # On the event date and after: NEW is in, OLD is out (change effective ON its date).
        assert on_and_after["NEW"].all()
        assert not on_and_after["OLD"].any()
        # The bystander never changes.
        assert mask["AAA"].all()

    def test_mask_never_marks_unknown_ticker_member(self):
        """A ticker that is neither current nor in any event stays False."""
        dates = pd.bdate_range("2025-01-02", periods=5)
        changes = _changes([("2025-06-02", "NEW", "OLD")])
        mask = get_membership_mask(dates, ["MYSTERY"], changes=changes, current=["AAA"])
        assert not mask["MYSTERY"].any()
