"""
sp500_history.py — point-in-time S&P 500 membership, reconstructed from
Wikipedia's "Selected changes" table.

Why this module exists
----------------------
`sp500_universe.py` scrapes the *current* constituent list, and using that
list for historical backtests creates survivorship bias: companies that were
removed (acquired, bankrupt, demoted) vanish from the sample, and recent
joiners are treated as if they were always members. This module fixes the
membership half of that problem.

The same Wikipedia page that lists current members also has a second table,
"Selected changes to the list of S&P 500 components", recording every
addition/removal with its date. Starting from today's known member list and
*rewinding* those changes backwards, we can reconstruct who was actually in
the index on any past date.

Analogy: think of the index as a 500-seat theatre and the changes table as
the door log. If you know who is seated NOW and you play the door log
backwards (each "X came in, Y left" becomes "X leaves, Y comes back"), you
recover exactly who was seated at any earlier moment.

What this does NOT fix (residual limitation)
--------------------------------------------
Knowing that a delisted company WAS a member doesn't conjure up its price
history — free data feeds often have no data for delisted/acquired tickers.
So ex-members may still be missing from the dataset even though membership
is now correct. `warn_missing_price_history()` reports exactly which ones.
See CLAUDE.md rule 7.

Run once to build the cache:
    python -m src.data.sp500_history
"""

import logging
from io import StringIO

import pandas as pd
import requests

from config import DATA_DIR, PRICES_DIR
from src.data.sp500_universe import get_symbols

log = logging.getLogger(__name__)

# Cached long-format event log: one row per change, columns date/added/removed.
CHANGES_FILE = DATA_DIR / "sp500_changes.parquet"

# One-time-per-process flag so the "no changes parquet, falling back to the
# survivorship-biased universe" warning doesn't spam the log (same pattern
# as sp500_universe._survivorship_warned).
_missing_changes_warned = False


# ---------------------------------------------------------------------------
# Scraping / parsing
# ---------------------------------------------------------------------------

def _normalise_ticker(raw: object) -> str | None:
    """Clean one ticker cell: strip whitespace, dot→hyphen (BRK.B → BRK-B,
    matching sp500_universe's Alpaca convention), empty/NaN → None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    ticker = str(raw).strip()
    if not ticker or ticker.lower() == "nan":
        return None
    return ticker.replace(".", "-")


def _find_changes_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pick the "Selected changes" table out of everything pd.read_html found.

    We identify it by its MultiIndex header (Date / Added / Removed / Reason
    top level) rather than by position, so a new table added to the page
    won't silently break us.
    """
    for table in tables:
        if not isinstance(table.columns, pd.MultiIndex):
            continue
        top_levels = {str(c[0]).strip().lower() for c in table.columns}
        # The date column is titled "Effective Date" (was plain "Date" in the
        # past) — match by substring so either variant works.
        has_date = any("date" in level for level in top_levels)
        if has_date and {"added", "removed"} <= top_levels:
            return table
    return None


def _find_column(columns: pd.MultiIndex, top: str, sub: str) -> tuple | None:
    """Locate the column whose header contains (top, sub), case-insensitively.

    Substring match on both levels so ("date", "date") finds the real page's
    ('Effective Date', 'Effective Date') column, and ("added", "ticker")
    finds ('Added', 'Ticker').
    """
    for col in columns:
        if top in str(col[0]).strip().lower() and sub in str(col[1]).strip().lower():
            return col
    return None


def fetch_membership_changes(save: bool = True) -> pd.DataFrame:
    """Scrape the S&P 500 membership-change log from Wikipedia.

    Returns a long-format DataFrame sorted oldest-first with columns:
        date    - pd.Timestamp of the change
        added   - ticker that joined the index on that date (or None)
        removed - ticker that left the index on that date (or None)

    Rows with an unparseable date are skipped and logged. Rows where both
    added and removed are empty are skipped silently (nothing happened).

    Args:
        save: If True (default), also write the result to CHANGES_FILE.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia 403s requests without a User-Agent (same as sp500_universe).
    headers = {"User-Agent": "trading-research/1.0 (educational project; python-requests)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    # StringIO wrapper: pandas deprecated passing literal HTML directly.
    tables = pd.read_html(StringIO(response.text), flavor="lxml")

    table = _find_changes_table(tables)
    if table is None:
        raise ValueError(
            "Could not find the 'Selected changes' table on the Wikipedia page "
            "(no table with a Date/Added/Removed MultiIndex header). The page "
            "layout may have changed — inspect it manually."
        )

    date_col    = _find_column(table.columns, "date", "date")
    added_col   = _find_column(table.columns, "added", "ticker")
    removed_col = _find_column(table.columns, "removed", "ticker")
    if date_col is None or added_col is None or removed_col is None:
        raise ValueError(
            f"Changes table found but expected columns are missing "
            f"(got {list(table.columns)}). Wikipedia layout may have changed."
        )

    rows: list[dict] = []
    n_bad_dates = 0
    for _, raw_row in table.iterrows():
        # Dates look like "October 1, 2024" — pd.to_datetime handles that.
        # errors="coerce" turns anything unparseable into NaT instead of crashing.
        date = pd.to_datetime(str(raw_row[date_col]), errors="coerce")
        if pd.isna(date):
            n_bad_dates += 1
            log.warning("Skipping changes row with unparseable date: %r", raw_row[date_col])
            continue

        added   = _normalise_ticker(raw_row[added_col])
        removed = _normalise_ticker(raw_row[removed_col])
        if added is None and removed is None:
            # A row with neither an addition nor a removal carries no event.
            continue

        rows.append({"date": date, "added": added, "removed": removed})

    if n_bad_dates:
        log.warning("Skipped %d changes rows with unparseable dates.", n_bad_dates)
    if not rows:
        raise ValueError(
            "Parsed the changes table but extracted zero events — the parser "
            "is probably broken against the current Wikipedia layout."
        )

    changes = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)

    if save:
        CHANGES_FILE.parent.mkdir(parents=True, exist_ok=True)
        changes.to_parquet(CHANGES_FILE, index=False)
        log.info("Saved %d membership-change events to %s", len(changes), CHANGES_FILE)

    return changes


def load_membership_changes(refresh: bool = False) -> pd.DataFrame:
    """Return the change log, scraping Wikipedia only if the cache is missing."""
    if CHANGES_FILE.exists() and not refresh:
        return pd.read_parquet(CHANGES_FILE)
    return fetch_membership_changes(save=True)


# ---------------------------------------------------------------------------
# Reconstruction (pure functions — no I/O, easy to test)
# ---------------------------------------------------------------------------

def _is_ticker(value: object) -> bool:
    """True if a parquet/DataFrame cell holds a real ticker (not None/NaN/'')."""
    return isinstance(value, str) and len(value) > 0


def reconstruct_membership(
    current: list[str],
    changes: pd.DataFrame,
    as_of: pd.Timestamp | str,
) -> set[str]:
    """Reconstruct the S&P 500 member set as of a past date. PURE function.

    How the rewind works — think of the changes table as a tape recording of
    substitutions ("NEW came in, OLD went out"). We know the ending state
    (today's members), so to find the state at `as_of` we play the tape
    BACKWARDS from today to `as_of`, undoing each substitution as we pass it:
    the ticker that was added gets kicked back out (discard), and the ticker
    that was removed gets put back in (add).

    Boundary rule: a change dated exactly ON `as_of` took effect that day, so
    it is already reflected in the `as_of` membership — we only rewind
    changes with date strictly GREATER than `as_of`.

    Args:
        current: today's member list (the known end state).
        changes: long-format event log with columns date/added/removed.
        as_of:   the past date to reconstruct membership for.

    Returns:
        Set of tickers that were members on `as_of`.
    """
    as_of = pd.Timestamp(as_of)
    members = set(current)

    if changes.empty:
        return members

    # Sort oldest-first (stable, so same-day events keep table order), then
    # walk the future events newest-first — that IS the backwards tape.
    future = changes[changes["date"] > as_of].sort_values("date", kind="stable")
    for row in reversed(list(future.itertuples())):
        if _is_ticker(row.added):
            members.discard(row.added)   # undo the addition
        if _is_ticker(row.removed):
            members.add(row.removed)     # undo the removal
    return members


def get_symbols_as_of(d: pd.Timestamp | str, refresh: bool = False) -> list[str]:
    """Return the S&P 500 member tickers as of date `d` (sorted).

    Combines today's list (sp500_universe.get_symbols) with the cached change
    log (scraped on first use) and rewinds. Note: tickers excluded by
    sp500_universe for Alpaca incompatibility (BRK-B, BF-B) stay excluded.
    """
    current = get_symbols(refresh)
    changes = load_membership_changes(refresh)
    return sorted(reconstruct_membership(current, changes, d))


def get_membership_mask(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    changes: pd.DataFrame | None = None,
    current: list[str] | None = None,
) -> pd.DataFrame:
    """Build a (date x ticker) boolean membership mask.

    True means the ticker was an S&P 500 member on that date. Efficient for
    ~1300 dates x 500 tickers because membership only changes on event dates:
    we reconstruct the set ONCE at the first date, then walk forward applying
    each event as we pass its date — never re-reconstructing per calendar day.

    Args:
        dates:   dates for the mask rows (sorted ascending internally).
        tickers: tickers for the mask columns.
        changes: event log; loaded from CHANGES_FILE if None.
        current: today's member list; get_symbols() if None. (Both injection
                 points exist mainly so tests don't need network/disk.)

    Returns:
        DataFrame of bools, index=dates (sorted), columns=tickers.
    """
    if changes is None:
        changes = load_membership_changes()
    if current is None:
        current = get_symbols()

    # Normalise to sorted, tz-naive timestamps so comparisons are consistent.
    dates = pd.DatetimeIndex(dates)
    if dates.tzinfo is not None:
        dates = dates.tz_localize(None)
    dates = dates.sort_values()

    if len(dates) == 0:
        return pd.DataFrame(columns=tickers, dtype=bool)

    # State at the very first date (one full rewind — the only expensive step).
    members = reconstruct_membership(current, changes, as_of=dates[0])

    # Events strictly after the first date, oldest-first, to replay forwards.
    events = changes[changes["date"] > dates[0]].sort_values("date", kind="stable")
    event_rows = list(events.itertuples())

    mask_rows: list[list[bool]] = []
    j = 0  # pointer into event_rows — advances monotonically, never rescans
    for d in dates:
        # Apply every event dated on-or-before d that we haven't applied yet
        # (consistent with reconstruct_membership's boundary rule: a change
        # ON date d counts as effective on d).
        while j < len(event_rows) and event_rows[j].date <= d:
            ev = event_rows[j]
            if _is_ticker(ev.added):
                members.add(ev.added)
            if _is_ticker(ev.removed):
                members.discard(ev.removed)
            j += 1
        mask_rows.append([t in members for t in tickers])

    return pd.DataFrame(mask_rows, index=dates, columns=tickers, dtype=bool)


def load_membership_mask(
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame | None:
    """Safe mask loader for feature/backtest code: returns None on any failure.

    If the changes parquet hasn't been built yet (or anything goes wrong),
    we warn ONCE per process and return None so callers can proceed with the
    old survivorship-biased universe instead of crashing. Fixing bias should
    never take the pipeline down.
    """
    global _missing_changes_warned
    if not CHANGES_FILE.exists():
        if not _missing_changes_warned:
            log.warning(
                "%s not found — point-in-time membership masking is DISABLED "
                "and the universe falls back to survivorship-biased current "
                "members. Run `python -m src.data.sp500_history` to build it.",
                CHANGES_FILE,
            )
            _missing_changes_warned = True
        return None
    try:
        changes = pd.read_parquet(CHANGES_FILE)
        return get_membership_mask(dates, tickers, changes=changes)
    except Exception:
        log.warning(
            "Failed to build point-in-time membership mask — proceeding "
            "unmasked (survivorship-biased).",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Residual-limitation reporting (CLAUDE.md rule 7)
# ---------------------------------------------------------------------------

def warn_missing_price_history(changes: pd.DataFrame | None = None) -> list[str]:
    """Log which ex-members have NO price parquet in data/prices/.

    Membership is now point-in-time correct, but free feeds often carry no
    price history for delisted/acquired tickers — those names silently drop
    out of features/backtests even on dates when they were members. This
    makes the residual bias visible instead of silent.

    Returns the sorted list of affected tickers (useful for tests/inspection).
    """
    if changes is None:
        changes = load_membership_changes()
    current = set(get_symbols())

    # Every ticker that was ever removed and is not a member today.
    ex_members = {
        t for t in changes["removed"].tolist() if _is_ticker(t)
    } - current

    missing = sorted(
        t for t in ex_members if not (PRICES_DIR / f"{t}.parquet").exists()
    )
    if missing:
        log.warning(
            "%d ex-member tickers have no price parquet in %s — they were "
            "index members historically but cannot appear in features or "
            "backtests (residual survivorship bias, see CLAUDE.md rule 7): %s",
            len(missing), PRICES_DIR, ", ".join(missing),
        )
    return missing


# ---------------------------------------------------------------------------
# CLI entry point — build the cache and print a sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
    )

    changes = fetch_membership_changes(save=True)
    # ASCII "->" (not a unicode arrow): Windows consoles default to cp1252,
    # which cannot print the arrow character and crashes the run.
    print(f"Fetched {len(changes)} membership-change events "
          f"({changes['date'].min().date()} -> {changes['date'].max().date()})")

    today = pd.Timestamp.today().normalize()
    three_years_ago = today - pd.DateOffset(years=3)
    n_now = len(get_symbols_as_of(today))
    n_then = len(get_symbols_as_of(three_years_ago))
    print(f"Members today ({today.date()}):        {n_now}")
    print(f"Members 3 years ago ({three_years_ago.date()}): {n_then}")

    warn_missing_price_history(changes)
