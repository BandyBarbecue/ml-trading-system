"""
Tests for the point-in-time universe masking in feature_matrix.py.

Two behaviours matter:
  1. When data/sp500_changes.parquet exists, (date, ticker) rows where the
     ticker was not an index member are DROPPED — and the cross-sectional
     target rank is computed only among actual members.
  2. When the parquet is missing, build_feature_matrix() warns once and
     proceeds unmasked — the bias fix must never crash the pipeline.

Reuses the synthetic-price fixtures and source-isolation helper from
test_feature_matrix.py (same tmp_path + monkeypatch pattern).
"""

import logging

import pandas as pd
import pytest

from src.data import sp500_history
from src.features.feature_matrix import build_feature_matrix
from tests.features.test_feature_matrix import (
    _isolate_other_feature_sources,
    _make_synthetic_prices,
    _write_prices,
)


def _install_synthetic_membership(
    tmp_path, monkeypatch, changes: pd.DataFrame, current: list[str]
) -> None:
    """Point sp500_history at a synthetic changes parquet + member list.

    _isolate_other_feature_sources() already redirected CHANGES_FILE to a
    nonexistent file (so the OTHER tests never mask); here we redirect it
    again to a real synthetic parquet, and stub get_symbols so no network
    or CSV cache is touched.
    """
    changes_path = tmp_path / "synthetic_changes.parquet"
    changes.to_parquet(changes_path, index=False)
    monkeypatch.setattr(sp500_history, "CHANGES_FILE", changes_path)
    monkeypatch.setattr(sp500_history, "get_symbols", lambda refresh=False: list(current))


def test_non_member_rows_are_dropped(tmp_path, monkeypatch):
    """AAA only joined the index mid-window: its rows before the join date
    must be absent, and BBB's target rank on those days must be computed as
    if BBB were the only competitor (rank 1.0)."""
    dates, prices = _make_synthetic_prices()
    _write_prices(tmp_path, prices)
    _isolate_other_feature_sources(tmp_path, monkeypatch)

    join_date = dates[290]  # deep enough that features/targets are defined around it
    changes = pd.DataFrame({"date": [join_date], "added": ["AAA"], "removed": [None]})
    _install_synthetic_membership(tmp_path, monkeypatch, changes, current=["AAA", "BBB"])

    fm = build_feature_matrix(["AAA", "BBB"], dates)  # use_pit_universe defaults True
    assert not fm.empty

    aaa_dates = set(fm.loc[fm["ticker"] == "AAA", "date"])
    bbb_dates = set(fm.loc[fm["ticker"] == "BBB", "date"])

    # AAA: nothing before the join, present from the join date onwards.
    assert all(d >= join_date for d in aaa_dates), "AAA has rows before it joined the index"
    assert join_date in aaa_dates
    assert dates[289] not in aaa_dates and dates[289] in bbb_dates

    # BBB (member throughout) keeps a row on an early date AAA lost.
    early = dates[280]
    assert early in bbb_dates

    # Masking happened BEFORE ranking: with AAA excluded on `early`, BBB is
    # the only competitor, so its percentile rank must be exactly 1.0 — if
    # ranks had been computed against both tickers and AAA merely dropped
    # afterwards, BBB (the slower grower) would show 0.5 here.
    bbb_early = fm[(fm["date"] == early) & (fm["ticker"] == "BBB")].iloc[0]
    assert bbb_early["target"] == pytest.approx(1.0)


def test_missing_changes_parquet_falls_back_unmasked(tmp_path, monkeypatch, caplog):
    """No changes parquet → warn ONCE, then build the full unmasked matrix."""
    dates, prices = _make_synthetic_prices()
    _write_prices(tmp_path, prices)
    # This helper already points CHANGES_FILE at a file that doesn't exist.
    _isolate_other_feature_sources(tmp_path, monkeypatch)
    # Reset the once-per-process warning flag so this test observes it fire.
    monkeypatch.setattr(sp500_history, "_missing_changes_warned", False)

    with caplog.at_level(logging.WARNING, logger="src.data.sp500_history"):
        fm = build_feature_matrix(["AAA", "BBB"], dates)

    assert not fm.empty
    # Unmasked: both tickers appear across the whole window.
    assert set(fm["ticker"].unique()) == {"AAA", "BBB"}
    assert fm.groupby("ticker")["date"].nunique().nunique() == 1  # same coverage for both

    pit_warnings = [r for r in caplog.records if "point-in-time" in r.getMessage()]
    assert len(pit_warnings) == 1, "Expected exactly one fallback warning"

    # Second build must NOT warn again (once per process).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.data.sp500_history"):
        build_feature_matrix(["AAA", "BBB"], dates)
    assert not [r for r in caplog.records if "point-in-time" in r.getMessage()]


def test_use_pit_universe_false_skips_masking(tmp_path, monkeypatch):
    """Opting out must ignore the changes parquet entirely."""
    dates, prices = _make_synthetic_prices()
    _write_prices(tmp_path, prices)
    _isolate_other_feature_sources(tmp_path, monkeypatch)

    join_date = dates[290]
    changes = pd.DataFrame({"date": [join_date], "added": ["AAA"], "removed": [None]})
    _install_synthetic_membership(tmp_path, monkeypatch, changes, current=["AAA", "BBB"])

    fm = build_feature_matrix(["AAA", "BBB"], dates, use_pit_universe=False)
    aaa_dates = set(fm.loc[fm["ticker"] == "AAA", "date"])
    assert dates[100] in aaa_dates  # pre-join rows retained when masking is off
