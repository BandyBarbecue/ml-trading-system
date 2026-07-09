"""
ic_tracker.py — Information Coefficient (IC) tracker with feature retirement flagging.

What is IC?
  IC = Information Coefficient = Spearman rank correlation between a feature
  (e.g. momentum score) and future returns.

  Analogy: Imagine you're ranking students by how tall they are and then checking
  if that ranking predicts their exam scores.  If taller students consistently score
  higher, there's a positive IC between "height" and "score."  In trading, we rank
  stocks by a feature (e.g. momentum) and check if that rank predicts next month's
  returns.

Why Spearman (rank correlation) and not Pearson?
  Pearson measures *linear* relationships and is sensitive to outliers.
  Spearman converts both series to ranks first, which makes it robust to extreme
  values (e.g. a stock that went up 200% in a day).  In finance, outliers are common,
  so Spearman is the standard for IC measurement.

Why track IC over time?
  A feature's predictive power changes with market regimes.  A momentum factor that
  worked in 2015 might be useless in 2020.  By tracking IC month-by-month, we can:
    1. Upweight features that have recently been predictive (higher IC).
    2. Retire (flag) features that have been near-zero for several months in a row —
       they're consuming model capacity without adding signal.
"""

import json
from datetime import date, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from config import IC_WINDOW, IC_RETIRE_MONTHS, IC_RETIRE_THRESH


def compute_ic(
    feature: pd.Series,
    forward_returns: pd.Series,
    dates: pd.Series | None = None,
    window: int | None = IC_WINDOW,
) -> float | None:
    """Spearman rank correlation between a feature and future returns.

    IC = Information Coefficient.  A feature with |IC| > 0.05 is considered
    genuinely predictive in quantitative finance.  Most real-world factors
    have ICs in the range 0.02–0.10; anything above 0.10 is exceptional.

    Args:
        feature:         Cross-sectional feature values (one observation per stock).
        forward_returns: The future return for each stock, aligned by index.
        dates:           Optional — the trading date each row of `feature` /
                          `forward_returns` belongs to, aligned by index (e.g.
                          feature_matrix["date"]). Required for `window` to have
                          any effect; without it there's no way to know which
                          rows are "recent".
        window:          Restrict the IC computation to the most recent
                          `window` distinct dates found in `dates` (default:
                          IC_WINDOW = 60 trading days — "dynamic" IC that
                          reflects the current regime instead of averaging
                          over the whole history). Has no effect if `dates`
                          is not supplied — in that case IC is computed over
                          the full input, same as before this parameter
                          existed (backward compatible).

    Returns:
        Spearman correlation coefficient in [-1, 1], or None if IC could not
        be measured at all (fewer than 10 overlapping observations, or a
        degenerate/NaN result e.g. zero-variance input).

        WHY None and not 0.0: 0.0 is a real, meaningful measurement — "this
        feature genuinely has no relationship with returns right now". None
        means "we don't know" — e.g. a brand-new feature (news_score before
        any news exists for these tickers) that simply has no overlapping
        data yet. Collapsing both cases to 0.0 (the old behaviour) made a
        feature that has literally never been measured look identical to one
        that was measured and found useless — callers (get_feature_weights,
        the weight floor / drop logic in daily_run.py) could not tell "no
        signal" apart from "no evidence", and silently downweighted or
        dropped features that had simply never had a chance to prove
        themselves. Analogy: a student who was ABSENT for an exam should not
        be recorded as having scored 0 — "no grade yet" and "failed" call for
        different responses.
    """
    # Align on the shared index and drop rows where either series is NaN.
    # This handles cases where not all stocks have both a feature value and a return.
    combined = pd.concat([feature, forward_returns], axis=1).dropna()

    if dates is not None and window is not None and not combined.empty:
        # Restrict to the most recent `window` distinct dates. Without this,
        # the call site in daily_run.py computes IC over the ENTIRE ~2-year
        # feature-matrix buffer every day, so one more day of data barely
        # moves a ~500-day-old average — "dynamic" IC weighting was dynamic
        # in name only. Using only the most recent `window` dates makes the
        # IC (and therefore the feature weights derived from it) actually
        # responsive to recent market regime changes.
        aligned_dates = dates.reindex(combined.index)
        recent_cutoff_dates = pd.Series(aligned_dates.dropna().unique())
        recent_cutoff_dates = recent_cutoff_dates.sort_values().iloc[-window:]
        combined = combined[aligned_dates.isin(set(recent_cutoff_dates))]

    if len(combined) < 10:
        # Not enough data to compute a meaningful correlation — this is "no
        # measurement", not "measured zero". See the docstring's None-vs-0.0
        # explanation above.
        # Analogy: you can't reliably judge a coin's fairness from 3 flips.
        return None
    corr, _ = spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    # NaN happens when one series has zero variance (e.g. every stock has the
    # same feature value) — Spearman correlation is undefined there, not zero.
    return float(corr) if not np.isnan(corr) else None


class ICTracker:
    """Tracks rolling IC observations for each feature and flags stale ones.

    Usage pattern:
        tracker = ICTracker()
        # After each monthly rebalance, compute the IC for each feature and update:
        tracker.update("mom_1m", compute_ic(feature_series, fwd_returns))
        # Check which features have lost predictive power:
        dead_features = tracker.get_retirement_warnings()
    """

    def __init__(self) -> None:
        # Dict mapping feature_name → list of recent IC floats (oldest first).
        # We keep at most IC_RETIRE_MONTHS + 1 entries per feature so memory
        # stays bounded regardless of how long the system runs.
        self._ic_history: dict[str, list[float]] = {}
        # Dict mapping feature_name → "YYYY-MM" string for the calendar month
        # of the most recent entry in _ic_history[feature_name]. Used by
        # update() to decide whether a new call should APPEND a new monthly
        # observation or be blended INTO the current month's entry.
        self._last_update_month: dict[str, str] = {}
        # Dict mapping feature_name → how many daily observations have been
        # blended into the current month's entry so far. Needed to compute a
        # correct running mean in update() (see below).
        self._month_obs_count: dict[str, int] = {}

    def update(
        self,
        feature_name: str,
        ic_value: float | None,
        as_of: "date | str | None" = None,
    ) -> None:
        """Record a new IC observation for a feature — idempotent per calendar month.

        ic_value may be None (see compute_ic's docstring) — this means "no
        overlapping data today", not "measured zero". A None is stored as a
        null entry in the history rather than being folded into the running
        mean, so it can never drag a real measured average toward zero.

        IC_RETIRE_MONTHS (see get_retirement_warnings) is defined in *months*,
        not calls. But daily_run.py calls update() once per trading day. If
        every call appended a new list entry, a feature would get flagged
        "near-zero for 3 months" after just 3 bad DAYS — the retirement
        window would be ~15x too aggressive. To fix this without changing the
        daily_run.py call site: if `as_of` (or "today", when omitted) falls in
        the same calendar month as the last recorded observation for this
        feature, `ic_value` is blended into that month's entry via a running
        mean, in place, instead of appending a new "month". A new entry is
        only appended when the month actually advances (or on the very first
        call for a feature).

        Args:
            feature_name: Name of the feature (e.g. "mom_1m").
            ic_value:     The IC measured this period (float in [-1, 1]), or
                          None if there was no overlapping data to measure it
                          from (see compute_ic's docstring).
            as_of:        The calendar date this observation belongs to.
                          Accepts a `datetime.date` or an ISO date string
                          ("YYYY-MM-DD"). Defaults to today's real date —
                          this is what makes daily_run.py's once-a-day calls
                          collapse into one observation per month automatically,
                          with no change needed at the call site. Pass this
                          explicitly in tests (or when simulating historical
                          days, e.g. inside a backtest loop) so the "current
                          month" isn't tied to the wall clock.
        """
        if as_of is None:
            as_of = datetime.now().date()
        elif isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of).date()
        month_key = f"{as_of.year:04d}-{as_of.month:02d}"

        hist = self._ic_history.setdefault(feature_name, [])
        last_month = self._last_update_month.get(feature_name)

        if last_month == month_key and hist:
            # Same calendar month as the last observation.
            if ic_value is None:
                # No data today — leave the month's entry exactly as it was.
                # WHY: folding None into the running mean would require
                # treating "no data" as some number, and there is no number
                # that means "no data" other than None itself. A no-data day
                # simply contributes nothing — it doesn't shift the average
                # and doesn't count toward the observation count.
                pass
            elif hist[-1] is None:
                # Earlier day(s) this month had no data, but today does —
                # this IS the first real observation for the month, so SET
                # (not blend) it, and start the running-mean count at 1.
                hist[-1] = ic_value
                self._month_obs_count[feature_name] = 1
            else:
                # Update the most recent entry in place with a running mean,
                # rather than treating today's single value as a whole new
                # "month" of data.
                n = self._month_obs_count.get(feature_name, 1)
                hist[-1] = (hist[-1] * n + ic_value) / (n + 1)
                self._month_obs_count[feature_name] = n + 1
        else:
            # New calendar month (or first-ever observation): append.
            # month_obs_count starts at 0 (not 1) when the first entry of the
            # month is itself a no-data None — there's no measured value yet
            # to average against.
            hist.append(ic_value)
            self._last_update_month[feature_name] = month_key
            self._month_obs_count[feature_name] = 0 if ic_value is None else 1

        # Trim to the rolling window: keep only the most recent entries.
        self._ic_history[feature_name] = hist[-(IC_RETIRE_MONTHS + 1):]

    def get_recent_ic(self, feature_name: str) -> float | None:
        """Return the most recent MEASURED IC for a feature, skipping any
        trailing no-data (None) months, or None if it has never been measured.

        WHY skip Nones instead of just returning hist[-1]: a feature can have
        real history from a few months ago and then hit a no-data month (e.g.
        a data source outage) — its last known signal is still the best
        estimate we have, not "unknown again". See compute_ic's docstring for
        the None-vs-0.0 distinction this preserves.

        Args:
            feature_name: Name of the feature.

        Returns:
            Most recent non-None IC float, or None if this feature has no
            measured history at all (never updated, or every entry so far is
            a no-data None).
        """
        hist = self._ic_history.get(feature_name, [])
        for value in reversed(hist):
            if value is not None:
                return value
        return None

    def save(self, path: Path | None = None) -> None:
        """Persist IC history to disk as JSON so it survives between daily runs.

        Also persists the month-tracking state (_last_update_month,
        _month_obs_count) that update() relies on for monthly idempotency —
        daily_run.py calls ``ICTracker.load_or_create()`` fresh every day, so
        without persisting this state, every reload would forget which month
        it last saw and start appending a new "month" on every call again.
        """
        import config
        save_path = path or (config.FEATURES_DIR / "ic_tracker.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ic_history": self._ic_history,
            "last_update_month": self._last_update_month,
            "month_obs_count": self._month_obs_count,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load_or_create(cls, path: Path | None = None) -> "ICTracker":
        """Load persisted IC history from disk, or start fresh if none exists yet.

        Analogy: like opening last month's notebook of student rankings rather
        than starting from a blank page each day.

        Backward compatible with files written before month-tracking existed
        (a bare ``{feature_name: [ic, ic, ...]}`` dict with no "ic_history"
        key) — those load with empty month-tracking state, which just means
        the next update() call is treated as a new month (same as a fresh
        tracker would behave).
        """
        import config
        load_path = path or (config.FEATURES_DIR / "ic_tracker.json")
        tracker = cls()
        if load_path.exists():
            with open(load_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "ic_history" in data:
                tracker._ic_history = data["ic_history"]
                tracker._last_update_month = data.get("last_update_month", {})
                tracker._month_obs_count = data.get("month_obs_count", {})
            else:
                # Legacy format: the file IS the ic_history dict.
                tracker._ic_history = data
        return tracker

    def get_retirement_warnings(self) -> list[str]:
        """Return names of features whose IC has been near-zero for IC_RETIRE_MONTHS.

        A feature is flagged if its absolute IC has been below IC_RETIRE_THRESH
        for every one of the last IC_RETIRE_MONTHS observations.

        Analogy: if a weather forecast has been wrong for 3 months in a row,
        you stop trusting it.  Same idea here — if a feature hasn't predicted
        anything for 3 months, flag it for review or removal.

        Returns:
            List of feature names that should be reviewed for retirement.
            Empty list if all features are performing adequately.
        """
        warnings: list[str] = []
        for name, hist in self._ic_history.items():
            if len(hist) >= IC_RETIRE_MONTHS:
                # Look at only the most recent IC_RETIRE_MONTHS observations.
                recent = hist[-IC_RETIRE_MONTHS:]
                if all(abs(v) < IC_RETIRE_THRESH for v in recent):
                    warnings.append(name)
        return warnings


def get_feature_weights(tracker: ICTracker) -> tuple[dict[str, float], list[str]]:
    """Return normalised feature weights based on the latest MEASURED IC.

    Features with higher absolute IC get upweighted.  The weights sum to 1.0.

    Why absolute IC?
        A feature with IC = -0.08 is just as predictive as IC = +0.08 — the
        model can learn to negate it.  What matters is magnitude, not sign.

    Features that have NEVER produced a measured (non-None) IC — e.g.
    news_score before enough overlapping data exists — are excluded from the
    weight dict entirely (they have no evidence either way, so they
    shouldn't be treated as "IC = 0 and therefore worthless"). Those feature
    names are returned separately in `no_data_features` so callers can
    distinguish "no evidence yet" from "measured and dropped".

    Args:
        tracker: An ICTracker instance with populated _ic_history.

    Returns:
        A (weights, no_data_features) tuple:
          weights:          Dict mapping feature_name → weight (floats
                             summing to 1.0), for every feature with at least
                             one measured IC. Empty dict if none qualify.
          no_data_features: List of feature names whose entire history is
                             None (tracked, but never successfully measured).
    """
    raw: dict[str, float] = {}
    no_data_features: list[str] = []
    for name in tracker._ic_history:
        # get_recent_ic already implements "latest non-None entry, else
        # None" — reuse it so there's one place that defines "latest
        # measured IC" for a feature.
        latest = tracker.get_recent_ic(name)
        if latest is None:
            no_data_features.append(name)
        else:
            # Floor of 1e-6 so a genuinely-measured IC of exactly 0.0 still
            # gets a tiny non-zero weight instead of vanishing from the dict
            # (it WAS measured — it just showed no signal right now).
            raw[name] = max(abs(latest), 1e-6)

    if not raw:
        return {}, no_data_features

    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()}
    return weights, no_data_features
