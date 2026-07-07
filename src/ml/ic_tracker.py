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
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from config import IC_WINDOW, IC_RETIRE_MONTHS, IC_RETIRE_THRESH


def compute_ic(feature: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank correlation between a feature and future returns.

    IC = Information Coefficient.  A feature with |IC| > 0.05 is considered
    genuinely predictive in quantitative finance.  Most real-world factors
    have ICs in the range 0.02–0.10; anything above 0.10 is exceptional.

    Args:
        feature:         Cross-sectional feature values (one observation per stock).
        forward_returns: The future return for each stock, aligned by index.

    Returns:
        Spearman correlation coefficient in [-1, 1].
        Returns 0.0 if there are fewer than 10 overlapping observations
        or if the result is NaN (e.g. zero-variance input).
    """
    # Align on the shared index and drop rows where either series is NaN.
    # This handles cases where not all stocks have both a feature value and a return.
    combined = pd.concat([feature, forward_returns], axis=1).dropna()
    if len(combined) < 10:
        # Not enough data to compute a meaningful correlation.
        # Analogy: you can't reliably judge a coin's fairness from 3 flips.
        return 0.0
    corr, _ = spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return float(corr) if not np.isnan(corr) else 0.0


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

    def update(self, feature_name: str, ic_value: float) -> None:
        """Append a new IC observation for a feature.

        Keeps only the last ``IC_RETIRE_MONTHS + 1`` values so the list never
        grows unboundedly.  The +1 ensures we always have at least one value
        before the retirement window, which makes debugging easier.

        Args:
            feature_name: Name of the feature (e.g. "mom_1m").
            ic_value:     The IC measured this period (float in [-1, 1]).
        """
        hist = self._ic_history.setdefault(feature_name, [])
        hist.append(ic_value)
        # Trim to the rolling window: keep only the most recent entries.
        self._ic_history[feature_name] = hist[-(IC_RETIRE_MONTHS + 1):]

    def get_recent_ic(self, feature_name: str) -> float:
        """Return the most recent IC for a feature, or 0.0 if no history yet.

        Args:
            feature_name: Name of the feature.

        Returns:
            Most recent IC float, or 0.0 if this feature has never been updated.
        """
        hist = self._ic_history.get(feature_name, [])
        return hist[-1] if hist else 0.0

    def save(self, path: Path | None = None) -> None:
        """Persist IC history to disk as JSON so it survives between daily runs."""
        import config
        save_path = path or (config.FEATURES_DIR / "ic_tracker.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self._ic_history, f)

    @classmethod
    def load_or_create(cls, path: Path | None = None) -> "ICTracker":
        """Load persisted IC history from disk, or start fresh if none exists yet.

        Analogy: like opening last month's notebook of student rankings rather
        than starting from a blank page each day.
        """
        import config
        load_path = path or (config.FEATURES_DIR / "ic_tracker.json")
        tracker = cls()
        if load_path.exists():
            with open(load_path) as f:
                tracker._ic_history = json.load(f)
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


def get_feature_weights(tracker: ICTracker) -> dict[str, float]:
    """Return normalised feature weights based on most-recent IC magnitude.

    Features with higher absolute IC get upweighted.  The weights sum to 1.0
    so they can be passed directly as LightGBM ``feature_weight`` inputs.

    Why absolute IC?
        A feature with IC = -0.08 is just as predictive as IC = +0.08 — the
        model can learn to negate it.  What matters is magnitude, not sign.

    Features with no history are excluded from the weight dict entirely
    (treating them as unknown rather than zero-weight).

    Args:
        tracker: An ICTracker instance with populated _ic_history.

    Returns:
        Dict mapping feature_name → weight (floats summing to 1.0).
        Returns an empty dict if the tracker has no history at all.
    """
    raw: dict[str, float] = {}
    for name, hist in tracker._ic_history.items():
        if hist:
            # Use a floor of 1e-6 so features with IC=0 still get a tiny
            # non-zero weight instead of being silently dropped.
            raw[name] = max(abs(hist[-1]), 1e-6)

    if not raw:
        return {}

    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}
