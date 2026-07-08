"""
attribution.py — Monthly SHAP-based attribution report with Claude Sonnet review.

After each full month of paper trading this module:
  1. Loads the most recent LightGBM model.
  2. Computes SHAP values on last month's feature data.
  3. Sends a structured summary to Claude Sonnet for post-mortem review.
  4. Saves a markdown report to data/attribution/YYYY-MM-attribution.md.

Analogy: think of SHAP values like "match statistics" after a football game.
The model scored a prediction — SHAP breaks down exactly how much each player
(feature) contributed to that final score, positive or negative.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import anthropic  # module-level so tests can patch src.execution.attribution.anthropic
import numpy as np
import pandas as pd
import shap  # module-level so tests can patch src.execution.attribution.shap

import config  # imported as module so tests can monkeypatch config.ATTR_DIR etc.
from src.ml.model import FEATURE_COLS


# ---------------------------------------------------------------------------
# Prompt template for Claude Sonnet
# ---------------------------------------------------------------------------

ATTRIBUTION_PROMPT = """You are reviewing a monthly attribution report for a paper trading system.

Month: {month}
Portfolio return (if tracked): {equity_change_pct:+.1%}

## Feature importance (SHAP mean absolute values)

{shap_table}

## Top holdings this month

{top_tickers}

Please provide:
1. Which features drove returns this month and why (2-3 sentences).
2. Any features with unexpectedly low or high importance (1-2 sentences).
3. A risk flag if any single feature has > 50% of total SHAP importance (brief).

Then end your response with a fenced JSON block of `proposals` — concrete
actions for next month. Each proposal must have:
  - type: "weight_adjustment" | "feature_retirement" | "regime_rule" | "risk_adjustment"
  - target: the feature name, risk parameter, or "regime:<label>"
  - action: what to change (e.g. "increase_weight", "retire", "set_max_drawdown=0.12")
  - rationale: one sentence

Format EXACTLY as:
```json
[
  {{"type": "weight_adjustment", "target": "feature_name", "action": "increase_weight", "rationale": "..."}}
]
```
If no changes are warranted, return: ```json
[]
```
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_shap_summary(
    model,
    feature_matrix: pd.DataFrame,
    n_background: int = 100,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values per feature.

    SHAP (SHapley Additive exPlanations) tells us how much each feature
    pushed the model's prediction up or down on average.  We take the
    *absolute* value because we care about magnitude of influence, not
    direction — a feature that strongly pulls predictions down is just as
    "important" as one that pushes them up.

    Args:
        model:          Trained lgb.LGBMRegressor.
        feature_matrix: Long-format DataFrame containing FEATURE_COLS.
        n_background:   Number of rows to sample for the SHAP background
                        dataset.  More rows → more accurate but slower.

    Returns:
        DataFrame with columns ["feature", "mean_abs_shap"],
        sorted descending by mean_abs_shap.
    """
    # Drop rows with missing features so SHAP sees a clean matrix
    clean = feature_matrix[FEATURE_COLS].dropna()

    # Sample a representative subset — SHAP TreeExplainer is fast but a full
    # month of data can be thousands of rows; n_background keeps it snappy.
    n = min(n_background, len(clean))
    sample = clean.sample(n, random_state=42)

    explainer = shap.TreeExplainer(model)
    # shap_values shape: (n_samples, n_features)
    shap_values = explainer.shap_values(sample)

    mean_abs = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs})
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def generate_attribution_report(
    model_or_shap_summary,
    feature_matrix_or_month,
    month: str | None = None,
    equity_change_pct: float | None = None,
    top_tickers: list[str] | None = None,
) -> Union[str, dict]:
    """Send SHAP summary + performance context to Claude Sonnet for review.

    Supports two call signatures:

    **New (structured) API** — returns a dict:
        generate_attribution_report(model, feature_matrix, month="YYYY-MM")

        Args:
            model_or_shap_summary:     Trained LightGBM model.
            feature_matrix_or_month:   Long-format DataFrame containing feature
                                       columns plus "date", "ticker", "target".
            month:                     "YYYY-MM" string (required).

        Returns:
            dict with keys:
              - "prose": full Claude response text (str)
              - "proposals": list of proposal dicts parsed from the JSON block
              - "saved_to": path of the saved .md report file (str)

    **Legacy API** — returns a plain string (kept for backward compatibility):
        generate_attribution_report(shap_summary, month, equity_change_pct, top_tickers)

        Args:
            model_or_shap_summary:       Output of compute_shap_summary().
            feature_matrix_or_month:     "YYYY-MM" string.
            month (unused in this path): ignored.
            equity_change_pct:           Portfolio return (0.032 = +3.2%).
            top_tickers:                 Top 5 tickers by total weight.

        Returns:
            Full markdown report text returned by Claude (str).
    """
    # --- Detect which call signature is being used ---
    # New API: first arg is a model (not a DataFrame), second arg is a DataFrame.
    # Legacy API: first arg IS a DataFrame (shap_summary), second arg is a str.
    if isinstance(model_or_shap_summary, pd.DataFrame):
        # ---- Legacy API path ----
        # Old call: generate_attribution_report(shap_summary, month, equity_change_pct, top_tickers)
        # Positional mapping:
        #   model_or_shap_summary   → shap_summary DataFrame
        #   feature_matrix_or_month → month string ("YYYY-MM")
        #   month                   → equity_change_pct float  (positional slot 2)
        #   equity_change_pct       → top_tickers list          (positional slot 3)
        shap_summary: pd.DataFrame = model_or_shap_summary
        _month: str = feature_matrix_or_month  # type: ignore[assignment]
        # Read equity_change_pct and top_tickers from the shifted positional slots
        _equity_change_pct: float = month if month is not None else 0.0  # type: ignore[assignment]
        _top_tickers: list[str] = equity_change_pct if equity_change_pct is not None else []  # type: ignore[assignment]

        shap_table = shap_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        top_tickers_str = ", ".join(_top_tickers)

        prompt = ATTRIBUTION_PROMPT.format(
            month=_month,
            equity_change_pct=_equity_change_pct,
            shap_table=shap_table,
            top_tickers=top_tickers_str,
        )

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=config.CLAUDE_REVIEW_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ---- New structured API path ----
    model = model_or_shap_summary
    feature_matrix: pd.DataFrame = feature_matrix_or_month  # type: ignore[assignment]

    if month is None:
        first_of_this_month = datetime.now().replace(day=1)
        month = (first_of_this_month - timedelta(days=1)).strftime("%Y-%m")

    # --- Try to read equity change from persisted equity curve ---
    _equity_change_pct = 0.0
    try:
        equity_curve_path = config.DATA_DIR / "equity_curve.parquet"
        if equity_curve_path.exists():
            eq = pd.read_parquet(equity_curve_path)
            # BUG (fixed): the old filter used `eq.index.astype(str).str.startswith(month)`,
            # but the equity curve has a plain RangeIndex (0, 1, 2, ...), not a date
            # index — so that filter matched nothing (or garbage) and
            # _equity_change_pct was silently always 0.0. The real date lives in
            # the "date" column, so filter on that instead.
            eq["date"] = pd.to_datetime(eq["date"])
            month_rows = eq[eq["date"].dt.strftime("%Y-%m") == month].sort_values("date")
            if len(month_rows) >= 2:
                first_equity = float(month_rows["equity"].iloc[0])
                last_equity  = float(month_rows["equity"].iloc[-1])
                if first_equity != 0:
                    _equity_change_pct = (last_equity - first_equity) / first_equity
    except Exception:
        _equity_change_pct = 0.0  # graceful fallback — never block reporting

    # --- Build top tickers from the feature matrix ---
    _top_tickers_list: list[str] = []
    if "ticker" in feature_matrix.columns:
        _top_tickers_list = (
            feature_matrix["ticker"].value_counts().head(5).index.tolist()
        )

    # --- Compute SHAP summary over feature_matrix ---
    # Determine which columns are features (exclude metadata columns)
    meta_cols = {"date", "ticker", "target"}
    feature_cols = [c for c in feature_matrix.columns if c not in meta_cols]

    clean = feature_matrix[feature_cols].dropna()
    n = min(100, len(clean))
    sample = clean.sample(n, random_state=42) if n > 0 else clean

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)

    shap_summary_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_table = shap_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    top_tickers_str = ", ".join(_top_tickers_list) if _top_tickers_list else "N/A"

    prompt = ATTRIBUTION_PROMPT.format(
        month=month,
        equity_change_pct=_equity_change_pct,
        shap_table=shap_table,
        top_tickers=top_tickers_str,
    )

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=config.CLAUDE_REVIEW_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    prose: str = response.content[0].text

    # --- Extract the JSON proposals block from prose ---
    json_match = re.search(r"```json\s*(\[.*?\])\s*```", prose, re.DOTALL)
    proposals: list[dict] = []
    if json_match:
        try:
            proposals = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            proposals = []

    # --- Save the markdown report ---
    out_path = save_report(prose, month)

    return {"prose": prose, "proposals": proposals, "saved_to": str(out_path)}


def save_report(report_text: str, month: str) -> Path:
    """Save report_text to config.ATTR_DIR / "{month}-attribution.md".

    Creates the parent directory if it does not exist yet.

    Args:
        report_text: Markdown string to write.
        month:       "YYYY-MM" string used to name the file.

    Returns:
        Path to the saved file.
    """
    attr_dir: Path = config.ATTR_DIR
    attr_dir.mkdir(parents=True, exist_ok=True)

    path = attr_dir / f"{month}-attribution.md"
    path.write_text(report_text, encoding="utf-8")
    return path


def run_attribution(month: str | None = None) -> Path:
    """Full attribution pipeline for a given month (default: last month).

    Steps:
        1. Load latest model from disk.
        2. Build the feature matrix for the full history.
        3. Filter to last month's data.
        4. Compute SHAP summary.
        5. Send to Claude Sonnet for a post-mortem review.
        6. Save and return the path to the markdown report.

    Args:
        month: "YYYY-MM" string.  If None, defaults to the previous calendar
               month (e.g. if today is 2026-06-09, month = "2026-05").

    Returns:
        Path to the saved attribution report.
    """
    # --- Default to last month ---
    if month is None:
        first_of_this_month = datetime.now().replace(day=1)
        month = (first_of_this_month - timedelta(days=1)).strftime("%Y-%m")

    # --- Step 1: Load model ---
    from src.ml.model import load_latest_model
    model = load_latest_model()
    if model is None:
        raise RuntimeError("No saved model found in config.MODELS_DIR.")

    # --- Step 2: Build feature matrix ---
    from src.features.feature_matrix import build_feature_matrix
    from src.data.sp500_universe import get_symbols

    symbols = get_symbols()
    # Build a date range covering at least the target month plus warm-up buffer.
    # We ask for MODEL_TRAIN_DAYS + 25 ending at the last day of the given month
    # so features computed on those dates are available.
    last_day = pd.Period(month, freq="M").end_time.date().strftime("%Y-%m-%d")
    dates = (
        pd.bdate_range(end=last_day, periods=config.MODEL_TRAIN_DAYS + 25)
        .strftime("%Y-%m-%d")
        .tolist()
    )
    feature_matrix = build_feature_matrix(symbols, dates)

    # --- Step 3: Filter to the target month ---
    if feature_matrix.empty:
        raise RuntimeError(f"Feature matrix is empty for month {month}.")

    month_data = feature_matrix[
        feature_matrix["date"].astype(str).str.startswith(month)
    ]
    if month_data.empty:
        raise RuntimeError(f"No feature rows found for month {month}.")

    # --- Step 4: SHAP summary ---
    shap_summary = compute_shap_summary(model, month_data)

    # --- Step 5: Claude review ---
    # We use placeholder values for equity_change_pct and top_tickers here
    # because the pipeline does not yet persist a monthly equity curve or
    # position history.  Callers that have this data should call
    # generate_attribution_report() directly with real values.
    equity_change_pct = 0.0   # unknown without equity-curve persistence
    top_tickers: list[str] = []  # unknown without position-history persistence

    report_text = generate_attribution_report(
        shap_summary, month, equity_change_pct, top_tickers
    )

    # Prepend a header so the file is self-contained even without the context.
    # We use to_string() rather than to_markdown() to avoid a hard dependency
    # on the optional `tabulate` package.
    shap_table_str = shap_summary.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    header = (
        f"# Monthly Attribution Report — {month}\n\n"
        f"**Portfolio return:** {equity_change_pct:+.1%}\n\n"
        f"## SHAP Feature Importance\n\n"
        f"```\n{shap_table_str}\n```\n\n"
        f"---\n\n"
    )
    full_report = header + report_text

    # --- Step 6: Save ---
    return save_report(full_report, month)
