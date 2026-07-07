"""Tests for src/execution/attribution.py

Covers:
  - compute_shap_summary: shape, column names, sort order
  - save_report: file is created at the correct path with correct contents
  - generate_attribution_report: Claude SDK is called with the right arguments
  - run_attribution: full pipeline wiring (all heavy deps mocked)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import after the project is on sys.path (pytest handles this via conftest /
# pyproject.toml).  We import the module early so monkeypatching works correctly.
import src.execution.attribution as attr_mod
import config
from src.ml.model import FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model():
    """Train a real (tiny) LightGBM model on synthetic data.

    SHAP's TreeExplainer needs the real model internals — you can't mock it.
    50 rows and 5 trees is enough to make SHAP happy without being slow.
    """
    import lightgbm as lgb

    X = pd.DataFrame(np.random.randn(50, len(FEATURE_COLS)), columns=FEATURE_COLS)
    y = np.random.rand(50)
    model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
    model.fit(X, y)
    return model


def _make_feature_df(n: int = 50) -> pd.DataFrame:
    """Create a synthetic feature DataFrame with all FEATURE_COLS."""
    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data["date"] = "2026-05-01"
    data["ticker"] = "AAPL"
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# test_shap_summary_shape
# ---------------------------------------------------------------------------

def test_shap_summary_shape():
    """compute_shap_summary returns exactly one row per feature column."""
    model = _make_tiny_model()
    feature_df = _make_feature_df(50)

    result = attr_mod.compute_shap_summary(model, feature_df, n_background=20)

    assert list(result.columns) == ["feature", "mean_abs_shap"], (
        f"Expected columns ['feature', 'mean_abs_shap'], got {list(result.columns)}"
    )
    assert len(result) == len(FEATURE_COLS), (
        f"Expected {len(FEATURE_COLS)} rows (one per feature), got {len(result)}"
    )


# ---------------------------------------------------------------------------
# test_shap_summary_sorted
# ---------------------------------------------------------------------------

def test_shap_summary_sorted():
    """mean_abs_shap column must be in descending order (most important first)."""
    model = _make_tiny_model()
    feature_df = _make_feature_df(50)

    result = attr_mod.compute_shap_summary(model, feature_df, n_background=20)

    values = result["mean_abs_shap"].tolist()
    assert values == sorted(values, reverse=True), (
        "mean_abs_shap is not sorted descending: " + str(values)
    )


# ---------------------------------------------------------------------------
# test_save_report_creates_file
# ---------------------------------------------------------------------------

def test_save_report_creates_file(tmp_path, monkeypatch):
    """save_report writes the correct content to the expected path."""
    # Redirect config.ATTR_DIR to a temp directory so we never touch real data/
    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)

    report_text = "## Test Report\n"
    returned_path = attr_mod.save_report(report_text, "2026-05")

    expected_path = tmp_path / "2026-05-attribution.md"
    assert returned_path == expected_path, (
        f"Returned path {returned_path!r} != expected {expected_path!r}"
    )
    assert expected_path.exists(), "File was not created on disk."
    assert expected_path.read_text(encoding="utf-8") == report_text, (
        "File contents do not match the report_text argument."
    )


# ---------------------------------------------------------------------------
# test_generate_attribution_report_calls_claude
# ---------------------------------------------------------------------------

def test_generate_attribution_report_calls_claude():
    """generate_attribution_report calls client.messages.create with the right model."""
    # Build a minimal shap_summary DataFrame
    shap_summary = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": np.linspace(0.1, 0.01, len(FEATURE_COLS)),
    })

    month = "2026-05"
    equity_change_pct = 0.032
    top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    # Mock the Anthropic class so no real HTTP call is made
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="# Claude Review\nAll good.")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    mock_anthropic_cls = MagicMock(return_value=mock_client)

    with patch("anthropic.Anthropic", mock_anthropic_cls):
        result = attr_mod.generate_attribution_report(
            shap_summary, month, equity_change_pct, top_tickers
        )

    # The SDK client must have been instantiated
    mock_anthropic_cls.assert_called_once()

    # messages.create must have been called
    mock_client.messages.create.assert_called_once()

    # The call must use the correct model
    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs.get("model") == config.CLAUDE_REVIEW_MODEL, (
        f"Expected model={config.CLAUDE_REVIEW_MODEL!r}, "
        f"got {call_kwargs.kwargs.get('model')!r}"
    )

    # The prompt must contain the month string
    messages_arg = call_kwargs.kwargs.get("messages", [])
    assert any(month in msg.get("content", "") for msg in messages_arg), (
        f"Month string {month!r} not found in messages payload."
    )

    # The returned value is the text from Claude's response
    assert result == "# Claude Review\nAll good."


# ---------------------------------------------------------------------------
# test_run_attribution_returns_path
# ---------------------------------------------------------------------------

def test_run_attribution_returns_path(monkeypatch):
    """run_attribution wires all steps and returns a Path."""
    fake_model = MagicMock()
    fake_feature_matrix = _make_feature_df(30)
    fake_feature_matrix["date"] = "2026-05-01"  # ensure month filter hits

    fake_shap_summary = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": np.linspace(0.1, 0.01, len(FEATURE_COLS)),
    })

    fake_path = Path("/fake/data/attribution/2026-05-attribution.md")
    fake_report_text = "# Fake Report\n"

    monkeypatch.setattr("src.ml.model.load_latest_model", lambda: fake_model)
    monkeypatch.setattr(
        "src.features.feature_matrix.build_feature_matrix",
        lambda symbols, dates: fake_feature_matrix,
    )
    monkeypatch.setattr(
        "src.data.sp500_universe.get_symbols",
        lambda: ["AAPL", "MSFT"],
    )
    monkeypatch.setattr(
        attr_mod, "compute_shap_summary",
        lambda model, df, n_background=100: fake_shap_summary,
    )
    monkeypatch.setattr(
        attr_mod, "generate_attribution_report",
        lambda shap_summary, month, equity_change_pct, top_tickers: fake_report_text,
    )
    monkeypatch.setattr(
        attr_mod, "save_report",
        lambda report_text, month: fake_path,
    )

    result = attr_mod.run_attribution("2026-05")

    assert isinstance(result, Path), (
        f"run_attribution should return a Path, got {type(result)}"
    )
    assert result == fake_path
