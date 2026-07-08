"""Tests for structured JSON proposals in src/execution/attribution.py

Covers:
  - generate_attribution_report(model, feature_matrix, month) returns a dict
    with "prose" and "proposals" keys
  - proposals is an empty list when the response has no JSON block
  - the markdown file is saved to config.ATTR_DIR
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import config


def test_attribution_returns_proposals(tmp_path, monkeypatch):
    from src.execution import attribution

    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)

    mock_model = MagicMock()
    mock_model.feature_name_ = ["mom_1m", "mom_3m"]

    fm = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-01", "2024-05-02"]),
        "ticker": ["AAPL", "MSFT"],
        "mom_1m": [0.5, -0.3],
        "mom_3m": [0.2, 0.1],
        "target": [0.7, 0.3],
    })

    claude_response = """## Attribution

Feature mom_1m drove most returns this month.

```json
[
  {"type": "weight_adjustment", "target": "mom_1m", "action": "increase_weight", "rationale": "highest IC this month"}
]
```"""

    with patch("src.execution.attribution.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text=claude_response)]
        with patch("src.execution.attribution.shap") as mock_shap:
            mock_explainer = MagicMock()
            mock_shap.TreeExplainer.return_value = mock_explainer
            mock_explainer.shap_values.return_value = [[0.3, 0.1], [0.2, 0.05]]
            result = attribution.generate_attribution_report(mock_model, fm, month="2024-05")

    assert "prose" in result
    assert "proposals" in result
    assert isinstance(result["proposals"], list)
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["target"] == "mom_1m"


def test_attribution_empty_proposals_on_no_json(tmp_path, monkeypatch):
    from src.execution import attribution

    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)
    mock_model = MagicMock()
    mock_model.feature_name_ = ["mom_1m"]
    fm = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-01"]),
        "ticker": ["AAPL"],
        "mom_1m": [0.5],
        "target": [0.7],
    })

    with patch("src.execution.attribution.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="No proposals needed.")]
        with patch("src.execution.attribution.shap") as mock_shap:
            mock_shap.TreeExplainer.return_value.shap_values.return_value = [[0.1]]
            result = attribution.generate_attribution_report(mock_model, fm, month="2024-05")

    # No JSON block in response → proposals should be empty list, not error
    assert result["proposals"] == []


def test_equity_change_pct_computed_from_real_curve(tmp_path, monkeypatch):
    """#17c: the month filter used to compare the RangeIndex against `month`
    instead of the "date" column, so _equity_change_pct was silently always
    0.0. With a real equity curve for the target month, the prompt sent to
    Claude must contain the ACTUAL non-zero return, not +0.0%."""
    from src.execution import attribution

    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    # A real +10% move within the target month (2024-05), plus a row from an
    # adjacent month that must NOT be included in the calculation.
    curve = pd.DataFrame({
        "date":   pd.to_datetime(["2024-04-30", "2024-05-01", "2024-05-15", "2024-05-30"]),
        "equity": [999_999.0,    100_000.0,    105_000.0,     110_000.0],
    })
    curve.to_parquet(tmp_path / "equity_curve.parquet", index=False)

    mock_model = MagicMock()
    mock_model.feature_name_ = ["mom_1m"]
    fm = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-01"]),
        "ticker": ["AAPL"],
        "mom_1m": [0.5],
        "target": [0.7],
    })

    with patch("src.execution.attribution.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text="Report.\n```json\n[]\n```")
        ]
        with patch("src.execution.attribution.shap") as mock_shap:
            mock_shap.TreeExplainer.return_value.shap_values.return_value = [[0.1]]
            attribution.generate_attribution_report(mock_model, fm, month="2024-05")

    call_kwargs = mock_client.messages.create.call_args
    prompt = call_kwargs.kwargs["messages"][0]["content"]
    # (110,000 - 100,000) / 100,000 = +10.0% — the ATTRIBUTION_PROMPT format
    # string renders this as "+10.0%".
    assert "+10.0%" in prompt, (
        f"Expected a real +10.0% equity change in the prompt, got:\n{prompt}"
    )


def test_attribution_saves_markdown(tmp_path, monkeypatch):
    from src.execution import attribution

    monkeypatch.setattr(config, "ATTR_DIR", tmp_path)
    mock_model = MagicMock()
    mock_model.feature_name_ = ["mom_1m"]
    fm = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-01"]),
        "ticker": ["AAPL"],
        "mom_1m": [0.5],
        "target": [0.7],
    })

    with patch("src.execution.attribution.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="Report text.\n```json\n[]\n```")]
        with patch("src.execution.attribution.shap") as mock_shap:
            mock_shap.TreeExplainer.return_value.shap_values.return_value = [[0.1]]
            result = attribution.generate_attribution_report(mock_model, fm, month="2024-05")

    # Check the markdown file was saved
    saved_path = tmp_path / "2024-05-attribution.md"
    assert saved_path.exists()
    assert "Report text" in saved_path.read_text(encoding="utf-8")
