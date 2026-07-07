import pandas as pd
import json
import pytest
from unittest.mock import MagicMock, patch

SAMPLE_HEADLINES = pd.DataFrame({
    "ticker": ["AAPL", "MSFT"],
    "headline": [
        "Apple beats Q3 EPS by 9%, raises guidance",
        "Microsoft cloud growth slows to 18%",
    ],
    "published_at": pd.to_datetime(["2024-01-02 14:00", "2024-01-02 09:30"]),
    "source": ["Reuters", "Bloomberg"],
})

MOCK_CLAUDE_RESPONSE = json.dumps([
    {
        "headline": "Apple beats Q3 EPS by 9%, raises guidance",
        "event_type": "earnings_beat",
        "primary_ticker": "AAPL",
        "direction": "bullish",
        "magnitude": "large",
        "time_horizon": "1month",
        "confidence": 0.92,
        "affected_tickers": [{"ticker": "TSMC", "direction": "bullish", "reason": "supplier"}],
    },
    {
        "headline": "Microsoft cloud growth slows to 18%",
        "event_type": "guidance_miss",
        "primary_ticker": "MSFT",
        "direction": "bearish",
        "magnitude": "medium",
        "time_horizon": "1month",
        "confidence": 0.78,
        "affected_tickers": [],
    },
])

def test_extract_signals_calls_claude(tmp_path, monkeypatch):
    from src.features import nlp_features
    monkeypatch.setattr(nlp_features, "CACHE_FILE", tmp_path / "cache.json")

    mock_content = MagicMock()
    mock_content.text = MOCK_CLAUDE_RESPONSE
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        signals = nlp_features.extract_signals(SAMPLE_HEADLINES)

    assert len(signals) == 2
    assert signals[0]["primary_ticker"] == "AAPL"
    assert signals[0]["direction"] == "bullish"

def test_supply_chain_score_propagated(tmp_path, monkeypatch):
    from src.features import nlp_features
    monkeypatch.setattr(nlp_features, "CACHE_FILE", tmp_path / "cache.json")

    signals = json.loads(MOCK_CLAUDE_RESPONSE)
    dates = pd.bdate_range("2024-01-02", "2024-01-03")
    scores = nlp_features._aggregate_scores(signals, dates, SAMPLE_HEADLINES)

    assert "AAPL" in scores["news_score"].columns
    # TSMC gets a supply chain score from Apple's signal
    assert "TSMC" in scores["supply_chain_score"].columns
    assert scores["supply_chain_score"]["TSMC"].iloc[0] > 0

def test_cache_prevents_reprocessing(tmp_path, monkeypatch):
    from src.features import nlp_features
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(nlp_features, "CACHE_FILE", cache_path)

    mock_content = MagicMock()
    mock_content.text = MOCK_CLAUDE_RESPONSE
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        nlp_features.extract_signals(SAMPLE_HEADLINES)
        call_count_first = MockClient.return_value.messages.create.call_count
        nlp_features.extract_signals(SAMPLE_HEADLINES)  # same headlines
        call_count_second = MockClient.return_value.messages.create.call_count

    # Second call should not trigger additional API calls
    assert call_count_second == call_count_first
