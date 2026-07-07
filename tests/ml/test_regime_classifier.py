import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_classify_regime_returns_valid_label():
    """classify_regime must always return one of the 5 valid labels."""
    from src.ml import regime_classifier

    macro_df = pd.DataFrame([{
        "date": "2024-01-01",
        "t10y2y":   0.5,
        "unrate":   3.7,
        "fedfunds": 5.25,
    }])

    with patch("src.ml.regime_classifier.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text='{"regime": "bull_trending", "reasoning": "positive yield curve"}')
        ]
        result = regime_classifier.classify_regime(macro_df)

    assert result in regime_classifier.VALID_REGIMES


def test_classify_regime_defaults_to_sideways_on_bad_json():
    """If Claude returns unparseable JSON, should default to 'sideways'."""
    from src.ml import regime_classifier

    macro_df = pd.DataFrame([{"date": "2024-01-01", "t10y2y": 0.0, "unrate": 4.0, "fedfunds": 5.0}])

    with patch("src.ml.regime_classifier.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text="not valid json at all")
        ]
        result = regime_classifier.classify_regime(macro_df)

    assert result == "sideways"


def test_classify_regime_defaults_to_sideways_on_invalid_label():
    """If Claude returns a valid JSON but unknown regime label, default to 'sideways'."""
    from src.ml import regime_classifier

    macro_df = pd.DataFrame([{"date": "2024-01-01", "t10y2y": 0.0, "unrate": 4.0, "fedfunds": 5.0}])

    with patch("src.ml.regime_classifier.anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text='{"regime": "extreme_chaos", "reasoning": "unknown regime"}')
        ]
        result = regime_classifier.classify_regime(macro_df)

    assert result == "sideways"


def test_regime_feature_weights_bull_trending():
    from src.ml.regime_classifier import get_regime_feature_weights

    weights = get_regime_feature_weights("bull_trending")
    assert weights.get("mom_12m", 1.0) >= 1.5   # momentum up-weighted
    assert all(v > 0 for v in weights.values())  # all weights positive


def test_regime_feature_weights_bear():
    from src.ml.regime_classifier import get_regime_feature_weights

    weights = get_regime_feature_weights("bear")
    assert weights.get("vol_20d", 1.0) >= 1.5   # vol signal up-weighted in bear
    assert weights.get("mom_12m", 1.0) <= 0.5   # momentum down-weighted


def test_regime_feature_weights_sideways_returns_empty():
    from src.ml.regime_classifier import get_regime_feature_weights

    weights = get_regime_feature_weights("sideways")
    assert weights == {}  # no modulation in sideways


def test_classify_regime_empty_df():
    """Empty DataFrame should return 'sideways'."""
    from src.ml import regime_classifier

    result = regime_classifier.classify_regime(pd.DataFrame())
    assert result == "sideways"
