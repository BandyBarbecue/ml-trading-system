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

def test_after_close_headline_shifts_to_next_day():
    """A headline published after the 16:00 ET close must not affect the
    same day's score — it belongs to the NEXT day's window instead.

    17:15 ET on 2024-01-02 is 22:15 UTC (EST = UTC-5 in January).
    """
    from src.features import nlp_features

    late_headline = pd.DataFrame({
        "ticker": ["AAPL"],
        "headline": ["Apple announces after-hours buyback expansion"],
        "published_at": pd.to_datetime(["2024-01-02 22:15:00"]).tz_localize("UTC"),
        "source": ["Reuters"],
    })
    signals = [{
        "headline": "Apple announces after-hours buyback expansion",
        "event_type": "other",
        "primary_ticker": "AAPL",
        "direction": "bullish",
        "magnitude": "medium",
        "time_horizon": "1week",
        "confidence": 0.9,
        "affected_tickers": [],
    }]

    dates = pd.bdate_range("2024-01-02", "2024-01-03")
    scores = nlp_features._aggregate_scores(signals, dates, late_headline)

    same_day_score = scores["news_score"].loc["2024-01-02", "AAPL"]
    next_day_score = scores["news_score"].loc["2024-01-03", "AAPL"]

    assert same_day_score == 0.0, "A post-16:00-ET headline must not count for the publish day"
    assert next_day_score > 0.0, "A post-16:00-ET headline must count (at 2x recency) for the next day"


def test_null_affected_tickers_does_not_crash():
    """affected_tickers can be present but explicitly null (not just missing) —
    must not raise a TypeError when iterating over it."""
    from src.features import nlp_features

    headlines = pd.DataFrame({
        "ticker": ["AAPL"],
        "headline": ["Apple reports strong quarter"],
        "published_at": pd.to_datetime(["2024-01-02 14:00:00"]).tz_localize("UTC"),
        "source": ["Reuters"],
    })
    signals = [{
        "headline": "Apple reports strong quarter",
        "event_type": "earnings_beat",
        "primary_ticker": "AAPL",
        "direction": "bullish",
        "magnitude": "large",
        "time_horizon": "1month",
        "confidence": 0.9,
        "affected_tickers": None,   # explicit null, not a missing key
    }]

    dates = pd.bdate_range("2024-01-02", "2024-01-02")
    scores = nlp_features._aggregate_scores(signals, dates, headlines)  # must not raise
    assert scores["news_score"].loc["2024-01-02", "AAPL"] > 0


def test_extract_signals_allow_api_false_skips_api_for_uncached(tmp_path, monkeypatch):
    """allow_api=False must NEVER call Claude — an uncached headline is
    simply dropped from the result instead of being sent to the API. This
    is the safety valve that lets compute_news_features() widen its date
    window to cover historical news without ever triggering new (paid)
    extraction for headlines Claude has never seen."""
    from src.features import nlp_features
    monkeypatch.setattr(nlp_features, "CACHE_FILE", tmp_path / "cache.json")

    headlines = pd.DataFrame({
        "ticker": ["AAPL"],
        "headline": ["A headline never processed before"],
        "published_at": pd.to_datetime(["2024-01-02 14:00"]),
        "source": ["Reuters"],
    })

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        signals = nlp_features.extract_signals(headlines, allow_api=False)

    MockClient.assert_not_called()
    assert signals == []


def test_extract_signals_allow_api_false_still_returns_cached(tmp_path, monkeypatch):
    """allow_api=False must still serve already-cached headlines from disk —
    it only blocks NEW extraction, not the cache lookup itself."""
    from src.features import nlp_features
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(nlp_features, "CACHE_FILE", cache_path)

    headline_text = "Apple beats Q3 EPS by 9%, raises guidance"
    cached_signal = {
        "headline": headline_text,
        "event_type": "earnings_beat",
        "primary_ticker": "AAPL",
        "direction": "bullish",
        "magnitude": "large",
        "time_horizon": "1month",
        "confidence": 0.9,
        "affected_tickers": [],
    }
    # Cache key format matches nlp_features._cache_key: headline + "|" + summary[:200].
    cache_path.write_text(json.dumps({headline_text + "|": cached_signal}))

    headlines = pd.DataFrame({
        "ticker": ["AAPL"],
        "headline": [headline_text],
        "published_at": pd.to_datetime(["2024-01-02 14:00"]),
        "source": ["Reuters"],
    })

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        signals = nlp_features.extract_signals(headlines, allow_api=False)

    MockClient.assert_not_called()
    assert len(signals) == 1
    assert signals[0]["primary_ticker"] == "AAPL"


def test_compute_news_features_widens_window_and_anchors_as_of(monkeypatch):
    """compute_news_features must load enough history to cover the full
    `dates` span (not just NLP_DECAY_DAYS), anchored on dates.max() rather
    than real-world today. Otherwise, when `dates` spans ~2 years (a full
    historical matrix build), load_recent_news(days=NLP_DECAY_DAYS) would
    only ever fetch the last few REAL-calendar-days of headlines, leaving
    every older date in `dates` with zero headlines to aggregate over."""
    from src.features import nlp_features
    from src.data import fetch_news

    dates = pd.bdate_range("2024-01-02", periods=500)  # ~2 years, like a real build

    mock_load = MagicMock(return_value=pd.DataFrame())  # empty -> short-circuits early
    monkeypatch.setattr(fetch_news, "load_recent_news", mock_load)

    nlp_features.compute_news_features(dates)

    mock_load.assert_called_once()
    _, kwargs = mock_load.call_args
    expected_span = (dates.max() - dates.min()).days + nlp_features.NLP_DECAY_DAYS + 1
    assert kwargs["days"] == expected_span
    assert kwargs["as_of"] == dates.max().date()


def test_compute_news_features_allow_api_propagates(monkeypatch):
    """allow_api passed into compute_news_features must reach the
    extract_signals() call unchanged, so callers can force a hard
    no-API-calls guarantee for a historical/CI-safe rebuild."""
    from src.features import nlp_features
    from src.data import fetch_news

    dates = pd.bdate_range("2024-01-02", periods=5)
    fake_headlines = pd.DataFrame({
        "ticker": ["AAPL"],
        "headline": ["Some headline"],
        "published_at": pd.to_datetime(["2024-01-02 14:00"]),
        "source": ["Reuters"],
    })
    monkeypatch.setattr(fetch_news, "load_recent_news", MagicMock(return_value=fake_headlines))

    mock_extract = MagicMock(return_value=[])
    monkeypatch.setattr(nlp_features, "extract_signals", mock_extract)

    nlp_features.compute_news_features(dates, allow_api=False)

    mock_extract.assert_called_once()
    _, kwargs = mock_extract.call_args
    assert kwargs.get("allow_api") is False


def _mock_message(text: str) -> MagicMock:
    """Build a fake Anthropic response whose .content[0].text is *text*."""
    content = MagicMock()
    content.text = text
    message = MagicMock()
    message.content = [content]
    return message


def test_crash_mid_run_keeps_completed_batches(tmp_path, monkeypatch):
    """The cache must be saved after EVERY batch, not once at the end.

    2026-07-10 incident: the run died at batch ~70 of 145 (credits
    exhausted) and all 69 completed — and paid-for — batches were lost,
    because saving only happened after the full loop. A crash mid-run must
    keep everything extracted so far."""
    from src.features import nlp_features
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(nlp_features, "CACHE_FILE", cache_path)

    # 25 unique headlines with NLP_BATCH_SIZE=20 → two batches.
    n = nlp_features.NLP_BATCH_SIZE + 5
    headlines = pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "headline": [f"Unique headline number {i}" for i in range(n)],
        "published_at": pd.to_datetime(["2024-01-02 14:00"] * n),
        "source": ["Reuters"] * n,
    })
    batch1_response = json.dumps([
        {
            "headline": f"Unique headline number {i}",
            "event_type": "other",
            "primary_ticker": "AAPL",
            "direction": "neutral",
            "magnitude": "small",
            "time_horizon": "1week",
            "confidence": 0.5,
            "affected_tickers": [],
        }
        for i in range(nlp_features.NLP_BATCH_SIZE)
    ])

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.side_effect = [
            _mock_message(batch1_response),
            RuntimeError("credit balance too low"),  # batch 2 dies
        ]
        with pytest.raises(RuntimeError):
            nlp_features.extract_signals(headlines)

    # Batch 1's results must have survived the batch-2 crash.
    assert cache_path.exists(), "cache file was never written before the crash"
    saved = json.loads(cache_path.read_text())
    assert len(saved) == nlp_features.NLP_BATCH_SIZE


def test_no_signal_headlines_cached_and_not_resent(tmp_path, monkeypatch):
    """A headline Claude deliberately skips (batch parsed OK, but no entry
    for it) must be cached as a no-signal sentinel: it produces no signal in
    the result, and — crucially — is never paid for again on later runs."""
    from src.features import nlp_features
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(nlp_features, "CACHE_FILE", cache_path)

    headlines = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "headline": ["Signal-worthy headline", "Boring headline Claude skips"],
        "published_at": pd.to_datetime(["2024-01-02 14:00", "2024-01-02 15:00"]),
        "source": ["Reuters", "Reuters"],
    })
    # Claude returns an entry for only ONE of the two items in the batch.
    response = json.dumps([{
        "headline": "Signal-worthy headline",
        "event_type": "other",
        "primary_ticker": "AAPL",
        "direction": "bullish",
        "magnitude": "medium",
        "time_horizon": "1week",
        "confidence": 0.8,
        "affected_tickers": [],
    }])

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_message(response)
        signals_first = nlp_features.extract_signals(headlines)
        first_calls = MockClient.return_value.messages.create.call_count
        signals_second = nlp_features.extract_signals(headlines)  # same input again
        second_calls = MockClient.return_value.messages.create.call_count

    # The skipped headline yields no signal but must not be re-sent.
    assert len(signals_first) == 1
    assert signals_first[0]["primary_ticker"] == "AAPL"
    assert signals_second == signals_first
    assert second_calls == first_calls, "no-signal headline was re-sent to the API"


def test_batch_parse_failure_is_not_cached(tmp_path, monkeypatch):
    """If a WHOLE batch fails to parse (truncated JSON etc.), that's a
    transient error — nothing may be cached for it, so the next run retries
    instead of permanently treating those headlines as no-signal."""
    from src.features import nlp_features
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(nlp_features, "CACHE_FILE", cache_path)

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_message(
            "definitely { not valid json"
        )
        nlp_features.extract_signals(SAMPLE_HEADLINES)
        nlp_features.extract_signals(SAMPLE_HEADLINES)
        assert MockClient.return_value.messages.create.call_count == 2, (
            "a parse failure must be retried on the next run, not swallowed"
        )

    assert json.loads(cache_path.read_text()) == {}


def test_api_cutoff_blocks_paid_extraction_of_old_headlines(tmp_path, monkeypatch):
    """Uncached headlines OLDER than api_cutoff must never reach the API,
    even with allow_api=True. This is the circuit-breaker that caps the
    damage of any future cache invalidation at a few days of headlines
    instead of the whole 2-year window (the 2026-07-10 credits burn)."""
    from src.features import nlp_features
    monkeypatch.setattr(nlp_features, "CACHE_FILE", tmp_path / "cache.json")

    headlines = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "headline": ["Ancient uncached headline", "Fresh uncached headline"],
        "published_at": pd.to_datetime(
            ["2024-01-02 14:00", "2024-03-01 14:00"]
        ).tz_localize("UTC"),
        "source": ["Reuters", "Reuters"],
    })
    response = json.dumps([{
        "headline": "Fresh uncached headline",
        "event_type": "other",
        "primary_ticker": "MSFT",
        "direction": "bullish",
        "magnitude": "medium",
        "time_horizon": "1week",
        "confidence": 0.8,
        "affected_tickers": [],
    }])

    with patch("src.features.nlp_features.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_message(response)
        signals = nlp_features.extract_signals(
            headlines,
            allow_api=True,
            api_cutoff=pd.Timestamp("2024-02-01", tz="UTC"),
        )
        prompt_sent = MockClient.return_value.messages.create.call_args.kwargs[
            "messages"
        ][0]["content"]

    # Only the fresh headline was sent; the old one was silently dropped.
    assert "Fresh uncached headline" in prompt_sent
    assert "Ancient uncached headline" not in prompt_sent
    assert len(signals) == 1
    assert signals[0]["primary_ticker"] == "MSFT"


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
