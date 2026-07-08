"""Tests for src/feedback/ai_reviewer.py

The Anthropic API is fully mocked — no network calls, no API key needed.
These tests pin down the hardening contract: review_signal() must NEVER
raise on API/response problems, and every failure path must come back as
a conservative ("reject", <explanation>) rather than a crash.
"""

import sys
from pathlib import Path
from typing import Any

import anthropic
import pytest

# Ensure the project root is on sys.path so `import config` works.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.feedback import ai_reviewer


# ---------------------------------------------------------------------------
# Fakes — minimal stand-ins for the Anthropic SDK response objects
# ---------------------------------------------------------------------------

class _Block:
    """Mimics one content block: has .type and (for text blocks) .text."""

    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        self.text = text


class _Response:
    """Mimics a Messages API response: .stop_reason and .content."""

    def __init__(self, stop_reason: str, content: list[_Block]) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    """Either returns a canned response or raises a canned exception."""

    def __init__(self, response: _Response | None = None,
                 error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_kwargs: dict[str, Any] | None = None  # captured for assertions

    def create(self, **kwargs: Any) -> _Response:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


class _FakeConnectionError(anthropic.APIConnectionError):
    """A raisable APIConnectionError without the SDK's httpx plumbing."""

    def __init__(self) -> None:
        # Bypass the parent __init__ (which demands an httpx.Request);
        # isinstance checks in the except clause still match.
        Exception.__init__(self, "simulated connection failure")


def _patch_client(monkeypatch: pytest.MonkeyPatch, messages: _FakeMessages) -> None:
    """Make ai_reviewer's `anthropic.Anthropic()` return our fake client."""
    monkeypatch.setattr(ai_reviewer.anthropic, "Anthropic",
                        lambda: _FakeClient(messages))


def _call() -> tuple[str, str]:
    """Invoke review_signal with a fixed, valid set of arguments."""
    return ai_reviewer.review_signal(
        symbol="MSFT",
        signal_type="buy",
        signal_date="2025-10-15",
        price=420.00,
        short_ma=415.00,
        long_ma=410.00,
        portfolio_value=10_000.00,
        risk_approved=True,
        risk_reason="approved",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_parses_decision_and_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _Response("end_turn", [
            _Block("thinking"),  # thinking block first, like the real API
            _Block("text", '{"decision": "approve", "reasoning": "Solid golden cross."}'),
        ])
        messages = _FakeMessages(response=response)
        _patch_client(monkeypatch, messages)

        decision, reasoning = _call()

        assert decision == "approve"
        assert reasoning == "Solid golden cross."

    def test_uses_config_model_and_bigger_token_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model must come from config (not be hardcoded) and max_tokens must
        leave room for adaptive thinking before the JSON is emitted."""
        response = _Response("end_turn", [
            _Block("text", '{"decision": "reject", "reasoning": "Weak cross."}'),
        ])
        messages = _FakeMessages(response=response)
        _patch_client(monkeypatch, messages)

        _call()

        assert messages.last_kwargs is not None
        assert messages.last_kwargs["model"] == config.CLAUDE_REVIEW_MODEL
        assert messages.last_kwargs["max_tokens"] == 2000


class TestGuards:
    """Every failure mode returns a conservative reject — never an exception."""

    def test_refusal_stop_reason_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = _FakeMessages(response=_Response("refusal", []))
        _patch_client(monkeypatch, messages)

        decision, reasoning = _call()

        assert decision == "reject"
        assert "refusal" in reasoning.lower()

    def test_max_tokens_stop_reason_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = _FakeMessages(response=_Response("max_tokens", [
            _Block("thinking"),  # got cut off before any text was emitted
        ]))
        _patch_client(monkeypatch, messages)

        decision, reasoning = _call()

        assert decision == "reject"
        assert "max_tokens" in reasoning

    def test_missing_text_block_rejects_instead_of_stopiteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old bare next(...) raised StopIteration here — must not crash."""
        messages = _FakeMessages(response=_Response("end_turn", [
            _Block("thinking"),  # no text block at all
        ]))
        _patch_client(monkeypatch, messages)

        decision, reasoning = _call()

        assert decision == "reject"
        assert "no text" in reasoning.lower()

    def test_api_connection_error_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = _FakeMessages(error=_FakeConnectionError())
        _patch_client(monkeypatch, messages)

        decision, reasoning = _call()

        assert decision == "reject"
        assert "unavailable" in reasoning.lower()

    def test_unparseable_json_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = _FakeMessages(response=_Response("end_turn", [
            _Block("text", "not json at all"),
        ]))
        _patch_client(monkeypatch, messages)

        decision, reasoning = _call()

        assert decision == "reject"
        assert "parsed" in reasoning.lower()
