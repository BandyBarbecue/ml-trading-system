"""AI-powered trade reviewer: asks Claude to analyse and decide on MA crossover signals.

Replaces the human prompt in the feedback loop with a Claude API call.
Claude receives the full signal context — price, moving averages, portfolio state,
and the risk engine verdict — then returns a structured approve/reject decision
with a written rationale.

Usage (called from ai_cli.py — not meant to be run directly):
    from src.feedback.ai_reviewer import review_signal

    decision, reasoning = review_signal(
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

Requires ANTHROPIC_API_KEY in .env (see .env.example).
"""

import json
import logging
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Make sure the project root is importable regardless of how this module is run,
# so `import config` works (same pattern as src/risk/engine.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

log = logging.getLogger(__name__)


def review_signal(
    symbol: str,
    signal_type: str,
    signal_date: str,
    price: float,
    short_ma: float,
    long_ma: float,
    portfolio_value: float,
    risk_approved: bool,
    risk_reason: str,
) -> tuple[str, str]:
    """Ask Claude to approve or reject a trade signal.

    Args:
        symbol:           Ticker, e.g. "MSFT".
        signal_type:      "buy" or "sell".
        signal_date:      ISO date string, e.g. "2025-10-15".
        price:            Current close price.
        short_ma:         20-day moving average.
        long_ma:          50-day moving average.
        portfolio_value:  Cash + open positions, in USD.
        risk_approved:    Whether the risk engine approved the trade.
        risk_reason:      Human-readable verdict from the risk engine.

    Returns:
        (decision, reasoning)
        decision is one of: "approve", "reject"
        reasoning is Claude's written rationale (one concise paragraph).

        This function NEVER raises on API/response problems. Any failure —
        network error, API error, safety refusal, truncated output, missing
        or unparseable JSON — returns ("reject", <explanation>). Rejecting
        is the conservative default: a trade that isn't reviewed isn't taken.
    """
    client = anthropic.Anthropic()

    # Explain the strategy context so Claude can reason about it correctly.
    trend = "bullish" if short_ma > long_ma else "bearish"
    ma_gap_pct = abs(short_ma - long_ma) / long_ma * 100
    signal_direction = "golden cross (short MA crossed above long MA)" if signal_type == "buy" \
        else "death cross (short MA crossed below long MA)"

    system_prompt = """You are a disciplined quantitative trading analyst reviewing trade signals
for a paper-trading (no real money) learning project.

The system uses a 20-day / 50-day moving-average crossover strategy on US equities.
A 'buy' signal fires when the 20-day MA crosses above the 50-day MA (golden cross).
A 'sell' signal fires when the 20-day MA crosses below the 50-day MA (death cross).

Your job: decide whether to APPROVE or REJECT this specific signal.

Consider:
- Is the MA gap meaningful (>1%) or is this likely a false cross?
- Does the risk engine verdict change your view?
- Is the portfolio well-positioned to take this trade?
- For sells: are we protecting gains or cutting a loser?

Be concise. One decision. One paragraph of reasoning. No hedging."""

    user_message = f"""SIGNAL REVIEW REQUEST
=====================
Symbol:          {symbol}
Signal type:     {signal_type.upper()} — {signal_direction}
Date:            {signal_date}

Price context:
  Close price:   ${price:.2f}
  20-day MA:     ${short_ma:.2f}
  50-day MA:     ${long_ma:.2f}
  Trend:         {trend.upper()} (MA gap: {ma_gap_pct:.2f}%)

Portfolio:
  Value:         ${portfolio_value:,.2f}

Risk engine:
  Verdict:       {"APPROVED" if risk_approved else "VETOED"}
  Reason:        {risk_reason}

Should this trade be approved or rejected?"""

    # Wrap the API call so a network blip or server error can't crash the
    # whole review loop — same graceful-fallback pattern as
    # src/ml/regime_classifier.py (log a warning, return a safe default).
    try:
        response = client.messages.create(
            # Model comes from config (single source of truth) — it was
            # previously hardcoded here, which silently diverged from
            # CLAUDE_REVIEW_MODEL.
            model=config.CLAUDE_REVIEW_MODEL,
            # 2000, not 512: adaptive thinking consumes output-token budget
            # BEFORE the JSON answer is emitted, so a small cap could cut the
            # response off mid-thought and yield no JSON at all.
            max_tokens=2000,
            thinking={"type": "adaptive"},
            system=system_prompt,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": ["approve", "reject"],
                                "description": "The trade decision.",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "One concise paragraph explaining the decision.",
                            },
                        },
                        "required": ["decision", "reasoning"],
                        "additionalProperties": False,
                    },
                }
            },
            messages=[{"role": "user", "content": user_message}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        log.warning("AI review API call failed, rejecting conservatively: %s", exc)
        return "reject", f"AI review unavailable ({type(exc).__name__}) — rejected conservatively, not on merit."

    # Guard the stop reason BEFORE reading content. A "refusal" response has
    # empty/partial content, and "max_tokens" means the JSON was cut off —
    # in both cases there is no trustworthy decision to extract.
    if response.stop_reason == "refusal":
        return "reject", "AI reviewer declined the request (safety refusal) — rejected conservatively, not on merit."
    if response.stop_reason == "max_tokens":
        return "reject", "AI review was truncated (hit max_tokens) — rejected conservatively, not on merit."

    # next(..., None) instead of bare next(...): a bare next() raises
    # StopIteration if no text block exists (e.g. thinking-only output),
    # which used to crash the caller.
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        return "reject", "AI review returned no text content — rejected conservatively, not on merit."

    try:
        parsed = json.loads(text)
        return parsed["decision"], parsed["reasoning"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("AI review returned unparseable output: %s", exc)
        return "reject", f"AI review output could not be parsed ({type(exc).__name__}) — rejected conservatively, not on merit."
