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

import os
from pathlib import Path
import json

import anthropic
from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


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

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
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

    # The response content is guaranteed to be valid JSON that matches our schema.
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    return parsed["decision"], parsed["reasoning"]
