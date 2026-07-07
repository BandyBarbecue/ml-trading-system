"""Paper-trading execution layer: converts portfolio weight deltas into
Alpaca market orders.

This module is the *only* place where orders are submitted.  It enforces two
invariants:
  1. Paper mode is ALWAYS on (config.ALPACA_PAPER is hardcoded True).
  2. Only integer share quantities are sent (Alpaca's paper API rejects
     fractional shares for most equity symbols).

Analogy: think of this as the cashier at a shop.  The strategy (upstream) hands
over a shopping list of weight deltas ("buy 10% more AAPL, sell 5% of MSFT").
The cashier converts those percentages into actual share counts, checks the
price sticker, and submits the transaction — but never buys fractions of a
share, and never spends more than the budget allows.
"""

import logging
import math

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _make_trading_client() -> TradingClient:
    """Mockable factory that creates an Alpaca TradingClient.

    CRITICAL: paper=config.ALPACA_PAPER is always True.  This is not a
    configurable option — it is hardcoded in config.py and must never be
    changed to False or driven by an environment variable.
    """
    return TradingClient(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        paper=config.ALPACA_PAPER,  # always True — never live
    )


def _make_data_client() -> StockHistoricalDataClient:
    """Mockable factory for the market-data client (quotes, bars).

    Alpaca separates trading operations (TradingClient) from market data
    (StockHistoricalDataClient) — you need both for a full rebalance.
    """
    return StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def get_account_value(client: TradingClient) -> float:
    """Return total account equity in dollars.

    Alpaca returns equity as a string (e.g. "50000.00"), so we convert it.

    Parameters
    ----------
    client:  An authenticated TradingClient.

    Returns
    -------
    Total equity as a float.
    """
    account = client.get_account()
    return float(account.equity)


def get_current_positions(client: TradingClient) -> pd.Series:
    """Return current portfolio weights indexed by ticker symbol.

    Weight = market_value / total_equity.  An empty portfolio returns an
    empty Series rather than raising an error.

    Think of this as a snapshot of "what fraction of my total money is
    currently invested in each stock."

    Parameters
    ----------
    client:  An authenticated TradingClient.

    Returns
    -------
    pd.Series  — index is ticker symbol (str), values are portfolio weights
                 (float, 0.0–1.0).  Empty if no positions.
    """
    positions = client.get_all_positions()

    if not positions:
        return pd.Series(dtype=float)

    account = client.get_account()
    total_equity = float(account.equity)

    weights = {
        pos.symbol: float(pos.market_value) / total_equity
        for pos in positions
    }
    return pd.Series(weights)


# ---------------------------------------------------------------------------
# Order submission
# ---------------------------------------------------------------------------

def submit_rebalance(
    client: TradingClient,
    trade_deltas: pd.Series,  # positive = buy, negative = sell
    account_value: float,     # total account equity in $
) -> list[dict]:
    """Convert weight deltas into integer share counts and submit market orders.

    Each entry in *trade_deltas* is a fractional weight delta (e.g. +0.10 means
    "increase this stock's portfolio weight by 10 percentage points").  We turn
    that into a dollar amount, divide by the current bid/ask price, and floor
    to the nearest whole share.

    - Buys  use ask_price (the price at which we can buy).
    - Sells use bid_price (the price at which we can sell).
    - If the resulting share count < 1, the trade is too small to execute and
      is skipped silently.
    - If the quote fetch fails for any ticker, that ticker is skipped with a
      warning (we never crash a full rebalance because one ticker is mis-quoted).

    Parameters
    ----------
    client:         An authenticated TradingClient.
    trade_deltas:   pd.Series of weight deltas, indexed by ticker symbol.
    account_value:  Total account equity in dollars (used to size orders).

    Returns
    -------
    List of order result dicts, one per successfully submitted order:
        {"ticker": str, "side": str, "shares": int, "status": str}
    """
    results: list[dict] = []
    data_client = _make_data_client()

    for ticker, delta in trade_deltas.items():
        # Skip zero deltas — nothing to do.
        if delta == 0:
            continue

        # --- Fetch current quote -----------------------------------------------
        # Quotes come from StockHistoricalDataClient, not TradingClient.
        try:
            quote_response = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=[ticker])
            )
            quote = quote_response[ticker]
        except Exception as exc:
            logger.warning("Could not fetch quote for %s: %s — skipping.", ticker, exc)
            continue

        # Use ask for buys, bid for sells.
        if delta > 0:
            price = float(quote.ask_price)
            side = OrderSide.BUY
        else:
            price = float(quote.bid_price)
            side = OrderSide.SELL

        # Guard against zero / missing price.
        if not price or price <= 0:
            logger.warning("Invalid price %.4f for %s — skipping.", price, ticker)
            continue

        # --- Convert weight delta to share count --------------------------------
        # Dollar amount = |weight delta| × total account value
        # Shares        = dollar amount / price per share  (floored to integer)
        dollar_amount = abs(delta) * account_value
        shares = math.floor(dollar_amount / price)

        if shares < 1:
            # Too small to trade; quietly skip rather than sending a 0-share order.
            logger.debug(
                "Skipping %s: delta=%.6f, dollar_amount=%.2f, price=%.2f → %d shares",
                ticker, delta, dollar_amount, price, shares,
            )
            continue

        # --- Submit the order ---------------------------------------------------
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        try:
            order = client.submit_order(order_request)
            results.append({
                "ticker": ticker,
                "side":   side.value,   # "buy" or "sell" string
                "shares": shares,
                "status": str(order.status),
            })
            logger.info(
                "Submitted %s %d shares of %s (id=%s, status=%s)",
                side.value, shares, ticker, order.id, order.status,
            )
        except Exception as exc:
            logger.warning("Order submission failed for %s: %s — skipping.", ticker, exc)
            continue

    return results
