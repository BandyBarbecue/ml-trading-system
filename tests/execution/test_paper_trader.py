"""Tests for src/execution/paper_trader.py

All Alpaca API calls are mocked — this test suite never touches the network.
We verify:
  - paper=True is always passed to TradingClient (the critical safety check)
  - account equity is correctly converted to float
  - positions are correctly converted to weight fractions
  - buy/sell orders are sized and submitted correctly
  - tiny deltas (< 1 share) are silently skipped
"""

import math
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a mock Alpaca position object
# ---------------------------------------------------------------------------

def _mock_position(symbol: str, market_value: float) -> MagicMock:
    """Create a lightweight stand-in for an alpaca Position object."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.market_value = str(market_value)  # Alpaca returns strings
    return pos


def _mock_account(equity: str) -> MagicMock:
    """Create a lightweight stand-in for an alpaca Account object."""
    acc = MagicMock()
    acc.equity = equity
    return acc


def _mock_quote(ask_price: float, bid_price: float) -> MagicMock:
    """Create a lightweight stand-in for an alpaca Quote object."""
    q = MagicMock()
    q.ask_price = ask_price
    q.bid_price = bid_price
    return q


# ---------------------------------------------------------------------------
# test_paper_flag_always_true
# ---------------------------------------------------------------------------

def test_paper_flag_always_true():
    """_make_trading_client() must always call TradingClient with paper=True.

    This is the critical safety check.  No matter what environment variables
    are set, the paper flag must be hardcoded True via config.ALPACA_PAPER.
    """
    with patch("src.execution.paper_trader.TradingClient") as mock_cls:
        mock_cls.return_value = MagicMock()

        from src.execution.paper_trader import _make_trading_client
        _make_trading_client()

        # Verify the constructor was called exactly once.
        mock_cls.assert_called_once()

        # Extract keyword arguments from the call.
        _, kwargs = mock_cls.call_args
        assert kwargs.get("paper") is True, (
            "TradingClient MUST be constructed with paper=True. "
            "Live trading is not permitted in this system."
        )


# ---------------------------------------------------------------------------
# test_get_account_value
# ---------------------------------------------------------------------------

def test_get_account_value():
    """get_account_value() converts equity string to float correctly."""
    from src.execution.paper_trader import get_account_value

    client = MagicMock()
    client.get_account.return_value = _mock_account(equity="50000.00")

    result = get_account_value(client)

    assert isinstance(result, float)
    assert result == 50000.0


# ---------------------------------------------------------------------------
# test_get_current_positions_empty
# ---------------------------------------------------------------------------

def test_get_current_positions_empty():
    """get_current_positions() with no holdings returns an empty Series."""
    from src.execution.paper_trader import get_current_positions

    client = MagicMock()
    client.get_all_positions.return_value = []

    result = get_current_positions(client)

    assert isinstance(result, pd.Series)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# test_get_current_positions_with_holdings
# ---------------------------------------------------------------------------

def test_get_current_positions_with_holdings():
    """Two positions are correctly converted to portfolio weight fractions.

    Setup:
      total_equity = $100,000
      AAPL market_value = $30,000  → weight = 0.30
      MSFT market_value = $20,000  → weight = 0.20
    """
    from src.execution.paper_trader import get_current_positions

    client = MagicMock()
    client.get_all_positions.return_value = [
        _mock_position("AAPL", 30_000.0),
        _mock_position("MSFT", 20_000.0),
    ]
    client.get_account.return_value = _mock_account(equity="100000.00")

    result = get_current_positions(client)

    assert isinstance(result, pd.Series)
    assert set(result.index) == {"AAPL", "MSFT"}
    assert abs(result["AAPL"] - 0.30) < 1e-9
    assert abs(result["MSFT"] - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# test_submit_rebalance_buy
# ---------------------------------------------------------------------------

def test_submit_rebalance_buy():
    """A +0.10 delta on a $100 stock with $100k account → BUY 100 shares.

    Math: delta=+0.10, account=$100,000 → $10,000 to invest.
          price (ask)=$100/share → 10,000/100 = 100.0 → floor = 100 shares.
    """
    from src.execution.paper_trader import submit_rebalance
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    # Mock quote: ask=$100, bid=$99
    quote = _mock_quote(ask_price=100.0, bid_price=99.0)
    mock_quote_response = {"AAPL": quote}

    # Mock submitted order
    mock_order = MagicMock()
    mock_order.id = "order-abc-123"
    mock_order.status = "accepted"
    mock_order.symbol = "AAPL"
    mock_order.qty = 100

    client = MagicMock()
    client.submit_order.return_value = mock_order

    trade_deltas = pd.Series({"AAPL": 0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert len(results) == 1
    order_result = results[0]
    assert order_result["ticker"] == "AAPL"
    assert order_result["side"] == "buy"
    assert order_result["shares"] == 100
    assert order_result["status"] == "accepted"

    # Check the submitted order used ask price (BUY) and integer qty
    client.submit_order.assert_called_once()
    submitted_request = client.submit_order.call_args[0][0]
    assert submitted_request.qty == 100
    assert submitted_request.side == OrderSide.BUY
    assert submitted_request.time_in_force == TimeInForce.DAY


# ---------------------------------------------------------------------------
# test_submit_rebalance_sell
# ---------------------------------------------------------------------------

def test_submit_rebalance_sell():
    """A -0.05 delta on a $200 stock with $100k account → SELL 25 shares.

    Math: delta=-0.05, account=$100,000 → $5,000 to sell.
          price (bid)=$200/share → 5,000/200 = 25.0 → floor = 25 shares.
    """
    from src.execution.paper_trader import submit_rebalance
    from alpaca.trading.enums import OrderSide

    quote = _mock_quote(ask_price=201.0, bid_price=200.0)
    mock_quote_response = {"MSFT": quote}

    mock_order = MagicMock()
    mock_order.id = "order-xyz-456"
    mock_order.status = "accepted"

    client = MagicMock()
    client.submit_order.return_value = mock_order

    trade_deltas = pd.Series({"MSFT": -0.05})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert len(results) == 1
    order_result = results[0]
    assert order_result["ticker"] == "MSFT"
    assert order_result["side"] == "sell"
    assert order_result["shares"] == 25

    submitted_request = client.submit_order.call_args[0][0]
    assert submitted_request.side == OrderSide.SELL
    assert submitted_request.qty == 25


# ---------------------------------------------------------------------------
# test_submit_rebalance_skips_tiny
# ---------------------------------------------------------------------------

def test_submit_rebalance_skips_tiny():
    """A very small delta produces < 1 share → order is skipped entirely.

    Math: delta=0.0001, account=$1,000 → $0.10 to invest.
          price (ask)=$100/share → 0.10/100 = 0.001 → floor = 0 shares → skip.
    """
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=100.0, bid_price=99.0)
    mock_quote_response = {"TINY": quote}

    client = MagicMock()
    client.get_latest_quote.return_value = mock_quote_response

    trade_deltas = pd.Series({"TINY": 0.0001})
    results = submit_rebalance(client, trade_deltas, account_value=1_000.0)

    # No orders should have been submitted.
    assert results == []
    client.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# test_submit_rebalance_skips_zero_delta
# ---------------------------------------------------------------------------

def test_submit_rebalance_skips_zero_delta():
    """A delta of exactly 0.0 produces no order and no quote fetch."""
    from src.execution.paper_trader import submit_rebalance

    client = MagicMock()

    trade_deltas = pd.Series({"AAPL": 0.0})
    results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert results == []
    client.get_latest_quote.assert_not_called()
    client.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# test_submit_rebalance_skips_on_quote_error
# ---------------------------------------------------------------------------

def test_submit_rebalance_skips_on_quote_error():
    """If the quote fetch raises an exception, the ticker is skipped gracefully."""
    from src.execution.paper_trader import submit_rebalance

    client = MagicMock()
    client.get_latest_quote.side_effect = RuntimeError("Network error")

    trade_deltas = pd.Series({"FAIL": 0.10})
    # Should not raise; should return empty list.
    results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert results == []
    client.submit_order.assert_not_called()
