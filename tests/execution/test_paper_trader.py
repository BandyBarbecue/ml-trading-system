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


def _mock_account(equity: str, last_equity: str | None = None) -> MagicMock:
    """Create a lightweight stand-in for an alpaca Account object."""
    acc = MagicMock()
    acc.equity = equity
    if last_equity is not None:
        acc.last_equity = last_equity
    else:
        del acc.last_equity  # simulate the attribute being genuinely absent
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
# test_get_previous_close_equity — kill-switch baseline fix (2026-08-10)
# ---------------------------------------------------------------------------

def test_get_previous_close_equity_returns_float():
    """get_previous_close_equity() converts account.last_equity to float.

    This is Alpaca's own record of equity at the PREVIOUS trading day's
    close — it is the preferred kill-switch baseline because it can't go
    stale the way our local equity_curve.parquet can (see equity_tracker.
    load_previous_equity's staleness guard for the bug this avoids)."""
    from src.execution.paper_trader import get_previous_close_equity

    client = MagicMock()
    client.get_account.return_value = _mock_account(
        equity="97940.69", last_equity="98552.31"
    )

    result = get_previous_close_equity(client)

    assert isinstance(result, float)
    assert result == pytest.approx(98552.31)


def test_get_previous_close_equity_none_when_attribute_absent():
    """If Alpaca's account object doesn't have last_equity at all, return
    None rather than raising — the caller falls back to the next baseline."""
    from src.execution.paper_trader import get_previous_close_equity

    client = MagicMock()
    client.get_account.return_value = _mock_account(equity="97940.69")  # no last_equity

    assert get_previous_close_equity(client) is None


def test_get_previous_close_equity_none_when_zero():
    """A last_equity of "0" is not a usable baseline (a real paper account
    should never be worth exactly $0) — treat it as absent, not as truth."""
    from src.execution.paper_trader import get_previous_close_equity

    client = MagicMock()
    client.get_account.return_value = _mock_account(
        equity="97940.69", last_equity="0"
    )

    assert get_previous_close_equity(client) is None


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

    We hold well over 25 shares of MSFT here so the oversell clamp (task #18)
    doesn't interfere with this test — that behaviour has its own dedicated
    tests below.
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
    client.get_all_positions.return_value = [_mock_position("MSFT", 40_000.0)]
    client.get_all_positions.return_value[0].qty = "200"  # comfortably more than 25

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


# ---------------------------------------------------------------------------
# test_submit_rebalance_clamps_oversell  (#18a)
# ---------------------------------------------------------------------------

def test_submit_rebalance_clamps_oversell():
    """A SELL delta implying more shares than we actually hold is clamped to
    the held quantity, never opening an accidental short.

    Setup: delta=-0.10 on a $100k account, price=$100 → naive calc wants to
    sell 100 shares, but we only hold 10.
    """
    from src.execution.paper_trader import submit_rebalance
    from alpaca.trading.enums import OrderSide

    quote = _mock_quote(ask_price=101.0, bid_price=100.0)
    mock_quote_response = {"AAPL": quote}

    mock_order = MagicMock()
    mock_order.id = "order-clamp-1"
    mock_order.status = "accepted"

    client = MagicMock()
    client.submit_order.return_value = mock_order
    held_pos = _mock_position("AAPL", 1_000.0)
    held_pos.qty = "10"
    client.get_all_positions.return_value = [held_pos]

    trade_deltas = pd.Series({"AAPL": -0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert len(results) == 1
    assert results[0]["shares"] == 10, "SELL must be clamped to the 10 shares actually held"
    submitted_request = client.submit_order.call_args[0][0]
    assert submitted_request.qty == 10
    assert submitted_request.side == OrderSide.SELL


def test_submit_rebalance_sell_skipped_when_nothing_held():
    """A SELL delta for a ticker we don't hold at all is clamped to 0 shares
    (skipped entirely) rather than opening a naked short."""
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=51.0, bid_price=50.0)
    mock_quote_response = {"GHOST": quote}

    client = MagicMock()
    client.get_all_positions.return_value = []  # we hold nothing

    trade_deltas = pd.Series({"GHOST": -0.05})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert results == []
    client.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# test_submit_rebalance_skips_open_order  (#18b)
# ---------------------------------------------------------------------------

def test_submit_rebalance_skips_open_order():
    """If an open order already exists for a symbol, submit_rebalance skips it
    instead of submitting a duplicate."""
    from src.execution.paper_trader import submit_rebalance

    client = MagicMock()
    client.get_orders.return_value = [MagicMock()]  # one open order already exists
    client.get_all_positions.return_value = []

    trade_deltas = pd.Series({"AAPL": 0.05})
    results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert results == []
    client.submit_order.assert_not_called()


def test_submit_rebalance_open_order_check_failure_does_not_block():
    """If checking for open orders raises (API hiccup), the order is still
    submitted rather than the whole rebalance being killed."""
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=100.0, bid_price=99.0)
    mock_quote_response = {"AAPL": quote}

    mock_order = MagicMock()
    mock_order.id = "order-1"
    mock_order.status = "accepted"

    client = MagicMock()
    client.get_orders.side_effect = RuntimeError("API hiccup")
    client.get_all_positions.return_value = []
    client.submit_order.return_value = mock_order

    trade_deltas = pd.Series({"AAPL": 0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert len(results) == 1
    client.submit_order.assert_called_once()


# ---------------------------------------------------------------------------
# test_submit_rebalance idempotency_key_prefix (preemption-replay guard)
# ---------------------------------------------------------------------------

def test_submit_rebalance_no_prefix_leaves_client_order_id_unset():
    """Default behaviour (idempotency_key_prefix=None) must be unchanged:
    no client_order_id is sent at all. This is what the intraday runner
    relies on — it re-trades the same ticker/side across bars and must
    never be blocked by a duplicate-order-id rejection."""
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=100.0, bid_price=99.0)
    mock_quote_response = {"AAPL": quote}

    mock_order = MagicMock()
    mock_order.id = "order-abc-123"
    mock_order.status = "accepted"

    client = MagicMock()
    client.submit_order.return_value = mock_order

    trade_deltas = pd.Series({"AAPL": 0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(client, trade_deltas, account_value=100_000.0)

    assert len(results) == 1
    submitted_request = client.submit_order.call_args[0][0]
    assert submitted_request.client_order_id is None


def test_submit_rebalance_with_prefix_sets_client_order_id_buy():
    """When idempotency_key_prefix is supplied, a BUY order carries
    client_order_id = f"{prefix}-{ticker}-buy"."""
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=100.0, bid_price=99.0)
    mock_quote_response = {"AAPL": quote}

    mock_order = MagicMock()
    mock_order.id = "order-abc-123"
    mock_order.status = "accepted"

    client = MagicMock()
    client.submit_order.return_value = mock_order

    trade_deltas = pd.Series({"AAPL": 0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(
            client, trade_deltas, account_value=100_000.0,
            idempotency_key_prefix="daily-2026-08-10",
        )

    assert len(results) == 1
    submitted_request = client.submit_order.call_args[0][0]
    assert submitted_request.client_order_id == "daily-2026-08-10-AAPL-buy"


def test_submit_rebalance_with_prefix_sets_client_order_id_sell():
    """When idempotency_key_prefix is supplied, a SELL order carries
    client_order_id = f"{prefix}-{ticker}-sell"."""
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=201.0, bid_price=200.0)
    mock_quote_response = {"MSFT": quote}

    mock_order = MagicMock()
    mock_order.id = "order-xyz-456"
    mock_order.status = "accepted"

    client = MagicMock()
    client.submit_order.return_value = mock_order
    held_pos = _mock_position("MSFT", 40_000.0)
    held_pos.qty = "200"  # comfortably more than the 25 shares this trade implies
    client.get_all_positions.return_value = [held_pos]

    trade_deltas = pd.Series({"MSFT": -0.05})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(
            client, trade_deltas, account_value=100_000.0,
            idempotency_key_prefix="daily-2026-08-10",
        )

    assert len(results) == 1
    submitted_request = client.submit_order.call_args[0][0]
    assert submitted_request.client_order_id == "daily-2026-08-10-MSFT-sell"


def test_submit_rebalance_duplicate_rejection_is_skipped_and_run_continues():
    """A duplicate-client_order_id rejection from Alpaca (the signature of a
    preemption replay) is swallowed: it must not raise, must not appear in
    results, and must not stop the rest of the batch from being processed —
    the very next ticker should still submit normally."""
    from src.execution.paper_trader import submit_rebalance

    quote_a = _mock_quote(ask_price=100.0, bid_price=99.0)
    quote_b = _mock_quote(ask_price=50.0, bid_price=49.0)

    mock_order_ok = MagicMock()
    mock_order_ok.id = "order-ok-1"
    mock_order_ok.status = "accepted"

    duplicate_exc = Exception(
        "order with client_order_id already exists: must be unique"
    )

    client = MagicMock()
    client.submit_order.side_effect = [duplicate_exc, mock_order_ok]

    # dict preserves insertion order → DUPTICK processed before OKTICK, and
    # the quote lookups are consumed in that same order.
    trade_deltas = pd.Series({"DUPTICK": 0.10, "OKTICK": 0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.side_effect = [
            {"DUPTICK": quote_a},
            {"OKTICK": quote_b},
        ]
        results = submit_rebalance(
            client, trade_deltas, account_value=100_000.0,
            idempotency_key_prefix="daily-2026-08-10",
        )

    tickers_in_results = {r["ticker"] for r in results}
    assert "DUPTICK" not in tickers_in_results, (
        "A duplicate-rejected order must not be counted as submitted."
    )
    assert "OKTICK" in tickers_in_results, (
        "Processing must continue to the next ticker after a duplicate rejection."
    )
    assert len(results) == 1
    assert client.submit_order.call_count == 2


def test_submit_rebalance_genuine_error_still_uses_warning_path():
    """A non-duplicate exception (e.g. a real network/API failure) must
    still take the pre-existing WARNING-and-skip path, not be mistaken for
    a duplicate-order rejection."""
    from src.execution.paper_trader import submit_rebalance

    quote = _mock_quote(ask_price=100.0, bid_price=99.0)
    mock_quote_response = {"FAILTICK": quote}

    client = MagicMock()
    client.submit_order.side_effect = RuntimeError("connection reset by peer")

    trade_deltas = pd.Series({"FAILTICK": 0.10})
    with patch("src.execution.paper_trader._make_data_client") as mock_data_fn:
        mock_data_fn.return_value.get_stock_latest_quote.return_value = mock_quote_response
        results = submit_rebalance(
            client, trade_deltas, account_value=100_000.0,
            idempotency_key_prefix="daily-2026-08-10",
        )

    assert results == []
    client.submit_order.assert_called_once()


# ---------------------------------------------------------------------------
# test run lock (#18c)
# ---------------------------------------------------------------------------

def test_acquire_run_lock_creates_file(tmp_path, monkeypatch):
    """acquire_run_lock() creates data/run.lock with pid + ISO timestamp."""
    import config
    from src.execution import paper_trader

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    assert paper_trader.acquire_run_lock() is True

    lock_path = tmp_path / "run.lock"
    assert lock_path.exists()
    import json
    payload = json.loads(lock_path.read_text())
    assert "pid" in payload and "timestamp" in payload

    paper_trader.release_run_lock()
    assert not lock_path.exists()


def test_second_run_refused_while_lock_held(tmp_path, monkeypatch):
    """A second acquire_run_lock() call while the first lock is fresh returns False."""
    import config
    from src.execution import paper_trader

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    assert paper_trader.acquire_run_lock() is True
    # Second call, lock still held and fresh → must be refused.
    assert paper_trader.acquire_run_lock() is False

    paper_trader.release_run_lock()
    # After release, a new run can acquire the lock again.
    assert paper_trader.acquire_run_lock() is True
    paper_trader.release_run_lock()


def test_stale_lock_is_reclaimed(tmp_path, monkeypatch):
    """A lock older than the staleness window is reclaimed, not treated as live."""
    import json
    from datetime import datetime, timedelta, timezone
    import config
    from src.execution import paper_trader

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    lock_path = tmp_path / "run.lock"
    tmp_path.mkdir(parents=True, exist_ok=True)
    stale_time = datetime.now(timezone.utc) - timedelta(hours=3)  # > 2h staleness window
    lock_path.write_text(json.dumps({"pid": 99999, "timestamp": stale_time.isoformat()}))

    # Even though a lock file exists, it's stale → must be reclaimed (True).
    assert paper_trader.acquire_run_lock() is True
    paper_trader.release_run_lock()
