"""Tests for the IBKR IdempotentOrderPlacer.

All tests use Mock objects standing in for ib_async — no TWS needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import pytest

from trading_algo.broker.base import OrderRequest, OrderResult
from trading_algo.broker.idempotent_placer import (
    IBKROrderbookLookupError,
    IdempotentOrderPlacer,
    _is_transient,
    _trade_to_result,
    _with_order_ref,
    find_trade_by_order_ref,
)
from trading_algo.instruments import InstrumentSpec


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _mk_request(**overrides) -> OrderRequest:
    defaults = dict(
        instrument=InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD"),
        side="BUY", quantity=10, order_type="LMT", limit_price=150.0,
    )
    defaults.update(overrides)
    return OrderRequest(**defaults)


def _mk_trade(order_ref: str, *, status: str = "Submitted", order_id: int = 42) -> Mock:
    """Build a Mock that duck-types as an ib_async Trade."""
    trade = Mock()
    trade.order = Mock()
    trade.order.orderRef = order_ref
    trade.order.orderId = order_id
    trade.orderStatus = Mock()
    trade.orderStatus.status = status
    return trade


def _mk_ib(
    *,
    open_trades: list | None = None,
    completed_orders: list | None = None,
    open_raises: Exception | None = None,
    completed_raises: Exception | None = None,
) -> Mock:
    ib = Mock()
    if open_raises:
        ib.openTrades.side_effect = open_raises
    else:
        ib.openTrades.return_value = open_trades or []
    if completed_raises:
        ib.reqCompletedOrders.side_effect = completed_raises
    else:
        ib.reqCompletedOrders.return_value = completed_orders or []
    return ib


# -----------------------------------------------------------------------------
# find_trade_by_order_ref
# -----------------------------------------------------------------------------

class TestFindTrade:
    def test_found_in_open(self) -> None:
        ib = _mk_ib(open_trades=[_mk_trade("TA_ABC", status="Submitted")])
        t = find_trade_by_order_ref(ib, "TA_ABC")
        assert t is not None
        assert t.order.orderRef == "TA_ABC"

    def test_found_in_completed(self) -> None:
        ib = _mk_ib(completed_orders=[_mk_trade("TA_DONE", status="Filled")])
        t = find_trade_by_order_ref(ib, "TA_DONE")
        assert t is not None

    def test_case_insensitive(self) -> None:
        ib = _mk_ib(open_trades=[_mk_trade("ta_lowercase")])
        t = find_trade_by_order_ref(ib, "TA_LOWERCASE")
        assert t is not None

    def test_not_found_returns_none(self) -> None:
        ib = _mk_ib(open_trades=[_mk_trade("TA_OTHER")])
        assert find_trade_by_order_ref(ib, "TA_WANTED") is None

    def test_empty_orderbook(self) -> None:
        ib = _mk_ib()
        assert find_trade_by_order_ref(ib, "TA_X") is None

    def test_open_trades_failure_raises(self) -> None:
        ib = _mk_ib(open_raises=RuntimeError("TWS disconnected"))
        with pytest.raises(IBKROrderbookLookupError, match="openTrades"):
            find_trade_by_order_ref(ib, "TA_X")

    def test_completed_orders_failure_raises(self) -> None:
        ib = _mk_ib(completed_raises=RuntimeError("timeout"))
        with pytest.raises(IBKROrderbookLookupError, match="reqCompletedOrders"):
            find_trade_by_order_ref(ib, "TA_X")

    def test_old_signature_without_apionly(self) -> None:
        """ib_async older versions: reqCompletedOrders takes no args."""
        ib = Mock()
        ib.openTrades.return_value = []
        ib.reqCompletedOrders.side_effect = [TypeError("no apiOnly"), []]
        t = find_trade_by_order_ref(ib, "TA_X")
        assert t is None
        # Called twice: once with apiOnly, once without.
        assert ib.reqCompletedOrders.call_count == 2

    def test_skips_malformed_trade_objects(self) -> None:
        """A Trade-like object without .order attribute is just skipped."""
        bad = Mock()
        bad.order = None
        good = _mk_trade("TA_GOOD")
        ib = _mk_ib(open_trades=[bad, good])
        assert find_trade_by_order_ref(ib, "TA_GOOD") is not None

    def test_none_ib_raises(self) -> None:
        with pytest.raises(IBKROrderbookLookupError):
            find_trade_by_order_ref(None, "TA_X")


# -----------------------------------------------------------------------------
# IdempotentOrderPlacer — happy path
# -----------------------------------------------------------------------------

class TestPlaceHappyPath:
    def test_no_existing_places_order(self) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.return_value = OrderResult(order_id="42", status="Submitted")

        placer = IdempotentOrderPlacer(broker)
        result = placer.place(_mk_request(), idempotency_key="AGENT_KEY_1")

        assert result.order_id == "42"
        # place_order called once, with orderRef set.
        broker.place_order.assert_called_once()
        req_sent = broker.place_order.call_args.args[0]
        assert req_sent.order_ref is not None
        assert req_sent.order_ref.startswith("TA")

    def test_existing_order_replays_not_retransmits(self) -> None:
        """If an order with our orderRef already exists on IBKR, do NOT
        call place_order again. Return the existing trade's state.
        """
        from trading_algo.idempotency import derive_order_ref
        key = "CRASHED_AGENT_KEY"
        expected_ref = derive_order_ref(key)

        existing = _mk_trade(expected_ref, status="Filled", order_id=99)
        broker = Mock()
        broker._ib = _mk_ib(open_trades=[existing])
        broker.place_order = Mock()

        placer = IdempotentOrderPlacer(broker)
        result = placer.place(_mk_request(), idempotency_key=key)

        assert result.order_id == "99"
        assert result.status == "Filled"
        broker.place_order.assert_not_called()

    def test_explicit_order_ref_respected(self) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.return_value = OrderResult(order_id="1", status="Submitted")

        placer = IdempotentOrderPlacer(broker)
        placer.place(_mk_request(), order_ref="CUSTOM_REF_01")

        req_sent = broker.place_order.call_args.args[0]
        assert req_sent.order_ref == "CUSTOM_REF_01"

    def test_req_already_has_order_ref(self) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.return_value = OrderResult(order_id="1", status="Submitted")

        placer = IdempotentOrderPlacer(broker)
        placer.place(_mk_request(order_ref="EXISTING_REF"))

        req_sent = broker.place_order.call_args.args[0]
        assert req_sent.order_ref == "EXISTING_REF"

    def test_no_key_generates_ephemeral(self, caplog) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.return_value = OrderResult(order_id="1", status="Submitted")

        placer = IdempotentOrderPlacer(broker)
        with caplog.at_level("WARNING"):
            placer.place(_mk_request())
        # Log warns about non-idempotent ephemeral ref.
        assert any("ephemeral" in r.message.lower() for r in caplog.records)


# -----------------------------------------------------------------------------
# Transient + orderbook-unreachable paths
# -----------------------------------------------------------------------------

class TestTransient:
    def test_transient_retries_if_not_in_orderbook(self, monkeypatch) -> None:
        """First place_order raises transient; orderbook check → not found
        → retry succeeds."""
        # Skip the sleeps.
        import trading_algo.broker.idempotent_placer as m
        monkeypatch.setattr(m, "_ORDERBOOK_POLL_DELAYS", (0.001, 0.001, 0.001, 0.001, 0.001))

        TransientException = type("NetworkException", (Exception,), {})
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.side_effect = [
            TransientException("timeout"),
            OrderResult(order_id="99", status="Submitted"),
        ]

        placer = IdempotentOrderPlacer(
            broker, max_attempts=3, initial_backoff_s=0.001,
        )
        result = placer.place(_mk_request(), idempotency_key="K1")

        assert result.order_id == "99"
        assert broker.place_order.call_count == 2

    def test_transient_finds_order_on_poll_no_retry(self, monkeypatch) -> None:
        """First place_order raises transient; orderbook check → FOUND
        (the order did land). Must NOT retry — replay the existing state.
        """
        import trading_algo.broker.idempotent_placer as m
        monkeypatch.setattr(m, "_ORDERBOOK_POLL_DELAYS", (0.001,) * 5)

        from trading_algo.idempotency import derive_order_ref
        key = "SURVIVED_THE_ERROR"
        expected_ref = derive_order_ref(key)

        # First place_order raises, but the orderbook check in _poll finds
        # our order (it DID land despite the error).
        TransientException = type("NetworkException", (Exception,), {})

        call_n = {"n": 0}
        def fake_place(req):
            call_n["n"] += 1
            raise TransientException("gateway timeout")

        def fake_open_trades():
            # Empty on the FIRST find_trade_by_order_ref call (pre-place)
            # Then populated on the polls after the transient error.
            if call_n["n"] == 0:
                return []
            return [_mk_trade(expected_ref, status="Submitted", order_id=77)]

        broker = Mock()
        broker._ib = Mock()
        broker._ib.openTrades = fake_open_trades
        broker._ib.reqCompletedOrders.return_value = []
        broker.place_order.side_effect = fake_place

        placer = IdempotentOrderPlacer(broker, max_attempts=3, initial_backoff_s=0.001)
        result = placer.place(_mk_request(), idempotency_key=key)

        assert result.order_id == "77"
        # place_order called ONCE despite the transient error, because we
        # confirmed the order landed via the orderbook poll.
        assert call_n["n"] == 1

    def test_orderbook_unreachable_refuses_to_place(self, monkeypatch) -> None:
        """If the pre-check orderbook lookup fails, REFUSE TO PLACE.
        Even safer than the Kite-algo flow: we don't even transmit once,
        because we can't verify that a prior attempt didn't already land.
        """
        import trading_algo.broker.idempotent_placer as m
        monkeypatch.setattr(m, "_ORDERBOOK_POLL_DELAYS", (0.001,) * 5)

        broker = Mock()
        ib = Mock()
        ib.openTrades.side_effect = RuntimeError("TWS connection lost")
        ib.reqCompletedOrders.return_value = []
        broker._ib = ib
        broker.place_order = Mock()

        placer = IdempotentOrderPlacer(broker, max_attempts=3, initial_backoff_s=0.001)
        with pytest.raises(IBKROrderbookLookupError):
            placer.place(_mk_request(), idempotency_key="K")

        # NEVER called — the pre-check failed, so we refused to place at all.
        # This is the double-fill defence in action.
        assert broker.place_order.call_count == 0

    def test_orderbook_unreachable_after_initial_check_no_retry(self, monkeypatch) -> None:
        """Pre-check succeeds (not found), place_order raises transient,
        THEN the orderbook poll fails → refuse to retry place_order.
        """
        import trading_algo.broker.idempotent_placer as m
        monkeypatch.setattr(m, "_ORDERBOOK_POLL_DELAYS", (0.001,) * 5)

        TransientException = type("NetworkException", (Exception,), {})
        broker = Mock()
        ib = Mock()

        # First call (pre-check): empty list → "not found, proceed".
        # Subsequent calls (poll after transient): raise → unreachable.
        state = {"calls": 0}
        def open_trades_dyn():
            state["calls"] += 1
            if state["calls"] == 1:
                return []
            raise RuntimeError("TWS lost mid-flight")
        ib.openTrades.side_effect = open_trades_dyn
        ib.reqCompletedOrders.return_value = []
        broker._ib = ib
        broker.place_order.side_effect = TransientException("timeout")

        placer = IdempotentOrderPlacer(broker, max_attempts=3, initial_backoff_s=0.001)
        with pytest.raises(IBKROrderbookLookupError):
            placer.place(_mk_request(), idempotency_key="K")

        # Exactly ONE place attempt — we refused to retry after the
        # poll-phase orderbook lookup failed.
        assert broker.place_order.call_count == 1


# -----------------------------------------------------------------------------
# Hard errors — never retry
# -----------------------------------------------------------------------------

class TestHardErrors:
    def test_value_error_not_retried(self) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.side_effect = ValueError("bad qty")

        placer = IdempotentOrderPlacer(broker)
        with pytest.raises(ValueError):
            placer.place(_mk_request(), idempotency_key="K")
        assert broker.place_order.call_count == 1

    def test_input_exception_not_retried(self) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        InputException = type("InputException", (Exception,), {})
        broker.place_order.side_effect = InputException("invalid contract")

        placer = IdempotentOrderPlacer(broker)
        with pytest.raises(Exception):
            placer.place(_mk_request(), idempotency_key="K")
        assert broker.place_order.call_count == 1

    def test_keyboard_interrupt_propagates(self) -> None:
        broker = Mock()
        broker._ib = _mk_ib()
        broker.place_order.side_effect = KeyboardInterrupt()

        placer = IdempotentOrderPlacer(broker)
        with pytest.raises(KeyboardInterrupt):
            placer.place(_mk_request(), idempotency_key="K")


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

class TestUtil:
    def test_with_order_ref_replaces(self) -> None:
        req = _mk_request()
        req2 = _with_order_ref(req, "NEW_REF")
        assert req2.order_ref == "NEW_REF"
        # Original unchanged (frozen).
        assert req.order_ref is None

    def test_trade_to_result(self) -> None:
        trade = _mk_trade("TA_X", status="Filled", order_id=123)
        r = _trade_to_result(trade)
        assert r.order_id == "123"
        assert r.status == "Filled"

    def test_is_transient_network(self) -> None:
        exc = type("NetworkException", (Exception,), {})("timeout")
        # Actually NetworkException isn't in _HARD_ERROR_NAMES → goes through
        # message check → "timeout" matches.
        assert _is_transient(exc)

    def test_is_transient_by_errorcode(self) -> None:
        exc = Exception("something")
        exc.errorCode = 1100  # type: ignore[attr-defined]
        assert _is_transient(exc)

    def test_not_transient_value_error(self) -> None:
        assert not _is_transient(ValueError("bad"))

    def test_not_transient_keyboard_interrupt(self) -> None:
        assert not _is_transient(KeyboardInterrupt())


class TestOrderbookLookupErrorClassification:
    def test_ibkr_orderbook_lookup_error_classified(self) -> None:
        """It should map to UNAVAILABLE in the classifier — critical for
        agents branching on exit codes."""
        from trading_algo.exit_codes import classify_exception, UNAVAILABLE
        exc = IBKROrderbookLookupError("cannot see orderbook")
        cls = classify_exception(exc)
        assert cls.exit_code == UNAVAILABLE
        assert cls.retryable is True
