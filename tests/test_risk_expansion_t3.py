"""Tests for T3.5 RiskManager expansion (max_loss_per_trade, max_orders_per_day,
symbol_deny_list, RiskViolation classification)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from trading_algo.broker.base import AccountSnapshot, MarketDataSnapshot
from trading_algo.exit_codes import VALIDATION, classify_exception
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.risk import RiskLimits, RiskManager, RiskViolation


def _inst(symbol: str = "AAPL") -> InstrumentSpec:
    return InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")


def _fake_account(**vals) -> AccountSnapshot:
    base = {"NetLiquidation": 100_000.0, "GrossPositionValue": 0.0,
            "MaintMarginReq": 0.0, "BuyingPower": 200_000.0}
    base.update(vals)
    return AccountSnapshot(account="DU1234567", values=base, timestamp_epoch_s=1.0)


def _fake_snapshot(price: float = 100.0) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        instrument=_inst(), bid=price - 0.01, ask=price + 0.01,
        last=price, close=price, volume=1_000_000, timestamp_epoch_s=1,
    )


def _mock_broker(account: AccountSnapshot, positions=None) -> MagicMock:
    b = MagicMock()
    b.get_account_snapshot.return_value = account
    b.get_positions.return_value = positions or []
    return b


class TestRiskViolationClassifies:
    def test_classifies_as_validation(self) -> None:
        exc = RiskViolation("test")
        ce = classify_exception(exc)
        assert ce.exit_code == VALIDATION
        assert ce.error_code == "VALIDATION"
        assert ce.retryable is False


class TestSymbolDenyList:
    def test_deny_symbol_rejected(self) -> None:
        lim = RiskLimits(symbol_deny_list=frozenset({"XYZ"}))
        rm = RiskManager(lim)
        intent = TradeIntent(instrument=_inst("XYZ"), side="BUY", quantity=1)
        with pytest.raises(RiskViolation, match="deny-list"):
            rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot())

    def test_allowed_symbol_passes(self) -> None:
        lim = RiskLimits(symbol_deny_list=frozenset({"XYZ"}))
        rm = RiskManager(lim)
        intent = TradeIntent(instrument=_inst("AAPL"), side="BUY", quantity=1)
        # Should not raise.
        rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot())


class TestMaxLossPerTrade:
    def test_no_stop_means_no_check(self) -> None:
        lim = RiskLimits(max_loss_per_trade=100.0)
        rm = RiskManager(lim)
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=10)
        rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot(100.0))

    def test_buy_stop_below_entry_within_limit(self) -> None:
        lim = RiskLimits(max_loss_per_trade=100.0)
        rm = RiskManager(lim)
        # entry 100, stop 95, qty 10 → loss = 5 * 10 = 50 OK.
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=10, stop_price=95.0)
        rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot(100.0))

    def test_buy_stop_below_entry_over_limit(self) -> None:
        lim = RiskLimits(max_loss_per_trade=40.0)
        rm = RiskManager(lim)
        # entry 100, stop 95, qty 10 → loss = 50 > 40.
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=10, stop_price=95.0)
        with pytest.raises(RiskViolation, match="max_loss_per_trade"):
            rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot(100.0))

    def test_sell_stop_above_entry(self) -> None:
        lim = RiskLimits(max_loss_per_trade=100.0, allow_short=True,
                         max_abs_position_per_symbol=100)
        rm = RiskManager(lim)
        # entry 100, stop 108 (short stop above), qty 5 → loss = 8 * 5 = 40.
        intent = TradeIntent(instrument=_inst(), side="SELL", quantity=5, stop_price=108.0)
        rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot(100.0))

    def test_buy_stop_above_entry_rejected_as_wrong_side(self) -> None:
        lim = RiskLimits(max_loss_per_trade=100.0)
        rm = RiskManager(lim)
        # BUY with stop above entry: stop can't reduce loss → violation.
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=10, stop_price=110.0)
        with pytest.raises(RiskViolation, match="wrong side"):
            rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot(100.0))


class TestMaxOrdersPerDay:
    def test_counter_increments(self) -> None:
        lim = RiskLimits(max_orders_per_day=3)
        rm = RiskManager(lim)
        broker = _mock_broker(_fake_account())
        for _ in range(3):
            intent = TradeIntent(instrument=_inst(), side="BUY", quantity=1)
            rm.validate(intent, broker, lambda i: _fake_snapshot())
        assert rm.orders_today_count == 3

    def test_exceeding_cap_rejected(self) -> None:
        lim = RiskLimits(max_orders_per_day=2)
        rm = RiskManager(lim)
        broker = _mock_broker(_fake_account())
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=1)
        rm.validate(intent, broker, lambda i: _fake_snapshot())
        rm.validate(intent, broker, lambda i: _fake_snapshot())
        with pytest.raises(RiskViolation, match="max_orders_per_day"):
            rm.validate(intent, broker, lambda i: _fake_snapshot())

    def test_rejected_orders_do_not_consume_slot(self) -> None:
        lim = RiskLimits(max_orders_per_day=2, max_order_quantity=1)
        rm = RiskManager(lim)
        broker = _mock_broker(_fake_account())
        # First violation — exceeds qty cap, should NOT count.
        with pytest.raises(RiskViolation):
            rm.validate(
                TradeIntent(instrument=_inst(), side="BUY", quantity=100),
                broker, lambda i: _fake_snapshot(),
            )
        assert rm.orders_today_count == 0
        # Two valid orders should both pass.
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=1)
        rm.validate(intent, broker, lambda i: _fake_snapshot())
        rm.validate(intent, broker, lambda i: _fake_snapshot())
        assert rm.orders_today_count == 2


class TestErrorMessages:
    def test_max_order_quantity_message_is_descriptive(self) -> None:
        lim = RiskLimits(max_order_quantity=5)
        rm = RiskManager(lim)
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=10)
        with pytest.raises(RiskViolation, match="10.*max_order_quantity.*5"):
            rm.validate(intent, _mock_broker(_fake_account()), lambda i: _fake_snapshot())

    def test_leverage_message_contains_numbers(self) -> None:
        lim = RiskLimits(max_leverage=1.0)
        rm = RiskManager(lim)
        # Gross 150k on NetLiq 100k → leverage 1.5 > 1.0.
        acct = _fake_account(GrossPositionValue=150_000.0)
        intent = TradeIntent(instrument=_inst(), side="BUY", quantity=1)
        with pytest.raises(RiskViolation, match="leverage"):
            rm.validate(intent, _mock_broker(acct), lambda i: _fake_snapshot())
