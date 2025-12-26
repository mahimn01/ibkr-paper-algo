from __future__ import annotations

import os
import tempfile
import time
import unittest
from dataclasses import dataclass

from trading_algo.broker.base import (
    AccountSnapshot,
    Bar,
    BracketOrderRequest,
    BracketOrderResult,
    MarketDataSnapshot,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    validate_order_request,
)
from trading_algo.config import TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.llm.config import LLMConfig
from trading_algo.llm.gemini import LLMClient
from trading_algo.llm.trader import LLMTrader
from trading_algo.risk import RiskLimits, RiskManager


class _FakeLLM(LLMClient):
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: int = 0

    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        _ = (prompt, system, use_google_search)
        self.calls += 1
        if not self._outputs:
            return "{\"decisions\":[]}"
        return self._outputs.pop(0)


@dataclass
class _FakeBroker:
    connected: bool = False
    next_order_id: int = 1
    placed: int = 0
    modified: int = 0
    cancelled: int = 0
    _statuses: dict[str, OrderStatus] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._statuses = {}

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_market_data_snapshot(self, instrument: InstrumentSpec) -> MarketDataSnapshot:
        if not self.connected:
            raise RuntimeError("not connected")
        return MarketDataSnapshot(
            instrument=instrument,
            bid=99.0,
            ask=101.0,
            last=100.0,
            close=100.0,
            volume=1_000_000.0,
            timestamp_epoch_s=time.time(),
        )

    def get_positions(self) -> list[Position]:
        if not self.connected:
            raise RuntimeError("not connected")
        return []

    def get_account_snapshot(self) -> AccountSnapshot:
        if not self.connected:
            raise RuntimeError("not connected")
        return AccountSnapshot(
            account="SIM",
            values={"NetLiquidation": 100_000.0, "GrossPositionValue": 0.0, "AvailableFunds": 100_000.0},
            timestamp_epoch_s=time.time(),
        )

    def place_order(self, req: OrderRequest) -> OrderResult:
        if not self.connected:
            raise RuntimeError("not connected")
        req = validate_order_request(req)
        self.placed += 1
        oid = str(self.next_order_id)
        self.next_order_id += 1
        self._statuses[oid] = OrderStatus(order_id=oid, status="Submitted", filled=0.0, remaining=req.quantity, avg_fill_price=None)
        return OrderResult(order_id=oid, status="Submitted")

    def modify_order(self, order_id: str, new_req: OrderRequest) -> OrderResult:
        if not self.connected:
            raise RuntimeError("not connected")
        _ = validate_order_request(new_req)
        if order_id not in self._statuses:
            raise KeyError(order_id)
        self.modified += 1
        self._statuses[order_id] = OrderStatus(order_id=order_id, status="Submitted", filled=0.0, remaining=new_req.quantity, avg_fill_price=None)
        return OrderResult(order_id=order_id, status="Submitted")

    def cancel_order(self, order_id: str) -> None:
        if not self.connected:
            raise RuntimeError("not connected")
        if order_id not in self._statuses:
            raise KeyError(order_id)
        self.cancelled += 1
        self._statuses[order_id] = OrderStatus(order_id=order_id, status="Cancelled", filled=0.0, remaining=0.0, avg_fill_price=None)

    def get_order_status(self, order_id: str) -> OrderStatus:
        if not self.connected:
            raise RuntimeError("not connected")
        if order_id not in self._statuses:
            raise KeyError(order_id)
        return self._statuses[order_id]

    def list_open_order_statuses(self) -> list[OrderStatus]:
        if not self.connected:
            raise RuntimeError("not connected")
        return [st for st in self._statuses.values() if st.status not in {"Filled", "Cancelled", "Rejected"}]

    def place_bracket_order(self, req: BracketOrderRequest) -> BracketOrderResult:
        raise NotImplementedError

    def get_historical_bars(
        self,
        instrument: InstrumentSpec,
        *,
        end_datetime: str | None = None,
        duration: str,
        bar_size: str,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[Bar]:
        _ = (instrument, end_datetime, duration, bar_size, what_to_show, use_rth)
        return []


class TestLLMTrader(unittest.TestCase):
    def test_llm_trader_place_and_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "audit.sqlite3")
            cfg = TradingConfig(broker="sim", dry_run=False, db_path=db_path)
            llm = LLMConfig(
                enabled=True,
                provider="gemini",
                gemini_api_key="test",
                allowed_symbols_csv="AAPL",
                max_orders_per_tick=3,
                max_qty=10.0,
            )
            fake = _FakeLLM(
                outputs=[
                    """{"decisions":[{"action":"PLACE","order":{"instrument":{"kind":"STK","symbol":"AAPL","exchange":"SMART","currency":"USD"},"side":"BUY","qty":1,"type":"MKT","tif":"DAY"}}]}""",
                    """{"decisions":[{"action":"CANCEL","order_id":"1"}]}""",
                ]
            )
            broker = _FakeBroker()
            trader = LLMTrader(
                broker=broker,
                trading=cfg,
                llm=llm,
                client=fake,
                risk=RiskManager(RiskLimits()),
                sleep_seconds=0.0,
                max_ticks=1,
            )
            trader.run_once()
            self.assertEqual(broker.placed, 1)
            trader.run_once()
            self.assertEqual(broker.cancelled, 1)

