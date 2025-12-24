import logging
import os
import tempfile
import unittest

from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.persistence import SqliteStore

logging.disable(logging.CRITICAL)


class TestPersistence(unittest.TestCase):
    def test_sqlite_store_writes_run_and_decision(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        try:
            store = SqliteStore(path)
            cfg = TradingConfig(broker="sim", live_enabled=False, ibkr=IBKRConfig())
            run_id = store.start_run(cfg)
            self.assertIsInstance(run_id, int)
            store.log_decision(
                run_id,
                strategy="unit",
                intent=TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1),
                accepted=False,
                reason="test",
            )
            store.end_run(run_id)
            store.close()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_sqlite_store_orders_and_status_events(self):
        from trading_algo.broker.base import OrderRequest, OrderStatus

        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        try:
            store = SqliteStore(path)
            cfg = TradingConfig(broker="sim", live_enabled=False, ibkr=IBKRConfig())
            run_id = store.start_run(cfg)
            req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="MKT")
            store.log_order(run_id, broker="sim", order_id="sim-1", request=req, status="Submitted")
            store.log_order_status_event(run_id, "sim", OrderStatus("sim-1", "Filled", 1.0, 0.0, None))
            store.end_run(run_id)
            store.close()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
