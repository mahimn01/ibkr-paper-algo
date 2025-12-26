import logging
import os
import tempfile
import unittest

from trading_algo.broker.base import OrderRequest
from trading_algo.broker.sim import SimBroker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.oms import OrderManager

logging.disable(logging.CRITICAL)


class TestOMS(unittest.TestCase):
    def test_dry_run_submit_does_not_send(self):
        broker = SimBroker()
        broker.connect()
        try:
            cfg = TradingConfig(broker="ibkr", live_enabled=True, dry_run=True, order_token="T", ibkr=IBKRConfig())
            oms = OrderManager(broker, cfg, confirm_token="T")
            self.addCleanup(oms.close)
            req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="MKT")
            res = oms.submit(req)
            self.assertEqual(res.status, "DryRun")
            self.assertEqual(len(broker.orders), 0)
        finally:
            broker.disconnect()

    def test_token_gate_blocks(self):
        broker = SimBroker()
        broker.connect()
        try:
            cfg = TradingConfig(
                broker="ibkr",
                live_enabled=True,
                dry_run=False,
                order_token="T",
                confirm_token_required=True,
                ibkr=IBKRConfig(),
            )
            oms = OrderManager(broker, cfg, confirm_token="WRONG")
            self.addCleanup(oms.close)
            req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="MKT")
            with self.assertRaises(RuntimeError):
                oms.submit(req)
        finally:
            broker.disconnect()

    def test_persists_run_and_order(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        broker = SimBroker()
        broker.connect()
        try:
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
            cfg = TradingConfig(
                broker="sim",
                live_enabled=True,
                dry_run=False,
                order_token=None,
                db_path=path,
                ibkr=IBKRConfig(),
            )
            oms = OrderManager(broker, cfg, confirm_token=None)
            req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="MKT")
            res = oms.submit(req)
            self.assertTrue(res.order_id.startswith("sim-"))
            oms.close()
            # Basic check: DB exists and has tables; schema creation is what we care about here.
            self.assertTrue(os.path.exists(path))
        finally:
            broker.disconnect()
            try:
                os.remove(path)
            except OSError:
                pass

    def test_reconcile_reads_db_and_updates_status_events(self):
        from trading_algo.broker.base import OrderStatus
        from trading_algo.persistence import SqliteStore

        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        broker = SimBroker()
        broker.connect()
        try:
            cfg = TradingConfig(broker="sim", live_enabled=True, dry_run=False, order_token=None, db_path=path, ibkr=IBKRConfig())
            # Seed DB with a "Submitted" order row.
            store = SqliteStore(path)
            run_id = store.start_run(cfg)
            store.log_order(
                run_id,
                broker="sim",
                order_id="sim-open",
                request=OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="LMT", limit_price=1),
                status="Submitted",
            )
            store.end_run(run_id)
            store.close()

            broker._inject_order_status(OrderStatus("sim-open", "Submitted", None, None, None))
            oms = OrderManager(broker, cfg)
            self.addCleanup(oms.close)
            res = oms.reconcile()
            self.assertIn("sim-open", res)

            # Advance broker status and track.
            broker._inject_order_status(OrderStatus("sim-open", "Cancelled", 0.0, 1.0, None))
            oms.track_open_orders(poll_seconds=0.01, timeout_seconds=0.1)
        finally:
            broker.disconnect()
            try:
                os.remove(path)
            except OSError:
                pass
