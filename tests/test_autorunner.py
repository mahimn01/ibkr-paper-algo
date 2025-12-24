import logging
import os
import sqlite3
import tempfile
import unittest

from trading_algo.autorun import AutoRunner
from trading_algo.broker.base import OrderStatus, OrderRequest
from trading_algo.broker.sim import SimBroker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.risk import RiskLimits, RiskManager

logging.disable(logging.CRITICAL)


class _OneBuyStrategy:
    name = "one-buy"

    def on_tick(self, ctx):
        _ = ctx.get_snapshot(InstrumentSpec(kind="STK", symbol="AAPL"))
        return [TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1)]


class TestAutoRunner(unittest.TestCase):
    def test_dry_run_creates_decision_no_order(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        broker = SimBroker()
        broker.connect()
        try:
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
            cfg = TradingConfig(
                broker="ibkr",
                live_enabled=True,
                dry_run=True,
                order_token="T",
                db_path=path,
                ibkr=IBKRConfig(),
            )
            runner = AutoRunner(
                broker=broker,
                config=cfg,
                strategy=_OneBuyStrategy(),
                risk=RiskManager(RiskLimits(allow_short=True)),
                confirm_token="T",
                sleep_seconds=0.0,
                max_ticks=1,
                track_every_ticks=0,
            )
            runner.run()
            self.assertEqual(len(broker.orders), 0)

            con = sqlite3.connect(path)
            try:
                decisions = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
                self.assertGreaterEqual(int(decisions), 1)
            finally:
                con.close()
        finally:
            broker.disconnect()
            try:
                os.remove(path)
            except OSError:
                pass

    def test_live_token_sends_order(self):
        broker = SimBroker()
        broker.connect()
        try:
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
            cfg = TradingConfig(
                broker="ibkr",
                live_enabled=True,
                dry_run=False,
                order_token="T",
                db_path=None,
                ibkr=IBKRConfig(),
            )
            runner = AutoRunner(
                broker=broker,
                config=cfg,
                strategy=_OneBuyStrategy(),
                risk=RiskManager(RiskLimits(allow_short=True)),
                confirm_token="T",
                sleep_seconds=0.0,
                max_ticks=1,
                track_every_ticks=0,
            )
            runner.run()
            self.assertEqual(len(broker.orders), 1)
        finally:
            broker.disconnect()

    def test_reconcile_records_status_event(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        broker = SimBroker()
        broker.connect()
        try:
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
            # Seed DB with a non-terminal order.
            from trading_algo.persistence import SqliteStore

            cfg = TradingConfig(
                broker="sim",
                live_enabled=True,
                dry_run=False,
                order_token=None,
                db_path=path,
                ibkr=IBKRConfig(),
            )
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

            broker._inject_order_status(OrderStatus(order_id="sim-open", status="Submitted", filled=None, remaining=None, avg_fill_price=None))

            runner = AutoRunner(
                broker=broker,
                config=cfg,
                strategy=_OneBuyStrategy(),
                risk=RiskManager(RiskLimits(allow_short=True)),
                confirm_token=None,
                sleep_seconds=0.0,
                max_ticks=0,
                track_every_ticks=0,
            )
            runner.run()

            con = sqlite3.connect(path)
            try:
                events = con.execute("SELECT COUNT(*) FROM order_status_events WHERE order_id='sim-open'").fetchone()[0]
                self.assertGreaterEqual(int(events), 1)
            finally:
                con.close()
        finally:
            broker.disconnect()
            try:
                os.remove(path)
            except OSError:
                pass

