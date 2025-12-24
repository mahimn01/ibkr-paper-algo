import logging
import unittest

from trading_algo.broker.sim import SimBroker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.engine import Engine, default_risk_manager
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent

logging.disable(logging.CRITICAL)


class _OneBuyStrategy:
    name = "one-buy"

    def on_tick(self, ctx):
        return [TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1)]


class _OneSellStrategy:
    name = "one-sell"

    def on_tick(self, ctx):
        return [TradeIntent(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="SELL", quantity=1)]


class TestEngineSafety(unittest.TestCase):
    def test_blocks_ibkr_when_live_disabled(self):
        broker = SimBroker()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
        cfg = TradingConfig(broker="ibkr", live_enabled=False, poll_seconds=1, ibkr=IBKRConfig())
        engine = Engine(broker=broker, config=cfg, strategy=_OneBuyStrategy(), risk=default_risk_manager())

        engine.run_once()
        self.assertEqual(len(broker.orders), 0)

    def test_risk_blocks_sell_by_default(self):
        broker = SimBroker()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
        cfg = TradingConfig(broker="sim", live_enabled=True, poll_seconds=1, ibkr=IBKRConfig())
        engine = Engine(broker=broker, config=cfg, strategy=_OneSellStrategy(), risk=default_risk_manager())

        engine.run_once()
        self.assertEqual(len(broker.orders), 0)

    def test_blocks_ibkr_without_order_token_when_live_enabled(self):
        broker = SimBroker()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
        cfg = TradingConfig(broker="ibkr", live_enabled=True, dry_run=False, order_token=None, poll_seconds=1, ibkr=IBKRConfig())
        engine = Engine(broker=broker, config=cfg, strategy=_OneBuyStrategy(), risk=default_risk_manager(), confirm_token=None)

        engine.run_once()
        self.assertEqual(len(broker.orders), 0)

    def test_allows_ibkr_when_token_matches(self):
        broker = SimBroker()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100)
        cfg = TradingConfig(
            broker="ibkr",
            live_enabled=True,
            dry_run=False,
            order_token="TOKEN",
            poll_seconds=1,
            ibkr=IBKRConfig(),
        )
        engine = Engine(broker=broker, config=cfg, strategy=_OneBuyStrategy(), risk=default_risk_manager(), confirm_token="TOKEN")

        engine.run_once()
        self.assertEqual(len(broker.orders), 1)
