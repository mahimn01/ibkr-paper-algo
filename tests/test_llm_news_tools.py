from __future__ import annotations

import unittest

from trading_algo.broker.sim import SimBroker
from trading_algo.llm.tools import ToolError, dispatch_tool
from trading_algo.oms import OrderManager
from trading_algo.config import TradingConfig


class TestLLMNewsTools(unittest.TestCase):
    def test_news_tools_work_on_sim_broker(self) -> None:
        broker = SimBroker()
        broker.connect()
        try:
            cfg = TradingConfig(broker="sim", dry_run=True, db_path=None)
            oms = OrderManager(broker, cfg, confirm_token=None)
            try:
                res = dispatch_tool(call_name="list_news_providers", call_args={}, broker=broker, oms=oms, allowed_kinds={"STK"}, allowed_symbols=set(), llm_client=None)
                self.assertEqual(res, [])

                res = dispatch_tool(
                    call_name="get_historical_news",
                    call_args={"kind": "STK", "symbol": "AAPL", "provider_codes": ["BRF"]},
                    broker=broker,
                    oms=oms,
                    allowed_kinds={"STK"},
                    allowed_symbols=set(),
                    enforce_allowlist=False,
                    llm_client=None,
                )
                self.assertEqual(res, [])

                res = dispatch_tool(
                    call_name="get_news_article",
                    call_args={"provider_code": "BRF", "article_id": "123", "format": "TEXT"},
                    broker=broker,
                    oms=oms,
                    allowed_kinds={"STK"},
                    allowed_symbols=set(),
                    enforce_allowlist=False,
                    llm_client=None,
                )
                self.assertEqual(res["provider_code"], "BRF")
                self.assertEqual(res["article_id"], "123")
            finally:
                oms.close()
        finally:
            broker.disconnect()

    def test_get_news_article_requires_ids(self) -> None:
        broker = SimBroker()
        broker.connect()
        try:
            cfg = TradingConfig(broker="sim", dry_run=True, db_path=None)
            oms = OrderManager(broker, cfg, confirm_token=None)
            try:
                with self.assertRaises(ToolError):
                    dispatch_tool(
                        call_name="get_news_article",
                        call_args={"provider_code": "BRF"},
                        broker=broker,
                        oms=oms,
                        allowed_kinds={"STK"},
                        allowed_symbols=set(),
                        enforce_allowlist=False,
                        llm_client=None,
                    )
            finally:
                oms.close()
        finally:
            broker.disconnect()
