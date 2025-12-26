from __future__ import annotations

import unittest

from trading_algo.llm.decision import enforce_llm_limits, parse_llm_decisions


class TestLLMDecisionParsing(unittest.TestCase):
    def test_parse_decisions_json(self) -> None:
        raw = """
        {
          "decisions": [
            {
              "action": "PLACE",
              "reason": "test",
              "order": {
                "instrument": {"kind": "STK", "symbol": "AAPL", "exchange": "SMART", "currency": "USD"},
                "side": "BUY",
                "qty": 1,
                "type": "MKT",
                "tif": "DAY"
              }
            },
            {"action":"CANCEL","order_id":"123","reason":"no longer needed"}
          ]
        }
        """
        decisions = parse_llm_decisions(raw)
        self.assertEqual(len(decisions), 2)

    def test_parse_code_fences(self) -> None:
        raw = """```json
        {"decisions":[{"action":"CANCEL","order_id":"1"}]}
        ```"""
        decisions = parse_llm_decisions(raw)
        self.assertEqual(len(decisions), 1)

    def test_enforce_limits(self) -> None:
        raw = """
        {"decisions":[
          {"action":"PLACE","order":{
            "instrument":{"kind":"STK","symbol":"AAPL","exchange":"SMART","currency":"USD"},
            "side":"BUY","qty":2,"type":"MKT","tif":"DAY"
          }}
        ]}
        """
        decisions = parse_llm_decisions(raw)
        enforce_llm_limits(
            decisions,
            allowed_kinds={"STK"},
            allowed_symbols={"AAPL"},
            max_orders=1,
            max_qty=5.0,
        )

    def test_reject_too_many(self) -> None:
        raw = """{"decisions":[{"action":"CANCEL","order_id":"1"},{"action":"CANCEL","order_id":"2"}]}"""
        decisions = parse_llm_decisions(raw)
        with self.assertRaises(ValueError):
            enforce_llm_limits(decisions, allowed_kinds={"STK"}, allowed_symbols=set(), max_orders=1, max_qty=1.0)

