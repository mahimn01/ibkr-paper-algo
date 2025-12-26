from __future__ import annotations

import os
import tempfile
import unittest

from trading_algo.broker.sim import SimBroker
from trading_algo.config import TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.llm.chat import ChatSession
from trading_algo.llm.config import LLMConfig
from trading_algo.llm.gemini import LLMClient
from trading_algo.risk import RiskLimits, RiskManager


class _FakeLLM(LLMClient):
    def __init__(self, outputs: list[dict]) -> None:
        self._outputs = list(outputs)

    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        _ = (prompt, system, use_google_search)
        return ""

    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False):
        _ = (prompt, system, use_google_search)
        yield ""

    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ) -> dict[str, object]:
        _ = (contents, system, tools, use_google_search)
        if not self._outputs:
            return {"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}}]}
        return self._outputs.pop(0)

    def stream_generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ):
        _ = (contents, system, tools, use_google_search)
        if not self._outputs:
            return
        # Not used in these tests (stream=False).
        yield self._outputs.pop(0)


class _ExplodingLLM(LLMClient):
    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        _ = (prompt, system, use_google_search)
        raise RuntimeError("HTTP 400 Bad Request")

    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False):
        _ = (prompt, system, use_google_search)
        raise RuntimeError("HTTP 400 Bad Request")

    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ) -> dict[str, object]:
        _ = (contents, system, tools, use_google_search)
        raise RuntimeError("HTTP 400 Bad Request")

    def stream_generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ):
        _ = (contents, system, tools, use_google_search)
        raise RuntimeError("HTTP 400 Bad Request")


class TestLLMChatSession(unittest.TestCase):
    def test_chat_executes_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "audit.sqlite3")
            cfg = TradingConfig(broker="sim", dry_run=False, db_path=db_path)
            llm = LLMConfig(enabled=True, provider="gemini", gemini_api_key="x", allowed_symbols_csv="AAPL")

            fake = _FakeLLM(
                outputs=[
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"text": "Placing a small test order.\n"},
                                        {
                                            "functionCall": {
                                                "name": "place_order",
                                                "args": {
                                                    "order": {
                                                        "instrument": {"kind": "STK", "symbol": "AAPL", "exchange": "SMART", "currency": "USD"},
                                                        "side": "BUY",
                                                        "qty": 1,
                                                        "type": "MKT",
                                                        "tif": "DAY",
                                                    }
                                                },
                                            },
                                            "thoughtSignature": "context_engineering_is_the_way_to_go",
                                        },
                                    ],
                                }
                            }
                        ]
                    },
                    {"candidates": [{"content": {"role": "model", "parts": [{"text": "Done."}]}}]},
                ]
            )

            broker = SimBroker()
            broker.connect()
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)
            try:
                session = ChatSession(
                    broker=broker,
                    trading=cfg,
                    llm=llm,
                    client=fake,
                    risk=RiskManager(RiskLimits()),
                    stream=False,
                )
                session.add_user_message("Buy 1 AAPL market.")
                executed: list[tuple[str, bool]] = []

                def _on_tool(call, ok, result):
                    _ = result
                    executed.append((call.name, ok))

                reply = session.run_turn(on_tool_executed=_on_tool)
                self.assertIn("Placing a small test order.", reply.assistant_message)
                self.assertIn("Done.", reply.assistant_message)
                self.assertEqual(len(broker.orders), 1)
                self.assertEqual(executed, [("place_order", True)])
            finally:
                broker.disconnect()

    def test_chat_blocks_disallowed_symbol(self) -> None:
        cfg = TradingConfig(broker="sim", dry_run=False, db_path=None)
        llm = LLMConfig(enabled=True, provider="gemini", gemini_api_key="x", allowed_symbols_csv="AAPL")
        fake = _FakeLLM(
            outputs=[
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "Attempting disallowed symbol.\n"},
                                    {
                                        "functionCall": {
                                            "name": "place_order",
                                            "args": {
                                                "order": {
                                                    "instrument": {"kind": "STK", "symbol": "TSLA", "exchange": "SMART", "currency": "USD"},
                                                    "side": "BUY",
                                                    "qty": 1,
                                                    "type": "MKT",
                                                    "tif": "DAY",
                                                }
                                            },
                                        },
                                        "thoughtSignature": "context_engineering_is_the_way_to_go",
                                    },
                                ],
                            }
                        }
                    ]
                },
                {"candidates": [{"content": {"role": "model", "parts": [{"text": "Acknowledged."}]}}]},
            ]
        )
        broker = SimBroker()
        broker.connect()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="TSLA"), last=100.0)
        try:
            session = ChatSession(
                broker=broker,
                trading=cfg,
                llm=llm,
                client=fake,
                risk=RiskManager(RiskLimits()),
                stream=False,
            )
            session.add_user_message("Buy 1 TSLA market.")
            executed: list[tuple[str, bool]] = []

            def _on_tool(call, ok, result):
                _ = result
                executed.append((call.name, ok))

            reply = session.run_turn(on_tool_executed=_on_tool)
            self.assertIn("Attempting disallowed symbol.", reply.assistant_message)
            self.assertEqual(len(broker.orders), 0)
            self.assertEqual(executed, [("place_order", False)])
        finally:
            broker.disconnect()

    def test_chat_handles_model_error_without_raising(self) -> None:
        cfg = TradingConfig(broker="sim", dry_run=False, db_path=None)
        llm = LLMConfig(enabled=True, provider="gemini", gemini_api_key="x", allowed_symbols_csv="AAPL")
        broker = SimBroker()
        broker.connect()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)
        try:
            session = ChatSession(
                broker=broker,
                trading=cfg,
                llm=llm,
                client=_ExplodingLLM(),
                risk=RiskManager(RiskLimits()),
                stream=False,
            )
            session.add_user_message("hi")
            reply = session.run_turn()
            self.assertIn("LLM request failed", reply.assistant_message)
        finally:
            broker.disconnect()
