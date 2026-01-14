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


class _PrefetchRecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._step = 0

    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        _ = (prompt, system, use_google_search)
        return ""

    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False):
        _ = (prompt, system, use_google_search)
        yield ""

    def count_tokens(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        cached_content: str | None = None,
    ) -> int:
        _ = (contents, system, tools, use_google_search, cached_content)
        return 1

    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        cached_content: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {"contents": contents, "system": system, "tools": tools, "use_google_search": use_google_search, "cached_content": cached_content}
        )
        self._step += 1

        # Step 1: search prefetch (no tools/functionDeclarations, google search enabled)
        if self._step == 1:
            return {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Research brief."}]}, "groundingMetadata": {"groundingChunks": [], "groundingSupports": []}}
                ]
            }
        # Step 2: tool call
        if self._step == 2:
            return {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Placing.\n"},
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
            }
        # Step 3: final answer
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": "Done."}]}}]}

    def stream_generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        cached_content: str | None = None,
    ):
        _ = (contents, system, tools, use_google_search, cached_content)
        yield {"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}}]}

    def create_cache(self, *, contents, system=None, ttl_seconds=600, display_name=None) -> str:
        _ = (contents, system, ttl_seconds, display_name)
        return "cachedContents/test"

    def delete_cache(self, name: str) -> None:
        _ = name


class TestChatSearchPrefetch(unittest.TestCase):
    def test_prefetch_runs_with_google_search_before_tool_loop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "audit.sqlite3")
            cfg = TradingConfig(broker="sim", dry_run=False, db_path=db_path)
            llm = LLMConfig(enabled=True, provider="gemini", gemini_api_key="x", allowed_symbols_csv="AAPL", gemini_use_google_search=True)
            broker = SimBroker()
            broker.connect()
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)
            try:
                client = _PrefetchRecordingLLM()
                session = ChatSession(
                    broker=broker,
                    trading=cfg,
                    llm=llm,
                    client=client,
                    risk=RiskManager(RiskLimits()),
                    stream=False,
                )
                session.add_user_message("Buy 1 AAPL market.")
                reply = session.run_turn()
                self.assertIn("Done.", reply.assistant_message)

                self.assertGreaterEqual(len(client.calls), 2)
                # First call should be google search prefetch.
                self.assertTrue(client.calls[0]["use_google_search"])
                self.assertIsNone(client.calls[0]["tools"])
                # Second call is tool loop; google search disabled there.
                self.assertFalse(client.calls[1]["use_google_search"])
                self.assertIsNotNone(client.calls[1]["tools"])
            finally:
                broker.disconnect()
