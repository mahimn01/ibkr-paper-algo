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


class _CacheAwareFakeLLM(LLMClient):
    def __init__(self) -> None:
        self.cache_created: int = 0
        self.cache_deleted: int = 0
        self.last_cached_content: str | None = None
        self.calls: int = 0

    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        _ = (prompt, system, use_google_search)
        return ""

    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False):
        _ = (prompt, system, use_google_search)
        yield ""

    def create_cache(self, *, contents, system=None, ttl_seconds=600, display_name=None) -> str:
        _ = (contents, system, ttl_seconds, display_name)
        self.cache_created += 1
        return "cachedContents/test"

    def delete_cache(self, name: str) -> None:
        _ = name
        self.cache_deleted += 1

    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        cached_content: str | None = None,
    ) -> dict[str, object]:
        _ = (contents, system, tools, use_google_search)
        self.calls += 1
        self.last_cached_content = cached_content

        if self.calls == 1:
            # First response: tool call.
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
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15, "cachedContentTokenCount": 5},
            }
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Done."}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15, "cachedContentTokenCount": 5},
        }

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
        return 10_000


class TestChatCaching(unittest.TestCase):
    def test_chat_uses_cache_for_tool_loop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "audit.sqlite3")
            cfg = TradingConfig(broker="sim", dry_run=False, db_path=db_path)
            llm = LLMConfig(
                enabled=True,
                provider="gemini",
                gemini_api_key="x",
                allowed_symbols_csv="AAPL",
                gemini_explicit_caching=True,
                gemini_cache_min_tokens=1,
            )
            broker = SimBroker()
            broker.connect()
            broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)
            try:
                fake = _CacheAwareFakeLLM()
                session = ChatSession(
                    broker=broker,
                    trading=cfg,
                    llm=llm,
                    client=fake,
                    risk=RiskManager(RiskLimits()),
                    stream=False,
                )
                # Add an earlier message so prefix is non-empty and cacheable.
                session.add_user_message("prior context")
                session.add_user_message("Buy 1 AAPL market.")
                reply = session.run_turn()
                self.assertIn("Done.", reply.assistant_message)
                self.assertEqual(fake.cache_created, 1)
                self.assertEqual(fake.cache_deleted, 1)
                self.assertEqual(fake.last_cached_content, "cachedContents/test")
            finally:
                broker.disconnect()

