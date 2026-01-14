from __future__ import annotations

import unittest

from trading_algo.llm.config import LLMConfig
from trading_algo.llm.context_manager import maybe_compact_history
from trading_algo.llm.gemini import LLMClient


class _FakeSummarizer(LLMClient):
    def __init__(self) -> None:
        self.count_calls: int = 0
        self.summary_calls: int = 0

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
        cached_content: str | None = None,
    ) -> dict[str, object]:
        _ = (contents, system, tools, use_google_search, cached_content)
        self.summary_calls += 1
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": "SUMMARY"}]}}]}

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
        self.count_calls += 1
        return 10_000


class TestContextManager(unittest.TestCase):
    def test_compacts_when_over_budget(self) -> None:
        client = _FakeSummarizer()
        llm_cfg = LLMConfig(
            enabled=True,
            provider="gemini",
            gemini_api_key="x",
            max_context_tokens=100,
            keep_recent_contents=2,
            summarize_trigger_ratio=0.01,
        )
        system = "SYS"
        contents = [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
            {"role": "user", "parts": [{"text": "more"}]},
            {"role": "model", "parts": [{"text": "ok"}]},
            {"role": "user", "parts": [{"text": "even more"}]},
        ]

        new_contents, stats, summary = maybe_compact_history(client=client, llm_cfg=llm_cfg, system_prompt=system, contents=contents)
        self.assertIsNotNone(summary)
        self.assertGreaterEqual(stats.approx_tokens, 1)
        self.assertEqual(client.count_calls, 1)
        self.assertEqual(client.summary_calls, 1)
        self.assertGreaterEqual(len(new_contents), 3)
        self.assertEqual(new_contents[0]["role"], "user")
        mem = new_contents[0]["parts"][0]["text"]
        self.assertIn("[conversation_memory]", str(mem))

    def test_no_compact_under_trigger(self) -> None:
        client = _FakeSummarizer()
        llm_cfg = LLMConfig(
            enabled=True,
            provider="gemini",
            gemini_api_key="x",
            max_context_tokens=10_000_000,
            keep_recent_contents=2,
            summarize_trigger_ratio=0.90,
        )
        contents = [{"role": "user", "parts": [{"text": "hi"}]}]
        new_contents, stats, summary = maybe_compact_history(client=client, llm_cfg=llm_cfg, system_prompt="SYS", contents=contents)
        self.assertEqual(new_contents, contents)
        self.assertIsNone(summary)
        self.assertEqual(client.count_calls, 0)

