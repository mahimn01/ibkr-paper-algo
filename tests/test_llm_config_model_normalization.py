from __future__ import annotations

import unittest

from trading_algo.llm.config import LLMConfig


class TestLLMConfigModelNormalization(unittest.TestCase):
    def test_normalizes_gemini_3_pro(self) -> None:
        cfg = LLMConfig(enabled=True, provider="gemini", gemini_api_key="x", gemini_model="gemini-3-pro", allowed_symbols_csv="AAPL")
        self.assertEqual(cfg.normalized_gemini_model(), "gemini-3-pro-preview")
