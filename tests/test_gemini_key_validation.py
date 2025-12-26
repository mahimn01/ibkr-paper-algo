from __future__ import annotations

import unittest

from trading_algo.llm.gemini import _validate_api_key


class TestGeminiKeyValidation(unittest.TestCase):
    def test_rejects_trailing_comma(self) -> None:
        with self.assertRaises(RuntimeError):
            _validate_api_key("AIzaSyabc123,")

    def test_rejects_whitespace(self) -> None:
        with self.assertRaises(RuntimeError):
            _validate_api_key("AIzaSyabc 123")

    def test_accepts_basic_format(self) -> None:
        _validate_api_key("AIzaSyabc123DEF_456-7890")

