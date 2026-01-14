from __future__ import annotations

import json
import unittest
from unittest import mock

from trading_algo.llm.gemini import GeminiClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestGeminiRestEndpoints(unittest.TestCase):
    def test_count_tokens_hits_endpoint_and_parses_total(self) -> None:
        client = GeminiClient(api_key="AIza" + "x" * 35, model="gemini-3-pro-preview")

        seen = {}

        def _fake_urlopen(req, timeout=None):
            _ = timeout
            seen["url"] = req.full_url
            body = json.loads(req.data.decode("utf-8"))
            seen["body"] = body
            return _FakeHTTPResponse({"totalTokens": 123})

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            total = client.count_tokens(contents=[{"role": "user", "parts": [{"text": "hi"}]}], system="SYS")

        self.assertEqual(total, 123)
        self.assertIn(":countTokens", seen["url"])
        self.assertIn("contents", seen["body"])
        self.assertIn("systemInstruction", seen["body"])

    def test_create_cache_and_delete_cache(self) -> None:
        client = GeminiClient(api_key="AIza" + "x" * 35, model="gemini-3-pro-preview")
        calls: list[tuple[str, str]] = []

        def _fake_urlopen(req, timeout=None):
            _ = timeout
            calls.append((req.get_method(), req.full_url))
            if req.get_method() == "POST":
                body = json.loads(req.data.decode("utf-8"))
                self.assertIn("model", body)
                self.assertIn("contents", body)
                self.assertIn("ttl", body)
                return _FakeHTTPResponse({"name": "cachedContents/abc123"})
            return _FakeHTTPResponse({})

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            name = client.create_cache(contents=[{"role": "user", "parts": [{"text": "hi"}]}], system="SYS", ttl_seconds=60, display_name="x")
            client.delete_cache(name)

        self.assertEqual(name, "cachedContents/abc123")
        self.assertEqual(calls[0][0], "POST")
        self.assertIn("/v1beta/cachedContents", calls[0][1])
        self.assertEqual(calls[1][0], "DELETE")
        self.assertIn("/v1beta/cachedContents/abc123", calls[1][1])

