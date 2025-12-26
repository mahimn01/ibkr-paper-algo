from __future__ import annotations

import io
import unittest

from trading_algo.llm.gemini import _iter_sse_json_objects


class TestGeminiSSEParser(unittest.TestCase):
    def test_parses_multiline_data_events(self) -> None:
        payload = (
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hel\"}]}}]}\n"
            b"\n"
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n"
            b"\n"
        )
        resp = io.BytesIO(payload)
        objs = list(_iter_sse_json_objects(resp))
        self.assertEqual(len(objs), 2)
        self.assertIn("candidates", objs[0])

    def test_ignores_done(self) -> None:
        payload = b"data: [DONE]\n\n"
        resp = io.BytesIO(payload)
        objs = list(_iter_sse_json_objects(resp))
        self.assertEqual(objs, [])

