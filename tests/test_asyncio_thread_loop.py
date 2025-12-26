import asyncio
import threading
import unittest


class TestEnsureThreadEventLoop(unittest.TestCase):
    def test_creates_event_loop_in_new_thread(self) -> None:
        from trading_algo.broker.ibkr import _ensure_thread_event_loop

        out: dict[str, object] = {}

        def worker() -> None:
            # A fresh thread should not have a loop by default.
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                pass

            _ensure_thread_event_loop()
            loop = asyncio.get_event_loop()
            out["loop"] = loop
            loop.close()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        self.assertIn("loop", out)
        self.assertIsInstance(out["loop"], asyncio.AbstractEventLoop)
