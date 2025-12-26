import logging
import os
import tempfile
import unittest


class TestLoggingSetup(unittest.TestCase):
    def test_configure_logging_file_only_no_console(self) -> None:
        from trading_algo.logging_setup import configure_logging

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tui.log")
            configure_logging(level=logging.INFO, log_file=path, console=False)
            root = logging.getLogger()

            # No console StreamHandler writing to stdout/stderr.
            console_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(console_handlers, [])
            self.assertTrue(any(isinstance(h, logging.FileHandler) for h in root.handlers))

