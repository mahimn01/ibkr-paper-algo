import logging
import os
import unittest

from trading_algo.autorun import main

logging.disable(logging.CRITICAL)


class TestAutorunEntrypoint(unittest.TestCase):
    def test_main_runs_one_tick_sim(self):
        env_backup = dict(os.environ)
        try:
            os.environ["TRADING_BROKER"] = "sim"
            os.environ["TRADING_DRY_RUN"] = "true"
            os.environ["TRADING_LIVE_ENABLED"] = "false"
            os.environ["TRADING_ORDER_TOKEN"] = ""
            os.environ["TRADING_DB_PATH"] = ""
            rc = main(["--broker", "sim", "--max-ticks", "1", "--sleep-seconds", "0"])
            self.assertEqual(rc, 0)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

