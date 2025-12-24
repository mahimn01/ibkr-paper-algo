import logging
import unittest

from trading_algo.broker.base import OrderRequest, validate_order_request
from trading_algo.instruments import InstrumentSpec

logging.disable(logging.CRITICAL)


class TestOrderRequestAdvancedFields(unittest.TestCase):
    def test_gtd_requires_good_till_date(self):
        with self.assertRaises(ValueError):
            validate_order_request(
                OrderRequest(
                    instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                    side="BUY",
                    quantity=1,
                    order_type="LMT",
                    limit_price=100,
                    tif="GTD",
                )
            )

