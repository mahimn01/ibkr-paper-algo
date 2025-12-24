import logging
import unittest

from trading_algo.broker.base import OrderRequest, validate_order_request
from trading_algo.instruments import InstrumentSpec

logging.disable(logging.CRITICAL)


class TestOrders(unittest.TestCase):
    def test_market_order_validation(self):
        req = validate_order_request(
            OrderRequest(
                instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                side="buy",
                quantity=1,
                order_type="mkt",
            )
        )
        self.assertEqual(req.side, "BUY")
        self.assertEqual(req.order_type, "MKT")

    def test_limit_order_requires_price(self):
        with self.assertRaises(ValueError):
            validate_order_request(
                OrderRequest(
                    instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                    side="BUY",
                    quantity=1,
                    order_type="LMT",
                )
            )

        with self.assertRaises(ValueError):
            validate_order_request(
                OrderRequest(
                    instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                    side="BUY",
                    quantity=1,
                    order_type="LMT",
                    limit_price=-1,
                )
            )

    def test_stop_orders_require_stop_price(self):
        with self.assertRaises(ValueError):
            validate_order_request(
                OrderRequest(
                    instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                    side="BUY",
                    quantity=1,
                    order_type="STP",
                )
            )

        req = validate_order_request(
            OrderRequest(
                instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                side="BUY",
                quantity=1,
                order_type="STP",
                stop_price=100,
            )
        )
        self.assertEqual(req.order_type, "STP")

        with self.assertRaises(ValueError):
            validate_order_request(
                OrderRequest(
                    instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                    side="BUY",
                    quantity=1,
                    order_type="STPLMT",
                    stop_price=100,
                )
            )

        req2 = validate_order_request(
            OrderRequest(
                instrument=InstrumentSpec(kind="STK", symbol="AAPL"),
                side="BUY",
                quantity=1,
                order_type="STPLMT",
                stop_price=100,
                limit_price=101,
            )
        )
        self.assertEqual(req2.order_type, "STPLMT")
