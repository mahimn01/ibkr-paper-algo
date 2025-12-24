import logging
import unittest

from trading_algo.instruments import InstrumentSpec, validate_instrument

logging.disable(logging.CRITICAL)


class TestInstruments(unittest.TestCase):
    def test_stock_defaults(self):
        spec = validate_instrument(InstrumentSpec(kind="STK", symbol="aapl"))
        self.assertEqual(spec.kind, "STK")
        self.assertEqual(spec.symbol, "AAPL")
        self.assertEqual(spec.exchange, "SMART")
        self.assertEqual(spec.currency, "USD")

    def test_future_requires_exchange_and_expiry(self):
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="FUT", symbol="ES", expiry="202503"))
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="FUT", symbol="ES", exchange="CME"))
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="FUT", symbol="ES", exchange="CME", expiry="BAD"))

        spec = validate_instrument(InstrumentSpec(kind="FUT", symbol="ES", exchange="cme", expiry="202503"))
        self.assertEqual(spec.kind, "FUT")
        self.assertEqual(spec.exchange, "CME")
        self.assertEqual(spec.expiry, "202503")
        self.assertEqual(spec.currency, "USD")

    def test_fx_pair_validation(self):
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="FX", symbol="EUR/USD"))
        spec = validate_instrument(InstrumentSpec(kind="FX", symbol="eurusd"))
        self.assertEqual(spec.kind, "FX")
        self.assertEqual(spec.symbol, "EURUSD")
        self.assertEqual(spec.exchange, "IDEALPRO")
