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

    def test_option_requires_expiry_right_strike(self):
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="OPT", symbol="AAPL"))
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="OPT", symbol="AAPL", expiry="20260116", right="C"))
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="OPT", symbol="AAPL", expiry="20260116", strike=200))
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="OPT", symbol="AAPL", expiry="BAD", right="C", strike=200))
        with self.assertRaises(ValueError):
            validate_instrument(InstrumentSpec(kind="OPT", symbol="AAPL", expiry="20260116", right="X", strike=200))

        spec = validate_instrument(InstrumentSpec(kind="OPT", symbol="aapl", expiry="20260116", right="c", strike=200))
        self.assertEqual(spec.kind, "OPT")
        self.assertEqual(spec.symbol, "AAPL")
        self.assertEqual(spec.expiry, "20260116")
        self.assertEqual(spec.right, "C")
        self.assertEqual(spec.strike, 200.0)
        self.assertEqual(spec.exchange, "SMART")
        self.assertEqual(spec.currency, "USD")
        self.assertEqual(spec.multiplier, "100")
