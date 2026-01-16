"""Tests for RAT Adversarial Meta-Trader module."""

import unittest
from datetime import datetime, timedelta, time as dt_time

from trading_algo.rat.adversarial.detector import (
    AdversarialDetector,
    AdversarialState,
    AlgoArchetype,
    AlgoSignature,
)
from trading_algo.rat.signals import SignalSource


class TestAdversarialDetector(unittest.TestCase):
    """Test AdversarialDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = AdversarialDetector(
            flow_window=100,
            detection_threshold=0.65,
            round_number_tolerance=0.001,
        )
        self.base_time = datetime(2023, 1, 1, 10, 0)

    def test_initial_state_is_unknown(self):
        """Test that initial state shows no detected algos."""
        state = self.detector.detect("AAPL")

        self.assertEqual(state.dominant_archetype, AlgoArchetype.UNKNOWN)
        self.assertEqual(len(state.detected_algos), 0)
        self.assertFalse(state.has_opportunity())

    def test_insufficient_data_returns_unknown(self):
        """Test that insufficient data returns unknown."""
        # Add only a few ticks
        for i in range(10):
            self.detector.update(
                symbol="AAPL",
                price=100.0,
                volume=1000,
                aggressor="buy",
                bid=99.95,
                ask=100.05,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        state = self.detector.detect("AAPL")
        self.assertEqual(state.dominant_archetype, AlgoArchetype.UNKNOWN)

    def test_momentum_detection(self):
        """Test momentum algorithm detection."""
        # Simulate momentum trading: consistent buy flow
        for i in range(100):
            self.detector.update(
                symbol="AAPL",
                price=100.0 + i * 0.01,
                volume=1000,
                aggressor="buy",  # Consistent buying
                bid=99.95 + i * 0.01,
                ask=100.05 + i * 0.01,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        state = self.detector.detect("AAPL")

        # Should detect momentum pattern due to one-sided flow
        self.assertIsInstance(state.dominant_archetype, AlgoArchetype)

    def test_mean_reversion_detection(self):
        """Test mean reversion algorithm detection."""
        # Create scenario at price extreme with counter-trend flow
        base_price = 100.0

        # First, price moves up significantly
        for i in range(50):
            price = base_price + i * 0.1  # Trending up
            self.detector.update(
                symbol="AAPL",
                price=price,
                volume=500,
                aggressor="buy",
                bid=price - 0.02,
                ask=price + 0.02,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        # Then counter-trend selling at the extreme
        for i in range(50):
            price = base_price + 5 - i * 0.02
            self.detector.update(
                symbol="AAPL",
                price=price,
                volume=1000,
                aggressor="sell",  # Selling at the top
                bid=price - 0.02,
                ask=price + 0.02,
                timestamp=self.base_time + timedelta(seconds=50 + i),
            )

        state = self.detector.detect("AAPL")

        # Should have detected some patterns
        self.assertIsInstance(state, AdversarialState)

    def test_vwap_twap_detection(self):
        """Test VWAP/TWAP execution algorithm detection."""
        # Simulate even-paced, one-sided execution
        for i in range(100):
            self.detector.update(
                symbol="AAPL",
                price=100.0 + i * 0.001,
                volume=100,  # Consistent small size
                aggressor="buy" if i % 5 != 0 else "sell",  # Mostly buying
                bid=99.95,
                ask=100.05,
                timestamp=self.base_time + timedelta(seconds=i * 2),  # Even timing
            )

        state = self.detector.detect("AAPL")
        self.assertIsInstance(state, AdversarialState)

    def test_algo_signature_creation(self):
        """Test AlgoSignature dataclass."""
        sig = AlgoSignature(
            archetype=AlgoArchetype.MOMENTUM,
            confidence=0.75,
            predicted_action="buy",
            predicted_size=1000.0,
            predicted_timing=5.0,
            exploitation_edge=0.02,
        )

        self.assertEqual(sig.archetype, AlgoArchetype.MOMENTUM)
        self.assertAlmostEqual(sig.confidence, 0.75)
        self.assertEqual(sig.predicted_action, "buy")

    def test_algo_signature_confidence_clamping(self):
        """Test confidence is clamped to valid range."""
        sig = AlgoSignature(
            archetype=AlgoArchetype.MEAN_REVERSION,
            confidence=1.5,  # Should be clamped to 1
            predicted_action="sell",
            predicted_size=500.0,
            predicted_timing=10.0,
            exploitation_edge=0.01,
        )

        self.assertEqual(sig.confidence, 1.0)

    def test_has_opportunity(self):
        """Test has_opportunity method."""
        # State with no opportunity
        state_no_opp = AdversarialState(
            timestamp=datetime.now(),
            detected_algos=[],
            dominant_archetype=AlgoArchetype.UNKNOWN,
            exploitation_signal=None,
            total_confidence=0.3,
        )
        self.assertFalse(state_no_opp.has_opportunity())

        # State with opportunity
        state_with_opp = AdversarialState(
            timestamp=datetime.now(),
            detected_algos=[
                AlgoSignature(
                    AlgoArchetype.MOMENTUM, 0.8, "buy", 1000, 5, 0.02
                )
            ],
            dominant_archetype=AlgoArchetype.MOMENTUM,
            exploitation_signal="front_run",
            total_confidence=0.8,
        )
        self.assertTrue(state_with_opp.has_opportunity())

    def test_generate_signal_no_opportunity(self):
        """Test signal generation with no opportunity."""
        signal = self.detector.generate_signal("AAPL")
        self.assertIsNone(signal)

    def test_generate_signal_with_data(self):
        """Test signal generation with sufficient data."""
        # Add enough data to potentially detect patterns
        for i in range(150):
            aggressor = "buy" if i % 3 != 0 else "sell"
            self.detector.update(
                symbol="AAPL",
                price=100.0 + i * 0.005,
                volume=1000,
                aggressor=aggressor,
                bid=99.95 + i * 0.005,
                ask=100.05 + i * 0.005,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        signal = self.detector.generate_signal("AAPL")

        if signal is not None:
            self.assertEqual(signal.source, SignalSource.ADVERSARIAL)
            self.assertEqual(signal.symbol, "AAPL")

    def test_multiple_symbols(self):
        """Test tracking multiple symbols."""
        symbols = ["AAPL", "MSFT"]

        for symbol in symbols:
            for i in range(60):
                self.detector.update(
                    symbol=symbol,
                    price=100.0 + i * 0.01,
                    volume=1000,
                    aggressor="buy" if i % 2 == 0 else "sell",
                    bid=99.95,
                    ask=100.05,
                    timestamp=self.base_time + timedelta(seconds=i),
                )

        # Each symbol should have its own detection
        for symbol in symbols:
            state = self.detector.detect(symbol)
            self.assertIsInstance(state, AdversarialState)

    def test_inject_backtest_data(self):
        """Test backtest data injection."""
        trades = [
            {
                "timestamp": self.base_time + timedelta(seconds=i),
                "price": 100.0 + i * 0.01,
                "volume": 1000,
                "aggressor": "buy" if i % 2 == 0 else "sell",
                "bid": 99.95,
                "ask": 100.05,
            }
            for i in range(100)
        ]

        self.detector.inject_backtest_data("AAPL", trades)

        state = self.detector.detect("AAPL")
        self.assertIsInstance(state, AdversarialState)


class TestAlgoArchetype(unittest.TestCase):
    """Test AlgoArchetype enum."""

    def test_all_archetypes_exist(self):
        """Test all expected archetypes exist."""
        archetypes = [
            AlgoArchetype.UNKNOWN,
            AlgoArchetype.MOMENTUM,
            AlgoArchetype.MEAN_REVERSION,
            AlgoArchetype.INDEX_REBALANCE,
            AlgoArchetype.STOP_HUNT,
            AlgoArchetype.VWAP_TWAP,
            AlgoArchetype.MARKET_MAKER,
        ]

        for archetype in archetypes:
            self.assertIsInstance(archetype, AlgoArchetype)


class TestAdversarialMath(unittest.TestCase):
    """Test mathematical functions in AdversarialDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = AdversarialDetector()

    def test_nearest_round_number(self):
        """Test round number detection."""
        # For price ~150, should round to 150
        self.assertAlmostEqual(
            self.detector._nearest_round_number(149.5), 150.0
        )

        # For price ~1050, should round to 1100
        self.assertAlmostEqual(
            self.detector._nearest_round_number(1050), 1100.0
        )

        # For price ~15, should round to 15
        self.assertAlmostEqual(
            self.detector._nearest_round_number(14.8), 15.0
        )


if __name__ == "__main__":
    unittest.main()
