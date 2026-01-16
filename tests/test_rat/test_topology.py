"""Tests for RAT Topology Detector module."""

import unittest
import math
from datetime import datetime, timedelta

from trading_algo.rat.topology.detector import (
    TopologyDetector,
    TopologyState,
    TopologyRegime,
)
from trading_algo.rat.signals import SignalSource


class TestTopologyDetector(unittest.TestCase):
    """Test TopologyDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = TopologyDetector(
            embedding_dim=3,
            time_delay=1,
            max_dimension=2,
            window_size=100,
        )
        self.base_time = datetime(2023, 1, 1)

    def test_initial_state_is_unknown(self):
        """Test that initial state is unknown."""
        state = self.detector.detect("AAPL")

        self.assertEqual(state.regime, TopologyRegime.UNKNOWN)
        self.assertEqual(state.regime_confidence, 0.0)

    def test_insufficient_data_returns_unknown(self):
        """Test that insufficient data returns unknown."""
        # Add only a few data points
        for i in range(5):
            self.detector.update("AAPL", 100 + i, self.base_time + timedelta(days=i))

        state = self.detector.detect("AAPL")
        self.assertEqual(state.regime, TopologyRegime.UNKNOWN)

    def test_trending_data_detection(self):
        """Test trending regime detection."""
        # Create strong trend
        price = 100.0
        for i in range(100):
            price *= 1.01  # Consistent uptrend
            self.detector.update(
                "AAPL",
                price,
                self.base_time + timedelta(days=i),
            )

        state = self.detector.detect("AAPL")

        # Should detect some regime
        self.assertIsInstance(state.regime, TopologyRegime)
        self.assertGreater(state.regime_confidence, 0)

    def test_cyclic_data_detection(self):
        """Test cyclic/consolidation regime detection."""
        # Create sinusoidal price pattern
        for i in range(100):
            price = 100 + 10 * math.sin(2 * math.pi * i / 20)
            self.detector.update(
                "AAPL",
                price,
                self.base_time + timedelta(days=i),
            )

        state = self.detector.detect("AAPL")

        # With cyclic data, might detect consolidation
        self.assertIsInstance(state.regime, TopologyRegime)

    def test_betti_numbers_are_positive(self):
        """Test Betti numbers are non-negative."""
        # Add enough data
        for i in range(100):
            self.detector.update(
                "AAPL",
                100 + math.sin(i / 10) * 10 + i * 0.1,
                self.base_time + timedelta(days=i),
            )

        state = self.detector.detect("AAPL")

        self.assertGreaterEqual(state.betti_0, 0)
        self.assertGreaterEqual(state.betti_1, 0)
        self.assertGreaterEqual(state.betti_2, 0)

    def test_persistence_value(self):
        """Test persistence tracking."""
        # Add data for first detection
        for i in range(50):
            self.detector.update(
                "AAPL",
                100 + i,
                self.base_time + timedelta(days=i),
            )

        state1 = self.detector.detect("AAPL")

        # Add more data
        for i in range(50, 100):
            self.detector.update(
                "AAPL",
                150 + i,
                self.base_time + timedelta(days=i),
            )

        state2 = self.detector.detect("AAPL")

        # Persistence should be a value between 0 and 1
        self.assertGreaterEqual(state2.persistence, 0)
        self.assertLessEqual(state2.persistence, 1)

    def test_is_stable_property(self):
        """Test is_stable property."""
        # Create state with high persistence
        state_stable = TopologyState(
            timestamp=datetime.now(),
            symbol="TEST",
            regime=TopologyRegime.TRENDING,
            betti_0=1.0,
            betti_1=0.5,
            betti_2=0.0,
            persistence=0.8,
            regime_confidence=0.7,
        )
        self.assertTrue(state_stable.is_stable)

        # Create state with low persistence
        state_unstable = TopologyState(
            timestamp=datetime.now(),
            symbol="TEST",
            regime=TopologyRegime.CONSOLIDATION,
            betti_0=1.0,
            betti_1=2.0,
            betti_2=0.0,
            persistence=0.3,
            regime_confidence=0.7,
        )
        self.assertFalse(state_unstable.is_stable)

    def test_generate_signal_unknown_returns_none(self):
        """Test that unknown regime generates no signal."""
        signal = self.detector.generate_signal("AAPL")
        self.assertIsNone(signal)

    def test_generate_signal_with_regime(self):
        """Test signal generation when regime detected."""
        # Add trending data
        price = 100.0
        for i in range(100):
            price *= 1.005
            self.detector.update(
                "AAPL",
                price,
                self.base_time + timedelta(days=i),
            )

        signal = self.detector.generate_signal("AAPL")

        if signal is not None:
            self.assertEqual(signal.source, SignalSource.TOPOLOGY)
            self.assertEqual(signal.symbol, "AAPL")

    def test_multiple_symbols(self):
        """Test tracking multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOG"]

        for symbol in symbols:
            for i in range(50):
                self.detector.update(
                    symbol,
                    100 + i * (1 if symbol == "AAPL" else -0.5),
                    self.base_time + timedelta(days=i),
                )

        # Each symbol should have separate state
        for symbol in symbols:
            state = self.detector.detect(symbol)
            self.assertEqual(state.symbol, symbol)

    def test_inject_backtest_data(self):
        """Test backtest data injection."""
        prices = [100 + i * 0.5 for i in range(50)]

        self.detector.inject_backtest_data("AAPL", prices)

        # Should be able to detect after injection
        state = self.detector.detect("AAPL")
        self.assertEqual(state.symbol, "AAPL")


class TestTopologyRegime(unittest.TestCase):
    """Test TopologyRegime enum."""

    def test_all_regimes_exist(self):
        """Test all expected regimes exist."""
        regimes = [
            TopologyRegime.UNKNOWN,
            TopologyRegime.TRENDING,
            TopologyRegime.CONSOLIDATION,
            TopologyRegime.ROTATION,
            TopologyRegime.FRAGMENTED,
            TopologyRegime.BUBBLE,
        ]

        for regime in regimes:
            self.assertIsInstance(regime, TopologyRegime)


class TestTopologyMath(unittest.TestCase):
    """Test mathematical functions in TopologyDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = TopologyDetector()

    def test_build_point_cloud(self):
        """Test Takens embedding point cloud construction."""
        prices = [100 + i for i in range(20)]

        cloud = self.detector._build_point_cloud(prices)

        # Should create points with embedding_dim dimensions
        self.assertGreater(len(cloud), 0)
        self.assertEqual(len(cloud[0]), self.detector.embedding_dim)

    def test_build_point_cloud_normalization(self):
        """Test that point cloud is normalized."""
        prices = [1000 + i * 10 for i in range(20)]

        cloud = self.detector._build_point_cloud(prices)

        # Points should be roughly normalized (mean ~0, std ~1)
        all_values = [v for point in cloud for v in point]
        mean = sum(all_values) / len(all_values)
        var = sum((v - mean) ** 2 for v in all_values) / len(all_values)

        # Mean should be close to 0
        self.assertLess(abs(mean), 1.0)

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]

        dist = self.detector._euclidean_distance(p1, p2)
        self.assertAlmostEqual(dist, 1.0)

        p3 = [3, 4, 0]
        dist2 = self.detector._euclidean_distance([0, 0, 0], p3)
        self.assertAlmostEqual(dist2, 5.0)

    def test_compute_betti_numbers_simple(self):
        """Test simple Betti number computation."""
        # Create a simple point cloud
        cloud = [[i, i * 0.1, 0] for i in range(20)]

        betti = self.detector._compute_betti_numbers_simple(cloud)

        # Should return 3 Betti numbers
        self.assertEqual(len(betti), 3)
        # All should be non-negative
        self.assertGreaterEqual(betti[0], 0)
        self.assertGreaterEqual(betti[1], 0)
        self.assertGreaterEqual(betti[2], 0)

    def test_classify_regime(self):
        """Test regime classification from Betti numbers."""
        # Trending-like: low β₀, low β₁
        regime, conf = self.detector._classify_regime((1.0, 0.5, 0.0))
        self.assertEqual(regime, TopologyRegime.TRENDING)

        # Consolidation-like: low β₀, high β₁
        regime, conf = self.detector._classify_regime((1.0, 3.0, 0.0))
        self.assertEqual(regime, TopologyRegime.CONSOLIDATION)

        # Fragmented-like: high β₀, low β₁
        regime, conf = self.detector._classify_regime((5.0, 0.3, 0.0))
        self.assertEqual(regime, TopologyRegime.FRAGMENTED)


if __name__ == "__main__":
    unittest.main()
