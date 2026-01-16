"""Tests for RAT config module."""

import os
import unittest

from trading_algo.rat.config import (
    RATConfig,
    AttentionConfig,
    ReflexivityConfig,
    TopologyConfig,
    AdversarialConfig,
    AlphaConfig,
    SignalConfig,
    RATBacktestConfig,
)
from trading_algo.rat.combiner.combiner import WeightingMethod


class TestAttentionConfig(unittest.TestCase):
    """Test AttentionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AttentionConfig()

        self.assertEqual(config.flow_window, 100)
        self.assertAlmostEqual(config.news_weight, 0.4)
        self.assertAlmostEqual(config.flow_weight, 0.35)
        self.assertAlmostEqual(config.price_weight, 0.25)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AttentionConfig(
            flow_window=200,
            news_weight=0.5,
            flow_weight=0.3,
            price_weight=0.2,
        )

        self.assertEqual(config.flow_window, 200)
        self.assertAlmostEqual(config.news_weight, 0.5)

    def test_frozen(self):
        """Test that config is immutable."""
        config = AttentionConfig()
        with self.assertRaises(Exception):
            config.flow_window = 200


class TestReflexivityConfig(unittest.TestCase):
    """Test ReflexivityConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReflexivityConfig()

        self.assertEqual(config.lookback, 50)
        self.assertEqual(config.lag_order, 5)
        self.assertAlmostEqual(config.significance_level, 0.05)


class TestTopologyConfig(unittest.TestCase):
    """Test TopologyConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TopologyConfig()

        self.assertEqual(config.embedding_dim, 3)
        self.assertEqual(config.time_delay, 1)
        self.assertEqual(config.max_dimension, 2)


class TestAdversarialConfig(unittest.TestCase):
    """Test AdversarialConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AdversarialConfig()

        self.assertEqual(config.flow_window, 500)
        self.assertAlmostEqual(config.detection_threshold, 0.65)


class TestAlphaConfig(unittest.TestCase):
    """Test AlphaConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlphaConfig()

        self.assertEqual(config.sharpe_window, 20)
        self.assertEqual(config.ic_window, 20)
        self.assertFalse(config.enable_llm_mutation)


class TestSignalConfig(unittest.TestCase):
    """Test SignalConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SignalConfig()

        self.assertEqual(config.weighting_method, WeightingMethod.SHARPE_WEIGHTED)
        self.assertEqual(config.min_signals_required, 2)
        self.assertAlmostEqual(config.confidence_threshold, 0.5)


class TestRATConfig(unittest.TestCase):
    """Test main RATConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RATConfig()

        self.assertIsInstance(config.attention, AttentionConfig)
        self.assertIsInstance(config.reflexivity, ReflexivityConfig)
        self.assertIsInstance(config.topology, TopologyConfig)
        self.assertIsInstance(config.adversarial, AdversarialConfig)
        self.assertIsInstance(config.alpha, AlphaConfig)
        self.assertIsInstance(config.signal, SignalConfig)
        self.assertIsInstance(config.backtest, RATBacktestConfig)

    def test_from_env(self):
        """Test loading config from environment."""
        # Set environment variables
        os.environ["RAT_ATTENTION_FLOW_WINDOW"] = "150"
        os.environ["RAT_SIGNAL_CONFIDENCE_THRESHOLD"] = "0.7"
        os.environ["RAT_ALPHA_ENABLE_LLM"] = "false"

        try:
            config = RATConfig.from_env()

            self.assertEqual(config.attention.flow_window, 150)
            self.assertAlmostEqual(config.signal.confidence_threshold, 0.7)
            self.assertFalse(config.alpha.enable_llm_mutation)

        finally:
            # Clean up
            del os.environ["RAT_ATTENTION_FLOW_WINDOW"]
            del os.environ["RAT_SIGNAL_CONFIDENCE_THRESHOLD"]
            del os.environ["RAT_ALPHA_ENABLE_LLM"]

    def test_nested_configs_are_frozen(self):
        """Test that nested configs are also frozen."""
        config = RATConfig()

        with self.assertRaises(Exception):
            config.attention.flow_window = 200


class TestRATBacktestConfig(unittest.TestCase):
    """Test RATBacktestConfig."""

    def test_default_values(self):
        """Test default backtest configuration."""
        config = RATBacktestConfig()

        self.assertAlmostEqual(config.initial_capital, 100000.0)
        self.assertAlmostEqual(config.commission_per_share, 0.005)
        self.assertAlmostEqual(config.slippage_bps, 5.0)
        self.assertAlmostEqual(config.max_daily_loss_pct, 0.02)


if __name__ == "__main__":
    unittest.main()
