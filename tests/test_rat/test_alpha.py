"""Tests for RAT Alpha Tracker and Mutator modules."""

import unittest
from datetime import datetime, timedelta

from trading_algo.rat.alpha.tracker import (
    AlphaTracker,
    AlphaFactor,
    AlphaState,
    DecayStage,
)
from trading_algo.rat.alpha.mutator import (
    AlphaMutator,
    MutationType,
    MutationResult,
)
from trading_algo.rat.signals import SignalSource


class TestAlphaTracker(unittest.TestCase):
    """Test AlphaTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = AlphaTracker(
            sharpe_window=10,
            ic_window=10,
            decay_threshold=0.5,
            crowding_threshold=0.7,
        )
        self.base_time = datetime(2023, 1, 1)

    def test_register_factor(self):
        """Test registering a new factor."""
        def simple_momentum(data):
            prices = data.get("prices", [])
            if len(prices) < 2:
                return 0.0
            return (prices[-1] - prices[0]) / prices[0]

        self.tracker.register_factor("momentum", simple_momentum)

        self.assertIn("momentum", self.tracker._factors)
        self.assertEqual(
            self.tracker._factors["momentum"].current_stage,
            DecayStage.FRESH,
        )

    def test_update_factor_performance(self):
        """Test updating factor performance."""
        def dummy_factor(data):
            return 0.5

        self.tracker.register_factor("dummy", dummy_factor)

        for i in range(15):
            self.tracker.update_factor_performance(
                name="dummy",
                prediction=0.5,
                actual=0.4 if i % 2 == 0 else 0.6,
                pnl=0.01 if i % 2 == 0 else -0.005,
                market_volume=100000,
                timestamp=self.base_time + timedelta(days=i),
            )

        factor = self.tracker._factors["dummy"]
        self.assertEqual(factor.trade_count, 15)
        self.assertGreater(len(factor.returns), 0)

    def test_factor_is_viable(self):
        """Test factor viability check."""
        factor = AlphaFactor(
            name="test",
            compute_fn=lambda x: 0,
            current_stage=DecayStage.ALPHA,
        )
        self.assertTrue(factor.is_viable())

        factor_dead = AlphaFactor(
            name="test_dead",
            compute_fn=lambda x: 0,
            current_stage=DecayStage.DEAD,
        )
        self.assertFalse(factor_dead.is_viable())

    def test_analyze_state(self):
        """Test analyzing alpha state."""
        def factor_a(data):
            return 0.5

        def factor_b(data):
            return -0.3

        self.tracker.register_factor("factor_a", factor_a)
        self.tracker.register_factor("factor_b", factor_b)

        state = self.tracker.analyze()

        self.assertIsInstance(state, AlphaState)
        self.assertEqual(len(state.active_factors), 2)
        self.assertEqual(len(state.decaying_factors), 0)

    def test_rolling_sharpe_computation(self):
        """Test rolling Sharpe ratio computation."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.01, -0.008, 0.012, 0.005, -0.002]

        sharpe = self.tracker._compute_rolling_sharpe(returns)

        # Sharpe should be a finite number
        self.assertFalse(float('inf') == sharpe)
        self.assertFalse(float('-inf') == sharpe)

    def test_information_coefficient_computation(self):
        """Test IC computation."""
        predictions = [0.5, 0.6, 0.4, 0.7, 0.3]
        actuals = [0.52, 0.58, 0.42, 0.65, 0.35]

        ic = self.tracker._compute_information_coefficient(predictions, actuals)

        # IC should be between -1 and 1
        self.assertGreaterEqual(ic, -1)
        self.assertLessEqual(ic, 1)
        # With correlated data, should be positive
        self.assertGreater(ic, 0)

    def test_generate_signal(self):
        """Test signal generation from best factor."""
        def good_factor(data):
            prices = data.get("prices", [])
            if len(prices) < 2:
                return 0.0
            return 0.5

        self.tracker.register_factor("good_factor", good_factor)

        # Update performance to make it viable
        for i in range(25):
            self.tracker.update_factor_performance(
                name="good_factor",
                prediction=0.5,
                actual=0.45,
                pnl=0.01,
                market_volume=100000,
            )

        data = {"prices": [100, 101, 102, 103, 104]}
        signal = self.tracker.generate_signal("AAPL", data)

        if signal is not None:
            self.assertEqual(signal.source, SignalSource.ALPHA)

    def test_inject_backtest_data(self):
        """Test backtest data injection."""
        def test_factor(data):
            return 0.3

        self.tracker.register_factor("test_factor", test_factor)

        history = [
            {
                "timestamp": self.base_time + timedelta(days=i),
                "prediction": 0.3,
                "actual": 0.28 + i * 0.01,
                "pnl": 0.01 if i % 2 == 0 else -0.005,
                "market_volume": 100000,
            }
            for i in range(30)
        ]

        self.tracker.inject_backtest_data("test_factor", history)

        factor = self.tracker._factors["test_factor"]
        self.assertEqual(factor.trade_count, 30)


class TestDecayStage(unittest.TestCase):
    """Test DecayStage enum."""

    def test_all_stages_exist(self):
        """Test all expected stages exist."""
        stages = [
            DecayStage.FRESH,
            DecayStage.ALPHA,
            DecayStage.MATURE,
            DecayStage.DECAYING,
            DecayStage.CROWDED,
            DecayStage.DEAD,
        ]

        for stage in stages:
            self.assertIsInstance(stage, DecayStage)


class TestAlphaMutator(unittest.TestCase):
    """Test AlphaMutator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = AlphaTracker()
        self.mutator = AlphaMutator(
            tracker=self.tracker,
            llm_client=None,
            enable_llm=False,
        )

    def test_generate_new_factors(self):
        """Test generating new factors from templates."""
        results = self.mutator.generate_new_factors(count=3)

        self.assertEqual(len(results), 3)

        for result in results:
            self.assertIsInstance(result, MutationResult)
            self.assertFalse(result.used_llm)
            self.assertIn(result.new_name, self.tracker._factors)

    def test_generate_factors_from_specific_template(self):
        """Test generating factors from specific template."""
        results = self.mutator.generate_new_factors(
            count=2,
            template_name="momentum",
        )

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("momentum", result.original_name)

    def test_mutate_factor_parameter_shift(self):
        """Test parameter shift mutation."""
        def base_factor(data):
            return 0.5

        self.tracker.register_factor("base", base_factor)

        result = self.mutator.mutate_factor(
            "base",
            mutation_type=MutationType.PARAMETER_SHIFT,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.mutation_type, MutationType.PARAMETER_SHIFT)
        self.assertFalse(result.used_llm)

    def test_mutate_factor_inverse(self):
        """Test inverse mutation."""
        def base_factor(data):
            return 0.5

        self.tracker.register_factor("base", base_factor)

        result = self.mutator.mutate_factor(
            "base",
            mutation_type=MutationType.INVERSE,
        )

        self.assertIsNotNone(result)

        # Test that inverse actually inverts
        test_data = {"prices": [100, 101, 102]}
        original_signal = base_factor(test_data)
        mutated_signal = result.compute_fn(test_data)

        self.assertAlmostEqual(mutated_signal, -original_signal)

    def test_mutate_factor_timeframe(self):
        """Test timeframe mutation."""
        def base_factor(data):
            prices = data.get("prices", [])
            return len(prices) / 100.0

        self.tracker.register_factor("base", base_factor)

        result = self.mutator.mutate_factor(
            "base",
            mutation_type=MutationType.TIMEFRAME_CHANGE,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.mutation_type, MutationType.TIMEFRAME_CHANGE)

    def test_mutate_nonexistent_factor(self):
        """Test mutation of nonexistent factor returns None."""
        result = self.mutator.mutate_factor("nonexistent")
        self.assertIsNone(result)

    def test_llm_not_used_when_disabled(self):
        """Test that LLM is not used when disabled."""
        def base_factor(data):
            return 0.5

        self.tracker.register_factor("base", base_factor)

        result = self.mutator.mutate_factor(
            "base",
            mutation_type=MutationType.LLM_FORMULA,
            force_llm=True,
        )

        # Should be None since LLM is disabled
        self.assertIsNone(result)

    def test_get_mutation_stats(self):
        """Test getting mutation statistics."""
        # Generate some mutations
        self.mutator.generate_new_factors(count=2)

        stats = self.mutator.get_mutation_stats()

        self.assertIn("total_mutations", stats)
        self.assertIn("llm_mutations", stats)
        self.assertIn("math_mutations", stats)
        self.assertEqual(stats["llm_mutations"], 0)

    def test_factor_templates_exist(self):
        """Test that factor templates are defined."""
        templates = self.mutator.FACTOR_TEMPLATES

        expected_templates = [
            "momentum",
            "mean_reversion",
            "rsi",
            "volatility_breakout",
            "order_flow_imbalance",
        ]

        for template_name in expected_templates:
            self.assertIn(template_name, templates)


class TestMutationType(unittest.TestCase):
    """Test MutationType enum."""

    def test_all_types_exist(self):
        """Test all expected mutation types exist."""
        types = [
            MutationType.PARAMETER_SHIFT,
            MutationType.TIMEFRAME_CHANGE,
            MutationType.INDICATOR_COMBINE,
            MutationType.REGIME_CONDITION,
            MutationType.DECAY_ADJUST,
            MutationType.INVERSE,
            MutationType.NORMALIZE,
            MutationType.LLM_FORMULA,
            MutationType.LLM_COMBINATION,
        ]

        for mut_type in types:
            self.assertIsInstance(mut_type, MutationType)


class TestFactorComputation(unittest.TestCase):
    """Test generated factor computations."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = AlphaTracker()
        self.mutator = AlphaMutator(tracker=self.tracker, enable_llm=False)

    def test_generated_factors_are_callable(self):
        """Test that generated factors can be called."""
        results = self.mutator.generate_new_factors(count=5)

        test_data = {
            "prices": [100 + i for i in range(50)],
            "highs": [101 + i for i in range(50)],
            "lows": [99 + i for i in range(50)],
            "volumes": [10000] * 50,
            "buy_volume": [6000] * 50,
            "sell_volume": [4000] * 50,
        }

        for result in results:
            factor = self.tracker._factors[result.new_name]
            try:
                signal = factor.compute_fn(test_data)
                self.assertIsInstance(signal, (int, float))
            except Exception as e:
                self.fail(f"Factor {result.new_name} failed: {e}")

    def test_generated_factors_handle_empty_data(self):
        """Test that generated factors handle empty data gracefully."""
        results = self.mutator.generate_new_factors(count=3)

        empty_data = {"prices": []}

        for result in results:
            factor = self.tracker._factors[result.new_name]
            try:
                signal = factor.compute_fn(empty_data)
                self.assertIsInstance(signal, (int, float))
            except Exception as e:
                self.fail(f"Factor {result.new_name} failed on empty data: {e}")


if __name__ == "__main__":
    unittest.main()
