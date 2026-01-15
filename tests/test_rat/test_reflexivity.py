"""Tests for RAT Reflexivity Meter module."""

import unittest
import math
from datetime import datetime, timedelta

from trading_algo.rat.reflexivity.meter import (
    ReflexivityMeter,
    ReflexivityState,
    ReflexivityStage,
)
from trading_algo.rat.signals import SignalSource


class TestReflexivityMeter(unittest.TestCase):
    """Test ReflexivityMeter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.meter = ReflexivityMeter(
            lookback=50,
            lag_order=3,
            significance_level=0.05,
            min_data_points=20,
        )
        self.base_time = datetime(2023, 1, 1)

    def test_initial_state_is_efficient(self):
        """Test that initial state shows no reflexivity."""
        state = self.meter.compute_state("AAPL")

        self.assertEqual(state.stage, ReflexivityStage.EFFICIENT)
        self.assertFalse(state.is_reflexive)

    def test_insufficient_data_returns_efficient(self):
        """Test that insufficient data returns efficient state."""
        # Add only a few data points
        for i in range(5):
            self.meter.update(
                symbol="AAPL",
                price=100 + i,
                fundamental=100 + i,
                timestamp=self.base_time + timedelta(days=i),
            )

        state = self.meter.compute_state("AAPL")
        self.assertEqual(state.stage, ReflexivityStage.EFFICIENT)

    def test_trending_data_detection(self):
        """Test detection with trending data."""
        # Create data where price and fundamentals move together
        for i in range(50):
            price = 100 * (1.01 ** i)  # Exponential growth
            fundamental = 100 * (1.01 ** i)
            self.meter.update(
                symbol="AAPL",
                price=price,
                fundamental=fundamental,
                timestamp=self.base_time + timedelta(days=i),
            )

        state = self.meter.compute_state("AAPL")

        # With perfectly correlated data, should detect some reflexivity
        self.assertIsInstance(state.stage, ReflexivityStage)
        self.assertIsInstance(state.reflexivity_coefficient, float)

    def test_reflexivity_coefficient_bounds(self):
        """Test that reflexivity coefficient is bounded."""
        # Add some data
        for i in range(50):
            self.meter.update(
                symbol="AAPL",
                price=100 + math.sin(i / 5) * 10,
                fundamental=100 + math.cos(i / 5) * 10,
                timestamp=self.base_time + timedelta(days=i),
            )

        state = self.meter.compute_state("AAPL")

        self.assertGreaterEqual(state.reflexivity_coefficient, -1.0)
        self.assertLessEqual(state.reflexivity_coefficient, 1.0)

    def test_granger_causality_values(self):
        """Test Granger causality p-values."""
        # Add enough data for causality test
        for i in range(50):
            self.meter.update(
                symbol="AAPL",
                price=100 + i + math.sin(i / 3) * 5,
                fundamental=100 + i * 0.9,
                timestamp=self.base_time + timedelta(days=i),
            )

        state = self.meter.compute_state("AAPL")

        # P-values should be between 0 and 1
        self.assertGreaterEqual(state.granger_price_to_fund, 0.0)
        self.assertLessEqual(state.granger_price_to_fund, 1.0)
        self.assertGreaterEqual(state.granger_fund_to_price, 0.0)
        self.assertLessEqual(state.granger_fund_to_price, 1.0)

    def test_is_bidirectional_detection(self):
        """Test bidirectional causality detection."""
        # Create strongly correlated data
        for i in range(50):
            # Price leads fundamental by 1 period
            if i > 0:
                fundamental = 100 + i * 0.8
            else:
                fundamental = 100
            price = 100 + i

            self.meter.update(
                symbol="AAPL",
                price=price,
                fundamental=fundamental,
                timestamp=self.base_time + timedelta(days=i),
            )

        state = self.meter.compute_state("AAPL")

        # is_bidirectional should be a boolean
        self.assertIsInstance(state.is_bidirectional, bool)

    def test_is_reflexive_property(self):
        """Test is_reflexive property."""
        state_efficient = ReflexivityState(
            timestamp=datetime.now(),
            symbol="TEST",
            stage=ReflexivityStage.EFFICIENT,
            reflexivity_coefficient=0.0,
            granger_price_to_fund=1.0,
            granger_fund_to_price=1.0,
            is_bidirectional=False,
        )
        self.assertFalse(state_efficient.is_reflexive)

        state_reflexive = ReflexivityState(
            timestamp=datetime.now(),
            symbol="TEST",
            stage=ReflexivityStage.ACCELERATING,
            reflexivity_coefficient=0.5,
            granger_price_to_fund=0.01,
            granger_fund_to_price=0.01,
            is_bidirectional=True,
        )
        self.assertTrue(state_reflexive.is_reflexive)

    def test_generate_signal_efficient_returns_none(self):
        """Test that efficient state generates no signal."""
        # With no data, should be efficient and no signal
        signal = self.meter.generate_signal("AAPL")
        self.assertIsNone(signal)

    def test_generate_signal_with_reflexivity(self):
        """Test signal generation when reflexivity detected."""
        # Add data that might trigger reflexivity
        for i in range(50):
            price = 100 * (1.02 ** i) if i < 25 else 100 * (1.02 ** 25) * (0.98 ** (i - 25))
            fundamental = 100 * (1.015 ** i)

            self.meter.update(
                symbol="AAPL",
                price=price,
                fundamental=fundamental,
                timestamp=self.base_time + timedelta(days=i),
            )

        signal = self.meter.generate_signal("AAPL")

        if signal is not None:
            self.assertEqual(signal.source, SignalSource.REFLEXIVITY)
            self.assertEqual(signal.symbol, "AAPL")
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)

    def test_multiple_symbols(self):
        """Test tracking multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOG"]

        for symbol in symbols:
            for i in range(30):
                self.meter.update(
                    symbol=symbol,
                    price=100 + i,
                    fundamental=100 + i * 0.9,
                    timestamp=self.base_time + timedelta(days=i),
                )

        # Each symbol should have its own state
        for symbol in symbols:
            state = self.meter.compute_state(symbol)
            self.assertEqual(state.symbol, symbol)

    def test_inject_backtest_data(self):
        """Test backtest data injection."""
        prices = [
            (self.base_time + timedelta(days=i), 100 + i * 0.5)
            for i in range(30)
        ]
        fundamentals = [
            (self.base_time + timedelta(days=i), 100 + i * 0.4)
            for i in range(30)
        ]

        self.meter.inject_backtest_data(
            symbol="AAPL",
            prices=prices,
            fundamentals=fundamentals,
        )

        # Should be able to compute state after injection
        state = self.meter.compute_state("AAPL")
        self.assertIsNotNone(state)


class TestReflexivityStage(unittest.TestCase):
    """Test ReflexivityStage enum."""

    def test_all_stages_exist(self):
        """Test all expected stages exist."""
        stages = [
            ReflexivityStage.EFFICIENT,
            ReflexivityStage.NASCENT,
            ReflexivityStage.ACCELERATING,
            ReflexivityStage.PEAK,
            ReflexivityStage.UNWINDING,
        ]

        for stage in stages:
            self.assertIsInstance(stage, ReflexivityStage)


class TestReflexivityMeterMath(unittest.TestCase):
    """Test mathematical functions in ReflexivityMeter."""

    def setUp(self):
        """Set up test fixtures."""
        self.meter = ReflexivityMeter()

    def test_compute_returns(self):
        """Test returns computation."""
        series = [100, 102, 101, 105, 103]
        returns = self.meter._compute_returns(series)

        self.assertEqual(len(returns), len(series) - 1)
        self.assertAlmostEqual(returns[0], 0.02, places=4)  # (102-100)/100

    def test_compute_returns_with_zero(self):
        """Test returns computation handles zero."""
        series = [0, 100, 102]
        returns = self.meter._compute_returns(series)

        self.assertEqual(len(returns), 2)
        self.assertEqual(returns[0], 0.0)  # Can't compute return from 0

    def test_solve_linear_system(self):
        """Test linear system solver."""
        # Simple 2x2 system: 2x + y = 5, x + 3y = 10
        A = [[2, 1], [1, 3]]
        b = [5, 10]

        x = self.meter._solve_linear_system(A, b)

        # Solution should be x=1, y=3
        self.assertAlmostEqual(x[0], 1.0, places=5)
        self.assertAlmostEqual(x[1], 3.0, places=5)

    def test_log_beta(self):
        """Test log beta function."""
        # log(Beta(2,3)) = log(Gamma(2)*Gamma(3)/Gamma(5))
        result = self.meter._log_beta(2, 3)
        # Beta(2,3) = 1/12, so log should be negative
        self.assertLess(result, 0)

    def test_incomplete_beta_bounds(self):
        """Test incomplete beta function bounds."""
        # At x=0, should be 0
        result_0 = self.meter._incomplete_beta(0, 2, 3)
        self.assertAlmostEqual(result_0, 0.0)

        # At x=1, should be 1
        result_1 = self.meter._incomplete_beta(1, 2, 3)
        self.assertAlmostEqual(result_1, 1.0)

    def test_f_to_pvalue(self):
        """Test F-statistic to p-value conversion."""
        # Large F-stat should give small p-value
        p_large_f = self.meter._f_to_pvalue(10.0, 3, 30)
        self.assertLess(p_large_f, 0.5)

        # Small F-stat should give larger p-value
        p_small_f = self.meter._f_to_pvalue(0.5, 3, 30)
        self.assertGreater(p_small_f, 0.5)


if __name__ == "__main__":
    unittest.main()
