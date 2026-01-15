"""Tests for RAT Signal Combiner and Filters modules."""

import unittest
from datetime import datetime, timedelta, time as dt_time

from trading_algo.rat.combiner.combiner import (
    SignalCombiner,
    CombinedDecision,
    WeightingMethod,
    SourcePerformance,
)
from trading_algo.rat.combiner.filters import (
    SignalFilter,
    FilterType,
    FilterResult,
)
from trading_algo.rat.signals import Signal, SignalType, SignalSource


class TestSignalCombiner(unittest.TestCase):
    """Test SignalCombiner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.combiner = SignalCombiner(
            weighting_method=WeightingMethod.EQUAL,
            min_signals_required=2,
            agreement_threshold=0.6,
            max_position_pct=0.25,
        )

    def test_combine_insufficient_signals(self):
        """Test combining with too few signals."""
        signals = [
            Signal(
                source=SignalSource.ATTENTION,
                signal_type=SignalType.LONG,
                symbol="AAPL",
                direction=0.5,
                confidence=0.7,
                urgency=0.5,
            ),
        ]

        decision = self.combiner.combine(signals)

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.position_size_pct, 0.0)

    def test_combine_agreeing_signals(self):
        """Test combining signals that agree."""
        signals = [
            Signal(
                source=SignalSource.ATTENTION,
                signal_type=SignalType.LONG,
                symbol="AAPL",
                direction=0.6,
                confidence=0.7,
                urgency=0.5,
            ),
            Signal(
                source=SignalSource.TOPOLOGY,
                signal_type=SignalType.LONG,
                symbol="AAPL",
                direction=0.8,
                confidence=0.8,
                urgency=0.6,
            ),
        ]

        decision = self.combiner.combine(signals)

        self.assertEqual(decision.symbol, "AAPL")
        self.assertEqual(decision.action, "buy")
        self.assertGreater(decision.direction, 0)
        self.assertGreater(decision.confidence, 0)

    def test_combine_disagreeing_signals(self):
        """Test combining signals that disagree."""
        signals = [
            Signal(
                source=SignalSource.ATTENTION,
                signal_type=SignalType.LONG,
                symbol="AAPL",
                direction=0.6,
                confidence=0.7,
                urgency=0.5,
            ),
            Signal(
                source=SignalSource.TOPOLOGY,
                signal_type=SignalType.SHORT,
                symbol="AAPL",
                direction=-0.5,
                confidence=0.6,
                urgency=0.4,
            ),
        ]

        decision = self.combiner.combine(signals)

        # With disagreement, confidence should be reduced
        self.assertLess(decision.confidence, 0.7)

    def test_equal_weighting(self):
        """Test equal weighting method."""
        combiner = SignalCombiner(weighting_method=WeightingMethod.EQUAL)

        signals = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.6, 0.7, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.LONG, "AAPL", 0.4, 0.8, 0.6),
        ]

        weights = combiner._calculate_weights(
            signals, None
        )

        # Equal weights
        self.assertAlmostEqual(weights[SignalSource.ATTENTION], 0.5)
        self.assertAlmostEqual(weights[SignalSource.TOPOLOGY], 0.5)

    def test_sharpe_weighting(self):
        """Test Sharpe-based weighting."""
        combiner = SignalCombiner(weighting_method=WeightingMethod.SHARPE_WEIGHTED)

        # Add some performance history
        for i in range(20):
            combiner.update_performance(
                SignalSource.ATTENTION,
                prediction=0.5,
                actual=0.45,
                pnl=0.02,  # Good performance
            )
            combiner.update_performance(
                SignalSource.TOPOLOGY,
                prediction=0.5,
                actual=0.4,
                pnl=-0.01,  # Poor performance
            )

        signals = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.6, 0.7, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.LONG, "AAPL", 0.4, 0.8, 0.6),
        ]

        weights = combiner._calculate_weights(signals, None)

        # Attention should have higher weight due to better Sharpe
        self.assertGreater(
            weights.get(SignalSource.ATTENTION, 0),
            weights.get(SignalSource.TOPOLOGY, 0),
        )

    def test_should_trade_method(self):
        """Test should_trade method on CombinedDecision."""
        decision_trade = CombinedDecision(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="buy",
            signal_type=SignalType.LONG,
            direction=0.6,
            confidence=0.7,
            urgency=0.5,
            position_size_pct=0.1,
            contributing_sources=[SignalSource.ATTENTION],
            weights_used={},
            raw_signals={},
        )
        self.assertTrue(decision_trade.should_trade(0.5))
        self.assertFalse(decision_trade.should_trade(0.8))

        decision_hold = CombinedDecision(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="hold",
            signal_type=SignalType.HOLD,
            direction=0.0,
            confidence=0.0,
            urgency=0.0,
            position_size_pct=0.0,
            contributing_sources=[],
            weights_used={},
            raw_signals={},
        )
        self.assertFalse(decision_hold.should_trade())

    def test_update_performance(self):
        """Test performance tracking update."""
        self.combiner.update_performance(
            SignalSource.ATTENTION,
            prediction=0.5,
            actual=0.48,
            pnl=0.01,
        )

        stats = self.combiner.get_source_stats()
        self.assertIn("ATTENTION", stats)
        self.assertEqual(stats["ATTENTION"]["n_observations"], 1)

    def test_calculate_agreement(self):
        """Test agreement calculation."""
        signals_agree = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.5, 0.5, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.LONG, "AAPL", 0.6, 0.5, 0.5),
            Signal(SignalSource.REFLEXIVITY, SignalType.LONG, "AAPL", 0.4, 0.5, 0.5),
        ]

        agreement = self.combiner._calculate_agreement(signals_agree)
        self.assertAlmostEqual(agreement, 1.0)

        signals_mixed = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.5, 0.5, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.SHORT, "AAPL", -0.6, 0.5, 0.5),
        ]

        agreement_mixed = self.combiner._calculate_agreement(signals_mixed)
        self.assertAlmostEqual(agreement_mixed, 0.5)


class TestSourcePerformance(unittest.TestCase):
    """Test SourcePerformance dataclass."""

    def test_mean_return(self):
        """Test mean return calculation."""
        perf = SourcePerformance(source=SignalSource.ATTENTION)
        perf.returns.extend([0.01, 0.02, -0.01, 0.015])

        self.assertAlmostEqual(perf.mean_return, 0.00875)

    def test_variance(self):
        """Test variance calculation."""
        perf = SourcePerformance(source=SignalSource.ATTENTION)
        perf.returns.extend([0.01, 0.01, 0.01, 0.01])

        self.assertAlmostEqual(perf.variance, 0.0)

    def test_sharpe(self):
        """Test Sharpe ratio calculation."""
        perf = SourcePerformance(source=SignalSource.ATTENTION)
        perf.returns.extend([0.01] * 20)

        sharpe = perf.sharpe
        self.assertGreater(sharpe, 0)

    def test_win_rate(self):
        """Test win rate calculation."""
        perf = SourcePerformance(source=SignalSource.ATTENTION)
        perf.returns.extend([0.01, -0.01, 0.02, -0.005])

        self.assertAlmostEqual(perf.win_rate, 0.5)


class TestSignalFilter(unittest.TestCase):
    """Test SignalFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = SignalFilter(
            confidence_threshold=0.5,
            urgency_threshold=0.3,
            trading_start=dt_time(9, 30),
            trading_end=dt_time(16, 0),
            max_signals_per_hour=10,
        )
        self.base_time = datetime(2023, 1, 1, 10, 0)  # During trading hours

    def test_confidence_filter_pass(self):
        """Test confidence filter passes high confidence."""
        signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.6,
            confidence=0.7,
            urgency=0.5,
        )

        result = self.filter.filter(signal, self.base_time)
        self.assertTrue(result.passed)

    def test_confidence_filter_fail(self):
        """Test confidence filter rejects low confidence."""
        signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.6,
            confidence=0.3,  # Below threshold
            urgency=0.5,
        )

        result = self.filter.filter(signal, self.base_time)
        self.assertFalse(result.passed)
        self.assertEqual(result.filter_type, FilterType.CONFIDENCE_THRESHOLD)

    def test_time_window_filter_pass(self):
        """Test time window filter during trading hours."""
        signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.6,
            confidence=0.7,
            urgency=0.5,
        )

        # 10:00 AM is during trading hours
        result = self.filter.filter(signal, self.base_time)
        self.assertTrue(result.passed)

    def test_time_window_filter_fail(self):
        """Test time window filter outside trading hours."""
        signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.6,
            confidence=0.7,
            urgency=0.5,
        )

        # 8:00 AM is before market open
        early_time = datetime(2023, 1, 1, 8, 0)
        result = self.filter.filter(signal, early_time)
        self.assertFalse(result.passed)
        self.assertEqual(result.filter_type, FilterType.TIME_WINDOW)

    def test_rate_limit_filter(self):
        """Test rate limiting."""
        base_signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.6,
            confidence=0.7,
            urgency=0.5,
        )

        # Add many signals
        for i in range(15):
            result = self.filter.filter(
                base_signal,
                self.base_time + timedelta(minutes=i),
            )

            if i < 10:
                self.assertTrue(result.passed, f"Signal {i} should pass")
            else:
                # After 10 signals, should hit rate limit
                self.assertFalse(result.passed, f"Signal {i} should be rate limited")

    def test_filter_batch(self):
        """Test batch filtering."""
        signals = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.6, 0.7, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.LONG, "MSFT", 0.5, 0.3, 0.4),  # Low confidence
            Signal(SignalSource.REFLEXIVITY, SignalType.SHORT, "GOOG", 0.7, 0.8, 0.6),
        ]

        passed = self.filter.filter_batch(signals, self.base_time)

        # One signal should be filtered out due to low confidence
        self.assertEqual(len(passed), 2)

    def test_add_remove_filter(self):
        """Test adding and removing filters."""
        self.filter.add_filter(FilterType.DRAWDOWN_GATE)
        self.assertIn(FilterType.DRAWDOWN_GATE, self.filter._active_filters)

        self.filter.remove_filter(FilterType.DRAWDOWN_GATE)
        self.assertNotIn(FilterType.DRAWDOWN_GATE, self.filter._active_filters)

    def test_set_regime(self):
        """Test setting current regime."""
        self.filter.set_regime("TRENDING")
        self.assertEqual(self.filter._current_regime, "TRENDING")

    def test_update_equity(self):
        """Test equity tracking."""
        initial_equity = self.filter._current_equity

        self.filter.update_equity(100)  # Profit
        self.assertGreater(self.filter._current_equity, initial_equity)

        self.filter.update_equity(-50)  # Loss
        self.assertGreater(self.filter._current_equity, initial_equity)

    def test_drawdown_filter(self):
        """Test drawdown filter."""
        self.filter.add_filter(FilterType.DRAWDOWN_GATE)

        # Simulate large drawdown
        self.filter._current_equity = 85000  # 15% loss from 100000
        self.filter._peak_equity = 100000

        signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.6,
            confidence=0.6,  # Not high enough during drawdown
            urgency=0.5,
        )

        result = self.filter.filter(signal, self.base_time)
        # Should fail because in drawdown and confidence < 0.8
        self.assertFalse(result.passed)

    def test_inject_backtest_state(self):
        """Test injecting backtest state."""
        self.filter.inject_backtest_state(
            equity=95000,
            peak_equity=100000,
            regime="CONSOLIDATION",
        )

        self.assertEqual(self.filter._current_equity, 95000)
        self.assertEqual(self.filter._peak_equity, 100000)
        self.assertEqual(self.filter._current_regime, "CONSOLIDATION")


class TestWeightingMethod(unittest.TestCase):
    """Test WeightingMethod enum."""

    def test_all_methods_exist(self):
        """Test all weighting methods exist."""
        methods = [
            WeightingMethod.EQUAL,
            WeightingMethod.INVERSE_VARIANCE,
            WeightingMethod.SHARPE_WEIGHTED,
            WeightingMethod.KELLY_OPTIMAL,
            WeightingMethod.REGIME_CONDITIONAL,
            WeightingMethod.ADAPTIVE,
        ]

        for method in methods:
            self.assertIsInstance(method, WeightingMethod)


if __name__ == "__main__":
    unittest.main()
