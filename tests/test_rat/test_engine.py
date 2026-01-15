"""Tests for RAT Engine module."""

import unittest
from datetime import datetime, timedelta

from trading_algo.rat.config import RATConfig
from trading_algo.rat.engine import RATEngine, RATState, Position
from trading_algo.rat.signals import SignalSource


class TestRATEngine(unittest.TestCase):
    """Test RATEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RATConfig.from_env()
        self.engine = RATEngine(
            config=self.config,
            broker=None,
            llm_client=None,
        )
        self.base_time = datetime(2023, 1, 1, 10, 0)

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        self.assertIsNotNone(self.engine._attention)
        self.assertIsNotNone(self.engine._reflexivity)
        self.assertIsNotNone(self.engine._topology)
        self.assertIsNotNone(self.engine._adversarial)
        self.assertIsNotNone(self.engine._alpha_tracker)
        self.assertIsNotNone(self.engine._combiner)
        self.assertIsNotNone(self.engine._filter)

    def test_start_stop(self):
        """Test engine start and stop."""
        self.engine.start()
        self.assertTrue(self.engine._is_running)

        self.engine.stop()
        self.assertFalse(self.engine._is_running)

    def test_process_tick(self):
        """Test processing a single tick."""
        state = self.engine.process_tick(
            symbol="AAPL",
            price=150.0,
            volume=10000,
            bid=149.95,
            ask=150.05,
            timestamp=self.base_time,
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.symbol, "AAPL")
        self.assertIsInstance(state.attention_score, float)

    def test_process_multiple_ticks(self):
        """Test processing multiple ticks builds state."""
        for i in range(50):
            state = self.engine.process_tick(
                symbol="AAPL",
                price=150.0 + i * 0.1,
                volume=10000,
                bid=149.95 + i * 0.1,
                ask=150.05 + i * 0.1,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        self.assertIsNotNone(state)
        # After many ticks, should have meaningful state
        self.assertIsInstance(state.topology_regime.name, str)

    def test_signals_generation(self):
        """Test that signals are generated from modules."""
        # Process enough data to potentially generate signals
        for i in range(100):
            self.engine.process_tick(
                symbol="AAPL",
                price=150.0 + i * 0.05,
                volume=10000 + i * 100,
                bid=149.9 + i * 0.05,
                ask=150.1 + i * 0.05,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        state = self.engine.get_last_state("AAPL")
        self.assertIsNotNone(state)
        self.assertIn(SignalSource.ATTENTION, state.signals)

    def test_multiple_symbols(self):
        """Test processing multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOG"]

        for i in range(30):
            for symbol in symbols:
                self.engine.process_tick(
                    symbol=symbol,
                    price=100.0 + i,
                    volume=10000,
                    bid=99.95 + i,
                    ask=100.05 + i,
                    timestamp=self.base_time + timedelta(seconds=i),
                )

        # Each symbol should have its own state
        for symbol in symbols:
            state = self.engine.get_last_state(symbol)
            self.assertIsNotNone(state)
            self.assertEqual(state.symbol, symbol)

    def test_update_pnl(self):
        """Test P&L update."""
        initial_pnl = self.engine._total_pnl

        self.engine.update_pnl("AAPL", 100.0)

        self.assertEqual(self.engine._total_pnl, initial_pnl + 100.0)

    def test_get_stats(self):
        """Test getting engine statistics."""
        # Process some data
        for i in range(10):
            self.engine.process_tick(
                symbol="AAPL",
                price=150.0 + i,
                volume=10000,
                bid=149.95 + i,
                ask=150.05 + i,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        stats = self.engine.get_stats()

        self.assertIn("total_pnl", stats)
        self.assertIn("trade_count", stats)
        self.assertIn("positions", stats)
        self.assertIn("alpha_health", stats)

    def test_check_alpha_health(self):
        """Test alpha health checking."""
        # Should return True initially (factors are fresh)
        healthy = self.engine.check_alpha_health()
        self.assertTrue(healthy)

    def test_inject_backtest_tick(self):
        """Test backtest tick injection."""
        state = self.engine.inject_backtest_tick(
            symbol="AAPL",
            timestamp=self.base_time,
            open_price=149.0,
            high=151.0,
            low=148.5,
            close=150.0,
            volume=100000,
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.symbol, "AAPL")

    def test_inject_news_event(self):
        """Test news event injection."""
        self.engine.inject_news_event(
            symbol="AAPL",
            timestamp=self.base_time,
            headline="Apple reports strong earnings",
            sentiment=0.8,
        )

        # News should be recorded in attention module
        # Process a tick to see effect
        state = self.engine.process_tick(
            symbol="AAPL",
            price=150.0,
            volume=10000,
            bid=149.95,
            ask=150.05,
            timestamp=self.base_time + timedelta(seconds=10),
        )

        self.assertIsNotNone(state)

    def test_reset_for_backtest(self):
        """Test resetting engine for backtest."""
        # Process some data
        for i in range(10):
            self.engine.process_tick(
                symbol="AAPL",
                price=150.0 + i,
                volume=10000,
                bid=149.95 + i,
                ask=150.05 + i,
                timestamp=self.base_time + timedelta(seconds=i),
            )

        self.engine.update_pnl("AAPL", 100)

        # Reset
        self.engine.reset_for_backtest()

        self.assertEqual(self.engine._total_pnl, 0.0)
        self.assertEqual(self.engine._trade_count, 0)
        self.assertEqual(len(self.engine._positions), 0)


class TestRATState(unittest.TestCase):
    """Test RATState dataclass."""

    def test_rat_state_creation(self):
        """Test creating RATState."""
        from trading_algo.rat.reflexivity.meter import ReflexivityStage
        from trading_algo.rat.topology.detector import TopologyRegime

        state = RATState(
            timestamp=datetime.now(),
            symbol="AAPL",
            attention_score=0.5,
            reflexivity_stage=ReflexivityStage.EFFICIENT,
            topology_regime=TopologyRegime.TRENDING,
            adversarial_archetype=None,
            alpha_health=0.8,
            signals={},
            decision=None,
            current_position=0.0,
            suggested_action="hold",
            suggested_size=0.0,
        )

        self.assertEqual(state.symbol, "AAPL")
        self.assertAlmostEqual(state.attention_score, 0.5)


class TestPosition(unittest.TestCase):
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test creating Position."""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
        )

        self.assertEqual(pos.symbol, "AAPL")
        self.assertEqual(pos.quantity, 100)
        self.assertAlmostEqual(pos.entry_price, 150.0)
        self.assertAlmostEqual(pos.unrealized_pnl, 0.0)


class TestEngineBacktest(unittest.TestCase):
    """Test engine in backtest mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RATConfig.from_env()
        self.engine = RATEngine(config=self.config)
        self.engine.reset_for_backtest()
        self.base_time = datetime(2023, 1, 1, 9, 35)

    def test_backtest_simulation(self):
        """Test running a backtest simulation."""
        # Simulate 100 bars
        price = 100.0
        states = []

        for i in range(100):
            price *= 1.001 if i % 3 != 0 else 0.999  # Trending with noise

            state = self.engine.inject_backtest_tick(
                symbol="AAPL",
                timestamp=self.base_time + timedelta(minutes=i),
                open_price=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=10000,
            )

            if state:
                states.append(state)

        self.assertEqual(len(states), 100)

        # Check final stats
        stats = self.engine.get_stats()
        self.assertIsNotNone(stats)

    def test_backtest_multiple_symbols(self):
        """Test backtest with multiple symbols."""
        symbols = ["AAPL", "MSFT"]

        for i in range(50):
            for j, symbol in enumerate(symbols):
                price = 100 + i + j * 10

                self.engine.inject_backtest_tick(
                    symbol=symbol,
                    timestamp=self.base_time + timedelta(minutes=i),
                    open_price=price,
                    high=price + 0.5,
                    low=price - 0.5,
                    close=price,
                    volume=10000,
                )

        # Both symbols should have state
        for symbol in symbols:
            state = self.engine.get_last_state(symbol)
            self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
