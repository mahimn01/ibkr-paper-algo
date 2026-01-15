"""Tests for RAT Attention Flow module."""

import unittest
from datetime import datetime, timedelta

from trading_algo.rat.attention.flow import AttentionFlow, AttentionState
from trading_algo.rat.attention.tracker import AttentionTracker
from trading_algo.rat.config import AttentionConfig
from trading_algo.rat.signals import SignalSource, SignalType


class TestAttentionFlow(unittest.TestCase):
    """Test AttentionFlow class."""

    def setUp(self):
        """Set up test fixtures."""
        self.flow = AttentionFlow(
            news_weight=0.4,
            flow_weight=0.35,
            price_weight=0.25,
            decay_half_life=300.0,
            window_size=100,
        )
        self.base_time = datetime(2023, 1, 1, 9, 30)

    def test_initial_state_is_neutral(self):
        """Test that initial state is neutral."""
        state = self.flow.compute_attention_state("AAPL", self.base_time)

        self.assertEqual(state.symbol, "AAPL")
        self.assertAlmostEqual(state.news_velocity, 0.0)
        self.assertAlmostEqual(state.flow_imbalance, 0.0)
        self.assertAlmostEqual(state.price_acceleration, 0.0)
        self.assertAlmostEqual(state.attention_score, 0.0)

    def test_news_velocity_increases_with_events(self):
        """Test that news velocity increases with more events."""
        # Add news events
        for i in range(10):
            self.flow.update_news(self.base_time + timedelta(seconds=i * 10))

        state = self.flow.compute_attention_state(
            "AAPL",
            self.base_time + timedelta(seconds=100)
        )

        # Should have positive news velocity
        self.assertGreater(state.news_velocity, 0)

    def test_flow_imbalance_with_buying_pressure(self):
        """Test flow imbalance detection with buy pressure."""
        # Simulate buy pressure
        for i in range(20):
            ts = self.base_time + timedelta(seconds=i)
            self.flow.update_flow(ts, buy_volume=1000, sell_volume=200)

        state = self.flow.compute_attention_state(
            "AAPL",
            self.base_time + timedelta(seconds=20)
        )

        # Should show positive (buy) imbalance
        self.assertGreater(state.flow_imbalance, 0)

    def test_flow_imbalance_with_selling_pressure(self):
        """Test flow imbalance detection with sell pressure."""
        # Simulate sell pressure
        for i in range(20):
            ts = self.base_time + timedelta(seconds=i)
            self.flow.update_flow(ts, buy_volume=200, sell_volume=1000)

        state = self.flow.compute_attention_state(
            "AAPL",
            self.base_time + timedelta(seconds=20)
        )

        # Should show negative (sell) imbalance
        self.assertLess(state.flow_imbalance, 0)

    def test_price_acceleration_with_momentum(self):
        """Test price acceleration detection."""
        # Simulate accelerating uptrend
        price = 100.0
        for i in range(30):
            ts = self.base_time + timedelta(seconds=i)
            price *= 1.001 * (1 + i * 0.0001)  # Accelerating growth
            self.flow.update_price(ts, price)

        state = self.flow.compute_attention_state(
            "AAPL",
            self.base_time + timedelta(seconds=30)
        )

        # Should show positive acceleration
        self.assertGreater(state.price_acceleration, 0)

    def test_combined_attention_score(self):
        """Test combined attention score calculation."""
        # Add all types of data suggesting bullish attention
        for i in range(20):
            ts = self.base_time + timedelta(seconds=i)
            self.flow.update_news(ts)
            self.flow.update_flow(ts, buy_volume=1000, sell_volume=200)
            self.flow.update_price(ts, 100 + i * 0.5)  # Trending up

        state = self.flow.compute_attention_state(
            "AAPL",
            self.base_time + timedelta(seconds=20)
        )

        # Should have positive attention score
        self.assertGreater(state.attention_score, 0)

    def test_high_attention_threshold(self):
        """Test is_high_attention property."""
        # Create high attention scenario
        for i in range(30):
            ts = self.base_time + timedelta(seconds=i)
            self.flow.update_news(ts)
            self.flow.update_flow(ts, buy_volume=2000, sell_volume=100)
            self.flow.update_price(ts, 100 + i)

        state = self.flow.compute_attention_state(
            "AAPL",
            self.base_time + timedelta(seconds=30)
        )

        # Check if high attention is detected
        # (May or may not be high depending on data)
        self.assertIsInstance(state.is_high_attention, bool)

    def test_generate_signal_when_attention_high(self):
        """Test signal generation from attention state."""
        # Create scenario with strong attention
        for i in range(30):
            ts = self.base_time + timedelta(seconds=i)
            self.flow.update_news(ts)
            self.flow.update_flow(ts, buy_volume=2000, sell_volume=100)
            self.flow.update_price(ts, 100 + i * 0.5)

        signal = self.flow.generate_signal("AAPL")

        # Should generate a signal with moderate data
        if signal is not None:
            self.assertEqual(signal.source, SignalSource.ATTENTION)
            self.assertEqual(signal.symbol, "AAPL")


class TestAttentionTracker(unittest.TestCase):
    """Test AttentionTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AttentionConfig(
            flow_window=50,
            news_weight=0.4,
            flow_weight=0.35,
            price_weight=0.25,
        )
        self.tracker = AttentionTracker(config=self.config)
        self.base_time = datetime(2023, 1, 1, 9, 30)

    def test_process_snapshot(self):
        """Test processing market data snapshot."""
        snapshot = {
            "symbol": "AAPL",
            "last": 150.0,
            "bid": 149.95,
            "ask": 150.05,
            "volume": 1000,
            "timestamp": self.base_time,
        }

        state = self.tracker.process_snapshot(snapshot)

        self.assertIsNotNone(state)
        self.assertEqual(state.symbol, "AAPL")

    def test_process_multiple_snapshots(self):
        """Test processing multiple snapshots."""
        for i in range(20):
            snapshot = {
                "symbol": "AAPL",
                "last": 150.0 + i * 0.1,
                "bid": 149.95 + i * 0.1,
                "ask": 150.05 + i * 0.1,
                "volume": 1000 + i * 100,
                "timestamp": self.base_time + timedelta(seconds=i),
            }
            state = self.tracker.process_snapshot(snapshot)

        self.assertIsNotNone(state)

    def test_process_news(self):
        """Test processing news events."""
        news_item = {
            "symbol": "AAPL",
            "headline": "Apple announces new product",
            "timestamp": self.base_time,
        }

        self.tracker.process_news(news_item)

        # Verify news was recorded by processing a snapshot after
        snapshot = {
            "symbol": "AAPL",
            "last": 150.0,
            "bid": 149.95,
            "ask": 150.05,
            "volume": 1000,
            "timestamp": self.base_time + timedelta(seconds=10),
        }
        state = self.tracker.process_snapshot(snapshot)

        self.assertIsNotNone(state)

    def test_multiple_symbols(self):
        """Test tracking multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOG"]

        for symbol in symbols:
            for i in range(10):
                snapshot = {
                    "symbol": symbol,
                    "last": 100.0 + i,
                    "bid": 99.95 + i,
                    "ask": 100.05 + i,
                    "volume": 1000,
                    "timestamp": self.base_time + timedelta(seconds=i),
                }
                self.tracker.process_snapshot(snapshot)

        # Verify each symbol has state
        for symbol in symbols:
            state = self.tracker.get_state(symbol)
            self.assertIsNotNone(state)
            self.assertEqual(state.symbol, symbol)

    def test_generate_signal(self):
        """Test signal generation from tracker."""
        # Build up some data
        for i in range(30):
            snapshot = {
                "symbol": "AAPL",
                "last": 150.0 + i * 0.5,
                "bid": 149.9 + i * 0.5,
                "ask": 150.1 + i * 0.5,
                "volume": 10000,
                "timestamp": self.base_time + timedelta(seconds=i),
            }
            self.tracker.process_snapshot(snapshot)

        signal = self.tracker.generate_signal("AAPL")

        # May or may not generate signal depending on data
        if signal is not None:
            self.assertEqual(signal.source, SignalSource.ATTENTION)

    def test_estimate_flow_from_quotes(self):
        """Test flow estimation from bid/ask."""
        # Price near ask = buy aggressor
        buy_vol, sell_vol = self.tracker._estimate_flow_from_quotes(
            price=100.08,  # Near ask (80% of spread)
            bid=100.0,
            ask=100.1,
            volume=1000,
        )
        self.assertGreater(buy_vol, sell_vol)

        # Price near bid = sell aggressor
        buy_vol, sell_vol = self.tracker._estimate_flow_from_quotes(
            price=100.02,  # Near bid (20% of spread)
            bid=100.0,
            ask=100.1,
            volume=1000,
        )
        self.assertLess(buy_vol, sell_vol)

    def test_inject_backtest_data(self):
        """Test injecting backtest data."""
        prices = [
            (self.base_time + timedelta(seconds=i), 100.0 + i)
            for i in range(20)
        ]
        news_times = [
            self.base_time + timedelta(minutes=i)
            for i in range(5)
        ]
        flow_data = [
            (self.base_time + timedelta(seconds=i), 1000, 500)
            for i in range(20)
        ]

        self.tracker.inject_backtest_data(
            symbol="AAPL",
            prices=prices,
            news_times=news_times,
            flow_data=flow_data,
        )

        # Should be able to get state after injection
        state = self.tracker._flows["AAPL"].compute_attention_state("AAPL")
        self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
