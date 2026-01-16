"""Tests for RAT signals module."""

import unittest
from datetime import datetime

from trading_algo.rat.signals import (
    Signal,
    SignalType,
    SignalSource,
    CombinedSignal,
    SignalBuffer,
)


class TestSignal(unittest.TestCase):
    """Test Signal dataclass."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.8,
            confidence=0.75,
            urgency=0.6,
        )

        self.assertEqual(signal.source, SignalSource.ATTENTION)
        self.assertEqual(signal.signal_type, SignalType.LONG)
        self.assertEqual(signal.symbol, "AAPL")
        self.assertAlmostEqual(signal.direction, 0.8)
        self.assertAlmostEqual(signal.confidence, 0.75)
        self.assertAlmostEqual(signal.urgency, 0.6)

    def test_signal_bounds_clamping(self):
        """Test that signal values are clamped to valid bounds."""
        signal = Signal(
            source=SignalSource.TOPOLOGY,
            signal_type=SignalType.SHORT,
            symbol="MSFT",
            direction=-1.5,  # Should be clamped to -1
            confidence=1.5,  # Should be clamped to 1
            urgency=-0.5,    # Should be clamped to 0
        )

        self.assertEqual(signal.direction, -1.0)
        self.assertEqual(signal.confidence, 1.0)
        self.assertEqual(signal.urgency, 0.0)

    def test_signal_is_actionable(self):
        """Test is_actionable property."""
        actionable = Signal(
            source=SignalSource.REFLEXIVITY,
            signal_type=SignalType.LONG,
            symbol="GOOG",
            direction=0.5,
            confidence=0.5,
            urgency=0.5,
        )
        self.assertTrue(actionable.is_actionable)

        not_actionable_low_direction = Signal(
            source=SignalSource.REFLEXIVITY,
            signal_type=SignalType.HOLD,
            symbol="GOOG",
            direction=0.05,  # Too low
            confidence=0.5,
            urgency=0.5,
        )
        self.assertFalse(not_actionable_low_direction.is_actionable)

        not_actionable_low_confidence = Signal(
            source=SignalSource.REFLEXIVITY,
            signal_type=SignalType.LONG,
            symbol="GOOG",
            direction=0.5,
            confidence=0.2,  # Too low
            urgency=0.5,
        )
        self.assertFalse(not_actionable_low_confidence.is_actionable)

    def test_signal_strength(self):
        """Test strength property."""
        signal = Signal(
            source=SignalSource.ADVERSARIAL,
            signal_type=SignalType.LONG,
            symbol="AMZN",
            direction=0.8,
            confidence=0.5,
            urgency=0.5,
        )
        self.assertAlmostEqual(signal.strength, 0.4)  # |0.8| * 0.5

    def test_signal_with_metadata(self):
        """Test signal with metadata."""
        signal = Signal(
            source=SignalSource.ALPHA,
            signal_type=SignalType.LONG,
            symbol="TSLA",
            direction=0.6,
            confidence=0.7,
            urgency=0.5,
            metadata={"factor": "momentum", "sharpe": 1.5},
        )

        self.assertEqual(signal.metadata["factor"], "momentum")
        self.assertEqual(signal.metadata["sharpe"], 1.5)


class TestSignalBuffer(unittest.TestCase):
    """Test SignalBuffer class."""

    def test_buffer_add_and_retrieve(self):
        """Test adding and retrieving signals."""
        buffer = SignalBuffer(max_size=100)

        signal1 = Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.5,
            confidence=0.6,
            urgency=0.5,
        )
        signal2 = Signal(
            source=SignalSource.TOPOLOGY,
            signal_type=SignalType.SHORT,
            symbol="MSFT",
            direction=-0.7,
            confidence=0.8,
            urgency=0.6,
        )

        buffer.add(signal1)
        buffer.add(signal2)

        recent = buffer.get_recent(10)
        self.assertEqual(len(recent), 2)

    def test_buffer_by_symbol(self):
        """Test retrieving signals by symbol."""
        buffer = SignalBuffer()

        for i in range(5):
            buffer.add(Signal(
                source=SignalSource.ATTENTION,
                signal_type=SignalType.LONG,
                symbol="AAPL",
                direction=0.5,
                confidence=0.5,
                urgency=0.5,
            ))
            buffer.add(Signal(
                source=SignalSource.ATTENTION,
                signal_type=SignalType.SHORT,
                symbol="MSFT",
                direction=-0.5,
                confidence=0.5,
                urgency=0.5,
            ))

        aapl_signals = buffer.get_by_symbol("AAPL")
        msft_signals = buffer.get_by_symbol("MSFT")

        self.assertEqual(len(aapl_signals), 5)
        self.assertEqual(len(msft_signals), 5)

    def test_buffer_by_source(self):
        """Test retrieving signals by source."""
        buffer = SignalBuffer()

        buffer.add(Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.5,
            confidence=0.5,
            urgency=0.5,
        ))
        buffer.add(Signal(
            source=SignalSource.TOPOLOGY,
            signal_type=SignalType.SHORT,
            symbol="AAPL",
            direction=-0.5,
            confidence=0.5,
            urgency=0.5,
        ))
        buffer.add(Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="MSFT",
            direction=0.6,
            confidence=0.5,
            urgency=0.5,
        ))

        attention_signals = buffer.get_by_source(SignalSource.ATTENTION)
        topology_signals = buffer.get_by_source(SignalSource.TOPOLOGY)

        self.assertEqual(len(attention_signals), 2)
        self.assertEqual(len(topology_signals), 1)

    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = SignalBuffer()

        buffer.add(Signal(
            source=SignalSource.ATTENTION,
            signal_type=SignalType.LONG,
            symbol="AAPL",
            direction=0.5,
            confidence=0.5,
            urgency=0.5,
        ))

        self.assertEqual(len(buffer.get_recent()), 1)

        buffer.clear()
        self.assertEqual(len(buffer.get_recent()), 0)


class TestCombinedSignal(unittest.TestCase):
    """Test CombinedSignal dataclass."""

    def test_combined_signal_creation(self):
        """Test creating combined signal."""
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
                confidence=0.6,
                urgency=0.4,
            ),
        ]

        combined = CombinedSignal(
            signals=signals,
            weights={SignalSource.ATTENTION: 0.5, SignalSource.TOPOLOGY: 0.5},
            final_direction=0.7,
            final_confidence=0.65,
            final_urgency=0.45,
        )

        self.assertEqual(combined.source_count, 2)
        self.assertAlmostEqual(combined.final_direction, 0.7)

    def test_combined_signal_agreement(self):
        """Test agreement calculation."""
        # All agree
        signals_agree = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.5, 0.5, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.LONG, "AAPL", 0.6, 0.5, 0.5),
            Signal(SignalSource.REFLEXIVITY, SignalType.LONG, "AAPL", 0.7, 0.5, 0.5),
        ]

        combined_agree = CombinedSignal(
            signals=signals_agree,
            weights={},
            final_direction=0.6,
            final_confidence=0.5,
            final_urgency=0.5,
        )
        self.assertAlmostEqual(combined_agree.agreement, 1.0)

        # 2/3 agree
        signals_partial = [
            Signal(SignalSource.ATTENTION, SignalType.LONG, "AAPL", 0.5, 0.5, 0.5),
            Signal(SignalSource.TOPOLOGY, SignalType.LONG, "AAPL", 0.6, 0.5, 0.5),
            Signal(SignalSource.REFLEXIVITY, SignalType.SHORT, "AAPL", -0.3, 0.5, 0.5),
        ]

        combined_partial = CombinedSignal(
            signals=signals_partial,
            weights={},
            final_direction=0.3,
            final_confidence=0.5,
            final_urgency=0.5,
        )
        self.assertAlmostEqual(combined_partial.agreement, 2/3)


if __name__ == "__main__":
    unittest.main()
