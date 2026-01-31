"""
RAT Signals: Signal dataclasses and types for inter-module communication.

All signals are immutable dataclasses for thread safety.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional


class SignalType(Enum):
    """Types of trading signals."""

    LONG = auto()       # Go long / buy
    SHORT = auto()      # Go short / sell
    FLAT = auto()       # Close position
    ENTER = auto()      # Enter position
    EXIT = auto()       # Exit position
    SCALE_IN = auto()   # Add to position
    SCALE_OUT = auto()  # Reduce position
    HOLD = auto()       # Maintain current position


class SignalSource(Enum):
    """Source module of a signal."""

    ATTENTION = auto()      # Attention Flow module
    REFLEXIVITY = auto()    # Reflexivity Meter module
    TOPOLOGY = auto()       # Topology Detector module
    ADVERSARIAL = auto()    # Adversarial Meta-Trader module
    ALPHA = auto()          # Self-Cannibalizing Alpha module
    COMBINED = auto()       # Combined signal from multiple sources


@dataclass(frozen=True)
class Signal:
    """
    Trading signal from a RAT module.

    Immutable for thread safety and reproducibility.
    """

    source: SignalSource
    signal_type: SignalType
    symbol: str
    direction: float        # -1 to 1 (negative = short, positive = long)
    confidence: float       # 0 to 1
    urgency: float          # 0 to 1 (how quickly to act)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate bounds
        if not -1 <= self.direction <= 1:
            object.__setattr__(self, 'direction', max(-1, min(1, self.direction)))
        if not 0 <= self.confidence <= 1:
            object.__setattr__(self, 'confidence', max(0, min(1, self.confidence)))
        if not 0 <= self.urgency <= 1:
            object.__setattr__(self, 'urgency', max(0, min(1, self.urgency)))

    @property
    def is_actionable(self) -> bool:
        """Check if signal warrants action."""
        return abs(self.direction) > 0.1 and self.confidence > 0.3

    @property
    def strength(self) -> float:
        """Combined signal strength."""
        return abs(self.direction) * self.confidence


@dataclass
class CombinedSignal:
    """Signal combined from multiple sources."""

    signals: List[Signal]
    weights: Dict[SignalSource, float]
    final_direction: float
    final_confidence: float
    final_urgency: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def source_count(self) -> int:
        return len(self.signals)

    @property
    def agreement(self) -> float:
        """How much sources agree on direction."""
        if not self.signals:
            return 0.0
        positive = sum(1 for s in self.signals if s.direction > 0)
        return max(positive, len(self.signals) - positive) / len(self.signals)


class SignalBuffer:
    """Thread-safe buffer for recent signals."""

    def __init__(self, max_size: int = 1000):
        self._buffer: Deque[Signal] = deque(maxlen=max_size)
        self._by_symbol: Dict[str, Deque[Signal]] = {}
        self._by_source: Dict[SignalSource, Deque[Signal]] = {}

    def add(self, signal: Signal) -> None:
        """Add a signal to the buffer."""
        self._buffer.append(signal)

        # Index by symbol
        if signal.symbol not in self._by_symbol:
            self._by_symbol[signal.symbol] = deque(maxlen=100)
        self._by_symbol[signal.symbol].append(signal)

        # Index by source
        if signal.source not in self._by_source:
            self._by_source[signal.source] = deque(maxlen=100)
        self._by_source[signal.source].append(signal)

    def get_recent(self, n: int = 10) -> List[Signal]:
        """Get n most recent signals."""
        return list(self._buffer)[-n:]

    def get_by_symbol(self, symbol: str, n: int = 10) -> List[Signal]:
        """Get recent signals for a symbol."""
        if symbol not in self._by_symbol:
            return []
        return list(self._by_symbol[symbol])[-n:]

    def get_by_source(self, source: SignalSource, n: int = 10) -> List[Signal]:
        """Get recent signals from a source."""
        if source not in self._by_source:
            return []
        return list(self._by_source[source])[-n:]

    def clear(self) -> None:
        """Clear all signals."""
        self._buffer.clear()
        self._by_symbol.clear()
        self._by_source.clear()
