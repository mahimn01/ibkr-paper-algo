"""
Strategy adapter protocol for multi-strategy portfolio management.

Every strategy that participates in the multi-strategy controller
must implement the TradingStrategy protocol (or ABC).  Adapters in
the ``adapters/`` subpackage wrap existing strategy implementations
to conform to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class StrategyState(Enum):
    """Lifecycle state of a strategy."""
    WARMING_UP = auto()   # Collecting data, not yet producing signals
    ACTIVE = auto()       # Producing signals normally
    PAUSED = auto()       # Temporarily disabled (e.g., risk throttle)
    DISABLED = auto()     # Permanently off


@dataclass
class StrategySignal:
    """
    Unified signal emitted by any strategy.

    The controller consumes these to build a target portfolio.
    """
    strategy_name: str
    symbol: str
    direction: int              # 1 = long, -1 = short, 0 = flat/exit
    target_weight: float        # Desired portfolio weight for this leg
    confidence: float           # 0-1 signal quality score
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_price: Optional[float] = None
    trade_type: str = ""        # e.g. "momentum", "mean_reversion", "breakout"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.direction != 0

    @property
    def is_exit(self) -> bool:
        return self.direction == 0


class TradingStrategy(ABC):
    """
    Abstract base class for all strategies in the multi-strategy system.

    Subclasses must implement the four abstract methods.
    Adapters wrap existing strategy objects to conform to this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g. 'Orchestrator', 'ORB')."""

    @property
    @abstractmethod
    def state(self) -> StrategyState:
        """Current lifecycle state."""

    @abstractmethod
    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """
        Feed a single bar of data to the strategy.

        Called for every symbol on every bar. The strategy should update
        its internal state (indicators, edge engines, etc.).
        """

    @abstractmethod
    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        """
        Produce zero or more signals for the current bar.

        The controller calls this once per bar after all ``update()``
        calls for that bar are complete.

        Returns:
            List of StrategySignal objects.  An empty list means
            "no opinion" for this bar.
        """

    def get_current_exposure(self) -> float:
        """
        Return the strategy's current gross exposure as a fraction
        of allocated capital (0.0 = flat, 1.0 = fully invested).
        Default implementation returns 0.
        """
        return 0.0

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Return strategy-level performance statistics.
        Default implementation returns an empty dict.
        """
        return {}

    def reset(self) -> None:
        """Reset strategy state for a new session or backtest run."""
