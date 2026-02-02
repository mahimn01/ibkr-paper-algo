"""
Event system for algorithm-dashboard communication.

This module provides a pub/sub event system that allows any trading algorithm
to communicate with the dashboard in a decoupled way.

Usage:
    # In your algorithm
    from trading_algo.dashboard.events import EventBus, SignalEvent

    bus = EventBus()
    bus.emit(SignalEvent(signal=my_signal))

    # In the dashboard
    bus.subscribe(SignalEvent, self.handle_signal)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from weakref import WeakSet
import threading
import queue

from .models import (
    Signal,
    Position,
    Trade,
    PnLSummary,
    MarketData,
    AlgorithmStatus,
    TradeDirection,
    TradeStatus,
)


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BaseEvent(ABC):
    """Base class for all dashboard events."""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source: str = "unknown"

    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the type name of this event."""
        pass


@dataclass
class SignalEvent(BaseEvent):
    """Emitted when the algorithm generates a new signal."""
    signal: Optional[Signal] = None

    @property
    def event_type(self) -> str:
        return "signal"


@dataclass
class PositionOpenedEvent(BaseEvent):
    """Emitted when a new position is opened."""
    position: Optional[Position] = None

    @property
    def event_type(self) -> str:
        return "position_opened"


@dataclass
class PositionUpdatedEvent(BaseEvent):
    """Emitted when a position is updated (price change, stop adjustment, etc.)."""
    position: Optional[Position] = None
    update_type: str = "price"  # "price", "stop", "target", "trailing"

    @property
    def event_type(self) -> str:
        return "position_updated"


@dataclass
class PositionClosedEvent(BaseEvent):
    """Emitted when a position is closed."""
    trade: Optional[Trade] = None  # The completed trade
    position: Optional[Position] = None  # The position that was closed

    @property
    def event_type(self) -> str:
        return "position_closed"


@dataclass
class TradeExecutedEvent(BaseEvent):
    """Emitted when a trade order is executed."""
    trade: Optional[Trade] = None
    order_id: str = ""
    fill_price: float = 0.0
    fill_quantity: int = 0

    @property
    def event_type(self) -> str:
        return "trade_executed"


@dataclass
class PnLUpdateEvent(BaseEvent):
    """Emitted when P&L changes (on price updates or trade completions)."""
    summary: Optional[PnLSummary] = None
    realized_change: float = 0.0
    unrealized_change: float = 0.0

    @property
    def event_type(self) -> str:
        return "pnl_update"


@dataclass
class MarketDataEvent(BaseEvent):
    """Emitted on market data updates."""
    data: Optional[MarketData] = None
    symbol: str = ""

    @property
    def event_type(self) -> str:
        return "market_data"


@dataclass
class AlgorithmStatusEvent(BaseEvent):
    """Emitted when algorithm status changes."""
    status: Optional[AlgorithmStatus] = None
    previous_status: Optional[str] = None

    @property
    def event_type(self) -> str:
        return "algorithm_status"


@dataclass
class ErrorEvent(BaseEvent):
    """Emitted when an error occurs."""
    error_type: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    priority: EventPriority = EventPriority.HIGH

    @property
    def event_type(self) -> str:
        return "error"


@dataclass
class LogEvent(BaseEvent):
    """Emitted for general log messages."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        return "log"


@dataclass
class HeartbeatEvent(BaseEvent):
    """Emitted periodically to show algorithm is alive."""
    uptime_seconds: int = 0
    signals_count: int = 0
    trades_count: int = 0

    @property
    def event_type(self) -> str:
        return "heartbeat"


# Type variable for event handlers
E = TypeVar('E', bound=BaseEvent)


class EventBus:
    """
    Central event bus for algorithm-dashboard communication.

    Thread-safe and supports both sync and async handlers.
    """

    def __init__(self):
        self._subscribers: Dict[Type[BaseEvent], List[Callable]] = {}
        self._async_subscribers: Dict[Type[BaseEvent], List[Callable]] = {}
        self._lock = threading.RLock()
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None

        # Event history for replay
        self._history: List[BaseEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: Type[E], handler: Callable[[E], None]) -> None:
        """Subscribe to events of a specific type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def subscribe_async(self, event_type: Type[E], handler: Callable[[E], Any]) -> None:
        """Subscribe an async handler to events of a specific type."""
        with self._lock:
            if event_type not in self._async_subscribers:
                self._async_subscribers[event_type] = []
            self._async_subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[E], handler: Callable[[E], None]) -> None:
        """Unsubscribe from events."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                except ValueError:
                    pass
            if event_type in self._async_subscribers:
                try:
                    self._async_subscribers[event_type].remove(handler)
                except ValueError:
                    pass

    def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to all subscribers.

        This is thread-safe and can be called from any thread.
        """
        # Add to history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Put in queue for async processing
        self._event_queue.put(event)

        # Also call sync handlers immediately
        self._dispatch_sync(event)

    def emit_sync(self, event: BaseEvent) -> None:
        """Emit an event synchronously (blocks until all handlers complete)."""
        self._dispatch_sync(event)

    def _dispatch_sync(self, event: BaseEvent) -> None:
        """Dispatch event to synchronous handlers."""
        with self._lock:
            handlers = list(self._subscribers.get(type(event), []))
            # Also dispatch to handlers subscribed to BaseEvent (catch-all)
            handlers.extend(self._subscribers.get(BaseEvent, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Don't let handler errors break the event bus
                print(f"Error in event handler: {e}")

    async def _dispatch_async(self, event: BaseEvent) -> None:
        """Dispatch event to async handlers."""
        with self._lock:
            handlers = list(self._async_subscribers.get(type(event), []))
            handlers.extend(self._async_subscribers.get(BaseEvent, []))

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in async event handler: {e}")

    def get_history(self, event_type: Optional[Type[E]] = None,
                    limit: int = 100) -> List[BaseEvent]:
        """Get recent event history, optionally filtered by type."""
        with self._lock:
            if event_type:
                events = [e for e in self._history if isinstance(e, event_type)]
            else:
                events = list(self._history)
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _global_bus
    _global_bus = None
