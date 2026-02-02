"""
Central data store for the trading dashboard.

The store manages all dashboard state and provides reactive updates to widgets.
It subscribes to the event bus and updates its state accordingly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Deque, Dict, List, Optional, Set
import threading

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
from .events import (
    EventBus,
    BaseEvent,
    SignalEvent,
    PositionOpenedEvent,
    PositionUpdatedEvent,
    PositionClosedEvent,
    TradeExecutedEvent,
    PnLUpdateEvent,
    MarketDataEvent,
    AlgorithmStatusEvent,
    ErrorEvent,
    LogEvent,
    HeartbeatEvent,
    get_event_bus,
)


@dataclass
class LogEntry:
    """A log entry for display."""
    timestamp: datetime
    level: str
    message: str
    source: str = ""


class DashboardStore:
    """
    Central state store for the dashboard.

    This class:
    1. Subscribes to the event bus
    2. Maintains current state (positions, trades, P&L, etc.)
    3. Notifies listeners when state changes
    4. Provides computed values (today's P&L, win rate, etc.)
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._bus = event_bus or get_event_bus()
        self._lock = threading.RLock()

        # State
        self._positions: Dict[str, Position] = {}  # symbol -> Position
        self._trades: List[Trade] = []
        self._signals: Deque[Signal] = deque(maxlen=200)
        self._market_data: Dict[str, MarketData] = {}  # symbol -> MarketData
        self._algorithm_status: Optional[AlgorithmStatus] = None
        self._logs: Deque[LogEntry] = deque(maxlen=500)
        self._errors: Deque[LogEntry] = deque(maxlen=100)

        # P&L tracking
        self._daily_pnl = PnLSummary(
            period="today",
            start_time=datetime.now().replace(hour=0, minute=0, second=0),
            end_time=datetime.now(),
        )

        # Listeners for reactive updates
        self._listeners: List[Callable[[], None]] = []

        # Subscribe to events
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to all relevant events."""
        self._bus.subscribe(SignalEvent, self._handle_signal)
        self._bus.subscribe(PositionOpenedEvent, self._handle_position_opened)
        self._bus.subscribe(PositionUpdatedEvent, self._handle_position_updated)
        self._bus.subscribe(PositionClosedEvent, self._handle_position_closed)
        self._bus.subscribe(TradeExecutedEvent, self._handle_trade_executed)
        self._bus.subscribe(PnLUpdateEvent, self._handle_pnl_update)
        self._bus.subscribe(MarketDataEvent, self._handle_market_data)
        self._bus.subscribe(AlgorithmStatusEvent, self._handle_algorithm_status)
        self._bus.subscribe(ErrorEvent, self._handle_error)
        self._bus.subscribe(LogEvent, self._handle_log)
        self._bus.subscribe(HeartbeatEvent, self._handle_heartbeat)

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Add a listener to be notified on state changes."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        """Remove a listener."""
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify_listeners(self) -> None:
        """Notify all listeners of a state change."""
        for listener in self._listeners:
            try:
                listener()
            except Exception as e:
                pass  # Don't let listener errors break the store

    # Event handlers
    def _handle_signal(self, event: SignalEvent) -> None:
        if event.signal:
            with self._lock:
                self._signals.append(event.signal)
            self._add_log("INFO", f"Signal: {event.signal.signal_type.value} {event.signal.symbol} @ ${event.signal.price:.2f}", "Signal")
            self._notify_listeners()

    def _handle_position_opened(self, event: PositionOpenedEvent) -> None:
        if event.position:
            with self._lock:
                self._positions[event.position.symbol] = event.position
            self._add_log("INFO", f"Opened {event.position.direction.value} {event.position.symbol} @ ${event.position.entry_price:.2f}", "Position")
            self._notify_listeners()

    def _handle_position_updated(self, event: PositionUpdatedEvent) -> None:
        if event.position:
            with self._lock:
                self._positions[event.position.symbol] = event.position
            self._notify_listeners()

    def _handle_position_closed(self, event: PositionClosedEvent) -> None:
        if event.trade:
            with self._lock:
                # Remove from positions
                if event.position and event.position.symbol in self._positions:
                    del self._positions[event.position.symbol]
                # Add to trades
                self._trades.append(event.trade)
                # Update daily P&L
                self._update_daily_pnl()
            pnl_str = f"+${event.trade.realized_pnl:.2f}" if event.trade.realized_pnl >= 0 else f"-${abs(event.trade.realized_pnl):.2f}"
            self._add_log("INFO", f"Closed {event.trade.symbol}: {pnl_str} ({event.trade.exit_reason})", "Trade")
            self._notify_listeners()

    def _handle_trade_executed(self, event: TradeExecutedEvent) -> None:
        # This is for immediate order fill notification
        self._notify_listeners()

    def _handle_pnl_update(self, event: PnLUpdateEvent) -> None:
        if event.summary:
            with self._lock:
                self._daily_pnl = event.summary
            self._notify_listeners()

    def _handle_market_data(self, event: MarketDataEvent) -> None:
        if event.data:
            with self._lock:
                self._market_data[event.symbol] = event.data
                # Update position prices
                if event.symbol in self._positions:
                    self._positions[event.symbol].update_price(
                        event.data.last_price,
                        event.data.last_update or datetime.now()
                    )
            self._notify_listeners()

    def _handle_algorithm_status(self, event: AlgorithmStatusEvent) -> None:
        if event.status:
            with self._lock:
                self._algorithm_status = event.status
            self._notify_listeners()

    def _handle_error(self, event: ErrorEvent) -> None:
        self._add_log("ERROR", f"{event.error_type}: {event.message}", event.source)
        with self._lock:
            self._errors.append(LogEntry(
                timestamp=event.timestamp,
                level="ERROR",
                message=f"{event.error_type}: {event.message}",
                source=event.source,
            ))
        self._notify_listeners()

    def _handle_log(self, event: LogEvent) -> None:
        self._add_log(event.level, event.message, event.source)
        self._notify_listeners()

    def _handle_heartbeat(self, event: HeartbeatEvent) -> None:
        # Just update algorithm status if available
        if self._algorithm_status:
            with self._lock:
                self._algorithm_status.uptime_seconds = event.uptime_seconds
                self._algorithm_status.signals_generated = event.signals_count
                self._algorithm_status.trades_executed = event.trades_count
            self._notify_listeners()

    def _add_log(self, level: str, message: str, source: str = "") -> None:
        with self._lock:
            self._logs.append(LogEntry(
                timestamp=datetime.now(),
                level=level,
                message=message,
                source=source,
            ))

    def _update_daily_pnl(self) -> None:
        """Recalculate daily P&L from today's trades."""
        today = datetime.now().date()
        today_trades = [t for t in self._trades if t.exit_time and t.exit_time.date() == today]
        self._daily_pnl.update_from_trades(today_trades)
        self._daily_pnl.open_positions = len(self._positions)
        self._daily_pnl.unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())

    # Public getters
    @property
    def positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        with self._lock:
            return dict(self._positions)

    @property
    def positions_list(self) -> List[Position]:
        """Get open positions as a list."""
        with self._lock:
            return list(self._positions.values())

    @property
    def trades(self) -> List[Trade]:
        """Get all trades."""
        with self._lock:
            return list(self._trades)

    def get_trades_today(self) -> List[Trade]:
        """Get trades from today."""
        today = datetime.now().date()
        with self._lock:
            return [t for t in self._trades if t.exit_time and t.exit_time.date() == today]

    def get_trades_yesterday(self) -> List[Trade]:
        """Get trades from yesterday."""
        yesterday = (datetime.now() - timedelta(days=1)).date()
        with self._lock:
            return [t for t in self._trades if t.exit_time and t.exit_time.date() == yesterday]

    @property
    def signals(self) -> List[Signal]:
        """Get recent signals."""
        with self._lock:
            return list(self._signals)

    @property
    def market_data(self) -> Dict[str, MarketData]:
        """Get market data."""
        with self._lock:
            return dict(self._market_data)

    @property
    def algorithm_status(self) -> Optional[AlgorithmStatus]:
        """Get algorithm status."""
        with self._lock:
            return self._algorithm_status

    @property
    def logs(self) -> List[LogEntry]:
        """Get recent logs."""
        with self._lock:
            return list(self._logs)

    @property
    def errors(self) -> List[LogEntry]:
        """Get recent errors."""
        with self._lock:
            return list(self._errors)

    @property
    def daily_pnl(self) -> PnLSummary:
        """Get daily P&L summary."""
        with self._lock:
            return self._daily_pnl

    # Computed values
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        with self._lock:
            return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_realized_pnl_today(self) -> float:
        """Total realized P&L for today."""
        return sum(t.realized_pnl for t in self.get_trades_today())

    @property
    def total_pnl_today(self) -> float:
        """Total P&L (realized + unrealized) for today."""
        return self.total_realized_pnl_today + self.total_unrealized_pnl

    @property
    def trade_count_today(self) -> int:
        """Number of trades today."""
        return len(self.get_trades_today())

    @property
    def win_rate_today(self) -> float:
        """Win rate for today's trades."""
        trades = self.get_trades_today()
        if not trades:
            return 0.0
        winners = sum(1 for t in trades if t.realized_pnl > 0)
        return (winners / len(trades)) * 100

    # Manual state updates (for testing/simulation)
    def add_position(self, position: Position) -> None:
        """Manually add a position."""
        with self._lock:
            self._positions[position.symbol] = position
        self._notify_listeners()

    def add_trade(self, trade: Trade) -> None:
        """Manually add a trade."""
        with self._lock:
            self._trades.append(trade)
            self._update_daily_pnl()
        self._notify_listeners()

    def add_signal(self, signal: Signal) -> None:
        """Manually add a signal."""
        with self._lock:
            self._signals.append(signal)
        self._notify_listeners()

    def set_algorithm_status(self, status: AlgorithmStatus) -> None:
        """Set algorithm status."""
        with self._lock:
            self._algorithm_status = status
        self._notify_listeners()

    def update_market_data(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update market data for a symbol."""
        ts = timestamp or datetime.now()
        with self._lock:
            if symbol not in self._market_data:
                self._market_data[symbol] = MarketData(symbol=symbol, last_price=price)
            self._market_data[symbol].update(price, ts)
            # Update position if exists
            if symbol in self._positions:
                self._positions[symbol].update_price(price, ts)
        self._notify_listeners()

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._positions.clear()
            self._trades.clear()
            self._signals.clear()
            self._market_data.clear()
            self._logs.clear()
            self._errors.clear()
            self._algorithm_status = None
        self._notify_listeners()


# Global store instance
_global_store: Optional[DashboardStore] = None


def get_store() -> DashboardStore:
    """Get the global store instance."""
    global _global_store
    if _global_store is None:
        _global_store = DashboardStore()
    return _global_store


def reset_store() -> None:
    """Reset the global store (mainly for testing)."""
    global _global_store
    _global_store = None
