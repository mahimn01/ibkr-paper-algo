"""
Algorithm adapter for the trading dashboard.

This module provides adapters that connect trading algorithms to the dashboard
event system. It translates algorithm-specific events into dashboard events.

Usage:
    from trading_algo.dashboard.adapter import OrchestratorAdapter

    # Create adapter
    adapter = OrchestratorAdapter(orchestrator, broker)

    # The adapter will automatically emit events to the dashboard
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import threading

from .events import (
    EventBus,
    get_event_bus,
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
)
from .models import (
    Signal,
    Position,
    Trade,
    PnLSummary,
    MarketData,
    AlgorithmStatus,
    TradeDirection,
    TradeStatus,
    SignalType,
    SignalStrength,
)


class BaseAlgorithmAdapter:
    """
    Base class for algorithm adapters.

    Subclass this to create adapters for specific algorithms.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        algorithm_name: str = "Unknown Algorithm",
        version: str = "1.0.0",
    ):
        self.bus = event_bus or get_event_bus()
        self.algorithm_name = algorithm_name
        self.version = version

        # Track state
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._signals_count = 0
        self._trades_count = 0
        self._start_time = datetime.now()
        self._is_running = False
        self._is_paused = False
        self._errors = 0

        # Heartbeat
        self._heartbeat_interval = 5.0  # seconds
        self._heartbeat_timer: Optional[threading.Timer] = None

    def start(self) -> None:
        """Start the adapter."""
        self._is_running = True
        self._start_time = datetime.now()
        self._emit_status_update()
        self._start_heartbeat()
        self.log("INFO", "Algorithm started")

    def stop(self) -> None:
        """Stop the adapter."""
        self._is_running = False
        self._stop_heartbeat()
        self._emit_status_update()
        self.log("INFO", "Algorithm stopped")

    def pause(self) -> None:
        """Pause the algorithm."""
        self._is_paused = True
        self._emit_status_update()
        self.log("WARNING", "Algorithm paused")

    def resume(self) -> None:
        """Resume the algorithm."""
        self._is_paused = False
        self._emit_status_update()
        self.log("INFO", "Algorithm resumed")

    def _start_heartbeat(self) -> None:
        """Start periodic heartbeat."""
        def heartbeat():
            if self._is_running:
                uptime = int((datetime.now() - self._start_time).total_seconds())
                self.bus.emit(HeartbeatEvent(
                    uptime_seconds=uptime,
                    signals_count=self._signals_count,
                    trades_count=self._trades_count,
                    source=self.algorithm_name,
                ))
                self._heartbeat_timer = threading.Timer(self._heartbeat_interval, heartbeat)
                self._heartbeat_timer.daemon = True
                self._heartbeat_timer.start()

        heartbeat()

    def _stop_heartbeat(self) -> None:
        """Stop heartbeat timer."""
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None

    def _emit_status_update(self) -> None:
        """Emit algorithm status update."""
        uptime = int((datetime.now() - self._start_time).total_seconds()) if self._is_running else 0
        status = AlgorithmStatus(
            name=self.algorithm_name,
            version=self.version,
            is_running=self._is_running,
            is_paused=self._is_paused,
            start_time=self._start_time if self._is_running else None,
            signals_generated=self._signals_count,
            trades_executed=self._trades_count,
            errors=self._errors,
            uptime_seconds=uptime,
        )
        self.bus.emit(AlgorithmStatusEvent(status=status, source=self.algorithm_name))

    def emit_signal(self, signal: Signal) -> None:
        """Emit a trading signal."""
        self._signals_count += 1
        self.bus.emit(SignalEvent(signal=signal, source=self.algorithm_name))

    def emit_position_opened(self, position: Position) -> None:
        """Emit position opened event."""
        self._positions[position.symbol] = position
        self.bus.emit(PositionOpenedEvent(position=position, source=self.algorithm_name))

    def emit_position_updated(self, position: Position, update_type: str = "price") -> None:
        """Emit position updated event."""
        self._positions[position.symbol] = position
        self.bus.emit(PositionUpdatedEvent(
            position=position,
            update_type=update_type,
            source=self.algorithm_name,
        ))

    def emit_position_closed(self, position: Position, exit_price: float, exit_reason: str = "") -> None:
        """Emit position closed event."""
        trade = Trade.from_position(
            position=position,
            exit_price=exit_price,
            exit_time=datetime.now(),
            exit_reason=exit_reason,
        )
        self._trades.append(trade)
        self._trades_count += 1

        if position.symbol in self._positions:
            del self._positions[position.symbol]

        self.bus.emit(PositionClosedEvent(
            trade=trade,
            position=position,
            source=self.algorithm_name,
        ))

    def emit_market_data(self, symbol: str, price: float, **kwargs) -> None:
        """Emit market data update."""
        data = MarketData(
            symbol=symbol,
            last_price=price,
            last_update=datetime.now(),
            **kwargs,
        )
        self.bus.emit(MarketDataEvent(data=data, symbol=symbol, source=self.algorithm_name))

        # Also update position price if we have one
        if symbol in self._positions:
            self._positions[symbol].update_price(price, datetime.now())
            self.emit_position_updated(self._positions[symbol], "price")

    def log(self, level: str, message: str, **context) -> None:
        """Emit a log message."""
        self.bus.emit(LogEvent(
            level=level,
            message=message,
            context=context,
            source=self.algorithm_name,
        ))

    def error(self, error_type: str, message: str, is_critical: bool = False) -> None:
        """Emit an error event."""
        self._errors += 1
        self.bus.emit(ErrorEvent(
            error_type=error_type,
            message=message,
            is_critical=is_critical,
            source=self.algorithm_name,
        ))


class OrchestratorAdapter(BaseAlgorithmAdapter):
    """
    Adapter specifically for the Orchestrator trading system.

    Translates Orchestrator signals and trades into dashboard events.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(
            event_bus=event_bus,
            algorithm_name="Orchestrator",
            version="2.0.0",
        )

    def on_orchestrator_signal(
        self,
        symbol: str,
        action: str,
        price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        confidence: float,
        consensus_score: float,
        edge_votes: Dict[str, Any],
        reason: str,
    ) -> None:
        """
        Handle signal from Orchestrator.

        This should be called whenever the Orchestrator generates a signal.
        """
        # Map action to signal type
        action_map = {
            "buy": SignalType.ENTRY_LONG,
            "short": SignalType.ENTRY_SHORT,
            "sell": SignalType.EXIT,
            "cover": SignalType.EXIT,
            "hold": SignalType.HOLD,
        }
        signal_type = action_map.get(action.lower(), SignalType.HOLD)

        # Map confidence to strength
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.STRONG
        elif confidence >= 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        signal = Signal(
            id=f"SIG-{datetime.now().strftime('%H%M%S')}-{symbol}",
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=price,
            suggested_stop=stop_loss,
            suggested_target=take_profit,
            algorithm="Orchestrator",
            reason=reason,
            metadata={"consensus_score": consensus_score},
            components=edge_votes,
        )

        self.emit_signal(signal)

        # If it's an entry signal, create position
        if action.lower() in ("buy", "short"):
            direction = TradeDirection.LONG if action.lower() == "buy" else TradeDirection.SHORT
            position = Position(
                id=f"POS-{datetime.now().strftime('%H%M%S')}-{symbol}",
                symbol=symbol,
                direction=direction,
                entry_price=price,
                entry_time=datetime.now(),
                quantity=1,  # This should come from position sizing
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=price,
                algorithm="Orchestrator",
                signal_id=signal.id,
            )
            self.emit_position_opened(position)

    def on_position_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        Handle position exit from Orchestrator.
        """
        if symbol in self._positions:
            self.emit_position_closed(
                self._positions[symbol],
                exit_price,
                exit_reason,
            )

    def on_price_update(
        self,
        symbol: str,
        price: float,
        high: float = 0,
        low: float = 0,
        volume: int = 0,
    ) -> None:
        """
        Handle price update.
        """
        self.emit_market_data(
            symbol=symbol,
            price=price,
            day_high=high,
            day_low=low,
            volume=volume,
        )


class GenericAdapter(BaseAlgorithmAdapter):
    """
    Generic adapter that can be used with any algorithm.

    Provides simple methods to emit events.
    """

    def __init__(
        self,
        algorithm_name: str = "Custom Algorithm",
        version: str = "1.0.0",
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(
            event_bus=event_bus,
            algorithm_name=algorithm_name,
            version=version,
        )

    def signal(
        self,
        symbol: str,
        signal_type: str,  # "buy", "sell", "short", "cover", "hold"
        price: float,
        confidence: float = 0.5,
        reason: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Signal:
        """
        Emit a trading signal.

        Returns the created Signal object.
        """
        type_map = {
            "buy": SignalType.ENTRY_LONG,
            "sell": SignalType.EXIT,
            "short": SignalType.ENTRY_SHORT,
            "cover": SignalType.EXIT,
            "hold": SignalType.HOLD,
        }

        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.STRONG
        elif confidence >= 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        signal = Signal(
            id=f"SIG-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=type_map.get(signal_type.lower(), SignalType.HOLD),
            strength=strength,
            confidence=confidence,
            price=price,
            suggested_stop=stop_loss,
            suggested_target=take_profit,
            algorithm=self.algorithm_name,
            reason=reason,
        )

        self.emit_signal(signal)
        return signal

    def open_position(
        self,
        symbol: str,
        direction: str,  # "long" or "short"
        entry_price: float,
        quantity: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """
        Open a position and emit event.

        Returns the created Position object.
        """
        dir_enum = TradeDirection.LONG if direction.lower() == "long" else TradeDirection.SHORT

        position = Position(
            id=f"POS-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            direction=dir_enum,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            algorithm=self.algorithm_name,
        )

        self.emit_position_opened(position)
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "",
    ) -> Optional[Trade]:
        """
        Close a position and emit event.

        Returns the created Trade object, or None if no position found.
        """
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        self.emit_position_closed(position, exit_price, reason)
        return self._trades[-1] if self._trades else None

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a symbol."""
        self.emit_market_data(symbol, price)
