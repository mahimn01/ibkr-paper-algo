"""
Data models for the trading dashboard.

These models are algorithm-agnostic and can be used with any trading strategy.
They provide a standardized way to represent trades, positions, signals, and P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Current status of a trade."""
    PENDING = auto()      # Order submitted, not filled
    OPEN = auto()         # Position is open
    CLOSED = auto()       # Position closed
    CANCELLED = auto()    # Order cancelled


class SignalType(Enum):
    """Type of trading signal."""
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"


class SignalStrength(Enum):
    """Strength/confidence of a signal."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class Signal:
    """
    A trading signal from an algorithm.

    This is algorithm-agnostic - any strategy can emit signals in this format.
    """
    id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0

    # Price information
    price: float
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None

    # Algorithm-specific metadata
    algorithm: str = "Unknown"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For ensemble systems - individual component votes/scores
    components: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"SIG-{self.timestamp.strftime('%H%M%S')}-{self.symbol}"


@dataclass
class Position:
    """
    An open trading position.

    Tracks all details needed for position management and P&L calculation.
    """
    id: str
    symbol: str
    direction: TradeDirection

    # Entry details
    entry_price: float
    entry_time: datetime
    quantity: int

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_active: bool = False

    # Current state
    current_price: float = 0.0
    best_price: float = 0.0  # Best price since entry (for trailing)
    worst_price: float = 0.0  # Worst price since entry
    last_update: Optional[datetime] = None

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Metadata
    algorithm: str = "Unknown"
    signal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_price(self, price: float, timestamp: datetime) -> None:
        """Update position with new price."""
        self.current_price = price
        self.last_update = timestamp

        # Track best/worst
        if self.direction == TradeDirection.LONG:
            self.best_price = max(self.best_price, price) if self.best_price > 0 else price
            self.worst_price = min(self.worst_price, price) if self.worst_price > 0 else price
        else:
            self.best_price = min(self.best_price, price) if self.best_price > 0 else price
            self.worst_price = max(self.worst_price, price) if self.worst_price > 0 else price

        # Calculate P&L
        if self.direction == TradeDirection.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_pct = ((self.entry_price - price) / self.entry_price) * 100

    @property
    def risk_reward_current(self) -> Optional[float]:
        """Current risk/reward ratio."""
        if not self.stop_loss or not self.take_profit:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else None

    @property
    def distance_to_stop_pct(self) -> Optional[float]:
        """Distance to stop loss as percentage."""
        if not self.stop_loss or self.current_price <= 0:
            return None
        return ((self.current_price - self.stop_loss) / self.current_price) * 100

    @property
    def distance_to_target_pct(self) -> Optional[float]:
        """Distance to take profit as percentage."""
        if not self.take_profit or self.current_price <= 0:
            return None
        if self.direction == TradeDirection.LONG:
            return ((self.take_profit - self.current_price) / self.current_price) * 100
        else:
            return ((self.current_price - self.take_profit) / self.current_price) * 100


@dataclass
class Trade:
    """
    A completed trade (position that has been closed).

    Contains full history of the trade from entry to exit.
    """
    id: str
    symbol: str
    direction: TradeDirection
    status: TradeStatus

    # Entry
    entry_price: float
    entry_time: datetime
    quantity: int

    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""

    # Risk management (at entry)
    initial_stop: Optional[float] = None
    initial_target: Optional[float] = None
    final_stop: Optional[float] = None  # After any adjustments

    # P&L
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0

    # Performance metrics
    best_price: float = 0.0
    worst_price: float = 0.0
    max_favorable_excursion: float = 0.0  # Max profit during trade
    max_adverse_excursion: float = 0.0    # Max drawdown during trade

    # Timing
    duration_seconds: int = 0

    # Metadata
    algorithm: str = "Unknown"
    signal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_position(cls, position: Position, exit_price: float,
                      exit_time: datetime, exit_reason: str = "") -> Trade:
        """Create a Trade from a closed Position."""
        if position.direction == TradeDirection.LONG:
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            realized_pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
            mfe = (position.best_price - position.entry_price) * position.quantity
            mae = (position.entry_price - position.worst_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
            realized_pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
            mfe = (position.entry_price - position.best_price) * position.quantity
            mae = (position.worst_price - position.entry_price) * position.quantity

        duration = int((exit_time - position.entry_time).total_seconds())

        return cls(
            id=position.id.replace("POS-", "TRD-"),
            symbol=position.symbol,
            direction=position.direction,
            status=TradeStatus.CLOSED,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            quantity=position.quantity,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=exit_reason,
            initial_stop=position.stop_loss,
            initial_target=position.take_profit,
            final_stop=position.trailing_stop or position.stop_loss,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            net_pnl=realized_pnl,  # Commission added separately
            best_price=position.best_price,
            worst_price=position.worst_price,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            duration_seconds=duration,
            algorithm=position.algorithm,
            signal_id=position.signal_id,
            metadata=position.metadata,
        )


@dataclass
class PnLSummary:
    """
    P&L summary for a time period.
    """
    period: str  # "today", "yesterday", "week", "month", "all"
    start_time: datetime
    end_time: datetime

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L
    gross_pnl: float = 0.0
    commissions: float = 0.0
    net_pnl: float = 0.0

    # Statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Current positions
    open_positions: int = 0
    unrealized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        """Total P&L including unrealized."""
        return self.net_pnl + self.unrealized_pnl

    def update_from_trades(self, trades: List[Trade]) -> None:
        """Update summary from a list of trades."""
        if not trades:
            return

        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.realized_pnl > 0)
        self.losing_trades = sum(1 for t in trades if t.realized_pnl < 0)

        self.gross_pnl = sum(t.realized_pnl for t in trades)
        self.commissions = sum(t.commission for t in trades)
        self.net_pnl = sum(t.net_pnl for t in trades)

        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100

        winners = [t.realized_pnl for t in trades if t.realized_pnl > 0]
        losers = [t.realized_pnl for t in trades if t.realized_pnl < 0]

        if winners:
            self.avg_win = sum(winners) / len(winners)
            self.largest_win = max(winners)

        if losers:
            self.avg_loss = sum(losers) / len(losers)
            self.largest_loss = min(losers)

        total_wins = sum(winners) if winners else 0
        total_losses = abs(sum(losers)) if losers else 0
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses


@dataclass
class MarketData:
    """
    Market data for a symbol.
    """
    symbol: str
    last_price: float
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    day_high: float = 0.0
    day_low: float = 0.0
    day_open: float = 0.0
    prev_close: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    last_update: Optional[datetime] = None

    def update(self, price: float, timestamp: datetime) -> None:
        """Update with new price."""
        if self.prev_close > 0:
            self.change = price - self.prev_close
            self.change_pct = (self.change / self.prev_close) * 100
        self.last_price = price
        self.last_update = timestamp
        if price > self.day_high or self.day_high == 0:
            self.day_high = price
        if price < self.day_low or self.day_low == 0:
            self.day_low = price


@dataclass
class AlgorithmStatus:
    """
    Status of a trading algorithm.
    """
    name: str
    version: str = "1.0.0"

    # State
    is_running: bool = False
    is_paused: bool = False
    start_time: Optional[datetime] = None

    # Counters
    signals_generated: int = 0
    trades_executed: int = 0
    errors: int = 0

    # Current activity
    current_action: str = "Idle"
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    last_error: Optional[str] = None

    # Performance
    uptime_seconds: int = 0

    @property
    def status_text(self) -> str:
        """Human-readable status."""
        if not self.is_running:
            return "Stopped"
        if self.is_paused:
            return "Paused"
        return "Running"
