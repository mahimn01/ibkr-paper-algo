"""
Trading Dashboard - Enterprise-Level Terminal UI

A comprehensive, algorithm-agnostic trading dashboard that provides:
- Real-time P&L tracking (realized + unrealized)
- Open positions with stop/target levels
- Trade history (today/yesterday)
- Signal log from any algorithm
- Activity log
- Keyboard controls

Usage:
    # Standalone
    from trading_algo.dashboard import run_dashboard
    run_dashboard(algorithm_name="My Algorithm")

    # With algorithm integration
    from trading_algo.dashboard import TradingDashboard, GenericAdapter

    adapter = GenericAdapter(algorithm_name="My Algo")
    adapter.start()

    # Your algorithm emits events via adapter
    adapter.signal("AAPL", "buy", 150.0, confidence=0.8)
    adapter.open_position("AAPL", "long", 150.0, stop_loss=145.0)

    # Run dashboard
    dashboard = TradingDashboard()
    dashboard.run()
"""

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

from .events import (
    EventBus,
    get_event_bus,
    reset_event_bus,
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
)

from .store import (
    DashboardStore,
    get_store,
    reset_store,
)

from .adapter import (
    BaseAlgorithmAdapter,
    OrchestratorAdapter,
    GenericAdapter,
)

from .app import (
    TradingDashboard,
    run_dashboard,
    run_dashboard_async,
)

__all__ = [
    # Models
    "Signal",
    "Position",
    "Trade",
    "PnLSummary",
    "MarketData",
    "AlgorithmStatus",
    "TradeDirection",
    "TradeStatus",
    "SignalType",
    "SignalStrength",
    # Events
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
    "BaseEvent",
    "SignalEvent",
    "PositionOpenedEvent",
    "PositionUpdatedEvent",
    "PositionClosedEvent",
    "TradeExecutedEvent",
    "PnLUpdateEvent",
    "MarketDataEvent",
    "AlgorithmStatusEvent",
    "ErrorEvent",
    "LogEvent",
    "HeartbeatEvent",
    # Store
    "DashboardStore",
    "get_store",
    "reset_store",
    # Adapters
    "BaseAlgorithmAdapter",
    "OrchestratorAdapter",
    "GenericAdapter",
    # App
    "TradingDashboard",
    "run_dashboard",
    "run_dashboard_async",
]
