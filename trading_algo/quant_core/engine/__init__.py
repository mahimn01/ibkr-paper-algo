"""
Quantitative Trading Engine

The unified orchestrator that combines all quantitative modules
into a coherent trading system.

Components:
    - QuantOrchestrator: Main trading engine
    - SignalAggregator: Multi-model signal combination
    - RiskController: Real-time risk management
    - PortfolioManager: Position sizing and allocation
    - ExecutionManager: Optimal order execution
    - TradingContext: Abstraction for live/backtest modes
    - IBKRBrokerAdapter: IBKR broker integration

Usage:
    # For backtesting
    from trading_algo.quant_core.engine import (
        QuantOrchestrator, EngineConfig, EngineMode
    )

    config = EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=['AAPL', 'MSFT', 'GOOGL'],
    )
    engine = QuantOrchestrator(config)
    results = engine.run_backtest(historical_data, timestamps)

    # For live/paper trading
    config = EngineConfig(mode=EngineMode.PAPER, ...)
    engine = QuantOrchestrator(config)
    engine.run_live(ibkr_broker, data_provider)
"""

from trading_algo.quant_core.engine.orchestrator import (
    QuantOrchestrator,
    EngineConfig,
    EngineState,
    EngineMode,
    BacktestResult,
)
from trading_algo.quant_core.engine.signal_aggregator import (
    SignalAggregator,
    AggregatedSignal,
    AggregatorConfig,
)
from trading_algo.quant_core.engine.risk_controller import (
    RiskController,
    RiskDecision,
    RiskAction,
    RiskConfig,
)
from trading_algo.quant_core.engine.portfolio_manager import (
    PortfolioManager,
    TargetPortfolio,
    TargetPosition,
    PortfolioConfig,
    SizingMethod,
)
from trading_algo.quant_core.engine.execution_manager import (
    ExecutionManager,
    ExecutionResult,
    OrderRequest,
    ExecutionConfig,
    ExecutionMethod,
)
from trading_algo.quant_core.engine.trading_context import (
    TradingContext,
    LiveContext,
    BacktestContext,
    MarketData,
    Position,
    Order,
    OrderSide,
    OrderType,
)
from trading_algo.quant_core.engine.ibkr_adapter import (
    IBKRBrokerAdapter,
    IBKRDataProvider,
    IBKRConfig,
    create_live_context,
)

__all__ = [
    # Orchestrator
    "QuantOrchestrator",
    "EngineConfig",
    "EngineState",
    "EngineMode",
    "BacktestResult",
    # Signal
    "SignalAggregator",
    "AggregatedSignal",
    "AggregatorConfig",
    # Risk
    "RiskController",
    "RiskDecision",
    "RiskAction",
    "RiskConfig",
    # Portfolio
    "PortfolioManager",
    "TargetPortfolio",
    "TargetPosition",
    "PortfolioConfig",
    "SizingMethod",
    # Execution
    "ExecutionManager",
    "ExecutionResult",
    "OrderRequest",
    "ExecutionConfig",
    "ExecutionMethod",
    # Context
    "TradingContext",
    "LiveContext",
    "BacktestContext",
    "MarketData",
    "Position",
    "Order",
    "OrderSide",
    "OrderType",
    # IBKR
    "IBKRBrokerAdapter",
    "IBKRDataProvider",
    "IBKRConfig",
    "create_live_context",
]
