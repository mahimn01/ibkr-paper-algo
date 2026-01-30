"""
Trading Strategies Package.

Contains high-level trading strategies that combine multiple analysis techniques.
"""

from trading_algo.strategies.orchestrator import (
    Orchestrator,
    OrchestratorSignal,
    MarketRegime,
    EdgeVote,
    TradeType,
    EdgeSignal,
    AssetState,
    MarketRegimeEngine,
    RelativeStrengthEngine,
    StatisticalExtremeDetector,
    VolumeProfileEngine,
    CrossAssetEngine,
    TimeOfDayEngine,
    create_orchestrator,
)

__all__ = [
    # Main strategy
    "Orchestrator",
    "OrchestratorSignal",
    "create_orchestrator",
    # Enums
    "MarketRegime",
    "EdgeVote",
    "TradeType",
    # Data structures
    "EdgeSignal",
    "AssetState",
    # Edge engines
    "MarketRegimeEngine",
    "RelativeStrengthEngine",
    "StatisticalExtremeDetector",
    "VolumeProfileEngine",
    "CrossAssetEngine",
    "TimeOfDayEngine",
]
