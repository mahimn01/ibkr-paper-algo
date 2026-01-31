"""
Trading Strategies Package.

Contains high-level trading strategies that combine multiple analysis techniques.

For backward compatibility, this module re-exports from the new modular
orchestrator package. Prefer importing directly from trading_algo.orchestrator.
"""

# Re-export from the modular orchestrator for backward compatibility
from trading_algo.orchestrator import (
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
