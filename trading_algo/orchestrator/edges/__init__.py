"""
Edge engines for the Orchestrator trading system.

Each edge provides an independent source of trading signals.
The Orchestrator combines votes from all edges to make decisions.
"""

from .market_regime import MarketRegimeEngine
from .relative_strength import RelativeStrengthEngine
from .statistics import StatisticalExtremeDetector
from .volume_profile import VolumeProfileEngine
from .cross_asset import CrossAssetEngine
from .time_of_day import TimeOfDayEngine

__all__ = [
    "MarketRegimeEngine",
    "RelativeStrengthEngine",
    "StatisticalExtremeDetector",
    "VolumeProfileEngine",
    "CrossAssetEngine",
    "TimeOfDayEngine",
]

# QuantEdge is optional — requires quant_core (numpy + scipy).
try:
    from .quant_edge import QuantEdge  # noqa: F401
    __all__.append("QuantEdge")
except ImportError:
    pass
