"""
Topology Detector Module

Detects market regimes using Topological Data Analysis:
- Persistent homology to find structural patterns
- Betti numbers for regime classification
- Takens embedding for time series

Requires: pip install ripser (optional, has fallback)
"""

from trading_algo.rat.topology.detector import (
    TopologyDetector,
    TopologyState,
    TopologyRegime,
)

__all__ = ["TopologyDetector", "TopologyState", "TopologyRegime"]
