"""
Portfolio construction and position sizing.

Modules:
    - KellyCriterion: Optimal position sizing based on Kelly (1956)
    - HierarchicalRiskParity: López de Prado (2016) HRP algorithm
    - NestedClusteredOptimization: NCO extension of HRP
    - PortfolioOptimizer: Combined portfolio optimization
"""

from trading_algo.quant_core.portfolio.kelly import KellyCriterion, KellyEstimate
from trading_algo.quant_core.portfolio.hrp import (
    HierarchicalRiskParity,
    NestedClusteredOptimization,
    HRPResult,
)
from trading_algo.quant_core.portfolio.optimizer import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationResult,
    OptimizationConstraints,
    black_litterman,
)

__all__ = [
    "KellyCriterion",
    "KellyEstimate",
    "HierarchicalRiskParity",
    "NestedClusteredOptimization",
    "HRPResult",
    "PortfolioOptimizer",
    "OptimizationMethod",
    "OptimizationResult",
    "OptimizationConstraints",
    "black_litterman",
]
