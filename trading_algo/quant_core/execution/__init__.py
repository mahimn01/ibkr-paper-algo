"""
Execution algorithms module.

Provides:
    - AlmgrenChrissExecutor: Optimal execution using Almgren-Chriss framework
    - TWAPExecutor: Time-Weighted Average Price execution
    - VWAPExecutor: Volume-Weighted Average Price execution
"""

from trading_algo.quant_core.execution.almgren_chriss import (
    AlmgrenChrissExecutor,
    ExecutionPlan,
    MarketImpactModel,
    ExecutionMetrics,
)
from trading_algo.quant_core.execution.twap_vwap import TWAPExecutor, VWAPExecutor

__all__ = [
    "AlmgrenChrissExecutor",
    "ExecutionPlan",
    "MarketImpactModel",
    "ExecutionMetrics",
    "TWAPExecutor",
    "VWAPExecutor",
]
