"""
Risk management module.

Provides:
    - ExpectedShortfall: CVaR/ES calculation and monitoring
    - TailRiskManager: Tail risk detection and hedging triggers
    - RiskMetrics: Comprehensive risk metrics calculation
"""

from trading_algo.quant_core.risk.expected_shortfall import ExpectedShortfall
from trading_algo.quant_core.risk.tail_risk import TailRiskManager
from trading_algo.quant_core.risk.metrics import RiskMetrics

__all__ = [
    "ExpectedShortfall",
    "TailRiskManager",
    "RiskMetrics",
]
