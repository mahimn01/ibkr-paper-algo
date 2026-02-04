"""
Backtest Validation Module.

Provides:
    - BacktestValidator: Comprehensive backtest validation
    - PBOCalculator: Probability of Backtest Overfitting
    - DeflatedSharpe: Deflated Sharpe Ratio for multiple testing
"""

from trading_algo.quant_core.validation.backtest_validator import (
    BacktestValidator,
    ValidationResult,
    OverfittingMetrics,
)
from trading_algo.quant_core.validation.pbo import (
    PBOCalculator,
    DeflatedSharpe,
    MultipleTestingCorrection,
)

__all__ = [
    "BacktestValidator",
    "ValidationResult",
    "OverfittingMetrics",
    "PBOCalculator",
    "DeflatedSharpe",
    "MultipleTestingCorrection",
]
