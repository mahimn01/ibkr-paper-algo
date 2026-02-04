"""
Quantitative Core Framework

A mathematically rigorous trading framework based on academic research
and institutional best practices.

Architecture:
    - models/: Statistical and mathematical models (OU, HMM, TSMOM)
    - risk/: Risk management (CVaR, Expected Shortfall, tail risk)
    - execution/: Optimal execution (Almgren-Chriss, TWAP, VWAP)
    - ml/: Machine learning signal enhancement
    - portfolio/: Portfolio construction (HRP, Kelly)
    - validation/: Backtesting validation (PBO, CSCV)
    - utils/: Utilities and helpers

Academic References:
    - Avellaneda & Lee (2010): Statistical Arbitrage in US Equities
    - Moskowitz, Ooi & Pedersen (2012): Time Series Momentum
    - Daniel & Moskowitz (2016): Momentum Crashes
    - Gu, Kelly & Xiu (2020): Empirical Asset Pricing via Machine Learning
    - Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
    - López de Prado (2016): Hierarchical Risk Parity
    - Bailey et al. (2014): Probability of Backtest Overfitting

Author: Quantitative Framework v2
"""

__version__ = "2.0.0"
__author__ = "Quantitative Research"

# Models
from trading_algo.quant_core.models import (
    OrnsteinUhlenbeck,
    OUParameters,
    OUSignal,
    HiddenMarkovRegime,
    RegimeState,
    TimeSeriesMomentum,
    TSMOMAssetSignal,
    VolatilityManagedMomentum,
    VolManagedSignal,
)

# Risk Management
from trading_algo.quant_core.risk import (
    ExpectedShortfall,
    TailRiskManager,
    RiskMetrics,
)

# Execution
from trading_algo.quant_core.execution import (
    AlmgrenChrissExecutor,
    ExecutionPlan,
    MarketImpactModel,
    ExecutionMetrics,
    TWAPExecutor,
    VWAPExecutor,
)

# Portfolio Construction
from trading_algo.quant_core.portfolio import (
    KellyCriterion,
    KellyEstimate,
    HierarchicalRiskParity,
    NestedClusteredOptimization,
    HRPResult,
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationResult,
    black_litterman,
)

# Machine Learning
from trading_algo.quant_core.ml import (
    SignalCombiner,
    CombinerConfig,
    FeatureEngine,
    FeatureSet,
    TimeSeriesCV,
    PurgedKFold,
)

# Validation
from trading_algo.quant_core.validation import (
    BacktestValidator,
    ValidationResult,
    OverfittingMetrics,
    PBOCalculator,
    DeflatedSharpe,
    MultipleTestingCorrection,
)

__all__ = [
    # Models
    "OrnsteinUhlenbeck",
    "OUParameters",
    "OUSignal",
    "HiddenMarkovRegime",
    "RegimeState",
    "TimeSeriesMomentum",
    "TSMOMAssetSignal",
    "VolatilityManagedMomentum",
    "VolManagedSignal",
    # Risk
    "ExpectedShortfall",
    "TailRiskManager",
    "RiskMetrics",
    # Execution
    "AlmgrenChrissExecutor",
    "ExecutionPlan",
    "MarketImpactModel",
    "ExecutionMetrics",
    "TWAPExecutor",
    "VWAPExecutor",
    # Portfolio
    "KellyCriterion",
    "KellyEstimate",
    "HierarchicalRiskParity",
    "NestedClusteredOptimization",
    "HRPResult",
    "PortfolioOptimizer",
    "OptimizationMethod",
    "OptimizationResult",
    "black_litterman",
    # ML
    "SignalCombiner",
    "CombinerConfig",
    "FeatureEngine",
    "FeatureSet",
    "TimeSeriesCV",
    "PurgedKFold",
    # Validation
    "BacktestValidator",
    "ValidationResult",
    "OverfittingMetrics",
    "PBOCalculator",
    "DeflatedSharpe",
    "MultipleTestingCorrection",
]
