"""
Mathematical and financial constants.

All constants are defined here for consistency across the framework.
"""

import numpy as np

# =============================================================================
# TIME CONSTANTS
# =============================================================================
TRADING_DAYS_PER_YEAR: int = 252
TRADING_HOURS_PER_DAY: float = 6.5
MINUTES_PER_TRADING_DAY: int = 390
SECONDS_PER_TRADING_DAY: int = 23400

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================
SQRT_252: float = np.sqrt(252)
SQRT_12: float = np.sqrt(12)  # For monthly to annual conversion
LN_2: float = np.log(2)  # For half-life calculations

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================
EPSILON: float = 1e-10  # Prevent division by zero
MIN_VARIANCE: float = 1e-8  # Minimum variance for stability
MAX_SHARPE: float = 10.0  # Cap Sharpe ratio for numerical stability
MIN_HALF_LIFE: float = 1.0  # Minimum half-life in days
MAX_HALF_LIFE: float = 252.0  # Maximum half-life in days

# =============================================================================
# DEFAULT MODEL PARAMETERS
# =============================================================================

# Ornstein-Uhlenbeck defaults (Avellaneda & Lee 2010)
OU_DEFAULT_LOOKBACK: int = 60  # Days for parameter estimation
OU_S_SCORE_ENTRY: float = 1.25  # Entry threshold
OU_S_SCORE_EXIT: float = 0.5  # Exit threshold
OU_S_SCORE_STOP: float = 4.0  # Stop loss threshold
OU_MIN_KAPPA: float = 8.4  # Minimum mean reversion speed (252/30)

# Kelly Criterion defaults
KELLY_FRACTION_CONSERVATIVE: float = 0.25  # Quarter Kelly
KELLY_FRACTION_MODERATE: float = 0.50  # Half Kelly
KELLY_FRACTION_AGGRESSIVE: float = 0.75  # Three-quarter Kelly
KELLY_MAX_POSITION: float = 0.25  # Max 25% in single position

# Volatility scaling defaults (Daniel & Moskowitz 2016)
VOL_TARGET_DEFAULT: float = 0.15  # 15% annualized target
VOL_LOOKBACK_DEFAULT: int = 60  # Days for vol estimation
VOL_FLOOR: float = 0.05  # Minimum vol estimate (5%)
VOL_CAP: float = 1.0  # Maximum vol estimate (100%)
VOL_SCALAR_MAX: float = 3.0  # Maximum leverage from vol scaling

# Time Series Momentum defaults (Moskowitz, Ooi & Pedersen 2012)
TSMOM_LOOKBACK: int = 252  # 12-month lookback
TSMOM_HOLDING_PERIOD: int = 21  # 1-month holding
TSMOM_VOL_LOOKBACK: int = 60  # For volatility estimation

# Hidden Markov Model defaults
HMM_N_STATES: int = 3  # Bull, Bear, Neutral
HMM_N_ITER: int = 100  # EM iterations
HMM_LOOKBACK: int = 252  # Training window

# Risk management defaults
VAR_CONFIDENCE: float = 0.95  # 95% VaR
ES_CONFIDENCE: float = 0.95  # 95% Expected Shortfall
MAX_LEVERAGE: float = 2.0
MAX_DRAWDOWN_THRESHOLD: float = 0.20  # 20% drawdown trigger

# Almgren-Chriss execution defaults
AC_RISK_AVERSION: float = 1e-6  # Risk aversion parameter λ
AC_TEMP_IMPACT_COEF: float = 0.1  # Temporary impact coefficient η
AC_PERM_IMPACT_COEF: float = 0.01  # Permanent impact coefficient γ

# Machine Learning defaults
ML_N_ESTIMATORS: int = 100  # Number of trees for GBRT
ML_MAX_DEPTH: int = 3  # Max depth for trees
ML_LEARNING_RATE: float = 0.1
ML_VALIDATION_SPLIT: float = 0.2
ML_PURGE_WINDOW: int = 5  # Days to purge between train/test

# Backtest validation defaults
CSCV_N_SPLITS: int = 16  # Number of CSCV combinations
MIN_TRADES_FOR_VALIDITY: int = 30
T_STAT_THRESHOLD: float = 3.0  # Harvey, Liu & Zhu (2016)
MAX_PBO_THRESHOLD: float = 0.5  # Max acceptable PBO
