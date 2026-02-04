"""
Statistical utilities for quantitative analysis.

Includes risk metrics, performance statistics, and statistical tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from trading_algo.quant_core.utils.constants import (
    TRADING_DAYS_PER_YEAR,
    SQRT_252,
    EPSILON,
)


# =============================================================================
# RISK-ADJUSTED RETURN METRICS
# =============================================================================

@dataclass(frozen=True)
class PerformanceMetrics:
    """Container for strategy performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    expectancy: float
    var_95: float
    cvar_95: float


def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Calculate Sharpe ratio.

    SR = (μ - r_f) / σ

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        annualize: Whether to annualize

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Convert to daily risk-free rate if annualized
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    excess_returns = returns - daily_rf
    mean_excess = np.mean(excess_returns)
    std = np.std(excess_returns, ddof=1)

    if std < EPSILON:
        return 0.0

    sr = mean_excess / std

    if annualize:
        sr *= SQRT_252

    return float(sr)


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Calculate Sortino ratio using downside deviation.

    Sortino = (μ - r_f) / σ_downside

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        target_return: Target return for downside calculation
        annualize: Whether to annualize

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    # Convert to daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    excess_returns = returns - daily_rf
    mean_excess = np.mean(excess_returns)

    # Downside deviation
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std < EPSILON:
        return float('inf') if mean_excess > 0 else 0.0

    sortino = mean_excess / downside_std

    if annualize:
        sortino *= SQRT_252

    return float(sortino)


def calmar_ratio(
    returns: NDArray[np.float64],
    annualize: bool = True
) -> float:
    """
    Calculate Calmar ratio.

    Calmar = Annualized Return / Max Drawdown

    Args:
        returns: Return series
        annualize: Whether to annualize return

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    ann_return = np.mean(returns) * TRADING_DAYS_PER_YEAR if annualize else np.mean(returns)
    max_dd = max_drawdown(returns)

    if abs(max_dd) < EPSILON:
        return float('inf') if ann_return > 0 else 0.0

    return float(ann_return / abs(max_dd))


# =============================================================================
# DRAWDOWN CALCULATIONS
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def max_drawdown(returns: NDArray[np.float64]) -> float:
    """
    Calculate maximum drawdown from return series.

    MDD = max(1 - P_t / P_peak)

    Args:
        returns: Return series

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.20 = 20%)
    """
    n = len(returns)
    if n < 1:
        return 0.0

    # Calculate cumulative returns
    cum_return = 1.0
    peak = 1.0
    max_dd = 0.0

    for i in range(n):
        cum_return *= (1 + returns[i])
        if cum_return > peak:
            peak = cum_return
        dd = 1 - cum_return / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd


@jit(nopython=True, cache=True, fastmath=True)
def drawdown_series(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate drawdown series.

    Args:
        returns: Return series

    Returns:
        Array of drawdowns at each point
    """
    n = len(returns)
    dd = np.empty(n, dtype=np.float64)

    cum_return = 1.0
    peak = 1.0

    for i in range(n):
        cum_return *= (1 + returns[i])
        if cum_return > peak:
            peak = cum_return
        dd[i] = 1 - cum_return / peak

    return dd


def max_drawdown_duration(returns: NDArray[np.float64]) -> int:
    """
    Calculate maximum drawdown duration in periods.

    Args:
        returns: Return series

    Returns:
        Maximum duration (number of periods) in drawdown
    """
    dd = drawdown_series(returns)
    max_duration = 0
    current_duration = 0

    for d in dd:
        if d > 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


# =============================================================================
# VALUE AT RISK AND EXPECTED SHORTFALL
# =============================================================================

def value_at_risk(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: "historical" or "parametric"

    Returns:
        VaR as a positive number (potential loss)
    """
    if len(returns) < 10:
        return 0.0

    if method == "historical":
        # Historical simulation
        var = np.percentile(returns, (1 - confidence) * 100)
    else:
        # Parametric (assumes normal distribution)
        from scipy import stats
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        var = mean - stats.norm.ppf(confidence) * std

    return float(-var) if var < 0 else 0.0


def expected_shortfall(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).

    ES = E[L | L > VaR]

    The average loss given that loss exceeds VaR.

    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95)
        method: "historical" or "parametric"

    Returns:
        Expected Shortfall as a positive number
    """
    if len(returns) < 10:
        return 0.0

    if method == "historical":
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) == 0:
            return value_at_risk(returns, confidence, method)
        es = np.mean(tail_losses)
    else:
        # Parametric (assumes normal distribution)
        from scipy import stats
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        var = stats.norm.ppf(1 - confidence)
        es = mean - std * stats.norm.pdf(var) / (1 - confidence)

    return float(-es) if es < 0 else 0.0


# =============================================================================
# TRADE STATISTICS
# =============================================================================

def win_rate(pnls: NDArray[np.float64]) -> float:
    """Calculate win rate from P&L series."""
    if len(pnls) == 0:
        return 0.0
    return float(np.sum(pnls > 0) / len(pnls))


def profit_factor(pnls: NDArray[np.float64]) -> float:
    """
    Calculate profit factor.

    PF = Σ(wins) / |Σ(losses)|

    Args:
        pnls: Array of trade P&Ls

    Returns:
        Profit factor (> 1 is profitable)
    """
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0

    if total_losses < EPSILON:
        return float('inf') if total_wins > 0 else 0.0

    return float(total_wins / total_losses)


def expectancy(pnls: NDArray[np.float64]) -> float:
    """
    Calculate expectancy (expected value per trade).

    E = P(win) * avg_win + P(loss) * avg_loss

    Args:
        pnls: Array of trade P&Ls

    Returns:
        Expected P&L per trade
    """
    if len(pnls) == 0:
        return 0.0

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    p_win = len(wins) / len(pnls)
    p_loss = len(losses) / len(pnls)

    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    return float(p_win * avg_win + p_loss * avg_loss)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

class AugmentedDickeyFullerResult(NamedTuple):
    """Result of ADF test for stationarity."""
    adf_statistic: float
    p_value: float
    lags_used: int
    n_obs: int
    critical_values: dict
    is_stationary: bool


def augmented_dickey_fuller(
    series: NDArray[np.float64],
    max_lags: Optional[int] = None,
    significance: float = 0.05
) -> AugmentedDickeyFullerResult:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    H0: Series has a unit root (non-stationary)
    H1: Series is stationary

    Args:
        series: Time series to test
        max_lags: Maximum lags to include
        significance: Significance level for stationarity decision

    Returns:
        ADF test result
    """
    from scipy import stats

    n = len(series)
    if n < 20:
        return AugmentedDickeyFullerResult(
            adf_statistic=0.0,
            p_value=1.0,
            lags_used=0,
            n_obs=n,
            critical_values={},
            is_stationary=False
        )

    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, maxlag=max_lags, autolag='AIC')
        return AugmentedDickeyFullerResult(
            adf_statistic=float(result[0]),
            p_value=float(result[1]),
            lags_used=int(result[2]),
            n_obs=int(result[3]),
            critical_values=dict(result[4]),
            is_stationary=result[1] < significance
        )
    except ImportError:
        # Simplified ADF without statsmodels
        # Use basic regression: Δy_t = α + β*y_{t-1} + ε
        y = np.diff(series)
        x = series[:-1]
        _, slope, _ = _simple_ols(y, x)

        # Approximate critical value for no constant/trend
        critical_5pct = -2.86  # Approximate for n > 100

        return AugmentedDickeyFullerResult(
            adf_statistic=slope / 0.1,  # Rough approximation
            p_value=0.5,  # Unknown without tables
            lags_used=0,
            n_obs=n,
            critical_values={'5%': critical_5pct},
            is_stationary=slope < 0 and abs(slope) > 0.1
        )


def _simple_ols(y: NDArray, x: NDArray) -> Tuple[float, float, float]:
    """Simple OLS for testing without numba."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.sum((x - x_mean) * (y - y_mean))
    var = np.sum((x - x_mean) ** 2)
    slope = cov / var if var > EPSILON else 0.0
    intercept = y_mean - slope * x_mean
    return intercept, slope, 0.0


def half_life_from_regression(series: NDArray[np.float64]) -> float:
    """
    Estimate half-life of mean reversion using regression.

    Half-life = -ln(2) / λ

    where λ is from regression: Δy_t = λ * y_{t-1} + ε

    Args:
        series: Time series

    Returns:
        Half-life in periods (days if daily data)
    """
    from trading_algo.quant_core.utils.constants import LN_2, MIN_HALF_LIFE, MAX_HALF_LIFE

    if len(series) < 20:
        return MAX_HALF_LIFE

    y = np.diff(series)
    x = series[:-1]

    _, slope, _ = _simple_ols(y, x)

    # Mean reversion requires negative slope
    if slope >= 0:
        return MAX_HALF_LIFE

    half_life = -LN_2 / slope

    # Clip to reasonable range
    return float(np.clip(half_life, MIN_HALF_LIFE, MAX_HALF_LIFE))


# =============================================================================
# PERFORMANCE CALCULATION
# =============================================================================

def calculate_performance_metrics(
    returns: NDArray[np.float64],
    pnls: Optional[NDArray[np.float64]] = None,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Daily return series
        pnls: Trade P&L series (optional)
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics dataclass
    """
    if len(returns) < 2:
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            var_95=0.0,
            cvar_95=0.0,
        )

    total_ret = float(np.prod(1 + returns) - 1)
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    ann_ret = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol = float(np.std(returns, ddof=1) * SQRT_252)

    # Use pnls if provided, otherwise use returns for trade stats
    trade_pnls = pnls if pnls is not None else returns

    return PerformanceMetrics(
        total_return=total_ret,
        annualized_return=ann_ret,
        volatility=vol,
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate),
        sortino_ratio=sortino_ratio(returns, risk_free_rate),
        calmar_ratio=calmar_ratio(returns),
        max_drawdown=max_drawdown(returns),
        max_drawdown_duration=max_drawdown_duration(returns),
        win_rate=win_rate(trade_pnls),
        profit_factor=profit_factor(trade_pnls),
        expectancy=expectancy(trade_pnls),
        var_95=value_at_risk(returns, 0.95),
        cvar_95=expected_shortfall(returns, 0.95),
    )
