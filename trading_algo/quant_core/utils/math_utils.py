"""
High-performance mathematical utilities using NumPy and Numba.

All computationally intensive operations are JIT-compiled for near-C performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# RETURNS CALCULATIONS
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def log_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate log returns from price series.

    log_return = ln(P_t / P_{t-1})

    Args:
        prices: Array of prices

    Returns:
        Array of log returns (length = len(prices) - 1)
    """
    n = len(prices)
    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        if prices[i - 1] > 0:
            returns[i - 1] = np.log(prices[i] / prices[i - 1])
        else:
            returns[i - 1] = 0.0
    return returns


@jit(nopython=True, cache=True, fastmath=True)
def simple_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate simple returns from price series.

    simple_return = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices: Array of prices

    Returns:
        Array of simple returns (length = len(prices) - 1)
    """
    n = len(prices)
    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        if prices[i - 1] > 0:
            returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1]
        else:
            returns[i - 1] = 0.0
    return returns


# =============================================================================
# ROLLING WINDOW CALCULATIONS
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def rolling_mean(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Calculate rolling mean with O(n) complexity.

    Uses cumulative sum approach for efficiency.

    Args:
        arr: Input array
        window: Rolling window size

    Returns:
        Rolling mean array (NaN-padded at start)
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    # First window
    window_sum = 0.0
    for i in range(window):
        window_sum += arr[i]
    result[window - 1] = window_sum / window

    # Subsequent windows using sliding
    for i in range(window, n):
        window_sum = window_sum - arr[i - window] + arr[i]
        result[i] = window_sum / window

    return result


@jit(nopython=True, cache=True, fastmath=True)
def rolling_std(arr: NDArray[np.float64], window: int, ddof: int = 1) -> NDArray[np.float64]:
    """
    Calculate rolling standard deviation using Welford's algorithm.

    Numerically stable one-pass algorithm.

    Args:
        arr: Input array
        window: Rolling window size
        ddof: Delta degrees of freedom (0 for population, 1 for sample)

    Returns:
        Rolling std array (NaN-padded at start)
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        # Calculate mean and variance for this window
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += arr[j]
        mean /= window

        variance = 0.0
        for j in range(i - window + 1, i + 1):
            variance += (arr[j] - mean) ** 2

        if window > ddof:
            variance /= (window - ddof)
            result[i] = np.sqrt(variance) if variance > 0 else 0.0

    return result


@jit(nopython=True, cache=True, fastmath=True)
def exponential_moving_average(
    arr: NDArray[np.float64],
    span: int
) -> NDArray[np.float64]:
    """
    Calculate exponential moving average.

    EMA_t = α * x_t + (1 - α) * EMA_{t-1}
    where α = 2 / (span + 1)

    Args:
        arr: Input array
        span: EMA span

    Returns:
        EMA array
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (span + 1)

    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]

    return result


# =============================================================================
# VOLATILITY CALCULATIONS
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def realized_volatility(
    returns: NDArray[np.float64],
    window: int,
    annualize: bool = True
) -> NDArray[np.float64]:
    """
    Calculate realized volatility (standard deviation of returns).

    σ_realized = sqrt(sum(r_i^2) / n) * sqrt(252) for annualized

    Args:
        returns: Return series
        window: Rolling window
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        Realized volatility array
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    sqrt_252 = np.sqrt(252.0) if annualize else 1.0

    for i in range(window - 1, n):
        sum_sq = 0.0
        for j in range(i - window + 1, i + 1):
            sum_sq += returns[j] ** 2
        result[i] = np.sqrt(sum_sq / window) * sqrt_252

    return result


@jit(nopython=True, cache=True, fastmath=True)
def ewma_volatility(
    returns: NDArray[np.float64],
    decay: float = 0.94  # RiskMetrics default
) -> NDArray[np.float64]:
    """
    Calculate EWMA volatility (RiskMetrics approach).

    σ²_t = λ * σ²_{t-1} + (1 - λ) * r²_{t-1}

    Args:
        returns: Return series
        decay: Decay factor (λ), typically 0.94 for daily

    Returns:
        EWMA volatility array (annualized)
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    sqrt_252 = np.sqrt(252.0)

    # Initialize with first return squared
    variance = returns[0] ** 2
    result[0] = np.sqrt(variance) * sqrt_252

    for i in range(1, n):
        variance = decay * variance + (1 - decay) * returns[i - 1] ** 2
        result[i] = np.sqrt(variance) * sqrt_252

    return result


@jit(nopython=True, cache=True, fastmath=True)
def garman_klass_volatility(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    open_: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Calculate Garman-Klass volatility estimator.

    More efficient than close-to-close volatility using OHLC data.
    σ²_GK = 0.5 * (ln(H/L))² - (2ln(2) - 1) * (ln(C/O))²

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices
        window: Rolling window

    Returns:
        Garman-Klass volatility (annualized)
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    sqrt_252 = np.sqrt(252.0)

    if n < window:
        return result

    # Calculate daily GK variance
    gk_var = np.empty(n, dtype=np.float64)
    for i in range(n):
        if low[i] > 0 and open_[i] > 0:
            log_hl = np.log(high[i] / low[i])
            log_co = np.log(close[i] / open_[i])
            gk_var[i] = 0.5 * log_hl ** 2 - (2 * np.log(2.0) - 1) * log_co ** 2
        else:
            gk_var[i] = 0.0

    # Rolling mean of GK variance
    for i in range(window - 1, n):
        sum_var = 0.0
        for j in range(i - window + 1, i + 1):
            sum_var += gk_var[j]
        result[i] = np.sqrt(sum_var / window) * sqrt_252

    return result


# =============================================================================
# LINEAR ALGEBRA UTILITIES
# =============================================================================

def robust_covariance(
    returns: NDArray[np.float64],
    shrinkage: float = 0.1
) -> NDArray[np.float64]:
    """
    Calculate shrinkage covariance matrix (Ledoit-Wolf style).

    Σ_shrunk = (1 - α) * Σ_sample + α * F
    where F is a structured target (scaled identity)

    Args:
        returns: Return matrix (T x N)
        shrinkage: Shrinkage intensity (0 to 1)

    Returns:
        Shrinkage covariance matrix (N x N)
    """
    sample_cov = np.cov(returns, rowvar=False)

    # Target: scaled identity matrix
    n = sample_cov.shape[0]
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)

    # Shrinkage
    shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target

    return shrunk_cov


def correlation_to_distance(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert correlation matrix to distance matrix.

    d_ij = sqrt(0.5 * (1 - ρ_ij))

    Used for hierarchical clustering in HRP.

    Args:
        corr: Correlation matrix

    Returns:
        Distance matrix
    """
    return np.sqrt(0.5 * (1 - corr))


# =============================================================================
# REGRESSION UTILITIES
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def ols_regression(
    y: NDArray[np.float64],
    x: NDArray[np.float64]
) -> Tuple[float, float, float]:
    """
    Simple OLS regression: y = α + β*x + ε

    Returns coefficients using closed-form solution.

    Args:
        y: Dependent variable
        x: Independent variable

    Returns:
        Tuple of (intercept, slope, r_squared)
    """
    n = len(y)

    # Calculate means
    x_mean = 0.0
    y_mean = 0.0
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    # Calculate covariance and variance
    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        x_diff = x[i] - x_mean
        cov_xy += x_diff * (y[i] - y_mean)
        var_x += x_diff ** 2

    # Calculate coefficients
    if var_x > 1e-10:
        slope = cov_xy / var_x
    else:
        slope = 0.0
    intercept = y_mean - slope * x_mean

    # Calculate R-squared
    ss_tot = 0.0
    ss_res = 0.0
    for i in range(n):
        y_pred = intercept + slope * x[i]
        ss_tot += (y[i] - y_mean) ** 2
        ss_res += (y[i] - y_pred) ** 2

    if ss_tot > 1e-10:
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = 0.0

    return intercept, slope, r_squared


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def zscore(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Calculate rolling z-score.

    z = (x - μ) / σ

    Args:
        arr: Input array
        window: Rolling window for mean and std

    Returns:
        Z-score array
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        # Calculate mean
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += arr[j]
        mean /= window

        # Calculate std
        variance = 0.0
        for j in range(i - window + 1, i + 1):
            variance += (arr[j] - mean) ** 2
        variance /= (window - 1)
        std = np.sqrt(variance) if variance > 0 else 1e-10

        result[i] = (arr[i] - mean) / std

    return result


@jit(nopython=True, cache=True, fastmath=True)
def percentile_rank(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Calculate rolling percentile rank.

    Returns value between 0 and 100.

    Args:
        arr: Input array
        window: Rolling window

    Returns:
        Percentile rank array
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        current = arr[i]
        count_below = 0
        for j in range(i - window + 1, i):  # Exclude current value
            if arr[j] < current:
                count_below += 1
        result[i] = (count_below / (window - 1)) * 100

    return result


def clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))
