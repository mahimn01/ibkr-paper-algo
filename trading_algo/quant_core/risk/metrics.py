"""
Comprehensive Risk Metrics Calculator

Provides a unified interface for calculating all risk metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Optional

from trading_algo.quant_core.utils.statistics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    value_at_risk,
    expected_shortfall,
)
from trading_algo.quant_core.utils.constants import SQRT_252, TRADING_DAYS_PER_YEAR


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Tail risk
    var_95: float
    var_99: float
    es_95: float
    es_99: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # Distribution
    skewness: float
    kurtosis: float

    # Other
    beta: float
    alpha: float
    tracking_error: float

    @classmethod
    def calculate(
        cls,
        returns: NDArray[np.float64],
        benchmark_returns: Optional[NDArray[np.float64]] = None,
        risk_free_rate: float = 0.0,
    ) -> "RiskMetrics":
        """Calculate all risk metrics from returns."""
        from scipy import stats

        n = len(returns)
        if n < 2:
            return cls._empty()

        # Basic statistics
        total_ret = float(np.prod(1 + returns) - 1)
        ann_ret = float((1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n) - 1)
        vol = float(np.std(returns, ddof=1) * SQRT_252)

        # Risk-adjusted
        sr = sharpe_ratio(returns, risk_free_rate)
        sortino = sortino_ratio(returns, risk_free_rate)
        calmar = calmar_ratio(returns)

        # Tail risk
        var95 = value_at_risk(returns, 0.95)
        var99 = value_at_risk(returns, 0.99)
        es95 = expected_shortfall(returns, 0.95)
        es99 = expected_shortfall(returns, 0.99)

        # Drawdown
        max_dd = max_drawdown(returns)
        dd_series = cls._drawdown_series(returns)
        avg_dd = float(np.mean(dd_series[dd_series > 0])) if np.any(dd_series > 0) else 0.0
        max_dd_duration = cls._max_dd_duration(dd_series)

        # Distribution
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))

        # Benchmark-relative
        if benchmark_returns is not None and len(benchmark_returns) == n:
            beta, alpha, ir, te = cls._benchmark_metrics(returns, benchmark_returns, risk_free_rate)
        else:
            beta, alpha, ir, te = 1.0, 0.0, 0.0, 0.0

        return cls(
            total_return=total_ret,
            annualized_return=ann_ret,
            volatility=vol,
            sharpe_ratio=sr,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=ir,
            var_95=var95,
            var_99=var99,
            es_95=es95,
            es_99=es99,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            skewness=skew,
            kurtosis=kurt,
            beta=beta,
            alpha=alpha,
            tracking_error=te,
        )

    @staticmethod
    def _drawdown_series(returns: NDArray) -> NDArray:
        """Calculate drawdown at each point."""
        cum_return = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_return)
        return 1 - cum_return / running_max

    @staticmethod
    def _max_dd_duration(dd_series: NDArray) -> int:
        """Calculate max drawdown duration."""
        max_dur = 0
        cur_dur = 0
        for dd in dd_series:
            if dd > 0:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    @staticmethod
    def _benchmark_metrics(
        returns: NDArray,
        benchmark: NDArray,
        rf: float,
    ) -> tuple:
        """Calculate benchmark-relative metrics."""
        # Beta
        cov = np.cov(returns, benchmark)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

        # Alpha (Jensen's)
        mean_ret = np.mean(returns) * TRADING_DAYS_PER_YEAR
        mean_bench = np.mean(benchmark) * TRADING_DAYS_PER_YEAR
        alpha = mean_ret - rf - beta * (mean_bench - rf)

        # Tracking error and IR
        excess = returns - benchmark
        te = np.std(excess, ddof=1) * SQRT_252
        ir = np.mean(excess) * SQRT_252 / te if te > 0 else 0.0

        return float(beta), float(alpha), float(ir), float(te)

    @classmethod
    def _empty(cls) -> "RiskMetrics":
        """Return empty metrics."""
        return cls(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            information_ratio=0.0, var_95=0.0, var_99=0.0,
            es_95=0.0, es_99=0.0, max_drawdown=0.0, avg_drawdown=0.0,
            max_drawdown_duration=0, skewness=0.0, kurtosis=0.0,
            beta=1.0, alpha=0.0, tracking_error=0.0,
        )
