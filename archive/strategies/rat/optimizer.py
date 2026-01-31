"""
Walk-Forward Optimization for RAT Framework

Implements proper out-of-sample validation to avoid overfitting:
1. Rolling window optimization
2. Parameter stability analysis
3. Cross-validation across regimes
4. Performance degradation detection

This is critical for real-world trading - backtest results without
proper out-of-sample testing are meaningless.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Any
from enum import Enum, auto


@dataclass
class ParameterSpec:
    """Specification for an optimizable parameter."""
    name: str
    min_value: float
    max_value: float
    step: float
    current: float

    def get_search_space(self) -> List[float]:
        """Get all possible values in search space."""
        values = []
        v = self.min_value
        while v <= self.max_value:
            values.append(round(v, 6))
            v += self.step
        return values


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""
    parameters: Dict[str, float]
    in_sample_sharpe: float
    in_sample_return: float
    in_sample_drawdown: float
    out_sample_sharpe: float
    out_sample_return: float
    out_sample_drawdown: float
    degradation: float  # IS Sharpe vs OOS Sharpe
    is_overfitting: bool

    @property
    def is_robust(self) -> bool:
        """Check if results are robust (not overfitted)."""
        return (
            not self.is_overfitting and
            self.degradation < 0.5 and
            self.out_sample_sharpe > 0.3
        )


@dataclass
class WalkForwardResult:
    """Result from walk-forward optimization."""
    periods: List[OptimizationResult]
    avg_is_sharpe: float
    avg_oos_sharpe: float
    avg_degradation: float
    parameter_stability: Dict[str, float]  # Std dev of each parameter
    best_parameters: Dict[str, float]
    is_strategy_robust: bool


class ObjectiveFunction(Enum):
    """Optimization objective functions."""
    SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    CALMAR_RATIO = auto()
    PROFIT_FACTOR = auto()
    RISK_ADJUSTED_RETURN = auto()


class WalkForwardOptimizer:
    """
    Walk-forward optimization with out-of-sample validation.

    Splits data into in-sample (IS) and out-of-sample (OOS) periods,
    optimizes on IS, validates on OOS, then rolls forward.
    """

    def __init__(
        self,
        backtest_fn: Callable[[Dict[str, float], List[Dict]], Dict],
        parameters: List[ParameterSpec],
        objective: ObjectiveFunction = ObjectiveFunction.SHARPE_RATIO,
        in_sample_pct: float = 0.7,
        min_trades_for_validity: int = 30,
    ):
        """
        Args:
            backtest_fn: Function that runs backtest with params and returns metrics
            parameters: List of parameters to optimize
            objective: What to optimize for
            in_sample_pct: Percentage of data for in-sample
            min_trades_for_validity: Minimum trades for valid result
        """
        self.backtest_fn = backtest_fn
        self.parameters = {p.name: p for p in parameters}
        self.objective = objective
        self.in_sample_pct = in_sample_pct
        self.min_trades = min_trades_for_validity

        # Results storage
        self._optimization_history: List[OptimizationResult] = []

    def optimize(
        self,
        data: List[Dict],
        n_periods: int = 4,
        method: str = "grid",
        n_iterations: int = 100,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            data: List of bar data dictionaries
            n_periods: Number of walk-forward periods
            method: "grid" for grid search, "random" for random search
            n_iterations: For random search, number of iterations

        Returns:
            WalkForwardResult with all metrics
        """
        n = len(data)
        period_size = n // n_periods
        results = []

        for i in range(n_periods):
            # Define in-sample and out-of-sample periods
            start_idx = i * period_size
            end_idx = start_idx + period_size
            if i == n_periods - 1:
                end_idx = n  # Use all remaining data for last period

            period_data = data[start_idx:end_idx]

            # Split into IS and OOS
            is_end = int(len(period_data) * self.in_sample_pct)
            is_data = period_data[:is_end]
            oos_data = period_data[is_end:]

            if len(is_data) < 50 or len(oos_data) < 20:
                continue

            # Optimize on in-sample
            if method == "grid":
                best_params, is_metrics = self._grid_search(is_data)
            else:
                best_params, is_metrics = self._random_search(is_data, n_iterations)

            # Validate on out-of-sample
            oos_metrics = self.backtest_fn(best_params, oos_data)

            # Calculate degradation
            is_sharpe = is_metrics.get('sharpe_ratio', 0)
            oos_sharpe = oos_metrics.get('sharpe_ratio', 0)
            degradation = 1 - (oos_sharpe / is_sharpe) if is_sharpe > 0 else 1.0

            # Check for overfitting
            is_overfitting = (
                degradation > 0.7 or
                (is_sharpe > 1.0 and oos_sharpe < 0) or
                oos_metrics.get('total_trades', 0) < self.min_trades * 0.3
            )

            result = OptimizationResult(
                parameters=best_params,
                in_sample_sharpe=is_sharpe,
                in_sample_return=is_metrics.get('total_return', 0),
                in_sample_drawdown=is_metrics.get('max_drawdown', 0),
                out_sample_sharpe=oos_sharpe,
                out_sample_return=oos_metrics.get('total_return', 0),
                out_sample_drawdown=oos_metrics.get('max_drawdown', 0),
                degradation=degradation,
                is_overfitting=is_overfitting,
            )

            results.append(result)
            self._optimization_history.append(result)

        if not results:
            return self._create_empty_result()

        # Aggregate results
        return self._aggregate_results(results)

    def _grid_search(
        self,
        data: List[Dict],
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Grid search over parameter space.

        For small parameter spaces. Exponentially expensive.
        """
        from itertools import product

        # Build search space
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name].get_search_space() for name in param_names]

        best_objective = float('-inf')
        best_params = {name: self.parameters[name].current for name in param_names}
        best_metrics = {}

        # Limit iterations for large spaces
        total_combos = 1
        for values in param_values:
            total_combos *= len(values)

        if total_combos > 10000:
            # Fall back to random search
            return self._random_search(data, 1000)

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            try:
                metrics = self.backtest_fn(params, data)
            except Exception:
                continue

            if metrics.get('total_trades', 0) < self.min_trades:
                continue

            obj_value = self._compute_objective(metrics)

            if obj_value > best_objective:
                best_objective = obj_value
                best_params = params.copy()
                best_metrics = metrics

        return best_params, best_metrics

    def _random_search(
        self,
        data: List[Dict],
        n_iterations: int,
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Random search over parameter space.

        More efficient for large parameter spaces.
        """
        best_objective = float('-inf')
        best_params = {name: spec.current for name, spec in self.parameters.items()}
        best_metrics = {}

        for _ in range(n_iterations):
            # Random parameters
            params = {}
            for name, spec in self.parameters.items():
                values = spec.get_search_space()
                params[name] = random.choice(values)

            try:
                metrics = self.backtest_fn(params, data)
            except Exception:
                continue

            if metrics.get('total_trades', 0) < self.min_trades:
                continue

            obj_value = self._compute_objective(metrics)

            if obj_value > best_objective:
                best_objective = obj_value
                best_params = params.copy()
                best_metrics = metrics

        return best_params, best_metrics

    def _compute_objective(self, metrics: Dict) -> float:
        """Compute objective value from metrics."""
        if self.objective == ObjectiveFunction.SHARPE_RATIO:
            return metrics.get('sharpe_ratio', 0)

        elif self.objective == ObjectiveFunction.SORTINO_RATIO:
            return metrics.get('sortino_ratio', 0)

        elif self.objective == ObjectiveFunction.CALMAR_RATIO:
            ret = metrics.get('total_return', 0)
            dd = abs(metrics.get('max_drawdown', 1))
            return ret / dd if dd > 0 else 0

        elif self.objective == ObjectiveFunction.PROFIT_FACTOR:
            return metrics.get('profit_factor', 0)

        elif self.objective == ObjectiveFunction.RISK_ADJUSTED_RETURN:
            ret = metrics.get('total_return', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            dd = abs(metrics.get('max_drawdown', 1))
            # Combine metrics
            return (ret * 0.3 + sharpe * 0.5 - dd * 0.2) if sharpe > 0 else -1

        return 0

    def _aggregate_results(
        self,
        results: List[OptimizationResult],
    ) -> WalkForwardResult:
        """Aggregate walk-forward results."""
        n = len(results)

        avg_is_sharpe = sum(r.in_sample_sharpe for r in results) / n
        avg_oos_sharpe = sum(r.out_sample_sharpe for r in results) / n
        avg_degradation = sum(r.degradation for r in results) / n

        # Parameter stability (lower std = more stable)
        param_stability = {}
        for param_name in self.parameters:
            values = [r.parameters[param_name] for r in results]
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                std_val = math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
                # Normalize by parameter range
                spec = self.parameters[param_name]
                range_size = spec.max_value - spec.min_value
                param_stability[param_name] = std_val / range_size if range_size > 0 else 0
            else:
                param_stability[param_name] = 0

        # Best parameters (weighted by OOS performance)
        best_params = {}
        total_weight = sum(max(0.01, r.out_sample_sharpe) for r in results)

        for param_name in self.parameters:
            weighted_sum = sum(
                r.parameters[param_name] * max(0.01, r.out_sample_sharpe)
                for r in results
            )
            best_params[param_name] = weighted_sum / total_weight if total_weight > 0 else \
                self.parameters[param_name].current

        # Is strategy robust?
        robust_periods = sum(1 for r in results if r.is_robust)
        is_robust = (
            robust_periods >= n * 0.6 and  # At least 60% of periods robust
            avg_degradation < 0.5 and       # Average degradation < 50%
            avg_oos_sharpe > 0.3             # Positive OOS Sharpe
        )

        return WalkForwardResult(
            periods=results,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_degradation=avg_degradation,
            parameter_stability=param_stability,
            best_parameters=best_params,
            is_strategy_robust=is_robust,
        )

    def _create_empty_result(self) -> WalkForwardResult:
        """Create empty result when optimization fails."""
        return WalkForwardResult(
            periods=[],
            avg_is_sharpe=0,
            avg_oos_sharpe=0,
            avg_degradation=1.0,
            parameter_stability={},
            best_parameters={name: spec.current for name, spec in self.parameters.items()},
            is_strategy_robust=False,
        )


# =============================================================================
# PARAMETER SPECIFICATIONS FOR RAT FRAMEWORK
# =============================================================================

def get_rat_parameters() -> List[ParameterSpec]:
    """
    Get default parameter specifications for RAT framework.

    Ranges based on academic research and industry practice.
    """
    return [
        # Momentum parameters (Jegadeesh & Titman suggest 3-12 months)
        ParameterSpec("momentum_short", min_value=3, max_value=10, step=1, current=5),
        ParameterSpec("momentum_long", min_value=15, max_value=60, step=5, current=20),

        # RSI parameters (Wilder: 14 is standard)
        ParameterSpec("rsi_period", min_value=7, max_value=21, step=2, current=14),
        ParameterSpec("rsi_overbought", min_value=65, max_value=80, step=5, current=70),
        ParameterSpec("rsi_oversold", min_value=20, max_value=35, step=5, current=30),

        # Mean reversion (Bollinger: 20 period, 2 std)
        ParameterSpec("bollinger_period", min_value=15, max_value=30, step=5, current=20),
        ParameterSpec("bollinger_std", min_value=1.5, max_value=3.0, step=0.25, current=2.0),

        # Position sizing
        ParameterSpec("position_size_pct", min_value=0.02, max_value=0.10, step=0.01, current=0.05),

        # Signal thresholds
        ParameterSpec("confidence_threshold", min_value=0.3, max_value=0.7, step=0.1, current=0.5),
        ParameterSpec("momentum_threshold", min_value=0.01, max_value=0.05, step=0.01, current=0.02),
    ]


def print_optimization_report(result: WalkForwardResult) -> None:
    """Print formatted optimization report."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD OPTIMIZATION REPORT")
    print("=" * 70)

    print(f"\nStrategy Robust: {'YES ✓' if result.is_strategy_robust else 'NO ✗'}")

    print("\n📊 AGGREGATE METRICS")
    print("-" * 40)
    print(f"  Avg In-Sample Sharpe:    {result.avg_is_sharpe:>10.3f}")
    print(f"  Avg Out-of-Sample Sharpe: {result.avg_oos_sharpe:>10.3f}")
    print(f"  Avg Degradation:         {result.avg_degradation:>10.1%}")

    print("\n📈 BEST PARAMETERS")
    print("-" * 40)
    for name, value in result.best_parameters.items():
        stability = result.parameter_stability.get(name, 0)
        stability_indicator = "●" if stability < 0.2 else "◐" if stability < 0.4 else "○"
        print(f"  {name:<25} {value:>10.3f}  {stability_indicator}")

    print("\n🔍 PERIOD BREAKDOWN")
    print("-" * 40)
    for i, period in enumerate(result.periods):
        status = "✓" if period.is_robust else "✗"
        print(f"  Period {i+1}: IS={period.in_sample_sharpe:.2f} "
              f"OOS={period.out_sample_sharpe:.2f} "
              f"Deg={period.degradation:.1%} {status}")

    print("\n" + "=" * 70)
