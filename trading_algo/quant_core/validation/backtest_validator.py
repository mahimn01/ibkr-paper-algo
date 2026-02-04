"""
Backtest Validation Framework

Comprehensive validation to detect overfitting and ensure
robust strategy performance.

Key Validation Methods:
    1. Walk-Forward Analysis: Rolling out-of-sample testing
    2. Combinatorial Purged CV: Multiple train/test combinations
    3. Monte Carlo Simulation: Random path testing
    4. Statistical Significance Tests: p-values for performance

Metrics Computed:
    - In-sample vs Out-of-sample performance degradation
    - Strategy stability across market regimes
    - Sensitivity to parameters
    - Statistical significance of returns

References:
    - Bailey, D., et al. (2014). "The Probability of Backtest Overfitting"
    - López de Prado (2018). "Advances in Financial Machine Learning"
    - Harvey, C., et al. (2016). "...and the Cross-Section of Expected Returns"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252, TRADING_DAYS_PER_YEAR
from trading_algo.quant_core.ml.cross_validation import (
    TimeSeriesCV,
    PurgedKFold,
    CombinatorialPurgedCV,
    walk_forward_cv,
)


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = auto()      # Strategy appears robust
    WARNING = auto()     # Some concerns, use caution
    FAILED = auto()      # Strong evidence of overfitting
    INCONCLUSIVE = auto()  # Not enough data


@dataclass
class OverfittingMetrics:
    """Metrics related to overfitting detection."""
    pbo: float                          # Probability of Backtest Overfitting
    deflated_sharpe: float              # Sharpe ratio adjusted for multiple testing
    is_oos_ratio: float                 # In-sample / Out-of-sample Sharpe ratio
    rank_correlation: float             # Rank correlation across CV folds
    parameter_sensitivity: float        # Sensitivity to parameter changes
    regime_stability: float             # Performance stability across regimes
    n_trials: int                       # Number of strategy variations tested
    lookback_bias: float                # Bias from lookback optimization


@dataclass
class ValidationResult:
    """Complete validation result."""
    status: ValidationStatus
    message: str
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    overfitting_metrics: OverfittingMetrics
    cv_results: List[Dict[str, float]]   # Results from each CV fold
    statistical_tests: Dict[str, float]  # p-values from statistical tests
    recommendations: List[str]           # Actionable recommendations


class BacktestValidator:
    """
    Comprehensive backtest validation.

    Performs multiple validation checks to assess strategy robustness
    and detect potential overfitting.

    Usage:
        validator = BacktestValidator()

        # Validate a strategy
        result = validator.validate(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            n_trials=100,  # Number of parameter combinations tested
        )

        if result.status == ValidationStatus.FAILED:
            print("Strategy appears overfit")
    """

    def __init__(
        self,
        n_cv_splits: int = 10,
        purge_length: int = 5,
        embargo_length: int = 1,
        min_obs_per_fold: int = 252,
        significance_level: float = 0.05,
    ):
        """
        Initialize validator.

        Args:
            n_cv_splits: Number of cross-validation splits
            purge_length: Samples to purge between train/test
            embargo_length: Additional embargo period
            min_obs_per_fold: Minimum observations per CV fold
            significance_level: Statistical significance level
        """
        self.n_cv_splits = n_cv_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length
        self.min_obs_per_fold = min_obs_per_fold
        self.significance_level = significance_level

    def validate(
        self,
        returns: NDArray[np.float64],
        benchmark_returns: Optional[NDArray[np.float64]] = None,
        n_trials: int = 1,
        regime_labels: Optional[NDArray[np.int64]] = None,
    ) -> ValidationResult:
        """
        Perform comprehensive validation.

        Args:
            returns: Strategy returns (daily)
            benchmark_returns: Benchmark returns for comparison
            n_trials: Number of parameter/strategy variations tested
            regime_labels: Market regime labels (optional)

        Returns:
            ValidationResult with comprehensive metrics
        """
        n_samples = len(returns)

        # Check minimum data requirement
        min_required = self.n_cv_splits * self.min_obs_per_fold
        if n_samples < min_required:
            return ValidationResult(
                status=ValidationStatus.INCONCLUSIVE,
                message=f"Insufficient data: {n_samples} < {min_required} required",
                in_sample_sharpe=0.0,
                out_of_sample_sharpe=0.0,
                overfitting_metrics=self._empty_metrics(),
                cv_results=[],
                statistical_tests={},
                recommendations=["Collect more data before validating"],
            )

        # 1. Cross-validation analysis
        cv_results = self._cross_validation_analysis(returns)

        # 2. Calculate in-sample and out-of-sample performance
        is_sharpe = self._calculate_sharpe(returns)
        oos_sharpes = [r["test_sharpe"] for r in cv_results if "test_sharpe" in r]
        oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        # 3. Calculate overfitting metrics
        overfitting = self._calculate_overfitting_metrics(
            returns, cv_results, n_trials
        )

        # 4. Statistical tests
        stat_tests = self._statistical_tests(returns, benchmark_returns)

        # 5. Regime analysis (if labels provided)
        if regime_labels is not None:
            regime_stability = self._regime_stability_analysis(returns, regime_labels)
            overfitting.regime_stability = regime_stability

        # 6. Determine validation status
        status, message = self._determine_status(overfitting, stat_tests)

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(
            status, overfitting, stat_tests
        )

        return ValidationResult(
            status=status,
            message=message,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            overfitting_metrics=overfitting,
            cv_results=cv_results,
            statistical_tests=stat_tests,
            recommendations=recommendations,
        )

    def _cross_validation_analysis(
        self,
        returns: NDArray[np.float64],
    ) -> List[Dict[str, float]]:
        """Perform cross-validation analysis."""
        results = []

        # Use Purged K-Fold
        cv = PurgedKFold(
            n_splits=self.n_cv_splits,
            purge_length=self.purge_length,
            embargo_length=self.embargo_length,
        )

        # Create dummy features (just for CV splitting)
        X = np.arange(len(returns)).reshape(-1, 1)

        for split in cv.split(X):
            train_returns = returns[split.train_indices]
            test_returns = returns[split.test_indices]

            train_sharpe = self._calculate_sharpe(train_returns)
            test_sharpe = self._calculate_sharpe(test_returns)

            results.append({
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "train_size": len(train_returns),
                "test_size": len(test_returns),
                "train_return": float(np.mean(train_returns) * TRADING_DAYS_PER_YEAR),
                "test_return": float(np.mean(test_returns) * TRADING_DAYS_PER_YEAR),
            })

        return results

    def _calculate_overfitting_metrics(
        self,
        returns: NDArray[np.float64],
        cv_results: List[Dict[str, float]],
        n_trials: int,
    ) -> OverfittingMetrics:
        """Calculate overfitting-related metrics."""
        if not cv_results:
            return self._empty_metrics()

        # In-sample vs out-of-sample ratio
        is_sharpes = [r["train_sharpe"] for r in cv_results]
        oos_sharpes = [r["test_sharpe"] for r in cv_results]

        is_mean = np.mean(is_sharpes)
        oos_mean = np.mean(oos_sharpes)

        is_oos_ratio = is_mean / (oos_mean + EPSILON) if oos_mean != 0 else float('inf')

        # Rank correlation between IS and OOS performance
        from scipy.stats import spearmanr
        if len(is_sharpes) >= 3:
            rank_corr, _ = spearmanr(is_sharpes, oos_sharpes)
        else:
            rank_corr = 0.0

        # Calculate PBO (simplified)
        # PBO = P(best IS strategy has negative OOS performance)
        n_positive_oos = sum(1 for s in oos_sharpes if s > 0)
        pbo = 1 - (n_positive_oos / len(oos_sharpes)) if oos_sharpes else 0.5

        # Deflated Sharpe Ratio
        observed_sharpe = self._calculate_sharpe(returns)
        deflated = self._deflated_sharpe(observed_sharpe, n_trials, len(returns))

        return OverfittingMetrics(
            pbo=pbo,
            deflated_sharpe=deflated,
            is_oos_ratio=is_oos_ratio,
            rank_correlation=rank_corr if not np.isnan(rank_corr) else 0.0,
            parameter_sensitivity=0.0,  # Requires parameter sweep
            regime_stability=0.0,       # Calculated separately
            n_trials=n_trials,
            lookback_bias=max(0, is_mean - oos_mean),
        )

    def _deflated_sharpe(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio.

        Adjusts for multiple testing using the expected maximum
        Sharpe ratio under the null hypothesis.

        From Bailey & López de Prado (2014).
        """
        from scipy.stats import norm

        if n_trials <= 1:
            return observed_sharpe

        # Expected maximum Sharpe under null (random)
        # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * log(n)) for large n
        expected_max = np.sqrt(2 * np.log(n_trials))

        # Variance of Sharpe ratio estimator
        # Var(SR) ≈ (1 + 0.5 * SR^2) / T
        var_sr = (1 + 0.5 * observed_sharpe**2) / n_observations
        std_sr = np.sqrt(var_sr)

        # Deflated Sharpe = (Observed - Expected) / std
        if std_sr > EPSILON:
            deflated = (observed_sharpe - expected_max * std_sr) / std_sr
            # Convert to probability
            p_value = 1 - norm.cdf(deflated)
            # Return adjusted Sharpe (original * (1 - p_value))
            return observed_sharpe * (1 - p_value)
        return observed_sharpe

    def _statistical_tests(
        self,
        returns: NDArray[np.float64],
        benchmark_returns: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Perform statistical significance tests."""
        from scipy import stats

        tests = {}

        # Test 1: Is mean return significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        tests["mean_return_pvalue"] = p_value

        # Test 2: Normality test (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        tests["normality_pvalue"] = jb_pvalue

        # Test 3: Autocorrelation test
        # Simple Durbin-Watson approximation
        diffs = np.diff(returns)
        dw_stat = np.sum(diffs**2) / np.sum(returns**2)
        tests["durbin_watson"] = dw_stat

        # Test 4: If benchmark provided, test for alpha significance
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Simple OLS regression for alpha/beta
            X = np.column_stack([np.ones(len(returns)), benchmark_returns])
            try:
                beta = np.linalg.lstsq(X, returns, rcond=None)[0]
                residuals = returns - X @ beta
                alpha = beta[0]

                # t-stat for alpha
                se = np.sqrt(np.sum(residuals**2) / (len(returns) - 2))
                se_alpha = se / np.sqrt(len(returns))
                t_alpha = alpha / se_alpha if se_alpha > EPSILON else 0
                p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), len(returns) - 2))

                tests["alpha_pvalue"] = p_alpha
                tests["alpha"] = float(alpha * TRADING_DAYS_PER_YEAR)
            except np.linalg.LinAlgError:
                pass

        return tests

    def _regime_stability_analysis(
        self,
        returns: NDArray[np.float64],
        regime_labels: NDArray[np.int64],
    ) -> float:
        """Analyze performance stability across regimes."""
        unique_regimes = np.unique(regime_labels)

        if len(unique_regimes) < 2:
            return 1.0  # Single regime, no stability to measure

        regime_sharpes = []
        for regime in unique_regimes:
            mask = regime_labels == regime
            if np.sum(mask) >= 20:  # Minimum samples
                regime_returns = returns[mask]
                sharpe = self._calculate_sharpe(regime_returns)
                regime_sharpes.append(sharpe)

        if len(regime_sharpes) < 2:
            return 1.0

        # Stability = 1 - coefficient of variation of Sharpe across regimes
        mean_sharpe = np.mean(regime_sharpes)
        std_sharpe = np.std(regime_sharpes)

        if abs(mean_sharpe) < EPSILON:
            return 0.0

        cv = std_sharpe / abs(mean_sharpe)
        stability = max(0.0, 1 - cv)

        return stability

    def _calculate_sharpe(self, returns: NDArray[np.float64]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret < EPSILON:
            return 0.0

        return float(mean_ret / std_ret * SQRT_252)

    def _determine_status(
        self,
        overfitting: OverfittingMetrics,
        stat_tests: Dict[str, float],
    ) -> Tuple[ValidationStatus, str]:
        """Determine overall validation status."""
        issues = []

        # Check PBO
        if overfitting.pbo > 0.5:
            issues.append("High probability of backtest overfitting")

        # Check IS/OOS ratio
        if overfitting.is_oos_ratio > 2.0:
            issues.append("Large in-sample/out-of-sample performance gap")

        # Check deflated Sharpe
        if overfitting.deflated_sharpe < 0:
            issues.append("Deflated Sharpe ratio is negative")

        # Check statistical significance
        if stat_tests.get("mean_return_pvalue", 1.0) > self.significance_level:
            issues.append("Mean return not statistically significant")

        # Determine status
        if len(issues) >= 3:
            return ValidationStatus.FAILED, "; ".join(issues)
        elif len(issues) >= 1:
            return ValidationStatus.WARNING, "; ".join(issues)
        else:
            return ValidationStatus.PASSED, "Strategy passes validation checks"

    def _generate_recommendations(
        self,
        status: ValidationStatus,
        overfitting: OverfittingMetrics,
        stat_tests: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if overfitting.pbo > 0.3:
            recommendations.append(
                "Reduce number of parameter combinations tested"
            )

        if overfitting.is_oos_ratio > 1.5:
            recommendations.append(
                "Simplify strategy to reduce overfitting"
            )

        if stat_tests.get("mean_return_pvalue", 1.0) > 0.05:
            recommendations.append(
                "Collect more data or improve signal quality"
            )

        if overfitting.regime_stability < 0.5:
            recommendations.append(
                "Strategy may not be robust across market regimes"
            )

        if status == ValidationStatus.PASSED:
            recommendations.append(
                "Consider paper trading before live deployment"
            )

        return recommendations

    def _empty_metrics(self) -> OverfittingMetrics:
        """Return empty overfitting metrics."""
        return OverfittingMetrics(
            pbo=0.5,
            deflated_sharpe=0.0,
            is_oos_ratio=1.0,
            rank_correlation=0.0,
            parameter_sensitivity=0.0,
            regime_stability=0.0,
            n_trials=1,
            lookback_bias=0.0,
        )


def monte_carlo_permutation_test(
    returns: NDArray[np.float64],
    n_simulations: int = 1000,
    statistic_fn: Optional[Callable] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo permutation test for strategy significance.

    Tests whether the strategy performance could have been
    achieved by chance.

    Args:
        returns: Strategy returns
        n_simulations: Number of random permutations
        statistic_fn: Statistic to test (default: Sharpe ratio)

    Returns:
        Tuple of (observed_statistic, p_value)
    """
    if statistic_fn is None:
        def statistic_fn(r):
            if len(r) < 2 or np.std(r) < EPSILON:
                return 0.0
            return np.mean(r) / np.std(r) * SQRT_252

    observed = statistic_fn(returns)

    # Generate permuted statistics
    permuted_stats = np.zeros(n_simulations)
    for i in range(n_simulations):
        permuted = np.random.permutation(returns)
        permuted_stats[i] = statistic_fn(permuted)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed))

    return float(observed), float(p_value)
