"""
Expected Shortfall (Conditional Value at Risk)

Expected Shortfall is the expected loss given that loss exceeds VaR.
It is a coherent risk measure (subadditive) and captures tail severity.

Definition:
    ES_α = E[L | L > VaR_α] = (1/(1-α)) ∫_α^1 VaR_u du

where:
    - α: Confidence level (e.g., 0.95 for 95%)
    - VaR_α: Value at Risk at confidence α
    - L: Loss (negative return)

Key Properties:
    - Coherent: Satisfies subadditivity (diversification reduces risk)
    - Tail-sensitive: Accounts for severity of extreme losses
    - Required under Basel III FRTB for market risk capital

Calculation Methods:
    1. Historical Simulation: Average of losses beyond VaR
    2. Parametric: Assumes normal distribution
    3. Monte Carlo: Simulated distribution
    4. Cornish-Fisher: Accounts for skewness and kurtosis

References:
    - Rockafellar & Uryasev (2000): "Optimization of Conditional Value-at-Risk"
    - https://blog.quantinsti.com/cvar-expected-shortfall/
    - https://www.pyquantnews.com/free-python-resources/risk-metrics-in-python-var-and-cvar-guide
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import (
    VAR_CONFIDENCE,
    ES_CONFIDENCE,
    EPSILON,
    TRADING_DAYS_PER_YEAR,
    SQRT_252,
)


class ESMethod(Enum):
    """Expected Shortfall calculation methods."""
    HISTORICAL = auto()     # Historical simulation
    PARAMETRIC = auto()     # Assumes normal distribution
    CORNISH_FISHER = auto() # Adjusts for skew/kurtosis
    MONTE_CARLO = auto()    # Simulation-based


@dataclass
class RiskMeasures:
    """Container for VaR and ES measures."""
    var_95: float           # 95% VaR
    var_99: float           # 99% VaR
    es_95: float            # 95% Expected Shortfall
    es_99: float            # 99% Expected Shortfall
    max_loss: float         # Maximum historical loss
    skewness: float         # Return distribution skewness
    kurtosis: float         # Return distribution kurtosis
    tail_ratio: float       # ES_95 / VaR_95 (measures tail heaviness)


@dataclass
class ESBreachEvent:
    """Record of an ES breach event."""
    date: str
    loss: float
    es_level: float
    var_level: float
    severity_ratio: float  # loss / ES


class ExpectedShortfall:
    """
    Expected Shortfall (CVaR) calculator and monitor.

    Calculates ES at various confidence levels and monitors
    for tail risk events.

    Usage:
        es = ExpectedShortfall(confidence=0.95)
        es.fit(returns)

        # Get current ES
        current_es = es.calculate(recent_returns)

        # Check for breach
        if es.check_breach(today_loss):
            trigger_risk_response()
    """

    def __init__(
        self,
        confidence: float = ES_CONFIDENCE,
        method: ESMethod = ESMethod.HISTORICAL,
        lookback: int = 252,
        decay_factor: Optional[float] = None,  # For weighted historical
    ):
        """
        Initialize ES calculator.

        Args:
            confidence: Confidence level (0.95 = 95%)
            method: Calculation method
            lookback: Historical window for estimation
            decay_factor: Exponential decay for weighted historical
        """
        self.confidence = confidence
        self.method = method
        self.lookback = lookback
        self.decay_factor = decay_factor

        # State
        self._returns_history: List[float] = []
        self._breach_events: List[ESBreachEvent] = []
        self._current_es: float = 0.0
        self._current_var: float = 0.0

    def calculate_var(
        self,
        returns: NDArray[np.float64],
        confidence: Optional[float] = None,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Return series
            confidence: Override confidence level

        Returns:
            VaR as positive number (potential loss)
        """
        confidence = confidence or self.confidence

        if len(returns) < 10:
            return 0.0

        if self.method == ESMethod.PARAMETRIC:
            from scipy import stats
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            var = -(mean - stats.norm.ppf(confidence) * std)

        elif self.method == ESMethod.CORNISH_FISHER:
            var = self._cornish_fisher_var(returns, confidence)

        else:  # HISTORICAL or MONTE_CARLO
            if self.decay_factor is not None:
                # Weighted historical
                var = self._weighted_var(returns, confidence)
            else:
                var = -np.percentile(returns, (1 - confidence) * 100)

        return max(0.0, float(var))

    def calculate_es(
        self,
        returns: NDArray[np.float64],
        confidence: Optional[float] = None,
    ) -> float:
        """
        Calculate Expected Shortfall.

        ES = E[L | L > VaR] = average of losses exceeding VaR

        Args:
            returns: Return series
            confidence: Override confidence level

        Returns:
            ES as positive number
        """
        confidence = confidence or self.confidence

        if len(returns) < 10:
            return 0.0

        if self.method == ESMethod.PARAMETRIC:
            es = self._parametric_es(returns, confidence)

        elif self.method == ESMethod.CORNISH_FISHER:
            es = self._cornish_fisher_es(returns, confidence)

        elif self.method == ESMethod.MONTE_CARLO:
            es = self._monte_carlo_es(returns, confidence)

        else:  # HISTORICAL
            var_threshold = -self.calculate_var(returns, confidence)
            tail_losses = returns[returns <= var_threshold]

            if len(tail_losses) == 0:
                return self.calculate_var(returns, confidence)

            es = -np.mean(tail_losses)

        self._current_es = max(0.0, float(es))
        return self._current_es

    def _parametric_es(
        self,
        returns: NDArray[np.float64],
        confidence: float,
    ) -> float:
        """
        Parametric ES assuming normal distribution.

        ES = μ + σ * φ(Φ⁻¹(α)) / (1-α)

        where φ is PDF and Φ is CDF of standard normal.
        """
        from scipy import stats

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        alpha = 1 - confidence
        z = stats.norm.ppf(confidence)
        pdf_at_z = stats.norm.pdf(z)

        es = -(mean - std * pdf_at_z / alpha)
        return float(es)

    def _cornish_fisher_var(
        self,
        returns: NDArray[np.float64],
        confidence: float,
    ) -> float:
        """
        Cornish-Fisher VaR adjusting for skewness and kurtosis.

        Expands quantile using Cornish-Fisher expansion.
        """
        from scipy import stats

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))  # Excess kurtosis

        z = stats.norm.ppf(confidence)

        # Cornish-Fisher expansion
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = -(mean - z_cf * std)
        return float(var)

    def _cornish_fisher_es(
        self,
        returns: NDArray[np.float64],
        confidence: float,
    ) -> float:
        """ES using Cornish-Fisher adjusted VaR."""
        # Get CF-VaR as threshold
        var = self._cornish_fisher_var(returns, confidence)
        tail_losses = returns[returns <= -var]

        if len(tail_losses) == 0:
            return var

        return float(-np.mean(tail_losses))

    def _weighted_var(
        self,
        returns: NDArray[np.float64],
        confidence: float,
    ) -> float:
        """
        Weighted historical VaR with exponential decay.

        More recent observations get higher weights.
        """
        n = len(returns)
        if n < 10:
            return 0.0

        # Calculate weights
        weights = np.array([self.decay_factor ** i for i in range(n-1, -1, -1)])
        weights /= weights.sum()

        # Sort returns with weights
        sorted_indices = np.argsort(returns)
        sorted_returns = returns[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Find weighted quantile
        cumsum = np.cumsum(sorted_weights)
        var_idx = np.searchsorted(cumsum, 1 - confidence)
        var_idx = min(var_idx, n - 1)

        return float(-sorted_returns[var_idx])

    def _monte_carlo_es(
        self,
        returns: NDArray[np.float64],
        confidence: float,
        n_simulations: int = 10000,
    ) -> float:
        """
        Monte Carlo ES using bootstrapped returns.
        """
        n = len(returns)
        if n < 20:
            return self._parametric_es(returns, confidence)

        # Bootstrap samples
        simulated = np.random.choice(returns, size=(n_simulations, n), replace=True)
        sim_returns = simulated.mean(axis=1)  # Average return per simulation

        # Calculate ES from simulated distribution
        var_threshold = np.percentile(sim_returns, (1 - confidence) * 100)
        tail_losses = sim_returns[sim_returns <= var_threshold]

        if len(tail_losses) == 0:
            return float(-var_threshold)

        return float(-np.mean(tail_losses))

    def calculate_all_measures(
        self,
        returns: NDArray[np.float64],
    ) -> RiskMeasures:
        """
        Calculate comprehensive risk measures.

        Args:
            returns: Return series

        Returns:
            RiskMeasures dataclass
        """
        from scipy import stats

        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        es_95 = self.calculate_es(returns, 0.95)
        es_99 = self.calculate_es(returns, 0.99)
        max_loss = float(-np.min(returns)) if len(returns) > 0 else 0.0

        skewness = float(stats.skew(returns)) if len(returns) > 2 else 0.0
        kurtosis = float(stats.kurtosis(returns)) if len(returns) > 3 else 0.0

        tail_ratio = es_95 / var_95 if var_95 > EPSILON else 1.0

        return RiskMeasures(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            max_loss=max_loss,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
        )

    def check_breach(
        self,
        loss: float,
        date: Optional[str] = None,
    ) -> bool:
        """
        Check if loss exceeds ES threshold.

        Args:
            loss: Today's loss (positive number)
            date: Date string for logging

        Returns:
            True if ES is breached
        """
        if loss <= self._current_es:
            return False

        # Record breach event
        event = ESBreachEvent(
            date=date or str(np.datetime64('now')),
            loss=loss,
            es_level=self._current_es,
            var_level=self._current_var,
            severity_ratio=loss / self._current_es if self._current_es > EPSILON else 0.0,
        )
        self._breach_events.append(event)

        return True

    def get_breach_history(self) -> List[ESBreachEvent]:
        """Get history of ES breach events."""
        return self._breach_events.copy()

    def annualize(
        self,
        es: float,
        horizon_days: int = 1,
    ) -> float:
        """
        Annualize ES assuming scaling by sqrt(T).

        Args:
            es: Daily ES
            horizon_days: Current horizon

        Returns:
            Annualized ES
        """
        return es * np.sqrt(TRADING_DAYS_PER_YEAR / horizon_days)

    def scale_to_horizon(
        self,
        es: float,
        target_days: int,
        current_days: int = 1,
    ) -> float:
        """
        Scale ES to different time horizon.

        Uses square root of time scaling (assumes iid returns).
        """
        return es * np.sqrt(target_days / current_days)
