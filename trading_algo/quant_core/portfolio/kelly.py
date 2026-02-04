"""
Kelly Criterion Position Sizing

Mathematical framework for optimal position sizing that maximizes
the expected logarithm of wealth (geometric growth rate).

Core Formula (continuous case):
    f* = (μ - r) / σ²

where:
    - f*: Optimal fraction of capital
    - μ: Expected return
    - r: Risk-free rate
    - σ²: Variance of returns

For discrete outcomes (binary):
    f* = (p*b - q) / b

where:
    - p: Probability of winning
    - q = 1-p: Probability of losing
    - b: Win/loss ratio (avg_win / avg_loss)

Key Insights:
    - Full Kelly maximizes long-run growth but has severe drawdowns
    - Fractional Kelly (25-50%) sacrifices growth for stability:
        - Half Kelly: 75% of growth, 50% less drawdown
        - Quarter Kelly: 50% of growth, 75% less drawdown
    - Over-betting (>1x Kelly) leads to negative expected growth

References:
    - Kelly, J.L. (1956): "A New Interpretation of Information Rate"
    - https://en.wikipedia.org/wiki/Kelly_criterion
    - https://www.quantifiedstrategies.com/kelly-criterion-position-sizing/
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import (
    KELLY_FRACTION_CONSERVATIVE,
    KELLY_FRACTION_MODERATE,
    KELLY_FRACTION_AGGRESSIVE,
    KELLY_MAX_POSITION,
    EPSILON,
    TRADING_DAYS_PER_YEAR,
)


class KellyMode(Enum):
    """Kelly fraction modes."""
    FULL = auto()           # Full Kelly (risky, max growth)
    THREE_QUARTER = auto()  # 75% Kelly
    HALF = auto()           # 50% Kelly (balanced)
    QUARTER = auto()        # 25% Kelly (conservative)
    EIGHTH = auto()         # 12.5% Kelly (very conservative)


@dataclass
class KellyEstimate:
    """
    Kelly criterion estimation result.

    Contains the optimal fraction and supporting statistics.
    """
    full_kelly: float          # Full Kelly fraction
    fractional_kelly: float    # After applying Kelly fraction
    position_size: float       # Final position size (after caps)
    win_rate: float           # Historical win rate
    win_loss_ratio: float     # Average win / average loss
    expected_return: float    # Expected return per trade
    variance: float           # Variance of returns
    sharpe_ratio: float       # Edge in Sharpe units
    confidence: float         # Confidence in estimate (0-1)
    n_samples: int            # Number of samples used

    @property
    def edge(self) -> float:
        """Calculate edge (expected value per dollar risked)."""
        return self.win_rate * self.win_loss_ratio - (1 - self.win_rate)


@dataclass
class KellyPortfolio:
    """
    Multi-asset Kelly portfolio allocation.

    Contains optimal weights for multiple assets.
    """
    weights: Dict[str, float]      # Symbol -> weight
    total_leverage: float          # Sum of absolute weights
    expected_return: float         # Portfolio expected return
    expected_variance: float       # Portfolio variance
    kelly_growth_rate: float       # Expected log growth rate


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator.

    Calculates optimal position sizes based on expected returns
    and variance, with support for:
    - Single asset Kelly
    - Multi-asset Kelly with correlation
    - Fractional Kelly for reduced risk
    - Dynamic Kelly based on recent performance

    Usage:
        kelly = KellyCriterion(mode=KellyMode.HALF)

        # From trade history
        size = kelly.calculate_from_trades(pnl_series)

        # From return statistics
        size = kelly.calculate_from_statistics(
            expected_return=0.001,  # 0.1% per trade
            variance=0.0004,        # 2% daily vol
            risk_free_rate=0.0
        )

        # Multi-asset
        portfolio = kelly.optimize_portfolio(
            expected_returns={'AAPL': 0.001, 'GOOGL': 0.0008},
            covariance_matrix=cov_matrix
        )
    """

    def __init__(
        self,
        mode: KellyMode = KellyMode.HALF,
        max_position: float = KELLY_MAX_POSITION,
        min_samples: int = 30,
        max_leverage: float = 2.0,
        shrinkage: float = 0.1,  # Shrink estimates toward 0
    ):
        """
        Initialize Kelly calculator.

        Args:
            mode: Kelly fraction to use (FULL, HALF, QUARTER, etc.)
            max_position: Maximum position size per asset
            min_samples: Minimum samples required for estimation
            max_leverage: Maximum total portfolio leverage
            shrinkage: Shrinkage factor for conservative estimates
        """
        self.mode = mode
        self.max_position = max_position
        self.min_samples = min_samples
        self.max_leverage = max_leverage
        self.shrinkage = shrinkage

        # Kelly fraction mapping
        self._kelly_fractions = {
            KellyMode.FULL: 1.0,
            KellyMode.THREE_QUARTER: 0.75,
            KellyMode.HALF: 0.50,
            KellyMode.QUARTER: 0.25,
            KellyMode.EIGHTH: 0.125,
        }

    @property
    def kelly_fraction(self) -> float:
        """Get current Kelly fraction."""
        return self._kelly_fractions[self.mode]

    def calculate_from_trades(
        self,
        pnl_returns: NDArray[np.float64],
        risk_free_rate: float = 0.0
    ) -> KellyEstimate:
        """
        Calculate Kelly position size from trade P&L returns.

        Uses the general Kelly formula for continuous returns:
            f* = (μ - r) / σ²

        Args:
            pnl_returns: Array of trade returns (not P&L dollars)
            risk_free_rate: Risk-free rate (same period as returns)

        Returns:
            KellyEstimate with optimal position size
        """
        n = len(pnl_returns)

        if n < self.min_samples:
            return self._empty_estimate(n)

        # Calculate statistics
        mean_return = np.mean(pnl_returns)
        variance = np.var(pnl_returns, ddof=1)

        # Calculate win rate and win/loss ratio
        wins = pnl_returns[pnl_returns > 0]
        losses = pnl_returns[pnl_returns < 0]

        win_rate = len(wins) / n if n > 0 else 0.5
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > EPSILON else 0.0

        # Calculate full Kelly
        full_kelly = self._calculate_continuous_kelly(
            mean_return, variance, risk_free_rate
        )

        # Apply shrinkage for conservative estimate
        full_kelly *= (1 - self.shrinkage)

        # Apply fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction

        # Apply position limits
        position_size = np.clip(fractional_kelly, -self.max_position, self.max_position)

        # Calculate confidence based on sample size and consistency
        confidence = self._calculate_confidence(pnl_returns, win_rate)

        # Calculate Sharpe ratio
        std = np.sqrt(variance) if variance > 0 else 1.0
        sharpe = (mean_return - risk_free_rate) / std if std > EPSILON else 0.0

        return KellyEstimate(
            full_kelly=float(full_kelly),
            fractional_kelly=float(fractional_kelly),
            position_size=float(position_size),
            win_rate=float(win_rate),
            win_loss_ratio=float(win_loss_ratio),
            expected_return=float(mean_return),
            variance=float(variance),
            sharpe_ratio=float(sharpe),
            confidence=float(confidence),
            n_samples=n,
        )

    def calculate_from_statistics(
        self,
        expected_return: float,
        variance: float,
        risk_free_rate: float = 0.0,
        n_samples: int = 100,
    ) -> KellyEstimate:
        """
        Calculate Kelly from expected return and variance directly.

        Args:
            expected_return: Expected return per period
            variance: Variance of returns
            risk_free_rate: Risk-free rate
            n_samples: Assumed sample size for confidence

        Returns:
            KellyEstimate
        """
        # Calculate full Kelly
        full_kelly = self._calculate_continuous_kelly(
            expected_return, variance, risk_free_rate
        )

        # Apply shrinkage
        full_kelly *= (1 - self.shrinkage)

        # Apply fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction

        # Apply limits
        position_size = np.clip(fractional_kelly, -self.max_position, self.max_position)

        # Sharpe
        std = np.sqrt(variance) if variance > 0 else 1.0
        sharpe = (expected_return - risk_free_rate) / std if std > EPSILON else 0.0

        return KellyEstimate(
            full_kelly=float(full_kelly),
            fractional_kelly=float(fractional_kelly),
            position_size=float(position_size),
            win_rate=0.5 + sharpe * 0.1,  # Approximate
            win_loss_ratio=1.0,
            expected_return=float(expected_return),
            variance=float(variance),
            sharpe_ratio=float(sharpe),
            confidence=0.5,
            n_samples=n_samples,
        )

    def calculate_binary_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> KellyEstimate:
        """
        Calculate Kelly for binary outcomes (win/loss only).

        Formula: f* = (p*b - q) / b
        where p = win_rate, q = 1-p, b = avg_win/avg_loss

        Args:
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)

        Returns:
            KellyEstimate
        """
        if avg_loss < EPSILON:
            return self._empty_estimate(0)

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss

        # Binary Kelly formula
        full_kelly = (p * b - q) / b if b > EPSILON else 0.0

        # Can be negative if edge is negative
        if full_kelly < 0:
            full_kelly = 0.0

        # Apply shrinkage and fraction
        full_kelly *= (1 - self.shrinkage)
        fractional_kelly = full_kelly * self.kelly_fraction
        position_size = np.clip(fractional_kelly, 0, self.max_position)

        # Calculate expected return
        expected_return = p * avg_win - q * avg_loss
        variance = p * (avg_win - expected_return)**2 + q * (avg_loss + expected_return)**2

        return KellyEstimate(
            full_kelly=float(full_kelly),
            fractional_kelly=float(fractional_kelly),
            position_size=float(position_size),
            win_rate=float(win_rate),
            win_loss_ratio=float(b),
            expected_return=float(expected_return),
            variance=float(variance),
            sharpe_ratio=expected_return / np.sqrt(variance) if variance > 0 else 0.0,
            confidence=0.5,
            n_samples=0,
        )

    def optimize_portfolio(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: NDArray[np.float64],
        symbols: List[str],
        risk_free_rate: float = 0.0,
    ) -> KellyPortfolio:
        """
        Calculate optimal multi-asset Kelly portfolio.

        Multi-asset Kelly formula:
            f* = Σ⁻¹ @ μ

        where:
            - Σ⁻¹: Inverse covariance matrix
            - μ: Expected returns vector

        Note: This is equivalent to mean-variance optimization
        with λ = 0.5 (maximizing log utility).

        Args:
            expected_returns: Dict of symbol -> expected return
            covariance_matrix: N x N covariance matrix
            symbols: List of symbols (order must match cov matrix)
            risk_free_rate: Risk-free rate

        Returns:
            KellyPortfolio with optimal weights
        """
        n = len(symbols)

        if n == 0 or covariance_matrix.shape != (n, n):
            return KellyPortfolio(
                weights={},
                total_leverage=0.0,
                expected_return=0.0,
                expected_variance=0.0,
                kelly_growth_rate=0.0,
            )

        # Build expected return vector
        mu = np.array([expected_returns.get(s, 0.0) - risk_free_rate for s in symbols])

        # Apply shrinkage to covariance matrix for stability
        cov = covariance_matrix.copy()
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))

        try:
            # Calculate inverse covariance
            cov_inv = np.linalg.inv(cov)

            # Full Kelly weights: f* = Σ⁻¹ @ μ
            full_kelly_weights = cov_inv @ mu

            # Apply fractional Kelly
            fractional_weights = full_kelly_weights * self.kelly_fraction

            # Apply leverage constraint
            total_leverage = np.sum(np.abs(fractional_weights))
            if total_leverage > self.max_leverage:
                fractional_weights *= self.max_leverage / total_leverage
                total_leverage = self.max_leverage

            # Apply position limits
            fractional_weights = np.clip(fractional_weights, -self.max_position, self.max_position)

            # Build weight dictionary
            weights = {symbols[i]: float(fractional_weights[i]) for i in range(n)}

            # Calculate portfolio statistics
            portfolio_return = float(mu @ fractional_weights)
            portfolio_variance = float(fractional_weights @ cov @ fractional_weights)

            # Kelly growth rate: g = μ'f - 0.5 * f'Σf
            growth_rate = portfolio_return - 0.5 * portfolio_variance

            return KellyPortfolio(
                weights=weights,
                total_leverage=float(np.sum(np.abs(fractional_weights))),
                expected_return=portfolio_return,
                expected_variance=portfolio_variance,
                kelly_growth_rate=float(growth_rate),
            )

        except np.linalg.LinAlgError:
            # Singular matrix - fall back to equal weights
            equal_weight = 1.0 / n * self.kelly_fraction
            weights = {s: equal_weight for s in symbols}
            return KellyPortfolio(
                weights=weights,
                total_leverage=float(abs(equal_weight) * n),
                expected_return=0.0,
                expected_variance=0.0,
                kelly_growth_rate=0.0,
            )

    def dynamic_kelly(
        self,
        recent_returns: NDArray[np.float64],
        historical_kelly: float,
        decay: float = 0.9,
    ) -> float:
        """
        Calculate dynamic Kelly that adapts to recent performance.

        Combines historical Kelly estimate with recent performance
        using exponential weighting.

        Args:
            recent_returns: Recent trade returns
            historical_kelly: Historical Kelly estimate
            decay: Weight on historical (1-decay on recent)

        Returns:
            Dynamic Kelly fraction
        """
        if len(recent_returns) < 5:
            return historical_kelly

        recent_estimate = self.calculate_from_trades(recent_returns)
        recent_kelly = recent_estimate.fractional_kelly

        # Exponential smoothing
        dynamic = decay * historical_kelly + (1 - decay) * recent_kelly

        # Apply limits
        return float(np.clip(dynamic, 0, self.max_position))

    def _calculate_continuous_kelly(
        self,
        mean_return: float,
        variance: float,
        risk_free_rate: float,
    ) -> float:
        """
        Calculate continuous Kelly: f* = (μ - r) / σ²

        Args:
            mean_return: Expected return
            variance: Variance
            risk_free_rate: Risk-free rate

        Returns:
            Full Kelly fraction
        """
        if variance < EPSILON:
            return 0.0

        excess_return = mean_return - risk_free_rate
        kelly = excess_return / variance

        # Can be negative if expected return < risk-free rate
        return max(0.0, kelly)

    def _calculate_confidence(
        self,
        returns: NDArray[np.float64],
        win_rate: float,
    ) -> float:
        """
        Calculate confidence in Kelly estimate.

        Based on:
        - Sample size
        - Consistency of returns
        - Distance from 50% win rate

        Returns:
            Confidence score (0-1)
        """
        n = len(returns)

        # Sample size factor (diminishing returns after 100)
        size_factor = min(1.0, n / 100)

        # Consistency factor (lower coefficient of variation = better)
        if len(returns) > 1:
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            cv = abs(std / mean) if abs(mean) > EPSILON else 10.0
            consistency_factor = min(1.0, 1.0 / (1 + cv))
        else:
            consistency_factor = 0.0

        # Edge factor (further from 50% = more confident in edge)
        edge_factor = min(1.0, abs(win_rate - 0.5) * 4)

        # Combine
        confidence = (size_factor * 0.4 + consistency_factor * 0.3 + edge_factor * 0.3)
        return float(np.clip(confidence, 0, 1))

    def _empty_estimate(self, n_samples: int) -> KellyEstimate:
        """Return empty estimate when calculation fails."""
        return KellyEstimate(
            full_kelly=0.0,
            fractional_kelly=0.0,
            position_size=0.0,
            win_rate=0.5,
            win_loss_ratio=1.0,
            expected_return=0.0,
            variance=0.0,
            sharpe_ratio=0.0,
            confidence=0.0,
            n_samples=n_samples,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """
    Quick Kelly calculation for binary outcomes.

    Args:
        win_rate: Win probability
        avg_win: Average win
        avg_loss: Average loss (positive)
        fraction: Kelly fraction to use

    Returns:
        Recommended position size as fraction of capital
    """
    if avg_loss < EPSILON:
        return 0.0

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p

    kelly = (p * b - q) / b if b > EPSILON else 0.0
    return max(0.0, kelly * fraction)


def kelly_from_sharpe(sharpe: float, fraction: float = 0.5) -> float:
    """
    Approximate Kelly from Sharpe ratio.

    For normally distributed returns:
        f* ≈ SR² / σ ≈ SR (when σ normalized to 1)

    Args:
        sharpe: Sharpe ratio (annualized)
        fraction: Kelly fraction

    Returns:
        Approximate Kelly position size
    """
    # Convert annualized Sharpe to per-trade
    daily_sharpe = sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)
    kelly = daily_sharpe ** 2
    return kelly * fraction
