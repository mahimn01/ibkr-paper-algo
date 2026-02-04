"""
Portfolio Optimizer

Unified interface for portfolio optimization methods:
    - Mean-Variance (Markowitz)
    - Hierarchical Risk Parity (HRP)
    - Risk Parity
    - Maximum Diversification
    - Minimum Variance

This module provides a single entry point for all portfolio
optimization approaches, with proper regularization and
constraint handling.

References:
    - Markowitz, H. (1952). "Portfolio Selection"
    - Choueifaty & Coignard (2008). "Toward Maximum Diversification"
    - Maillard et al. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto
from scipy.optimize import minimize

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.portfolio.kelly import KellyCriterion
from trading_algo.quant_core.portfolio.hrp import HierarchicalRiskParity


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = auto()          # Markowitz
    MIN_VARIANCE = auto()           # Minimum variance
    RISK_PARITY = auto()            # Equal risk contribution
    MAX_DIVERSIFICATION = auto()    # Maximum diversification ratio
    HRP = auto()                    # Hierarchical Risk Parity
    EQUAL_WEIGHT = auto()           # 1/N portfolio
    KELLY = auto()                  # Kelly criterion


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0         # Minimum weight per asset
    max_weight: float = 1.0         # Maximum weight per asset
    min_total_weight: float = 1.0   # Minimum sum of weights
    max_total_weight: float = 1.0   # Maximum sum of weights
    max_sector_weight: Optional[Dict[str, float]] = None  # Sector limits
    turnover_limit: Optional[float] = None  # Max turnover from current


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: NDArray[np.float64]
    symbols: List[str]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    risk_contributions: NDArray[np.float64]
    method: OptimizationMethod
    success: bool
    message: str

    def to_dict(self) -> Dict[str, float]:
        """Convert to symbol -> weight dictionary."""
        return {s: float(w) for s, w in zip(self.symbols, self.weights)}


class PortfolioOptimizer:
    """
    Unified portfolio optimizer.

    Provides access to multiple optimization methods through
    a single interface with consistent constraint handling.

    Usage:
        optimizer = PortfolioOptimizer(
            method=OptimizationMethod.HRP,
            risk_free_rate=0.02,
        )

        result = optimizer.optimize(
            returns=returns_matrix,
            symbols=symbols,
            expected_returns=expected_returns,  # optional
        )

        weights = result.to_dict()
    """

    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.HRP,
        risk_free_rate: float = 0.0,
        target_volatility: Optional[float] = None,
        regularization: float = 1e-6,
    ):
        """
        Initialize optimizer.

        Args:
            method: Optimization method to use
            risk_free_rate: Risk-free rate for Sharpe calculation
            target_volatility: Target portfolio volatility (optional)
            regularization: Regularization for covariance matrix
        """
        self.method = method
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.regularization = regularization

        # Initialize sub-optimizers
        self._hrp = HierarchicalRiskParity()
        self._kelly = KellyCriterion()

    def optimize(
        self,
        returns: NDArray[np.float64],
        symbols: Optional[List[str]] = None,
        expected_returns: Optional[NDArray[np.float64]] = None,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[NDArray[np.float64]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            returns: Historical returns matrix (T x N)
            symbols: Asset symbols
            expected_returns: Expected returns (if None, uses historical mean)
            constraints: Optimization constraints
            current_weights: Current portfolio weights (for turnover constraint)

        Returns:
            OptimizationResult with optimal weights
        """
        n_assets = returns.shape[1]
        if symbols is None:
            symbols = [f"Asset_{i}" for i in range(n_assets)]
        if constraints is None:
            constraints = OptimizationConstraints()

        # Calculate covariance with regularization
        cov = np.cov(returns, rowvar=False)
        cov += np.eye(n_assets) * self.regularization

        # Expected returns
        if expected_returns is None:
            expected_returns = np.mean(returns, axis=0) * SQRT_252  # Annualized

        # Dispatch to appropriate method
        if self.method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._mean_variance(expected_returns, cov, constraints)
        elif self.method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance(cov, constraints)
        elif self.method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(cov, constraints)
        elif self.method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification(cov, constraints)
        elif self.method == OptimizationMethod.HRP:
            hrp_result = self._hrp.optimize(returns, symbols)
            weights = hrp_result.weights
        elif self.method == OptimizationMethod.KELLY:
            kelly_result = self._kelly.optimize_portfolio(expected_returns, cov * 252)
            weights = kelly_result.weights
        else:  # EQUAL_WEIGHT
            weights = np.ones(n_assets) / n_assets

        # Apply constraints
        weights = self._apply_constraints(weights, constraints, current_weights)

        # Calculate portfolio metrics
        port_return = float(weights @ expected_returns)
        port_vol = float(np.sqrt(weights @ (cov * 252) @ weights))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > EPSILON else 0.0

        # Diversification ratio
        weighted_vols = np.sqrt(np.diag(cov * 252)) @ weights
        div_ratio = weighted_vols / port_vol if port_vol > EPSILON else 1.0

        # Risk contributions
        marginal_risk = (cov * 252) @ weights
        risk_contrib = weights * marginal_risk / (port_vol + EPSILON)

        return OptimizationResult(
            weights=weights,
            symbols=symbols,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=float(div_ratio),
            risk_contributions=risk_contrib,
            method=self.method,
            success=True,
            message="Optimization successful",
        )

    def _mean_variance(
        self,
        expected_returns: NDArray[np.float64],
        cov: NDArray[np.float64],
        constraints: OptimizationConstraints,
    ) -> NDArray[np.float64]:
        """
        Mean-variance optimization (Markowitz).

        Maximizes Sharpe ratio subject to constraints.
        """
        n = len(expected_returns)

        def objective(w):
            port_ret = w @ expected_returns
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_ret - self.risk_free_rate) / (port_vol + EPSILON)

        # Constraints
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight)] * n

        # Initial guess
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000},
        )

        if result.success:
            return result.x
        else:
            return np.ones(n) / n

    def _min_variance(
        self,
        cov: NDArray[np.float64],
        constraints: OptimizationConstraints,
    ) -> NDArray[np.float64]:
        """
        Minimum variance portfolio.
        """
        n = cov.shape[0]

        def objective(w):
            return w @ cov @ w

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(constraints.min_weight, constraints.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else np.ones(n) / n

    def _risk_parity(
        self,
        cov: NDArray[np.float64],
        constraints: OptimizationConstraints,
        tol: float = 1e-8,
    ) -> NDArray[np.float64]:
        """
        Risk parity (equal risk contribution).

        Each asset contributes equally to portfolio risk.
        """
        n = cov.shape[0]

        def risk_contribution_error(w):
            port_vol = np.sqrt(w @ cov @ w)
            marginal = cov @ w
            risk_contrib = w * marginal / (port_vol + EPSILON)
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(max(EPSILON, constraints.min_weight), constraints.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            risk_contribution_error,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else np.ones(n) / n

    def _max_diversification(
        self,
        cov: NDArray[np.float64],
        constraints: OptimizationConstraints,
    ) -> NDArray[np.float64]:
        """
        Maximum diversification portfolio.

        Maximizes the diversification ratio:
            DR = (w'σ) / sqrt(w'Σw)

        where σ is the vector of volatilities.
        """
        n = cov.shape[0]
        vols = np.sqrt(np.diag(cov))

        def neg_diversification(w):
            port_vol = np.sqrt(w @ cov @ w)
            weighted_vols = w @ vols
            return -weighted_vols / (port_vol + EPSILON)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(constraints.min_weight, constraints.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_diversification,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else np.ones(n) / n

    def _apply_constraints(
        self,
        weights: NDArray[np.float64],
        constraints: OptimizationConstraints,
        current_weights: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Apply final constraints to weights."""
        # Clip to bounds
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)

        # Turnover constraint
        if constraints.turnover_limit is not None and current_weights is not None:
            turnover = np.sum(np.abs(weights - current_weights))
            if turnover > constraints.turnover_limit:
                # Scale down changes
                scale = constraints.turnover_limit / turnover
                weights = current_weights + scale * (weights - current_weights)

        # Normalize to sum constraint
        total = np.sum(weights)
        if total > EPSILON:
            target = (constraints.min_total_weight + constraints.max_total_weight) / 2
            weights = weights * target / total

        return weights

    def efficient_frontier(
        self,
        returns: NDArray[np.float64],
        n_points: int = 50,
        symbols: Optional[List[str]] = None,
    ) -> List[Tuple[float, float, NDArray[np.float64]]]:
        """
        Generate mean-variance efficient frontier.

        Returns list of (return, volatility, weights) tuples.
        """
        n_assets = returns.shape[1]
        if symbols is None:
            symbols = [f"Asset_{i}" for i in range(n_assets)]

        cov = np.cov(returns, rowvar=False)
        cov += np.eye(n_assets) * self.regularization
        expected_returns = np.mean(returns, axis=0) * SQRT_252

        # Find min and max return portfolios
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)

        frontier = []
        target_returns = np.linspace(min_ret, max_ret, n_points)

        for target in target_returns:
            weights = self._mean_variance_with_target(
                expected_returns, cov, target
            )
            port_ret = float(weights @ expected_returns)
            port_vol = float(np.sqrt(weights @ (cov * 252) @ weights))
            frontier.append((port_ret, port_vol, weights))

        return frontier

    def _mean_variance_with_target(
        self,
        expected_returns: NDArray[np.float64],
        cov: NDArray[np.float64],
        target_return: float,
    ) -> NDArray[np.float64]:
        """Mean-variance with target return constraint."""
        n = len(expected_returns)

        def objective(w):
            return w @ cov @ w

        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: w @ expected_returns - target_return},
        ]

        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=cons
        )

        return result.x if result.success else np.ones(n) / n


def black_litterman(
    cov: NDArray[np.float64],
    market_weights: NDArray[np.float64],
    views: NDArray[np.float64],
    view_confidences: NDArray[np.float64],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Black-Litterman model for combining market equilibrium with views.

    Args:
        cov: Covariance matrix
        market_weights: Market capitalization weights
        views: View matrix (K x N) - views on asset returns
        view_confidences: Confidence in each view (K,)
        risk_aversion: Risk aversion parameter (λ)
        tau: Scaling factor for prior uncertainty

    Returns:
        Tuple of (posterior expected returns, posterior covariance)

    Reference:
        Black, F. & Litterman, R. (1992). "Global Portfolio Optimization"
    """
    n_assets = cov.shape[0]
    n_views = views.shape[0] if views.ndim > 1 else 1

    # Prior (equilibrium) returns: π = λΣw
    pi = risk_aversion * cov @ market_weights

    # View portfolio matrix P (which assets in each view)
    if views.ndim == 1:
        P = views.reshape(1, -1)
    else:
        P = views

    # View uncertainty matrix Ω (diagonal)
    omega = np.diag(1.0 / (view_confidences + EPSILON))

    # Posterior expected returns
    tau_cov = tau * cov
    M = np.linalg.inv(
        np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P
    )
    posterior_returns = M @ (
        np.linalg.inv(tau_cov) @ pi +
        P.T @ np.linalg.inv(omega) @ (P @ pi)  # Simplified: views relative to equilibrium
    )

    # Posterior covariance
    posterior_cov = cov + M

    return posterior_returns, posterior_cov
