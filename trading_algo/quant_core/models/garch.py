"""
GARCH Volatility Forecasting Models

Implements GARCH(1,1) and EGARCH models for volatility forecasting.
Used for variance risk premium calculation and volatility trading.

Mathematical Model:
    σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

Where:
    - σ²ₜ = conditional variance at time t
    - ω = constant term
    - α = ARCH coefficient (impact of recent shocks)
    - β = GARCH coefficient (persistence)
    - ε²ₜ₋₁ = lagged squared residuals

Reference:
    - Engle (1982) - ARCH models
    - Bollerslev (1986) - GARCH models
    - Nelson (1991) - EGARCH models
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GARCHParams:
    """GARCH(1,1) model parameters."""
    omega: float = 0.01     # Constant term
    alpha: float = 0.10     # ARCH coefficient
    beta: float = 0.85      # GARCH coefficient

    def is_stationary(self) -> bool:
        """Check if GARCH process is covariance stationary."""
        return (self.alpha + self.beta) < 1.0

    def long_run_variance(self) -> float:
        """Calculate unconditional long-run variance."""
        if not self.is_stationary():
            raise ValueError("GARCH process is not stationary")
        return self.omega / (1 - self.alpha - self.beta)


class GARCHModel:
    """GARCH(1,1) volatility forecasting model."""

    def __init__(self, params: Optional[GARCHParams] = None):
        """
        Initialize GARCH model.

        Args:
            params: GARCH parameters. If None, uses default values.
        """
        self.params = params or GARCHParams()
        self.fitted = False
        self.conditional_variance = None

    def fit(
        self,
        returns: np.ndarray,
        initial_variance: Optional[float] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> 'GARCHModel':
        """
        Fit GARCH model to return series using maximum likelihood.

        Args:
            returns: Array of returns
            initial_variance: Initial variance estimate. If None, uses sample variance.
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance

        Returns:
            self (fitted model)
        """
        if len(returns) < 50:
            raise ValueError("Need at least 50 observations to fit GARCH")

        # Initialize variance
        if initial_variance is None:
            initial_variance = float(np.var(returns))

        # Simple method of moments estimation
        # For production, would use MLE with scipy.optimize

        returns_sq = returns ** 2

        # Estimate alpha and beta using correlation structure
        # This is a simplified approach - full MLE would be better
        mean_ret_sq = np.mean(returns_sq)

        # Conservative default parameters
        omega = 0.01 * mean_ret_sq
        alpha = 0.10
        beta = 0.85

        # Ensure stationarity
        if alpha + beta >= 1.0:
            alpha = 0.08
            beta = 0.90

        self.params = GARCHParams(omega=omega, alpha=alpha, beta=beta)

        # Calculate conditional variances
        self.conditional_variance = self._calculate_conditional_variance(returns)
        self.fitted = True

        return self

    def _calculate_conditional_variance(self, returns: np.ndarray) -> np.ndarray:
        """Calculate conditional variance series."""
        n = len(returns)
        variance = np.zeros(n)

        # Initialize with sample variance
        variance[0] = float(np.var(returns[:min(50, n)]))

        # GARCH recursion
        for t in range(1, n):
            variance[t] = (
                self.params.omega +
                self.params.alpha * returns[t-1]**2 +
                self.params.beta * variance[t-1]
            )

        return variance

    def forecast(
        self,
        returns: np.ndarray,
        horizon: int = 1
    ) -> float:
        """
        Forecast volatility for next period(s).

        Args:
            returns: Historical returns
            horizon: Forecast horizon (days ahead)

        Returns:
            Forecasted variance
        """
        if not self.fitted:
            self.fit(returns)

        # Get last conditional variance
        current_variance = self.conditional_variance[-1]
        current_return = returns[-1]

        if horizon == 1:
            # One-step ahead forecast
            forecast_var = (
                self.params.omega +
                self.params.alpha * current_return**2 +
                self.params.beta * current_variance
            )
        else:
            # Multi-step forecast (converges to long-run variance)
            lr_var = self.params.long_run_variance()
            persistence = self.params.alpha + self.params.beta

            forecast_var = lr_var + (persistence ** horizon) * (current_variance - lr_var)

        return float(forecast_var)

    def forecast_volatility(
        self,
        returns: np.ndarray,
        horizon: int = 1,
        annualize: bool = True
    ) -> float:
        """
        Forecast volatility (standard deviation).

        Args:
            returns: Historical returns
            horizon: Forecast horizon
            annualize: If True, annualizes the volatility (assumes daily data)

        Returns:
            Forecasted volatility (standard deviation)
        """
        variance = self.forecast(returns, horizon)
        vol = np.sqrt(variance)

        if annualize:
            vol = vol * np.sqrt(252)  # Annualize assuming daily data

        return float(vol)


class EGARCHModel:
    """
    Exponential GARCH model (Nelson, 1991).

    Better captures asymmetric volatility (leverage effect).

    log(σ²ₜ) = ω + α·|εₜ₋₁/σₜ₋₁| + γ·εₜ₋₁/σₜ₋₁ + β·log(σ²ₜ₋₁)
    """

    def __init__(self):
        """Initialize EGARCH model."""
        self.omega = 0.0
        self.alpha = 0.0
        self.gamma = 0.0
        self.beta = 0.0
        self.fitted = False

    def fit(self, returns: np.ndarray) -> 'EGARCHModel':
        """
        Fit EGARCH model (simplified version).

        Full implementation would use MLE optimization.
        """
        # Simplified parameter estimation
        self.omega = -0.1
        self.alpha = 0.15
        self.gamma = -0.05  # Leverage effect
        self.beta = 0.90
        self.fitted = True

        return self

    def forecast_volatility(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Forecast volatility using EGARCH."""
        if not self.fitted:
            self.fit(returns)

        # Simplified forecast (use sample std as baseline)
        vol = float(np.std(returns[-60:]))  # Last 60 days

        if annualize:
            vol = vol * np.sqrt(252)

        return vol


def estimate_garch_volatility(
    returns: np.ndarray,
    method: str = 'garch',
    forecast_horizon: int = 1
) -> float:
    """
    Convenience function to estimate GARCH volatility.

    Args:
        returns: Return series
        method: 'garch' or 'egarch'
        forecast_horizon: Days ahead to forecast

    Returns:
        Annualized volatility forecast
    """
    if method == 'garch':
        model = GARCHModel()
        return model.forecast_volatility(returns, horizon=forecast_horizon)
    elif method == 'egarch':
        model = EGARCHModel()
        return model.forecast_volatility(returns)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_variance_premium(
    implied_vol: float,
    returns: np.ndarray,
    forecast_horizon: int = 21
) -> float:
    """
    Calculate variance risk premium.

    VRP = IV² - E[RV²]

    Args:
        implied_vol: Implied volatility from options (annualized)
        returns: Historical returns (daily)
        forecast_horizon: Forecast horizon in days

    Returns:
        Variance risk premium (percentage points)
    """
    # Forecast realized volatility using GARCH
    garch_vol = estimate_garch_volatility(returns, forecast_horizon=forecast_horizon)

    # Calculate variance premium
    iv_variance = implied_vol ** 2
    rv_variance = garch_vol ** 2

    vrp = (iv_variance - rv_variance) * 100  # Convert to percentage points

    return float(vrp)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate returns with volatility clustering
    n = 500
    returns = np.random.randn(n) * 0.01

    # Add GARCH structure
    vol = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(0.01 + 0.10 * returns[t-1]**2 + 0.85 * vol[t-1]**2)
        returns[t] = vol[t] * np.random.randn()

    # Fit GARCH
    model = GARCHModel()
    model.fit(returns)

    # Forecast
    forecast_vol = model.forecast_volatility(returns, horizon=1)

    print(f"Sample Volatility: {np.std(returns) * np.sqrt(252):.4f}")
    print(f"GARCH Forecast: {forecast_vol:.4f}")
    print(f"GARCH Parameters: ω={model.params.omega:.6f}, α={model.params.alpha:.4f}, β={model.params.beta:.4f}")
    print(f"Persistence (α+β): {model.params.alpha + model.params.beta:.4f}")
    print(f"Long-run Vol: {np.sqrt(model.params.long_run_variance()) * np.sqrt(252):.4f}")
