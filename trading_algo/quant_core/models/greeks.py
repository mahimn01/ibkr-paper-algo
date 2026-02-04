"""
Options Greeks Calculator

Implements Black-Scholes-Merton pricing and Greeks calculation.

Greeks:
    - Delta (Δ): Rate of change of option price with respect to underlying price
    - Gamma (Γ): Rate of change of delta with respect to underlying price
    - Theta (Θ): Rate of change of option price with respect to time
    - Vega (ν): Rate of change of option price with respect to volatility
    - Rho (ρ): Rate of change of option price with respect to interest rate

Reference:
    - Black & Scholes (1973)
    - Merton (1973)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class OptionSpec:
    """Option specification."""
    spot: float              # Current underlying price
    strike: float            # Strike price
    time_to_expiry: float    # Time to expiration (years)
    volatility: float        # Implied volatility (annualized)
    risk_free_rate: float    # Risk-free rate (annualized)
    dividend_yield: float = 0.0  # Dividend yield (annualized)
    option_type: Literal['call', 'put'] = 'call'


@dataclass
class Greeks:
    """Option Greeks."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class BlackScholesCalculator:
    """Black-Scholes-Merton options pricing and Greeks."""

    @staticmethod
    def _d1(S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0) -> float:
        """Calculate d1 parameter."""
        if T <= 0:
            return 0.0
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0) -> float:
        """Calculate d2 parameter."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesCalculator._d1(S, K, T, sigma, r, q)
        return d1 - sigma * np.sqrt(T)

    @classmethod
    def price(cls, spec: OptionSpec) -> float:
        """
        Calculate option price using Black-Scholes.

        Args:
            spec: Option specification

        Returns:
            Option price
        """
        S, K, T, sigma, r, q = (
            spec.spot, spec.strike, spec.time_to_expiry,
            spec.volatility, spec.risk_free_rate, spec.dividend_yield
        )

        if T <= 0:
            # Expired option - intrinsic value only
            if spec.option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = cls._d1(S, K, T, sigma, r, q)
        d2 = cls._d2(S, K, T, sigma, r, q)

        if spec.option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return float(price)

    @classmethod
    def delta(cls, spec: OptionSpec) -> float:
        """
        Calculate delta.

        Delta = ∂V/∂S
        """
        S, K, T, sigma, r, q = (
            spec.spot, spec.strike, spec.time_to_expiry,
            spec.volatility, spec.risk_free_rate, spec.dividend_yield
        )

        if T <= 0:
            if spec.option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = cls._d1(S, K, T, sigma, r, q)

        if spec.option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

        return float(delta)

    @classmethod
    def gamma(cls, spec: OptionSpec) -> float:
        """
        Calculate gamma.

        Gamma = ∂²V/∂S² = ∂Δ/∂S
        """
        S, K, T, sigma, r, q = (
            spec.spot, spec.strike, spec.time_to_expiry,
            spec.volatility, spec.risk_free_rate, spec.dividend_yield
        )

        if T <= 0:
            return 0.0

        d1 = cls._d1(S, K, T, sigma, r, q)

        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

        return float(gamma)

    @classmethod
    def theta(cls, spec: OptionSpec) -> float:
        """
        Calculate theta (per day).

        Theta = -∂V/∂t

        Returns theta per day (divide by 365).
        """
        S, K, T, sigma, r, q = (
            spec.spot, spec.strike, spec.time_to_expiry,
            spec.volatility, spec.risk_free_rate, spec.dividend_yield
        )

        if T <= 0:
            return 0.0

        d1 = cls._d1(S, K, T, sigma, r, q)
        d2 = cls._d2(S, K, T, sigma, r, q)

        term1 = -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))

        if spec.option_type == 'call':
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(d1)
            theta = term1 - term2 + term3
        else:  # put
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta = term1 + term2 + term3

        # Convert to per-day theta
        theta_per_day = theta / 365.0

        return float(theta_per_day)

    @classmethod
    def vega(cls, spec: OptionSpec) -> float:
        """
        Calculate vega (per 1% change in volatility).

        Vega = ∂V/∂σ

        Returns vega per 1% (0.01) change in volatility.
        """
        S, K, T, sigma, r, q = (
            spec.spot, spec.strike, spec.time_to_expiry,
            spec.volatility, spec.risk_free_rate, spec.dividend_yield
        )

        if T <= 0:
            return 0.0

        d1 = cls._d1(S, K, T, sigma, r, q)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Convert to per 1% change
        vega_percent = vega / 100.0

        return float(vega_percent)

    @classmethod
    def rho(cls, spec: OptionSpec) -> float:
        """
        Calculate rho (per 1% change in interest rate).

        Rho = ∂V/∂r
        """
        S, K, T, sigma, r, q = (
            spec.spot, spec.strike, spec.time_to_expiry,
            spec.volatility, spec.risk_free_rate, spec.dividend_yield
        )

        if T <= 0:
            return 0.0

        d2 = cls._d2(S, K, T, sigma, r, q)

        if spec.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        # Convert to per 1% change
        rho_percent = rho / 100.0

        return float(rho_percent)

    @classmethod
    def calculate_all_greeks(cls, spec: OptionSpec) -> Greeks:
        """Calculate all Greeks at once."""
        return Greeks(
            price=cls.price(spec),
            delta=cls.delta(spec),
            gamma=cls.gamma(spec),
            theta=cls.theta(spec),
            vega=cls.vega(spec),
            rho=cls.rho(spec)
        )


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: Literal['call', 'put'] = 'call',
    dividend_yield: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    Args:
        option_price: Observed option price
        spot: Current underlying price
        strike: Strike price
        time_to_expiry: Time to expiration (years)
        risk_free_rate: Risk-free rate
        option_type: 'call' or 'put'
        dividend_yield: Dividend yield
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility
    """
    # Initial guess
    sigma = 0.3

    for i in range(max_iterations):
        spec = OptionSpec(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=sigma,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type
        )

        # Calculate price and vega
        calc_price = BlackScholesCalculator.price(spec)
        vega = BlackScholesCalculator.vega(spec) * 100  # Vega per 1.0 vol change

        # Newton-Raphson update
        price_diff = calc_price - option_price

        if abs(price_diff) < tolerance:
            return sigma

        if vega < 1e-10:
            # Vega too small, can't converge
            break

        sigma = sigma - price_diff / vega

        # Keep sigma in reasonable bounds
        sigma = max(0.01, min(5.0, sigma))

    # If didn't converge, return last estimate
    return sigma


if __name__ == "__main__":
    # Example usage
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        time_to_expiry=30/365,  # 30 days
        volatility=0.25,
        risk_free_rate=0.05,
        option_type='call'
    )

    greeks = BlackScholesCalculator.calculate_all_greeks(spec)

    print(f"Option Price: ${greeks.price:.4f}")
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.4f}")
    print(f"Theta (per day): ${greeks.theta:.4f}")
    print(f"Vega (per 1%): ${greeks.vega:.4f}")
    print(f"Rho (per 1%): ${greeks.rho:.4f}")

    # Calculate implied vol
    iv = implied_volatility(
        option_price=greeks.price,
        spot=100.0,
        strike=100.0,
        time_to_expiry=30/365,
        risk_free_rate=0.05,
        option_type='call'
    )
    print(f"\nImplied Volatility: {iv:.4f} (should be 0.25)")
