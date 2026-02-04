"""
Volatility-Managed Momentum

Implementation based on Daniel & Moskowitz (2016):
"Momentum Crashes" - Journal of Financial Economics

Key Innovation:
    Scale position sizes inversely with forecasted volatility.
    This nearly doubles the Sharpe ratio while eliminating momentum crash risk.

Core Concept:
    Position_t = (σ_target / σ_t^forecast) × Signal_t

where:
    - σ_target: Target volatility (e.g., 15% annualized)
    - σ_t^forecast: Forecasted volatility
    - Signal_t: Momentum signal (-1 to +1)

Two approaches documented in literature:
    1. Constant Volatility Scaling (CVS) - Barroso & Santa-Clara (2015)
    2. Dynamic Volatility Scaling (DVS) - Daniel & Moskowitz (2016)

DVS additionally scales by predicted Sharpe ratio, making leverage
proportional to expected risk-adjusted return.

Research Findings:
    - Volatility-adjusted momentum has higher Sharpe and lower crash risk
    - Both CVS and DVS produce similar results when Sharpe is time-invariant
    - Critical for avoiding momentum crashes (e.g., 73% drawdown in 2009)

References:
    - Daniel & Moskowitz (2016): https://doi.org/10.1016/j.jfineco.2016.01.032
    - Barroso & Santa-Clara (2015): https://doi.org/10.1016/j.jfineco.2014.11.010
    - https://quantpedia.com/strategies/time-series-momentum-effect/
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import (
    VOL_TARGET_DEFAULT,
    VOL_LOOKBACK_DEFAULT,
    VOL_FLOOR,
    VOL_CAP,
    VOL_SCALAR_MAX,
    SQRT_252,
    EPSILON,
    TRADING_DAYS_PER_YEAR,
)
from trading_algo.quant_core.utils.math_utils import (
    realized_volatility,
    ewma_volatility,
    simple_returns,
)


class VolScalingMethod(Enum):
    """Volatility scaling methods."""
    CVS = auto()     # Constant Volatility Scaling (Barroso & Santa-Clara 2015)
    DVS = auto()     # Dynamic Volatility Scaling (Daniel & Moskowitz 2016)
    GARCH = auto()   # GARCH(1,1) based forecasting
    EWMA = auto()    # Exponentially Weighted Moving Average


@dataclass
class VolManagedSignal:
    """
    Volatility-managed trading signal.

    Contains both the raw signal and the volatility-scaled position size.
    """
    raw_signal: float          # Original momentum signal (-1 to +1)
    vol_scalar: float          # Volatility scaling factor
    position_size: float       # Scaled position size
    current_vol: float         # Current (forecasted) volatility
    target_vol: float          # Target volatility
    momentum_return: float     # Momentum return used for signal
    sharpe_forecast: float     # Forecasted Sharpe (for DVS)


@dataclass
class VolRegime:
    """Volatility regime classification."""
    current_vol: float         # Current annualized vol
    vol_percentile: float      # Percentile vs history (0-100)
    regime: str                # 'low', 'normal', 'high', 'crisis'
    vol_trend: float           # Rate of change of volatility


class VolatilityManagedMomentum:
    """
    Volatility-Managed Momentum Strategy.

    Scales momentum positions inversely with volatility to maintain
    consistent risk exposure and avoid momentum crashes.

    Key Formula:
        Position = (σ_target / σ_forecast) × momentum_signal

    This approach:
    - Nearly doubles Sharpe ratio (per Daniel & Moskowitz 2016)
    - Eliminates momentum crash risk
    - Maintains consistent risk exposure across market regimes

    Usage:
        vmm = VolatilityManagedMomentum(target_vol=0.15)
        signal = vmm.generate_signal(prices, returns)
        position_size = signal.position_size
    """

    def __init__(
        self,
        target_vol: float = VOL_TARGET_DEFAULT,
        vol_lookback: int = VOL_LOOKBACK_DEFAULT,
        momentum_lookback: int = 252,  # 12-month momentum
        method: VolScalingMethod = VolScalingMethod.CVS,
        max_scalar: float = VOL_SCALAR_MAX,
        vol_floor: float = VOL_FLOOR,
        vol_cap: float = VOL_CAP,
        ewma_decay: float = 0.94,
    ):
        """
        Initialize volatility-managed momentum.

        Args:
            target_vol: Target annualized volatility (default 15%)
            vol_lookback: Days for volatility estimation
            momentum_lookback: Days for momentum calculation
            method: Volatility scaling method (CVS, DVS, GARCH, EWMA)
            max_scalar: Maximum leverage from vol scaling
            vol_floor: Minimum volatility estimate
            vol_cap: Maximum volatility estimate
            ewma_decay: EWMA decay factor (lambda)
        """
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.momentum_lookback = momentum_lookback
        self.method = method
        self.max_scalar = max_scalar
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self.ewma_decay = ewma_decay

        # State tracking
        self._vol_history: Dict[str, NDArray] = {}
        self._return_history: Dict[str, NDArray] = {}

    def forecast_volatility(
        self,
        returns: NDArray[np.float64],
        method: Optional[VolScalingMethod] = None
    ) -> float:
        """
        Forecast next-period volatility.

        Args:
            returns: Historical return series
            method: Override default method

        Returns:
            Forecasted annualized volatility
        """
        if len(returns) < 20:
            return self.target_vol

        method = method or self.method

        if method == VolScalingMethod.EWMA:
            # EWMA volatility (RiskMetrics style)
            ewma_vol = ewma_volatility(returns, decay=self.ewma_decay)
            vol_forecast = ewma_vol[-1] if len(ewma_vol) > 0 else self.target_vol

        elif method == VolScalingMethod.GARCH:
            # Simple GARCH(1,1) approximation
            vol_forecast = self._garch_forecast(returns)

        else:
            # CVS and DVS use realized volatility
            recent_returns = returns[-self.vol_lookback:]
            vol_forecast = float(np.std(recent_returns, ddof=1) * SQRT_252)

        # Apply floor and cap
        vol_forecast = np.clip(vol_forecast, self.vol_floor, self.vol_cap)

        return float(vol_forecast)

    def calculate_momentum_signal(
        self,
        prices: NDArray[np.float64],
        lookback: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate momentum signal.

        Time-series momentum: Sign of past N-period return.
        Magnitude: Normalized by volatility.

        Args:
            prices: Price series
            lookback: Override momentum lookback

        Returns:
            Tuple of (signal, momentum_return)
            signal in [-1, +1], momentum_return is raw return
        """
        lookback = lookback or self.momentum_lookback

        if len(prices) < lookback + 1:
            return 0.0, 0.0

        # Calculate return over lookback period
        momentum_return = (prices[-1] / prices[-lookback - 1]) - 1

        # Calculate volatility for normalization
        returns = simple_returns(prices)
        if len(returns) >= 20:
            vol = np.std(returns[-self.vol_lookback:], ddof=1) * SQRT_252
            vol = max(vol, self.vol_floor)
        else:
            vol = self.target_vol

        # Normalize momentum by volatility (risk-adjusted)
        momentum_zscore = momentum_return / (vol / SQRT_252 * np.sqrt(lookback))

        # Convert to signal in [-1, +1] using tanh
        signal = np.tanh(momentum_zscore / 2)  # Divide by 2 to moderate

        return float(signal), float(momentum_return)

    def calculate_vol_scalar(
        self,
        forecasted_vol: float,
        sharpe_forecast: float = None
    ) -> float:
        """
        Calculate volatility scaling factor.

        CVS: scalar = σ_target / σ_forecast
        DVS: scalar = (σ_target / σ_forecast) × (SR_forecast / SR_avg)

        Args:
            forecasted_vol: Forecasted volatility
            sharpe_forecast: Forecasted Sharpe (for DVS only)

        Returns:
            Volatility scalar
        """
        if forecasted_vol < EPSILON:
            forecasted_vol = self.vol_floor

        # Base CVS scalar
        base_scalar = self.target_vol / forecasted_vol

        if self.method == VolScalingMethod.DVS and sharpe_forecast is not None:
            # DVS additionally scales by predicted Sharpe
            # Normalize to average Sharpe of ~0.5 for momentum
            avg_sharpe = 0.5
            sharpe_scalar = sharpe_forecast / avg_sharpe if avg_sharpe > 0 else 1.0
            sharpe_scalar = np.clip(sharpe_scalar, 0.5, 2.0)
            base_scalar *= sharpe_scalar

        # Apply maximum scalar limit
        scalar = np.clip(base_scalar, 0.0, self.max_scalar)

        return float(scalar)

    def generate_signal(
        self,
        prices: NDArray[np.float64],
        returns: Optional[NDArray[np.float64]] = None,
        symbol: str = "default"
    ) -> VolManagedSignal:
        """
        Generate volatility-managed momentum signal.

        Args:
            prices: Price series
            returns: Return series (calculated if not provided)
            symbol: Symbol identifier

        Returns:
            VolManagedSignal with scaled position size
        """
        if returns is None:
            returns = simple_returns(prices)

        # Calculate momentum signal
        raw_signal, momentum_return = self.calculate_momentum_signal(prices)

        # Forecast volatility
        current_vol = self.forecast_volatility(returns)

        # Calculate Sharpe forecast for DVS
        sharpe_forecast = self._forecast_sharpe(returns) if self.method == VolScalingMethod.DVS else 0.0

        # Calculate volatility scalar
        vol_scalar = self.calculate_vol_scalar(current_vol, sharpe_forecast)

        # Scale position
        position_size = raw_signal * vol_scalar

        # Store history
        if symbol not in self._vol_history:
            self._vol_history[symbol] = np.array([])
        self._vol_history[symbol] = np.append(self._vol_history[symbol], current_vol)[-252:]

        return VolManagedSignal(
            raw_signal=raw_signal,
            vol_scalar=vol_scalar,
            position_size=position_size,
            current_vol=current_vol,
            target_vol=self.target_vol,
            momentum_return=momentum_return,
            sharpe_forecast=sharpe_forecast,
        )

    def get_vol_regime(
        self,
        current_vol: float,
        symbol: str = "default"
    ) -> VolRegime:
        """
        Classify current volatility regime.

        Args:
            current_vol: Current volatility
            symbol: Symbol identifier

        Returns:
            VolRegime classification
        """
        vol_history = self._vol_history.get(symbol, np.array([current_vol]))

        if len(vol_history) < 10:
            percentile = 50.0
        else:
            percentile = float(np.sum(vol_history < current_vol) / len(vol_history) * 100)

        # Classify regime
        if percentile < 20:
            regime = "low"
        elif percentile < 80:
            regime = "normal"
        elif percentile < 95:
            regime = "high"
        else:
            regime = "crisis"

        # Calculate vol trend
        if len(vol_history) >= 5:
            recent_vol = vol_history[-5:]
            vol_trend = (recent_vol[-1] - recent_vol[0]) / recent_vol[0] if recent_vol[0] > 0 else 0.0
        else:
            vol_trend = 0.0

        return VolRegime(
            current_vol=current_vol,
            vol_percentile=percentile,
            regime=regime,
            vol_trend=float(vol_trend),
        )

    def _garch_forecast(self, returns: NDArray[np.float64]) -> float:
        """
        Simple GARCH(1,1) volatility forecast.

        σ²_{t+1} = ω + α*ε²_t + β*σ²_t

        Using typical parameters: α=0.1, β=0.85, ω=0.05*var
        """
        if len(returns) < 20:
            return self.target_vol

        # GARCH(1,1) parameters (typical values)
        alpha = 0.10
        beta = 0.85
        omega = (1 - alpha - beta) * np.var(returns, ddof=1)

        # Initialize variance with sample variance
        variance = np.var(returns[-self.vol_lookback:], ddof=1)

        # One-step ahead forecast
        last_return = returns[-1]
        next_variance = omega + alpha * last_return**2 + beta * variance

        return float(np.sqrt(next_variance) * SQRT_252)

    def _forecast_sharpe(self, returns: NDArray[np.float64]) -> float:
        """
        Forecast Sharpe ratio for DVS method.

        Uses recent realized Sharpe as a predictor.
        """
        if len(returns) < 60:
            return 0.5  # Assume average

        recent = returns[-60:]
        mean_return = np.mean(recent)
        std_return = np.std(recent, ddof=1)

        if std_return < EPSILON:
            return 0.0

        sharpe = mean_return / std_return * SQRT_252
        return float(np.clip(sharpe, 0.0, 2.0))


# =============================================================================
# MULTI-ASSET VOLATILITY MANAGEMENT
# =============================================================================

class VolatilityTargetingPortfolio:
    """
    Portfolio-level volatility targeting.

    Scales entire portfolio to maintain target volatility,
    considering asset correlations.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback: int = 60,
        rebalance_threshold: float = 0.05,  # Rebalance if vol deviates by 5%
    ):
        self.target_vol = target_vol
        self.lookback = lookback
        self.rebalance_threshold = rebalance_threshold

    def calculate_portfolio_scalar(
        self,
        weights: Dict[str, float],
        returns_matrix: NDArray[np.float64],
        symbols: list,
    ) -> float:
        """
        Calculate portfolio-level volatility scalar.

        Args:
            weights: Current portfolio weights
            returns_matrix: T x N return matrix
            symbols: List of symbols

        Returns:
            Scalar to apply to all positions
        """
        if returns_matrix.shape[0] < self.lookback:
            return 1.0

        n = len(symbols)
        weight_array = np.array([weights.get(s, 0.0) for s in symbols])

        # Calculate covariance matrix
        recent_returns = returns_matrix[-self.lookback:]
        cov_matrix = np.cov(recent_returns, rowvar=False) * TRADING_DAYS_PER_YEAR

        # Portfolio variance
        port_var = weight_array @ cov_matrix @ weight_array
        port_vol = np.sqrt(max(port_var, EPSILON))

        # Calculate scalar
        if port_vol < EPSILON:
            return 1.0

        scalar = self.target_vol / port_vol
        return float(np.clip(scalar, 0.5, 3.0))

    def scale_weights(
        self,
        weights: Dict[str, float],
        scalar: float
    ) -> Dict[str, float]:
        """Scale all weights by the portfolio scalar."""
        return {symbol: weight * scalar for symbol, weight in weights.items()}
