"""
Ornstein-Uhlenbeck Mean Reversion Model

Implementation based on Avellaneda & Lee (2010):
"Statistical Arbitrage in the U.S. Equities Market"

The OU process is defined by the SDE:
    dX_t = κ(θ - X_t)dt + σdW_t

where:
    - κ: Mean reversion speed (higher = faster reversion)
    - θ: Long-term mean
    - σ: Volatility
    - W_t: Brownian motion

Key metrics:
    - Half-life: τ = -ln(2)/κ (time to revert halfway to mean)
    - Equilibrium variance: σ²_eq = σ² / (2κ)
    - S-score: s = (X_t - θ) / σ_eq

Trading rules (Avellaneda & Lee):
    - Enter when |s| > 1.25 (expect reversion)
    - Exit when |s| < 0.50 (near equilibrium)
    - Stop when |s| > 4.0 (regime break)

References:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505
    - https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/optimal_mean_reversion/ou_model.html
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import (
    OU_DEFAULT_LOOKBACK,
    OU_S_SCORE_ENTRY,
    OU_S_SCORE_EXIT,
    OU_S_SCORE_STOP,
    OU_MIN_KAPPA,
    LN_2,
    MIN_HALF_LIFE,
    MAX_HALF_LIFE,
    EPSILON,
)
from trading_algo.quant_core.utils.math_utils import ols_regression


class OUSignal(Enum):
    """Trading signals from OU model."""
    LONG = auto()       # Enter long (s < -entry_threshold)
    SHORT = auto()      # Enter short (s > entry_threshold)
    EXIT_LONG = auto()  # Exit long position
    EXIT_SHORT = auto() # Exit short position
    STOP_LONG = auto()  # Stop loss on long
    STOP_SHORT = auto() # Stop loss on short
    HOLD = auto()       # No action


@dataclass
class OUParameters:
    """
    Ornstein-Uhlenbeck process parameters.

    Estimated from price data using regression method.
    """
    kappa: float          # Mean reversion speed
    theta: float          # Long-term mean
    sigma: float          # Volatility
    half_life: float      # Days to revert halfway
    sigma_eq: float       # Equilibrium standard deviation
    r_squared: float      # Regression R² (goodness of fit)
    is_valid: bool        # Whether parameters indicate mean reversion

    @classmethod
    def invalid(cls) -> "OUParameters":
        """Return invalid parameters when estimation fails."""
        return cls(
            kappa=0.0,
            theta=0.0,
            sigma=0.0,
            half_life=MAX_HALF_LIFE,
            sigma_eq=1.0,
            r_squared=0.0,
            is_valid=False,
        )


@dataclass
class OUState:
    """Current state of OU model for a symbol."""
    symbol: str
    params: OUParameters
    current_value: float
    s_score: float
    last_signal: OUSignal
    position: int  # 1 = long, -1 = short, 0 = flat
    entry_price: float
    entry_s_score: float


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck mean reversion model.

    Implements the statistical arbitrage approach from Avellaneda & Lee (2010).

    Usage:
        ou = OrnsteinUhlenbeck(lookback=60)
        ou.fit(price_series)
        signal = ou.get_signal(current_price)

    The model estimates OU parameters from price data using regression:
        Δy_t = κ(θ - y_{t-1})Δt + σε_t

    which can be rewritten as:
        Δy_t = a + b*y_{t-1} + ε_t

    where:
        b = -κΔt (should be negative for mean reversion)
        a = κθΔt
        κ = -b/Δt
        θ = -a/b
    """

    def __init__(
        self,
        lookback: int = OU_DEFAULT_LOOKBACK,
        entry_threshold: float = OU_S_SCORE_ENTRY,
        exit_threshold: float = OU_S_SCORE_EXIT,
        stop_threshold: float = OU_S_SCORE_STOP,
        min_kappa: float = OU_MIN_KAPPA,
        min_half_life: float = MIN_HALF_LIFE,
        max_half_life: float = MAX_HALF_LIFE,
    ):
        """
        Initialize OU model.

        Args:
            lookback: Number of periods for parameter estimation
            entry_threshold: S-score threshold for entry (default 1.25)
            exit_threshold: S-score threshold for exit (default 0.50)
            stop_threshold: S-score threshold for stop loss (default 4.0)
            min_kappa: Minimum mean reversion speed
            min_half_life: Minimum half-life in days
            max_half_life: Maximum half-life in days
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_threshold = stop_threshold
        self.min_kappa = min_kappa
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

        # Internal state
        self._params: Optional[OUParameters] = None
        self._states: dict[str, OUState] = {}
        self._price_history: dict[str, List[float]] = {}

    def fit(
        self,
        prices: NDArray[np.float64],
        symbol: str = "default"
    ) -> OUParameters:
        """
        Fit OU parameters to price series.

        Uses regression method:
            Δy_t = a + b*y_{t-1} + ε_t
            where b should be negative for mean reversion

        Args:
            prices: Price series (or spread series for pairs)
            symbol: Symbol identifier

        Returns:
            Estimated OUParameters
        """
        if len(prices) < self.lookback:
            return OUParameters.invalid()

        # Use recent lookback window
        y = prices[-self.lookback:]

        # Calculate changes
        delta_y = np.diff(y)
        y_lagged = y[:-1]

        # OLS regression: Δy = a + b*y_{-1}
        try:
            intercept, slope, r_squared = ols_regression(delta_y, y_lagged)
        except Exception:
            return OUParameters.invalid()

        # Extract OU parameters (assuming Δt = 1 day)
        # slope = -κ (should be negative for mean reversion)
        if slope >= 0:
            return OUParameters.invalid()

        kappa = -slope
        theta = intercept / kappa if kappa > EPSILON else 0.0

        # Calculate sigma from residuals
        predicted = intercept + slope * y_lagged
        residuals = delta_y - predicted
        sigma = np.std(residuals, ddof=2)

        # Calculate half-life
        half_life = -LN_2 / slope if slope < 0 else MAX_HALF_LIFE
        half_life = np.clip(half_life, self.min_half_life, self.max_half_life)

        # Calculate equilibrium standard deviation
        # σ_eq = σ / sqrt(2κ)
        sigma_eq = sigma / np.sqrt(2 * kappa) if kappa > EPSILON else sigma

        # Validate: kappa should be high enough for tradeable mean reversion
        is_valid = (
            kappa >= self.min_kappa and
            half_life <= self.max_half_life and
            sigma_eq > EPSILON and
            r_squared > 0.01  # Some explanatory power
        )

        params = OUParameters(
            kappa=float(kappa),
            theta=float(theta),
            sigma=float(sigma),
            half_life=float(half_life),
            sigma_eq=float(sigma_eq),
            r_squared=float(r_squared),
            is_valid=is_valid,
        )

        self._params = params
        return params

    def calculate_s_score(
        self,
        current_value: float,
        params: Optional[OUParameters] = None
    ) -> float:
        """
        Calculate s-score (standardized deviation from equilibrium).

        s = (X - θ) / σ_eq

        Args:
            current_value: Current price or spread value
            params: OU parameters (uses fitted params if None)

        Returns:
            S-score (number of equilibrium std devs from mean)
        """
        if params is None:
            params = self._params

        if params is None or not params.is_valid:
            return 0.0

        if params.sigma_eq < EPSILON:
            return 0.0

        return (current_value - params.theta) / params.sigma_eq

    def get_signal(
        self,
        current_value: float,
        symbol: str = "default",
        params: Optional[OUParameters] = None
    ) -> Tuple[OUSignal, float]:
        """
        Generate trading signal based on current s-score.

        Trading rules:
        - Enter LONG when s < -entry_threshold
        - Enter SHORT when s > entry_threshold
        - Exit when |s| < exit_threshold
        - Stop when |s| > stop_threshold (against position)

        Args:
            current_value: Current price or spread
            symbol: Symbol identifier
            params: OU parameters (uses fitted if None)

        Returns:
            Tuple of (signal, s_score)
        """
        if params is None:
            params = self._params

        if params is None or not params.is_valid:
            return OUSignal.HOLD, 0.0

        s_score = self.calculate_s_score(current_value, params)

        # Get or create state
        if symbol not in self._states:
            self._states[symbol] = OUState(
                symbol=symbol,
                params=params,
                current_value=current_value,
                s_score=s_score,
                last_signal=OUSignal.HOLD,
                position=0,
                entry_price=0.0,
                entry_s_score=0.0,
            )

        state = self._states[symbol]
        state.s_score = s_score
        state.current_value = current_value
        state.params = params

        # Determine signal based on position
        if state.position == 0:
            # No position - look for entry
            if s_score < -self.entry_threshold:
                signal = OUSignal.LONG
            elif s_score > self.entry_threshold:
                signal = OUSignal.SHORT
            else:
                signal = OUSignal.HOLD

        elif state.position > 0:
            # Long position - look for exit or stop
            if s_score > -self.exit_threshold:
                # Near equilibrium - take profit
                signal = OUSignal.EXIT_LONG
            elif s_score > self.stop_threshold:
                # S-score moved against us significantly - stop loss
                signal = OUSignal.STOP_LONG
            else:
                signal = OUSignal.HOLD

        else:
            # Short position - look for exit or stop
            if s_score < self.exit_threshold:
                # Near equilibrium - take profit
                signal = OUSignal.EXIT_SHORT
            elif s_score < -self.stop_threshold:
                # S-score moved against us significantly - stop loss
                signal = OUSignal.STOP_SHORT
            else:
                signal = OUSignal.HOLD

        state.last_signal = signal
        return signal, s_score

    def update_position(
        self,
        symbol: str,
        position: int,
        entry_price: float = 0.0
    ) -> None:
        """
        Update position state after trade execution.

        Args:
            symbol: Symbol identifier
            position: New position (1=long, -1=short, 0=flat)
            entry_price: Entry price if opening position
        """
        if symbol not in self._states:
            return

        state = self._states[symbol]
        state.position = position
        if position != 0:
            state.entry_price = entry_price
            state.entry_s_score = state.s_score
        else:
            state.entry_price = 0.0
            state.entry_s_score = 0.0

    def expected_return(
        self,
        current_value: float,
        horizon: int = 1,
        params: Optional[OUParameters] = None
    ) -> float:
        """
        Calculate expected return over given horizon.

        E[X_{t+Δ} | X_t] = θ + (X_t - θ) * exp(-κΔ)

        Args:
            current_value: Current value
            horizon: Number of periods ahead
            params: OU parameters

        Returns:
            Expected change in value
        """
        if params is None:
            params = self._params

        if params is None or not params.is_valid:
            return 0.0

        expected_value = params.theta + (current_value - params.theta) * np.exp(-params.kappa * horizon)
        return expected_value - current_value

    def variance_at_horizon(
        self,
        horizon: int = 1,
        params: Optional[OUParameters] = None
    ) -> float:
        """
        Calculate variance of value at given horizon.

        Var[X_{t+Δ}] = (σ²/2κ) * (1 - exp(-2κΔ))

        Args:
            horizon: Number of periods ahead
            params: OU parameters

        Returns:
            Variance at horizon
        """
        if params is None:
            params = self._params

        if params is None or not params.is_valid:
            return 0.0

        return (params.sigma ** 2 / (2 * params.kappa)) * (1 - np.exp(-2 * params.kappa * horizon))

    def optimal_entry_threshold(
        self,
        transaction_cost: float = 0.001,
        params: Optional[OUParameters] = None
    ) -> float:
        """
        Calculate optimal entry threshold considering transaction costs.

        Based on optimal stopping theory for OU process.

        Args:
            transaction_cost: Round-trip transaction cost as fraction
            params: OU parameters

        Returns:
            Optimal entry threshold in s-score units
        """
        if params is None:
            params = self._params

        if params is None or not params.is_valid:
            return self.entry_threshold

        # Cost in s-score units
        cost_in_sigma = transaction_cost / params.sigma_eq if params.sigma_eq > EPSILON else 0

        # Optimal threshold increases with cost and decreases with kappa
        # Approximation based on optimal stopping literature
        optimal = self.entry_threshold + cost_in_sigma * (1 + 1 / params.kappa)

        return min(optimal, 3.0)  # Cap at 3 sigma

    def get_state(self, symbol: str = "default") -> Optional[OUState]:
        """Get current state for a symbol."""
        return self._states.get(symbol)

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset model state.

        Args:
            symbol: Symbol to reset (None = reset all)
        """
        if symbol is None:
            self._states.clear()
            self._params = None
        elif symbol in self._states:
            del self._states[symbol]


# =============================================================================
# PAIRS TRADING EXTENSION
# =============================================================================

class OUPairsTrading:
    """
    Pairs trading using OU model on spread.

    Models the spread between two cointegrated assets as an OU process.
    """

    def __init__(
        self,
        lookback: int = OU_DEFAULT_LOOKBACK,
        entry_threshold: float = OU_S_SCORE_ENTRY,
        exit_threshold: float = OU_S_SCORE_EXIT,
    ):
        self.lookback = lookback
        self.ou_model = OrnsteinUhlenbeck(
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        self._hedge_ratio: float = 1.0
        self._spread_mean: float = 0.0

    def fit(
        self,
        prices_a: NDArray[np.float64],
        prices_b: NDArray[np.float64]
    ) -> Tuple[OUParameters, float]:
        """
        Fit OU model to spread between two assets.

        Estimates hedge ratio using OLS:
            log(P_A) = α + β * log(P_B) + ε

        Spread = log(P_A) - β * log(P_B)

        Args:
            prices_a: Prices of asset A (long leg)
            prices_b: Prices of asset B (short leg)

        Returns:
            Tuple of (OU parameters, hedge ratio)
        """
        if len(prices_a) != len(prices_b):
            raise ValueError("Price series must have same length")

        if len(prices_a) < self.lookback:
            return OUParameters.invalid(), 1.0

        # Calculate log prices
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        # Estimate hedge ratio using regression
        _, hedge_ratio, _ = ols_regression(log_a[-self.lookback:], log_b[-self.lookback:])
        self._hedge_ratio = hedge_ratio

        # Calculate spread
        spread = log_a - hedge_ratio * log_b

        # Fit OU to spread
        params = self.ou_model.fit(spread, symbol="spread")

        return params, hedge_ratio

    def get_spread(
        self,
        price_a: float,
        price_b: float
    ) -> float:
        """Calculate current spread value."""
        return np.log(price_a) - self._hedge_ratio * np.log(price_b)

    def get_signal(
        self,
        price_a: float,
        price_b: float
    ) -> Tuple[OUSignal, float]:
        """
        Generate trading signal for the pair.

        Returns:
            Tuple of (signal, s_score)
            LONG = long A, short B
            SHORT = short A, long B
        """
        spread = self.get_spread(price_a, price_b)
        return self.ou_model.get_signal(spread, symbol="spread")

    @property
    def hedge_ratio(self) -> float:
        """Get current hedge ratio."""
        return self._hedge_ratio
