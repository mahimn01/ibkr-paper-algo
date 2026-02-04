"""
Volatility Maximizer Strategy for FX Markets

Philosophy: PROFIT MAXIMIZATION, not risk minimization
- Accept 50-80% drawdowns for 100%+ returns
- Exploit volatility explosions
- Use leverage aggressively when edge is present
- Statistically sophisticated but profit-focused

Target: 50-150% annual returns in FX/volatile markets
Expected Max Drawdown: 50-80%

Components:
1. GARCH volatility forecasting
2. Carry trade signals (interest rate differentials)
3. Volatility breakout detection
4. Correlation breakdown exploitation
5. Beyond-Kelly position sizing for max growth
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
import warnings
warnings.filterwarnings('ignore')


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    ULTRA_LOW = auto()      # Vol < 5th percentile - DANGER, vol explosion coming
    LOW = auto()            # Vol < 25th percentile
    NORMAL = auto()         # 25th-75th percentile
    HIGH = auto()           # Vol > 75th percentile
    CRISIS = auto()         # Vol > 95th percentile - opportunity for reversals


class TradingOpportunity(Enum):
    """Type of profit opportunity."""
    CARRY = auto()              # Interest rate differential
    MOMENTUM = auto()           # Trend continuation
    VOLATILITY_BREAKOUT = auto()  # Vol explosion
    MEAN_REVERSION = auto()     # Overbought/oversold in ranging market
    CORRELATION_BREAKDOWN = auto()  # Normal correlations breaking


@dataclass
class GARCHForecast:
    """GARCH(1,1) volatility forecast."""
    current_vol: float
    forecast_vol: float
    vol_regime: VolatilityRegime
    vol_percentile: float

    # GARCH parameters
    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0


@dataclass
class VolMaxSignal:
    """Volatility maximizer signal."""
    symbol: str
    opportunity_type: TradingOpportunity
    conviction: float          # 0 to 1 - how strong is the signal
    expected_return: float     # Expected return (can be >1 for leverage)
    expected_vol: float        # Expected volatility
    position_size: float       # Target position (-2 to +2 for 200% leverage)

    # Supporting data
    carry: float = 0.0
    momentum_score: float = 0.0
    vol_forecast: Optional[GARCHForecast] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class VolMaxConfig:
    """Configuration for volatility maximizer."""

    # Volatility forecasting
    vol_lookback: int = 60              # Days for vol estimation
    garch_estimation_window: int = 252  # Days for GARCH calibration

    # Signal weights (sum to 1.0)
    carry_weight: float = 0.20
    momentum_weight: float = 0.40
    vol_breakout_weight: float = 0.30
    correlation_weight: float = 0.10

    # Position sizing (AGGRESSIVE)
    target_leverage: float = 2.5        # 250% gross exposure
    max_position: float = 1.0           # 100% in single position allowed
    beyond_kelly_multiplier: float = 1.5  # Use 1.5x Kelly for max growth

    # Opportunity thresholds
    min_carry: float = 0.01             # 1% annual carry minimum
    min_momentum: float = 0.05          # 5% momentum threshold
    vol_breakout_threshold: float = 1.5  # 1.5x normal vol = breakout

    # Risk tolerance (HIGH)
    max_drawdown: float = 0.80          # Accept 80% drawdown
    max_correlation: float = 0.95       # Almost no diversification requirement
    accept_fat_tails: bool = True       # Don't reduce for tail risk

    # Volatility targeting
    target_portfolio_vol: float = 0.40  # 40% annual vol target
    vol_scale_positions: bool = True

    # Profit taking / Stop loss
    use_stops: bool = False             # No stops - let winners run
    profit_take_multiple: float = 5.0  # Take profit at 5x expected
    trailing_stop: float = 0.30         # 30% trailing stop from peak


class VolatilityMaximizer:
    """
    Sophisticated volatility exploitation for maximum profits.

    Uses GARCH forecasting, carry trades, momentum, and volatility
    breakouts to generate aggressive profit-seeking trades.

    NOT designed for risk-adjusted returns. Designed for ABSOLUTE returns.
    """

    def __init__(self, config: Optional[VolMaxConfig] = None):
        self.config = config or VolMaxConfig()

        # State
        self._returns_history: Dict[str, NDArray] = {}
        self._vol_history: Dict[str, List[float]] = {}
        self._carry_rates: Dict[str, float] = {}  # Simulated carry
        self._garch_params: Dict[str, GARCHForecast] = {}
        self._correlation_matrix: Optional[NDArray] = None

    def estimate_garch(
        self,
        returns: NDArray[np.float64],
    ) -> GARCHForecast:
        """
        Estimate GARCH(1,1) parameters and forecast volatility.

        GARCH(1,1): σ²(t+1) = ω + α*ε²(t) + β*σ²(t)

        Args:
            returns: Historical returns

        Returns:
            GARCHForecast with parameters and forecast
        """
        if len(returns) < 30:
            # Not enough data - use simple vol
            vol = float(np.std(returns) * np.sqrt(252))
            return GARCHForecast(
                current_vol=vol,
                forecast_vol=vol,
                vol_regime=VolatilityRegime.NORMAL,
                vol_percentile=0.5,
            )

        # Use maximum likelihood estimation (simplified)
        # In production, use arch library: arch_model(returns, vol='Garch', p=1, q=1)

        # Simple variance targeting GARCH
        unconditional_var = np.var(returns)

        # Estimate with moment matching
        returns_sq = returns ** 2

        # Initial guesses
        omega = unconditional_var * 0.05
        alpha = 0.10
        beta = 0.85

        # Simple one-pass estimation (in production, use MLE optimization)
        var_t = unconditional_var
        forecast_var = omega + alpha * returns[-1]**2 + beta * var_t
        forecast_vol = float(np.sqrt(forecast_var * 252))
        current_vol = float(np.std(returns[-20:]) * np.sqrt(252))

        # Classify regime
        n_periods = len(returns) // 20
        if n_periods >= 5:
            vol_history = np.std(returns[:n_periods*20].reshape(-1, 20), axis=1) * np.sqrt(252)
            percentile = float(np.mean(vol_history <= current_vol))
        else:
            percentile = 0.5  # Not enough history

        if percentile < 0.05:
            regime = VolatilityRegime.ULTRA_LOW
        elif percentile < 0.25:
            regime = VolatilityRegime.LOW
        elif percentile > 0.95:
            regime = VolatilityRegime.CRISIS
        elif percentile > 0.75:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.NORMAL

        return GARCHForecast(
            current_vol=current_vol,
            forecast_vol=forecast_vol,
            vol_regime=regime,
            vol_percentile=percentile,
            omega=omega,
            alpha=alpha,
            beta=beta,
        )

    def calculate_carry(
        self,
        symbol: str,
        price_history: NDArray[np.float64],
    ) -> float:
        """
        Calculate carry trade opportunity.

        In real FX trading, this would use interest rate differentials.
        For simulation, we estimate from price drift vs volatility.

        Args:
            symbol: Currency pair
            price_history: Historical prices

        Returns:
            Annualized carry (can be negative)
        """
        if len(price_history) < 60:
            return 0.0

        # Calculate drift vs volatility
        returns = np.diff(price_history) / price_history[:-1]

        # Carry = drift component after removing momentum
        # Use longer period for drift, shorter for vol
        drift = float(np.mean(returns[-252:]) * 252 if len(returns) >= 252 else np.mean(returns) * 252)
        vol = float(np.std(returns[-60:]) * np.sqrt(252))

        # Carry is persistent drift that's not explained by momentum
        # Simple proxy: if drift/vol ratio is stable, that's carry
        sharpe = drift / vol if vol > 0 else 0

        # Positive Sharpe = positive carry opportunity
        carry = drift if abs(sharpe) > 0.3 else 0.0

        return carry

    def detect_volatility_breakout(
        self,
        garch_forecast: GARCHForecast,
        returns: NDArray[np.float64],
    ) -> Tuple[bool, float]:
        """
        Detect volatility breakout opportunities.

        Volatility explosions create profit opportunities:
        - Long volatility when it's about to spike
        - Long directional when vol spike confirms trend

        Args:
            garch_forecast: GARCH volatility forecast
            returns: Recent returns

        Returns:
            (is_breakout, breakout_strength 0-1)
        """
        if len(returns) < 20:
            return False, 0.0

        # Check for vol expansion
        recent_vol = float(np.std(returns[-10:]) * np.sqrt(252))
        normal_vol = garch_forecast.current_vol

        vol_ratio = recent_vol / normal_vol if normal_vol > 0 else 1.0

        # Breakout if vol expanding rapidly
        is_breakout = vol_ratio > self.config.vol_breakout_threshold

        # Strength based on how extreme the breakout
        strength = min(1.0, (vol_ratio - 1.0) / 2.0)  # 0 at 1x, 1.0 at 3x vol

        # Ultra-low vol regime = imminent breakout
        if garch_forecast.vol_regime == VolatilityRegime.ULTRA_LOW:
            strength = max(strength, 0.6)  # High conviction on vol expansion

        return is_breakout, float(strength)

    def calculate_beyond_kelly(
        self,
        expected_return: float,
        expected_vol: float,
        conviction: float,
    ) -> float:
        """
        Calculate position size using Beyond-Kelly criterion.

        Kelly = μ / σ²
        Beyond Kelly = Kelly * multiplier (for max growth, accepting higher variance)

        Args:
            expected_return: Expected return (annualized)
            expected_vol: Expected volatility (annualized)
            conviction: Conviction 0-1

        Returns:
            Position size (can be >1 for leverage)
        """
        if expected_vol <= 0 or expected_return <= 0:
            return 0.0

        # Classic Kelly fraction
        kelly = expected_return / (expected_vol ** 2)

        # Beyond Kelly for maximum growth (accepts more variance)
        beyond_kelly = kelly * self.config.beyond_kelly_multiplier

        # Scale by conviction
        position = beyond_kelly * conviction

        # Cap at max position
        position = min(position, self.config.max_position)

        return float(position)

    def generate_signals(
        self,
        symbols: List[str],
        prices: Dict[str, NDArray[np.float64]],
        current_time: Optional[datetime] = None,
    ) -> Dict[str, VolMaxSignal]:
        """
        Generate volatility maximizer signals.

        Args:
            symbols: List of symbols
            prices: Dict of symbol -> price history
            current_time: Current timestamp

        Returns:
            Dict of symbol -> VolMaxSignal
        """
        signals = {}
        returns_dict = {}

        # 1. Calculate returns and GARCH forecasts
        for symbol in symbols:
            if symbol not in prices or len(prices[symbol]) < 60:
                continue

            price_array = prices[symbol]
            returns = np.diff(price_array) / price_array[:-1]
            returns_dict[symbol] = returns

            # GARCH forecast
            garch = self.estimate_garch(returns[-252:] if len(returns) >= 252 else returns)
            self._garch_params[symbol] = garch

            # Calculate carry
            carry = self.calculate_carry(symbol, price_array)
            self._carry_rates[symbol] = carry

        # 2. Update correlation matrix
        if len(returns_dict) >= 2:
            min_len = min(len(r) for r in returns_dict.values())
            if min_len >= 30:
                returns_matrix = np.column_stack([
                    returns_dict[s][-min_len:] for s in symbols if s in returns_dict
                ])
                self._correlation_matrix = np.corrcoef(returns_matrix.T)

        # 3. Generate signals for each symbol
        for symbol in symbols:
            if symbol not in returns_dict or symbol not in self._garch_params:
                continue

            returns = returns_dict[symbol]
            garch = self._garch_params[symbol]
            carry = self._carry_rates.get(symbol, 0.0)

            # Calculate momentum
            lookback = min(60, len(returns))
            momentum = float((prices[symbol][-1] / prices[symbol][-lookback]) - 1)
            momentum_score = np.tanh(momentum * 5)  # Scale to -1 to +1

            # Detect volatility breakout
            is_breakout, breakout_strength = self.detect_volatility_breakout(garch, returns[-30:])

            # Determine primary opportunity type
            opportunities = []

            # Carry opportunity
            if abs(carry) >= self.config.min_carry:
                carry_conviction = min(1.0, abs(carry) / 0.05)  # 5% carry = 100% conviction
                opportunities.append((
                    TradingOpportunity.CARRY,
                    carry_conviction * self.config.carry_weight,
                    carry,
                    garch.forecast_vol * 0.5,  # Carry trades are lower vol
                ))

            # Momentum opportunity
            if abs(momentum) >= self.config.min_momentum:
                mom_conviction = min(1.0, abs(momentum) / 0.20)  # 20% move = 100% conviction
                opportunities.append((
                    TradingOpportunity.MOMENTUM,
                    mom_conviction * self.config.momentum_weight,
                    momentum,
                    garch.forecast_vol,
                ))

            # Volatility breakout opportunity
            if is_breakout or garch.vol_regime == VolatilityRegime.ULTRA_LOW:
                # Trade in direction of momentum during vol breakout
                direction = np.sign(momentum) if abs(momentum) > 0.02 else 0
                vol_return = abs(momentum) * 2 if is_breakout else 0.10  # Expect big move
                opportunities.append((
                    TradingOpportunity.VOLATILITY_BREAKOUT,
                    breakout_strength * self.config.vol_breakout_weight,
                    vol_return * direction,
                    garch.forecast_vol * 1.5,  # Higher vol during breakout
                ))

            # Mean reversion in crisis vol
            if garch.vol_regime == VolatilityRegime.CRISIS:
                # Fade extreme moves in crisis
                if abs(momentum_score) > 0.7:
                    reversion_conviction = abs(momentum_score) * 0.5
                    opportunities.append((
                        TradingOpportunity.MEAN_REVERSION,
                        reversion_conviction * 0.3,
                        -momentum * 0.5,  # Fade half the move
                        garch.current_vol,
                    ))

            # Select best opportunity
            if not opportunities:
                continue

            opp_type, conviction, expected_ret, expected_vol = max(
                opportunities, key=lambda x: x[1]
            )

            # Calculate position size using Beyond-Kelly
            position = self.calculate_beyond_kelly(
                abs(expected_ret),
                expected_vol,
                conviction,
            )

            # Apply sign
            position *= np.sign(expected_ret)

            # Volatility scaling
            if self.config.vol_scale_positions:
                target_contribution = self.config.target_portfolio_vol / len(symbols)
                vol_scalar = target_contribution / expected_vol if expected_vol > 0 else 1.0
                position *= vol_scalar

            signals[symbol] = VolMaxSignal(
                symbol=symbol,
                opportunity_type=opp_type,
                conviction=conviction,
                expected_return=expected_ret,
                expected_vol=expected_vol,
                position_size=position,
                carry=carry,
                momentum_score=momentum_score,
                vol_forecast=garch,
                metadata={
                    'is_breakout': is_breakout,
                    'breakout_strength': breakout_strength,
                }
            )

        return signals

    def scale_to_target_leverage(
        self,
        signals: Dict[str, VolMaxSignal],
    ) -> Dict[str, float]:
        """
        Scale positions to target leverage.

        Args:
            signals: Raw signals

        Returns:
            Dict of symbol -> final position weight
        """
        # Sum absolute positions
        gross_exposure = sum(abs(s.position_size) for s in signals.values())

        if gross_exposure == 0:
            return {}

        # Scale to target leverage
        scale_factor = self.config.target_leverage / gross_exposure

        weights = {
            symbol: signal.position_size * scale_factor
            for symbol, signal in signals.items()
            if abs(signal.position_size) > 0.01
        }

        return weights

    def reset(self) -> None:
        """Reset strategy state."""
        self._returns_history.clear()
        self._vol_history.clear()
        self._carry_rates.clear()
        self._garch_params.clear()
        self._correlation_matrix = None
