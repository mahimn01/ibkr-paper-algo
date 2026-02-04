"""
Time Series Momentum (TSMOM)

Implementation based on Moskowitz, Ooi & Pedersen (2012):
"Time Series Momentum" - Journal of Financial Economics

Key Findings:
    - Positive predictability from a security's own past returns
    - Works across all 58 liquid futures/forward contracts tested
    - Diversified TSMOM portfolios achieve Sharpe ratios > 1.0
    - Effect persists for about 12 months, then partially reverses

Strategy:
    - Go LONG if past 12-month return is positive
    - Go SHORT if past 12-month return is negative
    - Scale positions by inverse volatility (risk parity)
    - Hold for 1 month, then rebalance

Critical Insight from Research:
    - TSMOM only significantly outperforms buy-and-hold if volatility
      scaling is applied
    - Without scaling, TSMOM alpha drops from 1.08% to 0.39%

Portfolio Construction:
    - Each position sized to equal risk contribution
    - Position_i = (σ_target / σ_i) × sign(r_{12m})
    - σ_target typically 10-20% per position

References:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463
    - https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data
    - https://quantpedia.com/strategies/time-series-momentum-effect/
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from collections import deque

from trading_algo.quant_core.utils.constants import (
    TSMOM_LOOKBACK,
    TSMOM_HOLDING_PERIOD,
    TSMOM_VOL_LOOKBACK,
    SQRT_252,
    EPSILON,
    TRADING_DAYS_PER_YEAR,
)
from trading_algo.quant_core.utils.math_utils import (
    simple_returns,
    rolling_std,
    exponential_moving_average,
)


class TSMOMSignal(Enum):
    """TSMOM position signal."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TSMOMAssetSignal:
    """Signal for a single asset in TSMOM strategy."""
    symbol: str
    signal: TSMOMSignal
    momentum_return: float     # Past 12-month return
    volatility: float          # Current volatility estimate
    raw_position: float        # Unscaled position (-1 to +1)
    scaled_position: float     # Volatility-scaled position size
    contribution_to_risk: float  # % of portfolio risk


@dataclass
class TSMOMPortfolioSignal:
    """Aggregate TSMOM portfolio signal."""
    timestamp: str
    asset_signals: Dict[str, TSMOMAssetSignal]
    portfolio_weights: Dict[str, float]
    long_positions: List[str]
    short_positions: List[str]
    gross_exposure: float
    net_exposure: float
    expected_vol: float


class TimeSeriesMomentum:
    """
    Time Series Momentum Strategy.

    Implements the TSMOM strategy from Moskowitz, Ooi & Pedersen (2012).

    Key features:
    - 12-month lookback for momentum signal
    - Volatility scaling for risk parity
    - Monthly rebalancing
    - Works across multiple assets

    Usage:
        tsmom = TimeSeriesMomentum(lookback=252, target_vol=0.10)

        # Single asset signal
        signal = tsmom.generate_signal(prices)

        # Portfolio signal
        portfolio = tsmom.generate_portfolio_signal(
            prices_dict={'AAPL': prices_aapl, 'GOOGL': prices_googl}
        )
    """

    def __init__(
        self,
        lookback: int = TSMOM_LOOKBACK,
        vol_lookback: int = TSMOM_VOL_LOOKBACK,
        holding_period: int = TSMOM_HOLDING_PERIOD,
        target_vol: float = 0.10,  # Per-asset target vol
        portfolio_vol_target: float = 0.15,  # Portfolio target vol
        max_position: float = 0.25,  # Max position per asset
        use_vol_scaling: bool = True,
    ):
        """
        Initialize TSMOM strategy.

        Args:
            lookback: Momentum lookback period (default 252 = 12 months)
            vol_lookback: Volatility estimation lookback
            holding_period: Position holding period (default 21 = 1 month)
            target_vol: Target volatility per position
            portfolio_vol_target: Portfolio-level target volatility
            max_position: Maximum position size per asset
            use_vol_scaling: Whether to apply volatility scaling
        """
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.holding_period = holding_period
        self.target_vol = target_vol
        self.portfolio_vol_target = portfolio_vol_target
        self.max_position = max_position
        self.use_vol_scaling = use_vol_scaling

        # State tracking
        self._last_rebalance: Dict[str, int] = {}
        self._positions: Dict[str, float] = {}
        self._price_history: Dict[str, deque] = {}
        self._return_history: Dict[str, deque] = {}

    def calculate_momentum(
        self,
        prices: NDArray[np.float64],
        lookback: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate momentum signal.

        Returns:
            Tuple of (momentum_return, t_statistic)
        """
        lookback = lookback or self.lookback

        if len(prices) < lookback + 1:
            return 0.0, 0.0

        # Calculate return over lookback period
        momentum_return = prices[-1] / prices[-lookback - 1] - 1

        # Calculate t-statistic for significance
        returns = simple_returns(prices[-lookback - 1:])
        if len(returns) > 0:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > EPSILON else 0.0
        else:
            t_stat = 0.0

        return float(momentum_return), float(t_stat)

    def estimate_volatility(
        self,
        returns: NDArray[np.float64],
        method: str = "realized"
    ) -> float:
        """
        Estimate volatility for position sizing.

        Args:
            returns: Return series
            method: "realized" or "ewma"

        Returns:
            Annualized volatility estimate
        """
        if len(returns) < 20:
            return self.target_vol

        lookback = min(self.vol_lookback, len(returns))
        recent_returns = returns[-lookback:]

        if method == "ewma":
            # EWMA with lambda = 0.94
            variance = recent_returns[-1] ** 2
            for i in range(len(recent_returns) - 2, -1, -1):
                variance = 0.94 * variance + 0.06 * recent_returns[i] ** 2
            vol = np.sqrt(variance) * SQRT_252
        else:
            # Realized volatility
            vol = np.std(recent_returns, ddof=1) * SQRT_252

        # Floor and cap
        vol = np.clip(vol, 0.05, 1.0)

        return float(vol)

    def generate_signal(
        self,
        prices: NDArray[np.float64],
        symbol: str = "default"
    ) -> TSMOMAssetSignal:
        """
        Generate TSMOM signal for a single asset.

        Args:
            prices: Price series
            symbol: Asset identifier

        Returns:
            TSMOMAssetSignal
        """
        # Calculate momentum
        momentum_return, t_stat = self.calculate_momentum(prices)

        # Determine direction
        if momentum_return > 0:
            signal = TSMOMSignal.LONG
            raw_position = 1.0
        elif momentum_return < 0:
            signal = TSMOMSignal.SHORT
            raw_position = -1.0
        else:
            signal = TSMOMSignal.FLAT
            raw_position = 0.0

        # Calculate volatility
        returns = simple_returns(prices)
        volatility = self.estimate_volatility(returns)

        # Scale position by volatility (inverse vol weighting)
        if self.use_vol_scaling and volatility > EPSILON:
            vol_scalar = self.target_vol / volatility
            scaled_position = raw_position * vol_scalar
        else:
            scaled_position = raw_position

        # Apply position limit
        scaled_position = np.clip(scaled_position, -self.max_position, self.max_position)

        return TSMOMAssetSignal(
            symbol=symbol,
            signal=signal,
            momentum_return=momentum_return,
            volatility=volatility,
            raw_position=raw_position,
            scaled_position=scaled_position,
            contribution_to_risk=volatility * abs(scaled_position),
        )

    def generate_portfolio_signal(
        self,
        prices_dict: Dict[str, NDArray[np.float64]],
        returns_dict: Optional[Dict[str, NDArray[np.float64]]] = None,
    ) -> TSMOMPortfolioSignal:
        """
        Generate TSMOM signals for a portfolio of assets.

        Implements the full MOP (2012) approach:
        - Individual TSMOM signals per asset
        - Volatility scaling for equal risk contribution
        - Portfolio-level volatility targeting

        Args:
            prices_dict: Dict of symbol -> price series
            returns_dict: Optional dict of symbol -> return series

        Returns:
            TSMOMPortfolioSignal with all positions
        """
        asset_signals: Dict[str, TSMOMAssetSignal] = {}
        long_positions: List[str] = []
        short_positions: List[str] = []

        # Generate individual signals
        for symbol, prices in prices_dict.items():
            signal = self.generate_signal(prices, symbol)
            asset_signals[symbol] = signal

            if signal.signal == TSMOMSignal.LONG:
                long_positions.append(symbol)
            elif signal.signal == TSMOMSignal.SHORT:
                short_positions.append(symbol)

        # Calculate portfolio weights
        weights = self._calculate_portfolio_weights(asset_signals, returns_dict)

        # Calculate exposures
        gross_exposure = sum(abs(w) for w in weights.values())
        net_exposure = sum(weights.values())

        # Estimate portfolio volatility
        expected_vol = self._estimate_portfolio_volatility(
            weights, returns_dict, list(prices_dict.keys())
        ) if returns_dict else self.portfolio_vol_target

        return TSMOMPortfolioSignal(
            timestamp=str(np.datetime64('now')),
            asset_signals=asset_signals,
            portfolio_weights=weights,
            long_positions=long_positions,
            short_positions=short_positions,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            expected_vol=expected_vol,
        )

    def _calculate_portfolio_weights(
        self,
        asset_signals: Dict[str, TSMOMAssetSignal],
        returns_dict: Optional[Dict[str, NDArray[np.float64]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights with volatility targeting.

        Implements risk parity weighting where each asset
        contributes equally to portfolio risk.
        """
        weights = {}

        if not asset_signals:
            return weights

        # Start with individual scaled positions
        total_risk = sum(s.contribution_to_risk for s in asset_signals.values())

        if total_risk < EPSILON:
            # Equal weight if no risk info
            equal_weight = 1.0 / len(asset_signals)
            for symbol, signal in asset_signals.items():
                weights[symbol] = signal.raw_position * equal_weight
            return weights

        # Risk parity weights
        for symbol, signal in asset_signals.items():
            # Weight proportional to scaled position
            weights[symbol] = signal.scaled_position

        # Scale to portfolio vol target
        if self.portfolio_vol_target > 0:
            # Simple scaling - divide by number of assets and target
            n_assets = len(weights)
            scaling_factor = self.portfolio_vol_target / (total_risk / n_assets)
            scaling_factor = np.clip(scaling_factor, 0.5, 2.0)

            weights = {s: w * scaling_factor for s, w in weights.items()}

        return weights

    def _estimate_portfolio_volatility(
        self,
        weights: Dict[str, float],
        returns_dict: Optional[Dict[str, NDArray[np.float64]]],
        symbols: List[str],
    ) -> float:
        """Estimate portfolio volatility from weights and returns."""
        if returns_dict is None or not weights:
            return self.portfolio_vol_target

        # Build return matrix
        min_len = min(len(r) for r in returns_dict.values())
        if min_len < 20:
            return self.portfolio_vol_target

        n = len(symbols)
        returns_matrix = np.zeros((min_len, n))
        weight_array = np.zeros(n)

        for i, symbol in enumerate(symbols):
            if symbol in returns_dict and symbol in weights:
                returns_matrix[:, i] = returns_dict[symbol][-min_len:]
                weight_array[i] = weights[symbol]

        # Calculate portfolio variance
        cov_matrix = np.cov(returns_matrix, rowvar=False) * TRADING_DAYS_PER_YEAR
        port_var = weight_array @ cov_matrix @ weight_array
        port_vol = np.sqrt(max(port_var, EPSILON))

        return float(port_vol)

    def should_rebalance(
        self,
        symbol: str,
        current_bar: int,
    ) -> bool:
        """
        Check if position should be rebalanced.

        Args:
            symbol: Asset symbol
            current_bar: Current bar index

        Returns:
            True if rebalancing is needed
        """
        last_rebal = self._last_rebalance.get(symbol, 0)
        return (current_bar - last_rebal) >= self.holding_period

    def record_rebalance(self, symbol: str, bar: int) -> None:
        """Record rebalance timestamp."""
        self._last_rebalance[symbol] = bar


# =============================================================================
# CROSS-SECTIONAL MOMENTUM (XSMOM) EXTENSION
# =============================================================================

class CrossSectionalMomentum:
    """
    Cross-sectional momentum (XSMOM).

    Ranks assets by momentum and goes long winners, short losers.
    Often combined with TSMOM for diversification.
    """

    def __init__(
        self,
        lookback: int = 252,
        n_long: int = 5,
        n_short: int = 5,
        skip_period: int = 21,  # Skip most recent month
    ):
        self.lookback = lookback
        self.n_long = n_long
        self.n_short = n_short
        self.skip_period = skip_period

    def rank_assets(
        self,
        prices_dict: Dict[str, NDArray[np.float64]]
    ) -> Dict[str, Tuple[float, int]]:
        """
        Rank assets by momentum.

        Returns:
            Dict of symbol -> (momentum_return, rank)
        """
        momentum_returns = {}

        for symbol, prices in prices_dict.items():
            if len(prices) < self.lookback + self.skip_period + 1:
                continue

            # Calculate momentum (skip most recent period)
            end_idx = -self.skip_period - 1 if self.skip_period > 0 else -1
            start_idx = end_idx - self.lookback

            momentum = prices[end_idx] / prices[start_idx] - 1
            momentum_returns[symbol] = momentum

        # Sort by momentum
        sorted_symbols = sorted(momentum_returns.keys(),
                               key=lambda s: momentum_returns[s],
                               reverse=True)

        # Assign ranks
        rankings = {}
        for i, symbol in enumerate(sorted_symbols):
            rankings[symbol] = (momentum_returns[symbol], i + 1)

        return rankings

    def generate_portfolio(
        self,
        rankings: Dict[str, Tuple[float, int]]
    ) -> Dict[str, float]:
        """
        Generate long/short portfolio from rankings.

        Returns:
            Dict of symbol -> weight
        """
        sorted_by_rank = sorted(rankings.items(), key=lambda x: x[1][1])

        weights = {}

        # Long winners
        for symbol, (mom, rank) in sorted_by_rank[:self.n_long]:
            weights[symbol] = 1.0 / self.n_long

        # Short losers
        for symbol, (mom, rank) in sorted_by_rank[-self.n_short:]:
            weights[symbol] = -1.0 / self.n_short

        return weights
