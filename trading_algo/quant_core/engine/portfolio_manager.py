"""
Portfolio Manager

Determines target portfolio weights and position sizes using:
    - HRP for allocation across assets
    - Kelly Criterion for position sizing
    - Risk parity for risk allocation

Responsibilities:
    - Convert signals to target weights
    - Apply risk-based position sizing
    - Handle rebalancing logic
    - Respect turnover constraints
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import logging

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.portfolio.kelly import KellyCriterion, KellyEstimate, KellyMode
from trading_algo.quant_core.portfolio.hrp import HierarchicalRiskParity, HRPResult
from trading_algo.quant_core.portfolio.optimizer import (
    PortfolioOptimizer, OptimizationMethod, OptimizationConstraints
)
from trading_algo.quant_core.engine.signal_aggregator import AggregatedSignal
from trading_algo.quant_core.engine.risk_controller import RiskDecision


logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods."""
    EQUAL_WEIGHT = auto()      # Equal allocation
    SIGNAL_SCALED = auto()     # Proportional to signal strength
    VOLATILITY_SCALED = auto() # Inverse volatility
    KELLY = auto()             # Kelly criterion
    HRP = auto()               # Hierarchical risk parity
    RISK_PARITY = auto()       # Equal risk contribution


@dataclass
class TargetPosition:
    """Target position for a single asset."""
    symbol: str
    target_weight: float       # Target weight in portfolio
    target_shares: float       # Target number of shares
    current_shares: float      # Current shares held
    delta_shares: float        # Shares to trade
    signal_strength: float     # Underlying signal strength
    sizing_confidence: float   # Confidence in sizing

    @property
    def is_entry(self) -> bool:
        """Is this a new position entry?"""
        return abs(self.current_shares) < EPSILON and abs(self.target_shares) > EPSILON

    @property
    def is_exit(self) -> bool:
        """Is this a position exit?"""
        return abs(self.current_shares) > EPSILON and abs(self.target_shares) < EPSILON

    @property
    def is_rebalance(self) -> bool:
        """Is this a rebalancing trade?"""
        return not self.is_entry and not self.is_exit and abs(self.delta_shares) > EPSILON


@dataclass
class TargetPortfolio:
    """Complete target portfolio."""
    positions: Dict[str, TargetPosition]
    total_long_weight: float
    total_short_weight: float
    gross_exposure: float
    net_exposure: float
    expected_turnover: float
    timestamp: Optional[Any] = None

    @property
    def is_balanced(self) -> bool:
        """Check if portfolio is approximately market-neutral."""
        return abs(self.net_exposure) < 0.1


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration."""
    # Sizing method
    sizing_method: SizingMethod = SizingMethod.KELLY

    # Kelly parameters
    kelly_fraction: float = 0.25        # Fractional Kelly (25%)
    max_kelly_leverage: float = 2.0     # Max Kelly-implied leverage

    # Volatility targeting
    target_volatility: float = 0.15     # 15% annualized
    vol_lookback: int = 20              # Days for vol calculation

    # Allocation constraints
    min_weight: float = 0.01            # 1% minimum position
    max_weight: float = 0.10            # 10% maximum position
    max_gross_exposure: float = 1.0     # 100% gross
    max_net_exposure: float = 0.50      # 50% net
    min_confidence: float = 0.05        # Minimum signal confidence

    # Rebalancing
    min_trade_value: float = 1000       # Minimum trade in $
    min_rebalance_threshold: float = 0.02  # 2% weight change to trigger
    max_turnover: float = 0.50          # 50% daily turnover limit

    # Long/short
    allow_shorting: bool = True
    long_short_ratio: float = 1.0       # 1.0 = equal long/short


class PortfolioManager:
    """
    Portfolio construction and position sizing manager.

    Converts trading signals into target portfolio positions
    with proper risk-based sizing.

    Usage:
        manager = PortfolioManager(config)

        # Generate target portfolio from signals
        target = manager.construct_portfolio(
            signals=aggregated_signals,
            equity=100000,
            current_positions=current_positions,
            returns_history=returns,
        )

        # Execute trades to reach target
        for symbol, pos in target.positions.items():
            if abs(pos.delta_shares) > 0:
                execute_trade(symbol, pos.delta_shares)
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio manager.

        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()

        # Initialize sub-components
        # Map kelly_fraction to KellyMode
        kelly_mode = self._get_kelly_mode(self.config.kelly_fraction)
        self.kelly = KellyCriterion(
            mode=kelly_mode,
            max_leverage=self.config.max_kelly_leverage,
        )
        self.hrp = HierarchicalRiskParity(
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        self.optimizer = PortfolioOptimizer(
            method=OptimizationMethod.HRP,
            target_volatility=self.config.target_volatility,
        )

        # Historical data for sizing
        self._returns_history: Dict[str, List[float]] = {}
        self._vol_estimates: Dict[str, float] = {}

    def _get_kelly_mode(self, fraction: float) -> KellyMode:
        """Map kelly fraction to KellyMode enum."""
        if fraction >= 0.9:
            return KellyMode.FULL
        elif fraction >= 0.65:
            return KellyMode.THREE_QUARTER
        elif fraction >= 0.4:
            return KellyMode.HALF
        elif fraction >= 0.2:
            return KellyMode.QUARTER
        else:
            return KellyMode.EIGHTH

    def construct_portfolio(
        self,
        signals: Dict[str, AggregatedSignal],
        equity: float,
        current_positions: Dict[str, float],  # symbol -> shares
        current_prices: Dict[str, float],
        risk_decision: Optional[RiskDecision] = None,
        returns_matrix: Optional[NDArray[np.float64]] = None,
    ) -> TargetPortfolio:
        """
        Construct target portfolio from signals.

        Args:
            signals: Dict of symbol -> AggregatedSignal
            equity: Current portfolio equity
            current_positions: Current position shares
            current_prices: Current asset prices
            risk_decision: Risk controller decision
            returns_matrix: Historical returns for allocation

        Returns:
            TargetPortfolio with target positions
        """
        if not signals or equity <= 0:
            return self._empty_portfolio()

        # Apply risk decision multiplier
        exposure_mult = 1.0
        if risk_decision:
            exposure_mult = risk_decision.exposure_multiplier
            if not risk_decision.can_trade:
                # If can't trade, target is to reduce to risk limits
                exposure_mult = min(exposure_mult, 0.5)

        # Step 1: Filter signals
        active_signals = self._filter_signals(signals)
        if not active_signals:
            return self._empty_portfolio()

        # Step 2: Calculate base weights
        if self.config.sizing_method == SizingMethod.HRP and returns_matrix is not None:
            base_weights = self._hrp_weights(active_signals, returns_matrix)
        elif self.config.sizing_method == SizingMethod.KELLY:
            base_weights = self._kelly_weights(active_signals)
        elif self.config.sizing_method == SizingMethod.VOLATILITY_SCALED:
            base_weights = self._vol_scaled_weights(active_signals)
        else:
            base_weights = self._signal_scaled_weights(active_signals)

        # Step 3: Apply signal direction and strength
        target_weights = self._apply_signal_direction(base_weights, active_signals)

        # Step 4: Apply risk constraints
        target_weights = self._apply_constraints(target_weights, exposure_mult)

        # Step 5: Convert to shares
        positions = self._weights_to_shares(
            target_weights, equity, current_positions, current_prices, active_signals
        )

        # Step 6: Apply turnover constraint
        positions = self._apply_turnover_constraint(
            positions, current_positions, current_prices, equity
        )

        # Calculate portfolio metrics
        total_long = sum(w for w in target_weights.values() if w > 0)
        total_short = abs(sum(w for w in target_weights.values() if w < 0))
        gross = total_long + total_short
        net = total_long - total_short

        # Calculate turnover
        current_weights = {
            s: current_positions.get(s, 0) * current_prices.get(s, 0) / equity
            for s in target_weights
        }
        turnover = sum(abs(target_weights.get(s, 0) - current_weights.get(s, 0))
                       for s in set(target_weights) | set(current_weights))

        return TargetPortfolio(
            positions=positions,
            total_long_weight=total_long,
            total_short_weight=total_short,
            gross_exposure=gross,
            net_exposure=net,
            expected_turnover=turnover,
        )

    def _filter_signals(
        self,
        signals: Dict[str, AggregatedSignal],
    ) -> Dict[str, AggregatedSignal]:
        """Filter signals to only include actionable ones."""
        filtered = {}

        for symbol, signal in signals.items():
            # Must have meaningful signal strength
            if abs(signal.signal) < self.config.min_weight:
                continue

            # Must have reasonable confidence
            if signal.confidence < self.config.min_confidence:
                continue

            # Skip if shorting disabled and signal is short
            if signal.signal < 0 and not self.config.allow_shorting:
                continue

            filtered[symbol] = signal

        return filtered

    def _signal_scaled_weights(
        self,
        signals: Dict[str, AggregatedSignal],
    ) -> Dict[str, float]:
        """Calculate weights proportional to signal strength."""
        weights = {}
        total_signal = sum(abs(s.signal) for s in signals.values())

        if total_signal < EPSILON:
            return {s: 1.0 / len(signals) for s in signals}

        for symbol, signal in signals.items():
            weights[symbol] = abs(signal.signal) / total_signal

        return weights

    def _vol_scaled_weights(
        self,
        signals: Dict[str, AggregatedSignal],
    ) -> Dict[str, float]:
        """Calculate inverse volatility weights."""
        weights = {}
        inv_vols = []

        for symbol in signals:
            vol = self._vol_estimates.get(symbol, 0.20)  # Default 20%
            inv_vols.append((symbol, 1.0 / (vol + EPSILON)))

        total_inv_vol = sum(iv for _, iv in inv_vols)

        for symbol, inv_vol in inv_vols:
            weights[symbol] = inv_vol / total_inv_vol

        return weights

    def _kelly_weights(
        self,
        signals: Dict[str, AggregatedSignal],
    ) -> Dict[str, float]:
        """Calculate Kelly-based weights."""
        weights = {}

        for symbol, signal in signals.items():
            # Use signal strength as edge proxy
            edge = signal.signal * signal.confidence

            # Estimate win rate from signal confidence
            win_rate = 0.5 + signal.confidence * 0.25  # 50-75%

            # Simple Kelly approximation
            kelly_f = (win_rate - (1 - win_rate)) / 1.0  # Assuming even odds

            # Apply fractional Kelly
            weight = kelly_f * self.config.kelly_fraction

            # Clamp to reasonable range
            weight = np.clip(weight, -self.config.max_weight, self.config.max_weight)
            weights[symbol] = abs(weight)

        # Normalize
        total = sum(weights.values())
        if total > EPSILON:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _hrp_weights(
        self,
        signals: Dict[str, AggregatedSignal],
        returns_matrix: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Calculate HRP weights."""
        symbols = list(signals.keys())

        try:
            # Need returns for all symbols
            if returns_matrix.shape[1] != len(symbols):
                # Fallback to equal weight
                return {s: 1.0 / len(symbols) for s in symbols}

            result = self.hrp.optimize(returns_matrix, symbols)
            return result.to_dict()

        except Exception as e:
            logger.warning(f"HRP optimization failed: {e}")
            return {s: 1.0 / len(symbols) for s in symbols}

    def _apply_signal_direction(
        self,
        base_weights: Dict[str, float],
        signals: Dict[str, AggregatedSignal],
    ) -> Dict[str, float]:
        """Apply signal direction to base weights."""
        directed_weights = {}

        for symbol, weight in base_weights.items():
            signal = signals.get(symbol)
            if signal is None:
                continue

            # Apply direction from signal
            if signal.signal > 0:
                directed_weights[symbol] = weight * min(1.0, abs(signal.signal) * 2)
            else:
                directed_weights[symbol] = -weight * min(1.0, abs(signal.signal) * 2)

        return directed_weights

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        exposure_mult: float,
    ) -> Dict[str, float]:
        """Apply portfolio constraints."""
        constrained = {}

        # Calculate current totals
        long_total = sum(w for w in weights.values() if w > 0)
        short_total = abs(sum(w for w in weights.values() if w < 0))
        gross = long_total + short_total

        # Apply exposure multiplier
        max_gross = self.config.max_gross_exposure * exposure_mult
        max_net = self.config.max_net_exposure * exposure_mult

        # Scale if needed
        if gross > max_gross:
            scale = max_gross / gross
            weights = {s: w * scale for s, w in weights.items()}
            long_total *= scale
            short_total *= scale

        # Check net exposure
        net = long_total - short_total
        if abs(net) > max_net:
            # Reduce the larger side
            if net > 0:
                # Reduce longs
                scale = (max_net + short_total) / long_total
                for s, w in weights.items():
                    if w > 0:
                        weights[s] = w * scale
            else:
                # Reduce shorts
                scale = (max_net + long_total) / short_total
                for s, w in weights.items():
                    if w < 0:
                        weights[s] = w * scale

        # Apply individual position limits
        for symbol, weight in weights.items():
            if abs(weight) > self.config.max_weight:
                constrained[symbol] = np.sign(weight) * self.config.max_weight
            elif abs(weight) < self.config.min_weight:
                constrained[symbol] = 0.0
            else:
                constrained[symbol] = weight

        return constrained

    def _weights_to_shares(
        self,
        weights: Dict[str, float],
        equity: float,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        signals: Dict[str, AggregatedSignal],
    ) -> Dict[str, TargetPosition]:
        """Convert weights to share positions."""
        positions = {}

        for symbol, weight in weights.items():
            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            target_value = weight * equity
            target_shares = target_value / price
            current_shares = current_positions.get(symbol, 0)
            delta_shares = target_shares - current_shares

            signal = signals.get(symbol)
            signal_strength = signal.signal if signal else 0.0
            confidence = signal.confidence if signal else 0.0

            positions[symbol] = TargetPosition(
                symbol=symbol,
                target_weight=weight,
                target_shares=target_shares,
                current_shares=current_shares,
                delta_shares=delta_shares,
                signal_strength=signal_strength,
                sizing_confidence=confidence,
            )

        return positions

    def _apply_turnover_constraint(
        self,
        positions: Dict[str, TargetPosition],
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        equity: float,
    ) -> Dict[str, TargetPosition]:
        """Apply maximum turnover constraint."""
        # Calculate proposed turnover
        total_turnover = 0.0
        for symbol, pos in positions.items():
            price = current_prices.get(symbol, 0)
            trade_value = abs(pos.delta_shares) * price
            total_turnover += trade_value

        turnover_pct = total_turnover / equity if equity > 0 else 0

        if turnover_pct <= self.config.max_turnover:
            return positions

        # Scale down trades to meet turnover limit
        scale = self.config.max_turnover / turnover_pct

        scaled_positions = {}
        for symbol, pos in positions.items():
            new_delta = pos.delta_shares * scale
            new_target = pos.current_shares + new_delta

            scaled_positions[symbol] = TargetPosition(
                symbol=symbol,
                target_weight=pos.target_weight * scale,
                target_shares=new_target,
                current_shares=pos.current_shares,
                delta_shares=new_delta,
                signal_strength=pos.signal_strength,
                sizing_confidence=pos.sizing_confidence,
            )

        return scaled_positions

    def update_volatility_estimates(
        self,
        symbol: str,
        returns: NDArray[np.float64],
    ) -> None:
        """Update volatility estimate for a symbol."""
        if len(returns) < self.config.vol_lookback:
            return

        recent_returns = returns[-self.config.vol_lookback:]
        vol = float(np.std(recent_returns, ddof=1) * SQRT_252)
        self._vol_estimates[symbol] = vol

    def _empty_portfolio(self) -> TargetPortfolio:
        """Return empty portfolio."""
        return TargetPortfolio(
            positions={},
            total_long_weight=0.0,
            total_short_weight=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            expected_turnover=0.0,
        )
