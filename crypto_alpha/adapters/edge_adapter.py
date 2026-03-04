"""
Generic adapter: wraps any CryptoEdge into the TradingStrategy protocol.

Each edge engine has its own thin adapter subclass that provides
edge-specific configuration. The base adapter handles:
  - Feeding bars to the edge
  - Maintaining price/state history
  - Converting EdgeSignal votes to StrategySignal objects
  - Warmup logic

Follows the pattern from trading_algo/multi_strategy/adapters/momentum_adapter.py
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from crypto_alpha.edges.base_edge import CryptoEdge
from crypto_alpha.types import CryptoAssetState, CryptoEdgeVote, EdgeSignal

logger = logging.getLogger(__name__)

# Map vote to direction and weight multiplier
VOTE_MAP = {
    CryptoEdgeVote.STRONG_LONG:  (1, 1.3),
    CryptoEdgeVote.LONG:         (1, 1.0),
    CryptoEdgeVote.NEUTRAL:      (0, 0.0),
    CryptoEdgeVote.SHORT:        (-1, 1.0),
    CryptoEdgeVote.STRONG_SHORT: (-1, 1.3),
    CryptoEdgeVote.VETO_LONG:    (0, 0.0),
    CryptoEdgeVote.VETO_SHORT:   (0, 0.0),
}


class CryptoEdgeAdapter(TradingStrategy):
    """
    Wraps a CryptoEdge as a TradingStrategy for the controller.

    Maintains per-symbol state (prices, volumes, returns) and
    builds CryptoAssetState objects for the edge's get_vote() method.
    """

    def __init__(
        self,
        edge: CryptoEdge,
        base_weight: float = 0.10,
        max_history: int = 500,
        leverage_scalar_edge: Optional[str] = None,  # Name of edge providing leverage
    ):
        self._edge = edge
        self._base_weight = base_weight
        self._max_history = max_history
        self._leverage_scalar_edge = leverage_scalar_edge
        self._state = StrategyState.WARMING_UP

        # Per-symbol market data
        self._prices: Dict[str, List[float]] = defaultdict(list)
        self._volumes: Dict[str, List[float]] = defaultdict(list)
        self._returns: Dict[str, List[float]] = defaultdict(list)
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

        # Crypto-specific data
        self._funding_rates: Dict[str, float] = defaultdict(float)
        self._spot_prices: Dict[str, float] = {}
        self._open_interest: Dict[str, float] = {}

        # Leverage scalar from RADL edge (if connected)
        self._leverage_scalar: float = 1.0

    @property
    def name(self) -> str:
        return self._edge.name

    @property
    def state(self) -> StrategyState:
        return self._state

    def set_leverage_scalar(self, scalar: float) -> None:
        """Set leverage scalar from RADL edge."""
        self._leverage_scalar = scalar

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        # Track history
        self._prices[symbol].append(close)
        self._volumes[symbol].append(volume)
        self._bars_per_symbol[symbol] += 1

        # Compute return
        if len(self._prices[symbol]) >= 2:
            prev = self._prices[symbol][-2]
            ret = close / prev - 1 if prev > 0 else 0.0
            self._returns[symbol].append(ret)

        # Trim history
        if len(self._prices[symbol]) > self._max_history:
            self._prices[symbol] = self._prices[symbol][-self._max_history:]
            self._volumes[symbol] = self._volumes[symbol][-self._max_history:]
            self._returns[symbol] = self._returns[symbol][-self._max_history:]

        # Feed edge with crypto-specific kwargs
        kwargs = {
            'high': high,
            'low': low,
        }
        if symbol in self._funding_rates:
            kwargs['funding_rate'] = self._funding_rates[symbol]
        if symbol in self._spot_prices:
            kwargs['spot_price'] = self._spot_prices[symbol]
        if symbol in self._open_interest:
            kwargs['open_interest'] = self._open_interest[symbol]

        self._edge.update(symbol, timestamp, close, volume, **kwargs)

        # Check warmup
        if self._state == StrategyState.WARMING_UP:
            max_bars = max(self._bars_per_symbol.values()) if self._bars_per_symbol else 0
            if max_bars >= self._edge.warmup_bars:
                self._state = StrategyState.ACTIVE

    def update_crypto_data(
        self,
        symbol: str,
        funding_rate: Optional[float] = None,
        spot_price: Optional[float] = None,
        open_interest: Optional[float] = None,
    ) -> None:
        """Update crypto-specific data fields used by edges."""
        if funding_rate is not None:
            self._funding_rates[symbol] = funding_rate
        if spot_price is not None:
            self._spot_prices[symbol] = spot_price
        if open_interest is not None:
            self._open_interest[symbol] = open_interest

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        signals = []
        for symbol in symbols:
            if not self._prices.get(symbol):
                continue

            # Build asset state for edge
            asset_state = CryptoAssetState(
                symbol=symbol,
                timestamp=timestamp,
                price=self._prices[symbol][-1],
                volume_24h=sum(self._volumes[symbol][-288:]) if len(self._volumes[symbol]) >= 288 else sum(self._volumes[symbol]),
                prices=self._prices[symbol],
                volumes=self._volumes[symbol],
                returns=self._returns[symbol],
                funding_rate=self._funding_rates.get(symbol, 0.0),
                basis=0.0,  # Computed by specific edges
            )

            # Get edge vote
            edge_signal = self._edge.get_vote(symbol, asset_state)

            # Convert to StrategySignal
            direction, weight_mult = VOTE_MAP.get(
                edge_signal.vote, (0, 0.0)
            )

            if direction == 0:
                # Emit exit signal if edge says neutral but we might have a position
                if edge_signal.data.get('action') in ('exit', 'stop'):
                    signals.append(StrategySignal(
                        strategy_name=self.name,
                        symbol=symbol,
                        direction=0,
                        target_weight=0.0,
                        confidence=0.0,
                        trade_type="exit",
                        metadata=edge_signal.data,
                    ))
                continue

            # Compute target weight
            weight = self._base_weight * weight_mult * edge_signal.confidence

            # Apply leverage scalar from RADL
            leverage = self._leverage_scalar
            if 'leverage_scalar' in edge_signal.data:
                leverage = edge_signal.data['leverage_scalar']

            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                direction=direction,
                target_weight=weight,
                confidence=edge_signal.confidence,
                trade_type=self._edge.name.lower(),
                metadata={
                    **edge_signal.data,
                    'edge_vote': edge_signal.vote.name,
                    'reason': edge_signal.reason,
                    'leverage': leverage,
                },
            ))

        return signals

    def get_current_exposure(self) -> float:
        return 0.0

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self._edge.reset()
        self._prices.clear()
        self._volumes.clear()
        self._returns.clear()
        self._bars_per_symbol.clear()
        self._funding_rates.clear()
        self._spot_prices.clear()
        self._open_interest.clear()
        self._state = StrategyState.WARMING_UP
