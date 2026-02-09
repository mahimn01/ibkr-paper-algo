"""
Short-Term Reversal Strategy

Based on: Jegadeesh (1990) "Evidence of Predictable Behavior of Security
          Returns", Journal of Finance 45(3), 881-898.

Key finding: Stocks that have declined over the past 1-5 days tend to
outperform over the next 1-5 days, and vice versa.  The effect is
strongest at the 1-week horizon and among small/mid-cap stocks with
higher idiosyncratic volatility.

Implementation:
    1. Rank all symbols by their 5-day return (loser to winner).
    2. Go long the bottom quintile (losers), short the top quintile
       (winners).
    3. Size positions inversely proportional to volatility.
    4. Rebalance weekly (or configurable period).

Expected: 20-30% standalone Sharpe, ~2-3% portfolio contribution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ReversalConfig:
    """Configuration for the short-term reversal strategy."""

    lookback_days: int = 5
    """Number of bars to compute the return for ranking (5-day default)."""

    rebalance_frequency: int = 5
    """Rebalance every N bars (5 = weekly for daily bars)."""

    long_quantile: float = 0.20
    """Buy the bottom N% of performers (losers)."""

    short_quantile: float = 0.20
    """Short the top N% of performers (winners)."""

    position_size: float = 0.05
    """Target position weight per symbol (5%)."""

    max_positions: int = 10
    """Maximum number of positions (long + short combined)."""

    vol_scale: bool = True
    """Scale position size inversely by realized vol."""

    vol_lookback: int = 20
    """Lookback for volatility computation."""

    target_vol: float = 0.20
    """Target annualized volatility for vol scaling."""

    min_price: float = 5.0
    """Skip symbols below this price (avoid penny stocks)."""


@dataclass
class ReversalSignal:
    """Signal produced by the short-term reversal strategy."""
    symbol: str
    direction: int          # 1 = long (loser), -1 = short (winner)
    past_return: float      # The N-day return that generated this signal
    rank_percentile: float  # 0.0 (worst performer) to 1.0 (best performer)
    position_size: float    # Target weight
    confidence: float       # 0-1


class ShortTermReversalStrategy:
    """
    Implements Jegadeesh's 1-week reversal effect.

    Usage::

        strategy = ShortTermReversalStrategy()

        # Feed daily closing prices
        for bar in bars:
            strategy.update(symbol, close)

        # Generate signals (weekly)
        signals = strategy.generate_signals(symbols)
    """

    def __init__(self, config: Optional[ReversalConfig] = None):
        self.config = config or ReversalConfig()
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._bars_since_rebalance: int = 0
        self._current_signals: Dict[str, ReversalSignal] = {}

    def update(self, symbol: str, price: float) -> None:
        """Feed a new price observation."""
        self._price_history[symbol].append(price)

        max_len = max(self.config.lookback_days, self.config.vol_lookback) + 10
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

    def generate_signals(
        self,
        symbols: List[str],
    ) -> List[ReversalSignal]:
        """
        Generate reversal signals for all eligible symbols.

        Returns signals regardless of rebalance frequency — the caller
        should track rebalance timing if desired.
        """
        returns: Dict[str, float] = {}
        volatilities: Dict[str, float] = {}

        for symbol in symbols:
            hist = self._price_history.get(symbol)
            if hist is None or len(hist) < self.config.lookback_days + 1:
                continue

            current_price = hist[-1]
            if current_price < self.config.min_price:
                continue

            # N-day return
            past_price = hist[-(self.config.lookback_days + 1)]
            if past_price <= 0:
                continue
            ret = (current_price - past_price) / past_price
            returns[symbol] = ret

            # Volatility
            if len(hist) >= self.config.vol_lookback + 1:
                prices = np.array(hist[-(self.config.vol_lookback + 1):])
                daily_rets = np.diff(prices) / prices[:-1]
                volatilities[symbol] = float(np.std(daily_rets) * np.sqrt(252))

        if len(returns) < 3:
            return []

        # Rank by return (ascending: worst performers first)
        sorted_symbols = sorted(returns.keys(), key=lambda s: returns[s])
        n = len(sorted_symbols)

        # Compute percentile ranks
        percentiles = {s: i / (n - 1) for i, s in enumerate(sorted_symbols)} if n > 1 else {s: 0.5 for s in sorted_symbols}

        # Select longs (bottom quantile) and shorts (top quantile)
        n_long = max(1, int(n * self.config.long_quantile))
        n_short = max(1, int(n * self.config.short_quantile))

        longs = sorted_symbols[:n_long]
        shorts = sorted_symbols[-n_short:]

        # Respect max positions
        half_max = self.config.max_positions // 2
        longs = longs[:half_max]
        shorts = shorts[:half_max]

        signals: List[ReversalSignal] = []

        for symbol in longs:
            size = self.config.position_size
            if self.config.vol_scale and symbol in volatilities and volatilities[symbol] > 0:
                vol_scalar = min(2.0, self.config.target_vol / volatilities[symbol])
                size *= vol_scalar

            # Confidence based on how extreme the loser is
            extremity = 1.0 - percentiles[symbol]  # 0 for median, 1 for worst
            confidence = min(1.0, 0.5 + extremity)

            signals.append(ReversalSignal(
                symbol=symbol,
                direction=1,
                past_return=returns[symbol],
                rank_percentile=percentiles[symbol],
                position_size=size,
                confidence=confidence,
            ))

        for symbol in shorts:
            size = self.config.position_size
            if self.config.vol_scale and symbol in volatilities and volatilities[symbol] > 0:
                vol_scalar = min(2.0, self.config.target_vol / volatilities[symbol])
                size *= vol_scalar

            extremity = percentiles[symbol]
            confidence = min(1.0, 0.5 + extremity)

            signals.append(ReversalSignal(
                symbol=symbol,
                direction=-1,
                past_return=returns[symbol],
                rank_percentile=percentiles[symbol],
                position_size=size,
                confidence=confidence,
            ))

        self._bars_since_rebalance = 0
        return signals

    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance."""
        self._bars_since_rebalance += 1
        return self._bars_since_rebalance >= self.config.rebalance_frequency

    def reset(self) -> None:
        """Reset all state."""
        self._price_history.clear()
        self._bars_since_rebalance = 0
        self._current_signals.clear()
