"""
Adapter wrapping the Short-Term Reversal strategy (Jegadeesh 1990).

Reversal is a contrarian strategy: long recent losers, short recent
winners.  It rebalances on a configurable schedule (default weekly).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.short_term_reversal import (
    ShortTermReversalStrategy,
    ReversalConfig,
)

logger = logging.getLogger(__name__)


class ReversalStrategyAdapter(TradingStrategy):
    """
    Wraps ShortTermReversalStrategy as a TradingStrategy.

    Needs at least ``lookback_days + vol_lookback`` bars to produce
    its first signal.
    """

    def __init__(self, config: Optional[ReversalConfig] = None):
        self._reversal = ShortTermReversalStrategy(config)
        self._config = config or ReversalConfig()
        self._state = StrategyState.WARMING_UP
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)
        self._current_prices: Dict[str, float] = {}
        self._bar_counter = 0

    @property
    def name(self) -> str:
        return "ShortTermReversal"

    @property
    def state(self) -> StrategyState:
        return self._state

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
        self._reversal.update(symbol, close)
        self._current_prices[symbol] = close
        self._bars_per_symbol[symbol] += 1
        self._bar_counter += 1

        # Activate once we have enough history for ranking
        min_bars = self._config.lookback_days + self._config.vol_lookback + 5
        if self._state == StrategyState.WARMING_UP:
            ready = sum(
                1 for n in self._bars_per_symbol.values()
                if n >= min_bars
            )
            if ready >= 3:  # Need at least 3 symbols for ranking
                self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        # Only rebalance on schedule
        if not self._reversal.should_rebalance():
            return []

        rev_signals = self._reversal.generate_signals(symbols)

        signals: List[StrategySignal] = []
        for rs in rev_signals:
            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=rs.symbol,
                direction=rs.direction,
                target_weight=rs.position_size,
                confidence=rs.confidence,
                entry_price=self._current_prices.get(rs.symbol),
                trade_type="reversal_long" if rs.direction > 0 else "reversal_short",
                metadata={
                    "past_return": rs.past_return,
                    "rank_percentile": rs.rank_percentile,
                },
            ))

        return signals

    def get_current_exposure(self) -> float:
        return 0.0  # Tracked externally

    def reset(self) -> None:
        self._reversal.reset()
        self._bars_per_symbol.clear()
        self._current_prices.clear()
        self._bar_counter = 0
        self._state = StrategyState.WARMING_UP
