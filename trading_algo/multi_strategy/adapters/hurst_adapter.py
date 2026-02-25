"""
Adapter wrapping the Hurst-Adaptive strategy.

The Hurst-Adaptive strategy dynamically switches between momentum and
mean-reversion sub-strategies based on the rolling Hurst exponent of
5-minute returns.  It abstains from trading during random-walk regimes
(0.45 <= H <= 0.55).

This adapter maps the inner HurstSignal dicts into per-symbol
StrategySignal objects for the multi-strategy controller.
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
from trading_algo.quant_core.strategies.hurst_adaptive import (
    HurstAdaptiveStrategy,
    HurstConfig,
)

logger = logging.getLogger(__name__)


class HurstAdaptiveAdapter(TradingStrategy):
    """
    Wraps HurstAdaptiveStrategy as a TradingStrategy.

    Needs at least ``config.warmup`` bars (default 100) per symbol
    before producing signals.  After warmup, it emits signals in
    trending or mean-reverting regimes and stays flat during
    random-walk periods.
    """

    def __init__(self, config: Optional[HurstConfig] = None):
        self._strategy = HurstAdaptiveStrategy(config)
        self._config = config or HurstConfig()
        self._state = StrategyState.WARMING_UP

        # Per-symbol bar counts and latest prices
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)
        self._current_prices: Dict[str, float] = {}

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "HurstAdaptive"

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
        # Feed the underlying strategy
        self._strategy.update(symbol, close, high, low)

        # Track per-symbol state
        self._current_prices[symbol] = close
        self._bars_per_symbol[symbol] += 1

        # Activate once any symbol has enough history
        if self._state == StrategyState.WARMING_UP:
            ready = sum(
                1 for n in self._bars_per_symbol.values()
                if n >= self._config.warmup
            )
            if ready >= 1:
                self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        # Filter to symbols with sufficient history
        eligible = [
            s for s in symbols
            if self._bars_per_symbol.get(s, 0) >= self._config.warmup
        ]

        if not eligible:
            return []

        # Generate inner signals
        hurst_signals = self._strategy.generate_signals(eligible)

        signals: List[StrategySignal] = []
        for hs in hurst_signals:
            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=hs.symbol,
                direction=hs.direction,
                target_weight=hs.weight,
                confidence=hs.confidence,
                stop_loss=hs.stop_loss,
                take_profit=hs.take_profit,
                entry_price=self._current_prices.get(hs.symbol),
                trade_type=hs.trade_type,
                metadata=hs.metadata,
            ))

        return signals

    def get_current_exposure(self) -> float:
        # Exposure is tracked externally by the controller
        return 0.0

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self._strategy.reset()
        self._bars_per_symbol.clear()
        self._current_prices.clear()
        self._state = StrategyState.WARMING_UP
