"""
Adapter wrapping the Intraday Time-Adaptive strategy.

The strategy adapts its sub-strategy to the time of day:
  - 9:30-10:30:  Opening range breakout (momentum)
  - 10:30-14:00: No new entries (random walk period)
  - 14:00-15:30: VWAP mean-reversion (afternoon fade)
  - 15:30-16:00: Close momentum (follow 30-min trend)

The adapter maps TimeAdaptiveSignal objects into per-symbol
StrategySignals for the multi-strategy controller.
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
from trading_algo.quant_core.strategies.intraday_time_adaptive import (
    IntradayTimeAdaptive,
    TimeAdaptiveConfig,
)

logger = logging.getLogger(__name__)


class TimeAdaptiveAdapter(TradingStrategy):
    """
    Wraps IntradayTimeAdaptive as a TradingStrategy.

    Needs at least ``config.warmup`` bars (default 20) before
    producing signals.  After warmup, it emits signals based on
    time-of-day sub-strategy selection.
    """

    def __init__(self, config: Optional[TimeAdaptiveConfig] = None):
        self._strategy = IntradayTimeAdaptive(config)
        self._config = config or TimeAdaptiveConfig()
        self._state = StrategyState.WARMING_UP

        # Per-symbol OHLCV history (bounded buffer)
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._volume_history: Dict[str, List[float]] = defaultdict(list)
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "TimeAdaptive"

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
        self._strategy.update(
            symbol=symbol,
            timestamp=timestamp,
            open_price=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )

        # Keep our own history for exposure tracking
        self._price_history[symbol].append(close)
        self._volume_history[symbol].append(volume)
        self._bars_per_symbol[symbol] += 1

        # Trim to bounded length (keep ~2 days of 5-min bars)
        max_len = 200
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]
            self._volume_history[symbol] = self._volume_history[symbol][-max_len:]

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

        # Filter to symbols with enough data
        eligible = [
            s for s in symbols
            if self._bars_per_symbol.get(s, 0) >= self._config.warmup
        ]

        if not eligible:
            return []

        # Generate time-adaptive signals
        ta_signals = self._strategy.generate_signals(
            symbols=eligible,
            timestamp=timestamp,
        )

        signals: List[StrategySignal] = []
        for ts in ta_signals:
            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=ts.symbol,
                direction=ts.direction,
                target_weight=ts.weight,
                confidence=ts.confidence,
                stop_loss=ts.stop_loss,
                take_profit=ts.take_profit,
                entry_price=ts.entry_price,
                trade_type=ts.trade_type,
                metadata={
                    "time_window": ts.time_window,
                },
            ))

        return signals

    def get_current_exposure(self) -> float:
        # Intraday strategy: positions are short-lived
        # Sum active position weights from the underlying strategy state
        total_exposure = 0.0
        for symbol, st in self._strategy._state.items():
            if st.active_direction != 0:
                total_exposure += self._config.max_weight
        return total_exposure

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self._strategy.reset()
        self._price_history.clear()
        self._volume_history.clear()
        self._bars_per_symbol.clear()
        self._state = StrategyState.WARMING_UP
