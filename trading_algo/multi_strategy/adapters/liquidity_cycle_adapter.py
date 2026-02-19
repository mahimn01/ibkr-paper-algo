"""
Adapter wrapping the Intraday Liquidity Cycle strategy.

Liquidity Cycles exploits predictable intraday liquidity patterns
caused by market-maker inventory management and institutional trading
schedules.  Different parts of the trading day (opening auction,
morning trend, lunch lull, afternoon, closing, MOC window) exhibit
fundamentally different microstructure characteristics and each calls
for a distinct sub-strategy (mean-reversion, momentum, or MOC
imbalance).

The underlying strategy operates on intraday bars via ``update_bar()``
and ``generate_signal()`` (singular -- one symbol at a time).  This
adapter converts the per-bar OHLCV update interface of the controller
protocol into the underlying strategy's bar-level API and maps
LiquidityCycleSignal objects into StrategySignals.
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
from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
    LiquidityCycleStrategy,
    LiquidityCycleConfig,
)

logger = logging.getLogger(__name__)

# Minimum bars before the strategy can activate.
_WARMUP_BARS = 20


class LiquidityCycleAdapter(TradingStrategy):
    """
    Wraps LiquidityCycleStrategy as a TradingStrategy.

    Needs at least ~20 intraday bars before the underlying strategy has
    enough data for ATR calculation, volume scoring, and momentum
    confirmation.  After warmup it emits at most one signal per symbol
    per bar (the strategy enforces one active signal per symbol
    internally).
    """

    def __init__(self, config: Optional[LiquidityCycleConfig] = None):
        self._strategy = LiquidityCycleStrategy(config)
        self._config = config or LiquidityCycleConfig()
        self._state = StrategyState.WARMING_UP

        # Track total bars ingested per symbol
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

        # Cache the latest signal per symbol for exposure calculation
        self._active_signals: Dict[str, float] = {}

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "LiquidityCycles"

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
        # Feed the underlying strategy (it handles day-boundary resets
        # and per-symbol state internally).
        self._strategy.update_bar(
            symbol, timestamp, open_price, high, low, close, volume
        )
        self._bars_per_symbol[symbol] += 1

        # Activate once any symbol has accumulated enough bars
        if self._state == StrategyState.WARMING_UP:
            ready = sum(
                1
                for n in self._bars_per_symbol.values()
                if n >= _WARMUP_BARS
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

        signals: List[StrategySignal] = []

        for symbol in symbols:
            if self._bars_per_symbol.get(symbol, 0) < _WARMUP_BARS:
                continue

            # The underlying strategy returns a single signal or None
            try:
                lc_signal = self._strategy.generate_signal(symbol, timestamp)
            except Exception:
                logger.debug(
                    "Liquidity cycle signal generation failed for %s",
                    symbol,
                    exc_info=True,
                )
                continue

            if lc_signal is None:
                continue

            # Map continuous direction [-1, +1] to discrete
            if lc_signal.direction > 0:
                direction = 1
            elif lc_signal.direction < 0:
                direction = -1
            else:
                continue

            self._active_signals[symbol] = lc_signal.position_size

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=direction,
                    target_weight=lc_signal.position_size,
                    confidence=lc_signal.confidence,
                    entry_price=lc_signal.entry_price,
                    stop_loss=lc_signal.stop_price,
                    take_profit=lc_signal.target_price,
                    trade_type=lc_signal.strategy_type,
                    metadata={
                        "regime": lc_signal.regime.name,
                        "expected_holding_minutes": lc_signal.expected_holding_minutes,
                    },
                )
            )

        return signals

    def get_current_exposure(self) -> float:
        return sum(abs(w) for w in self._active_signals.values())

    def get_performance_stats(self) -> Dict[str, float]:
        stats = self._strategy.get_session_stats()
        return {
            "total_signals": float(stats.get("total_signals", 0)),
            "symbols_active": float(stats.get("symbols_active", 0)),
        }

    def reset(self) -> None:
        self._strategy.reset()
        self._bars_per_symbol.clear()
        self._active_signals.clear()
        self._state = StrategyState.WARMING_UP
