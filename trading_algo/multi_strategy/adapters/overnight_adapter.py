"""
Adapter wrapping the Overnight Returns strategy.

The strategy enters long positions near market close (15:55) and
exits shortly after market open the next day (9:35).
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
from trading_algo.quant_core.strategies.overnight_returns import (
    OvernightReturnsStrategy,
    OvernightConfig,
)

logger = logging.getLogger(__name__)


class OvernightReturnsAdapter(TradingStrategy):
    """
    Wraps OvernightReturnsStrategy as a TradingStrategy.

    Active window: ~15:55 (entry) and ~9:35 next day (exit).
    Requires multiple days of history to compute overnight stats.
    """

    MIN_WARMUP_DAYS = 10  # bars, roughly maps to days for daily data

    def __init__(self, config: Optional[OvernightConfig] = None):
        self._strategy = OvernightReturnsStrategy(config)
        self._config = config or OvernightConfig()
        self._state = StrategyState.WARMING_UP
        self._bars_seen = 0
        self._current_prices: Dict[str, float] = {}
        self._day_count = 0
        self._last_date: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "OvernightReturns"

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
        self._strategy.update(symbol, timestamp, open_price, close, volume)
        self._current_prices[symbol] = close
        self._bars_seen += 1

        # Track day transitions for EOD recording
        current_date = timestamp.date()
        if self._last_date is None or current_date != self._last_date:
            # Record EOD for previous day's symbols
            if self._last_date is not None:
                for sym, price in self._current_prices.items():
                    self._strategy.record_eod(sym, price)
                self._day_count += 1

            self._last_date = current_date

        if self._state == StrategyState.WARMING_UP and self._day_count >= self.MIN_WARMUP_DAYS:
            self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        on_signals = self._strategy.generate_signals(
            symbols, timestamp, self._current_prices,
        )

        signals: List[StrategySignal] = []
        for ons in on_signals:
            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=ons.symbol,
                direction=ons.direction,
                target_weight=ons.position_size,
                confidence=ons.confidence,
                entry_price=ons.entry_price,
                trade_type="overnight_long",
                metadata={
                    "avg_overnight_return": ons.avg_overnight_return,
                    "overnight_sharpe": ons.overnight_sharpe,
                },
            ))

        return signals

    def get_current_exposure(self) -> float:
        return 0.0

    def reset(self) -> None:
        self._strategy.reset()
        self._current_prices.clear()
        self._bars_seen = 0
        self._day_count = 0
        self._last_date = None
        self._state = StrategyState.WARMING_UP
