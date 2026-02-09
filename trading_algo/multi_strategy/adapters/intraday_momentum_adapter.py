"""
Adapter wrapping the Intraday Momentum strategy (Gao et al. 2018).

The strategy only operates during two narrow time windows:
  - 9:30-10:00: Measuring the opening return (passive observation).
  - 15:30-15:55: Entering positions in the opening-return direction.

Outside these windows it produces no signals.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Dict, List, Optional

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
    IntradayMomentumStrategy,
    IntradayMomentumConfig,
)

logger = logging.getLogger(__name__)


class IntradayMomentumAdapter(TradingStrategy):
    """
    Wraps IntradayMomentumStrategy as a TradingStrategy.

    Active window: 15:30 - 15:55 (after opening return captured).
    """

    MIN_WARMUP_BARS = 10

    def __init__(self, config: Optional[IntradayMomentumConfig] = None):
        self._strategy = IntradayMomentumStrategy(config)
        self._config = config or IntradayMomentumConfig()
        self._state = StrategyState.WARMING_UP
        self._bars_seen = 0

    @property
    def name(self) -> str:
        return "IntradayMomentum"

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
        self._strategy.update(symbol, timestamp, close, volume)
        self._bars_seen += 1
        if self._state == StrategyState.WARMING_UP and self._bars_seen >= self.MIN_WARMUP_BARS:
            self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        mom_signals = self._strategy.generate_signals(symbols, timestamp)

        signals: List[StrategySignal] = []
        for ms in mom_signals:
            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=ms.symbol,
                direction=ms.direction,
                target_weight=self._config.position_size,
                confidence=ms.confidence,
                stop_loss=ms.stop_loss,
                take_profit=ms.target_price,
                entry_price=ms.entry_price,
                trade_type="intraday_momentum",
                metadata={
                    "opening_return": ms.opening_return,
                    "midday_return": ms.midday_return,
                },
            ))

        return signals

    def get_current_exposure(self) -> float:
        return 0.0  # Positions are held < 30 min

    def reset(self) -> None:
        self._strategy.reset()
        self._state = StrategyState.WARMING_UP
        self._bars_seen = 0
