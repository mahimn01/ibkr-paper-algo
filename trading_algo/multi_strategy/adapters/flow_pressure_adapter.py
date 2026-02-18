"""
Adapter wrapping the Flow Pressure strategy.

Flow Pressure exploits structural institutional rebalancing flows:
turn-of-month pension inflows, quarter-end window dressing, options
expiry pinning, tax-loss selling, and index reconstitution pressure.

The underlying strategy has a date-based interface: ``update(dt, prices)``
accepts a datetime and a dict of symbol -> close price, while
``generate_signals(dt, prices)`` accepts a datetime and a dict of
symbol -> price array.

The adapter accumulates per-symbol OHLCV data, converts it to the
formats the underlying strategy expects, and maps FlowSignal objects
into per-symbol StrategySignals for the controller.
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
from trading_algo.quant_core.strategies.flow_pressure import (
    FlowPressureStrategy,
    FlowPressureConfig,
)

logger = logging.getLogger(__name__)

# Minimum bars before the strategy can activate.
_WARMUP_BARS = 30


class FlowPressureAdapter(TradingStrategy):
    """
    Wraps FlowPressureStrategy as a TradingStrategy.

    Needs at least ~30 bars of price data before the underlying
    sub-strategies have enough history for confidence estimation and
    momentum ranking.  After warmup it emits one signal per symbol
    per active flow type (turn-of-month, quarter-end, etc.).
    """

    def __init__(self, config: Optional[FlowPressureConfig] = None):
        self._strategy = FlowPressureStrategy(config)
        self._config = config or FlowPressureConfig()
        self._state = StrategyState.WARMING_UP

        # Per-symbol close-price history
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

        # Track last target weights for exposure calculation
        self._last_weights: Dict[str, float] = {}

        # Latest timestamp (used to feed strategy.update)
        self._latest_timestamp: Optional[datetime] = None

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "FlowPressure"

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
        self._price_history[symbol].append(close)
        self._bars_per_symbol[symbol] += 1
        self._latest_timestamp = timestamp

        # Trim to bounded length
        max_history = max(
            self._config.tom_lookback,
            self._config.qe_momentum_lookback,
            self._config.tax_loss_momentum_lookback,
            self._config.expiry_lookback,
        ) + 30
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]

        # Feed the underlying strategy with a single-symbol price dict.
        # The strategy's update() expects (datetime, Dict[str, float]).
        self._strategy.update(timestamp, {symbol: close})

        # Activate once any symbol has enough bars
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

        # Build numpy price arrays from our history
        prices: Dict[str, np.ndarray] = {}
        for s in symbols:
            hist = self._price_history.get(s)
            if hist and len(hist) >= _WARMUP_BARS:
                prices[s] = np.array(hist, dtype=np.float64)

        if not prices:
            return []

        # Generate flow-pressure signals
        try:
            flow_signals = self._strategy.generate_signals(timestamp, prices)
        except Exception:
            logger.debug(
                "Flow pressure signal generation failed",
                exc_info=True,
            )
            return []

        # Compute target weights for exposure tracking
        try:
            self._last_weights = self._strategy.get_target_weights(
                timestamp, prices
            )
        except Exception:
            self._last_weights = {}

        signals: List[StrategySignal] = []
        for fs in flow_signals:
            # Map continuous direction [-1, +1] to discrete
            if fs.direction > 0:
                direction = 1
            elif fs.direction < 0:
                direction = -1
            else:
                continue

            entry_price = None
            if fs.symbol in prices:
                entry_price = float(prices[fs.symbol][-1])

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    symbol=fs.symbol,
                    direction=direction,
                    target_weight=abs(fs.position_size),
                    confidence=fs.confidence,
                    entry_price=entry_price,
                    trade_type="flow_pressure",
                    metadata={
                        "flow_type": fs.flow_type,
                        "expected_magnitude": fs.expected_magnitude,
                        "expected_duration": fs.expected_duration,
                    },
                )
            )

        return signals

    def get_current_exposure(self) -> float:
        return sum(abs(w) for w in self._last_weights.values())

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self._strategy.reset()
        self._price_history.clear()
        self._bars_per_symbol.clear()
        self._last_weights.clear()
        self._latest_timestamp = None
        self._state = StrategyState.WARMING_UP
