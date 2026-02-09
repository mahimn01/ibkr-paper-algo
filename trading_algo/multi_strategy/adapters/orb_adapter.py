"""
Adapter wrapping the Opening Range Breakout strategy.

The ORB strategy is time-windowed: it only operates during the first
60-90 minutes of trading (9:30-10:30 AM ET). Outside this window it
produces no signals.

The adapter normalises the ORB's Dict-based signal format into the
unified StrategySignal protocol.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, time
from typing import Dict, List, Optional

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.intraday.opening_range_breakout import (
    OpeningRangeBreakout,
    ORBConfig,
)

logger = logging.getLogger(__name__)


class ORBStrategyAdapter(TradingStrategy):
    """
    Wraps OpeningRangeBreakout as a TradingStrategy.

    Active window: 9:30 AM to 10:30 AM (configurable).
    After the opening range is established, breakout signals are
    generated until the window closes.
    """

    # Time window for ORB activity
    RANGE_END = time(10, 0)    # Opening range established by 10:00
    SIGNAL_END = time(10, 30)  # Stop generating new entries after 10:30

    def __init__(self, config: Optional[ORBConfig] = None):
        self._orb = OpeningRangeBreakout(config)
        self._config = config or ORBConfig()
        self._state = StrategyState.WARMING_UP
        self._bars_seen = 0

        # Track bar counts per symbol per day for opening range detection
        self._daily_bar_counts: Dict[str, int] = defaultdict(int)
        self._current_day: Optional[datetime] = None

        # Volume tracking for confirmation
        self._volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._current_prices: Dict[str, float] = {}
        self._current_volumes: Dict[str, float] = {}

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ORB"

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
        # Detect day change and reset bar counts
        current_date = timestamp.date()
        if self._current_day is None or current_date != self._current_day:
            self._current_day = current_date
            self._daily_bar_counts.clear()

        self._daily_bar_counts[symbol] += 1
        bar_num = self._daily_bar_counts[symbol]

        # Feed the ORB strategy
        self._orb.update_opening_range(symbol, high, low, bar_num, timestamp)

        # Track prices and volumes
        self._current_prices[symbol] = close
        self._current_volumes[symbol] = volume
        self._volume_history[symbol].append(volume)

        self._bars_seen += 1
        if self._state == StrategyState.WARMING_UP and self._bars_seen >= 3:
            self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        # Only generate signals during the ORB window
        current_time = timestamp.time()
        if current_time < self.RANGE_END or current_time > self.SIGNAL_END:
            return []

        signals: List[StrategySignal] = []

        for symbol in symbols:
            price = self._current_prices.get(symbol)
            volume = self._current_volumes.get(symbol, 0)
            if price is None:
                continue

            # Calculate average volume for confirmation
            vol_hist = self._volume_history.get(symbol)
            avg_volume = (
                sum(vol_hist) / len(vol_hist) if vol_hist and len(vol_hist) > 0 else volume
            )

            # Generate ORB signals
            orb_signals = self._orb.generate_signals(
                symbol=symbol,
                current_price=price,
                current_volume=volume,
                avg_volume=avg_volume,
                current_date=timestamp,
                is_new_day=(self._daily_bar_counts[symbol] <= 1),
            )

            for orb_sig in orb_signals:
                direction = 1 if orb_sig.get("action") == "buy" else -1
                range_size = orb_sig.get("range_size", 0)
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=direction,
                    target_weight=self._config.position_size,
                    confidence=min(1.0, range_size * 50),  # Larger range = higher confidence
                    stop_loss=orb_sig.get("stop"),
                    take_profit=orb_sig.get("target"),
                    entry_price=orb_sig.get("price"),
                    trade_type="breakout",
                    metadata={
                        "range_high": orb_sig.get("range_high"),
                        "range_low": orb_sig.get("range_low"),
                        "range_size": range_size,
                    },
                ))

        return signals

    def get_current_exposure(self) -> float:
        n_positions = len(self._orb.positions) if hasattr(self._orb, "positions") else 0
        return n_positions * self._config.position_size

    def get_performance_stats(self) -> Dict[str, float]:
        try:
            return self._orb.get_performance_stats()
        except Exception:
            return {}

    def reset(self) -> None:
        self._orb = OpeningRangeBreakout(self._config)
        self._state = StrategyState.WARMING_UP
        self._bars_seen = 0
        self._daily_bar_counts.clear()
        self._current_day = None
