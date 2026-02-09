"""
Intraday Momentum Strategy

Based on: Gao, Han, Li, Zhou (2018) "Market Intraday Momentum"
          Review of Financial Studies, 31(12), 4889-4941.

Key finding: The first 30-minute return of the trading day predicts the
last 30-minute return with ~25% correlation (after controlling for
overnight returns).  The effect is strongest on high-volume days and
when institutional order flow is directionally persistent.

Implementation:
    1. Capture the first-30-minute return (9:30 - 10:00 AM ET).
    2. Wait for confirmation (don't trade during 10:00 - 15:00).
    3. At 15:30, enter in the direction of the opening return if:
       a) The opening return is above a threshold (default 0.3%).
       b) The midday return doesn't fully reverse the opening move.
    4. Exit at 15:55 (before MOC).

Expected: ~20-30% standalone Sharpe, ~2-3% portfolio contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional

import numpy as np


@dataclass
class IntradayMomentumConfig:
    """Configuration for the intraday momentum strategy."""

    # Time windows (ET)
    opening_window_end: time = time(10, 0)
    """End of the opening measurement window (default 10:00 AM)."""

    entry_time: time = time(15, 30)
    """Time to enter positions (default 3:30 PM)."""

    exit_time: time = time(15, 55)
    """Time to exit positions (default 3:55 PM)."""

    # Signal thresholds
    min_opening_return: float = 0.003
    """Minimum absolute opening return to trigger a signal (0.3%)."""

    max_reversal_ratio: float = 0.80
    """If midday return reverses more than this fraction of the opening
    return, the signal is cancelled.  0.80 = midday can reverse up to
    80% of the opening move and the signal still holds."""

    # Position sizing
    position_size: float = 0.10
    """Target portfolio weight per signal (10%)."""

    # Stop loss
    stop_pct: float = 0.005
    """Intraday stop loss as fraction of entry price (0.5%)."""


@dataclass
class IntradayMomSignal:
    """Signal produced by the intraday momentum strategy."""
    symbol: str
    direction: int          # 1 = long, -1 = short
    opening_return: float   # Signed opening return
    midday_return: float    # Return from open-window-end to now
    confidence: float       # Signal quality 0-1
    entry_price: float
    stop_loss: float
    target_price: float


class IntradayMomentumStrategy:
    """
    Implements the intraday momentum anomaly.

    Usage::

        strategy = IntradayMomentumStrategy()

        # Feed bars throughout the day
        for bar in bars:
            strategy.update(symbol, bar.timestamp, bar.close, bar.volume)

        # At 15:30, check for signals
        signals = strategy.generate_signals(symbols, current_time, current_prices)
    """

    def __init__(self, config: Optional[IntradayMomentumConfig] = None):
        self.config = config or IntradayMomentumConfig()

        # Per-symbol daily tracking
        self._day_open: Dict[str, float] = {}          # First price of the day
        self._opening_close: Dict[str, float] = {}     # Price at end of opening window
        self._current_prices: Dict[str, float] = {}
        self._current_volumes: Dict[str, float] = {}
        self._daily_volume: Dict[str, float] = {}
        self._avg_daily_volume: Dict[str, float] = {}
        self._current_date: Optional[datetime] = None

        # Volume history for relative volume computation
        self._volume_history: Dict[str, List[float]] = {}

    def new_day(self, date: datetime) -> None:
        """Reset for a new trading day."""
        # Save previous day's volume for avg computation
        for symbol, vol in self._daily_volume.items():
            if symbol not in self._volume_history:
                self._volume_history[symbol] = []
            self._volume_history[symbol].append(vol)
            # Keep 20 days
            if len(self._volume_history[symbol]) > 20:
                self._volume_history[symbol] = self._volume_history[symbol][-20:]
            self._avg_daily_volume[symbol] = np.mean(self._volume_history[symbol])

        self._day_open.clear()
        self._opening_close.clear()
        self._daily_volume.clear()
        self._current_date = date

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float = 0,
    ) -> None:
        """Feed a bar of data."""
        # Detect day change
        if self._current_date is None or timestamp.date() != self._current_date.date():
            self.new_day(timestamp)

        current_time = timestamp.time()
        self._current_prices[symbol] = price
        self._current_volumes[symbol] = volume
        self._daily_volume[symbol] = self._daily_volume.get(symbol, 0) + volume

        # Capture day open
        if symbol not in self._day_open:
            self._day_open[symbol] = price

        # Capture opening window close price
        if current_time <= self.config.opening_window_end:
            self._opening_close[symbol] = price

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[IntradayMomSignal]:
        """
        Generate intraday momentum signals.

        Should be called at or after ``config.entry_time`` (default 15:30).
        """
        current_time = timestamp.time()

        # Only generate signals during the entry window
        if current_time < self.config.entry_time or current_time > self.config.exit_time:
            return []

        signals: List[IntradayMomSignal] = []

        for symbol in symbols:
            open_price = self._day_open.get(symbol)
            opening_close = self._opening_close.get(symbol)
            price = (current_prices or {}).get(symbol) or self._current_prices.get(symbol)

            if open_price is None or opening_close is None or price is None:
                continue
            if open_price <= 0:
                continue

            # Opening return (first 30 min)
            opening_return = (opening_close - open_price) / open_price

            # Check minimum threshold
            if abs(opening_return) < self.config.min_opening_return:
                continue

            # Midday return: from opening close to current price
            midday_return = (price - opening_close) / opening_close if opening_close > 0 else 0

            # Check reversal — if the midday has reversed too much of
            # the opening move, the momentum has dissipated
            if opening_return != 0:
                reversal_ratio = -midday_return / opening_return
                if reversal_ratio > self.config.max_reversal_ratio:
                    continue

            # Direction follows opening return
            direction = 1 if opening_return > 0 else -1

            # Confidence based on opening return magnitude and volume
            magnitude_score = min(1.0, abs(opening_return) / 0.01)  # 1% → full score
            vol_ratio = 1.0
            if symbol in self._avg_daily_volume and self._avg_daily_volume[symbol] > 0:
                vol_ratio = min(2.0, self._daily_volume.get(symbol, 0) / self._avg_daily_volume[symbol])
            volume_score = min(1.0, vol_ratio)
            confidence = 0.5 * magnitude_score + 0.5 * volume_score

            # Stop/target
            stop_loss = price * (1 - self.config.stop_pct * direction)
            target_price = price * (1 + abs(opening_return) * direction)

            signals.append(IntradayMomSignal(
                symbol=symbol,
                direction=direction,
                opening_return=opening_return,
                midday_return=midday_return,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop_loss,
                target_price=target_price,
            ))

        return signals

    def reset(self) -> None:
        """Reset all state."""
        self._day_open.clear()
        self._opening_close.clear()
        self._current_prices.clear()
        self._current_volumes.clear()
        self._daily_volume.clear()
        self._avg_daily_volume.clear()
        self._volume_history.clear()
        self._current_date = None
