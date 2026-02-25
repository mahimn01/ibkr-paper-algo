"""
Intraday Time-Adaptive Strategy

Exploits the time-of-day autocorrelation structure discovered in pattern
analysis.  Different market microstructure regimes dominate different
parts of the trading day:

    Morning   (9:30-10:30):  AutoCorr = +0.016 -> MOMENTUM
    Midday    (10:30-14:00): AutoCorr = -0.005 -> RANDOM (skip)
    Afternoon (14:00-15:30): AutoCorr = -0.041 -> MEAN-REVERSION
    Close     (15:30-16:00): AutoCorr = +0.030, Sharpe = +0.210 -> MOMENTUM

Sub-strategies per time window:

    Morning Momentum (9:30-10:30):
        Opening range breakout variant.  Track the first 30-min range,
        trade breakouts with VWAP confirmation.  Stop below/above the
        opposite end of the opening range.

    Midday Quiet (10:30-14:00):
        No new entries (random walk period).  Manage existing positions
        only.  Close unprofitable morning positions by 11:30.

    Afternoon Mean-Reversion (14:00-15:30):
        Fade moves > 1.5 std from intraday VWAP.  Target: return to VWAP.
        Stop: 2 std from VWAP.

    Close Momentum (15:30-16:00):
        Follow the last 30-min trend direction using 6-bar momentum.
        Close all positions by 15:55.

All intraday state (opening range, VWAP, cumulative volume) is reset
at the start of each new trading day.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TimeAdaptiveConfig",
    "IntradayTimeAdaptive",
    "TimeAdaptiveSignal",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TimeAdaptiveConfig:
    """
    Configuration for the intraday time-adaptive strategy.

    Attributes:
        morning_start: Start of morning momentum window (hour, minute).
        morning_end: End of morning momentum window.
        midday_start: Start of midday quiet period.
        midday_end: End of midday quiet period.
        afternoon_start: Start of afternoon mean-reversion window.
        afternoon_end: End of afternoon mean-reversion window.
        close_start: Start of close momentum window.
        close_end: End of close momentum window.
        opening_range_bars: Number of 5-min bars to define the opening
            range (6 bars = 30 minutes).
        vwap_std_entry: Number of intraday VWAP standard deviations to
            trigger a mean-reversion entry in the afternoon.
        vwap_std_stop: Number of VWAP std for mean-reversion stop loss.
        close_momentum_bars: Number of bars for close-session momentum
            calculation (6 bars = 30 minutes).
        max_weight: Maximum portfolio weight per symbol per signal.
        warmup: Minimum number of 5-min bars before generating signals.
    """

    morning_start: tuple = (9, 30)
    morning_end: tuple = (10, 30)
    midday_start: tuple = (10, 30)
    midday_end: tuple = (14, 0)
    afternoon_start: tuple = (14, 0)
    afternoon_end: tuple = (15, 30)
    close_start: tuple = (15, 30)
    close_end: tuple = (16, 0)
    opening_range_bars: int = 6   # 30 minutes of 5-min bars
    vwap_std_entry: float = 1.5
    vwap_std_stop: float = 2.0
    close_momentum_bars: int = 6
    max_weight: float = 0.08
    warmup: int = 20


# =============================================================================
# SIGNAL DATA CLASS
# =============================================================================

@dataclass
class TimeAdaptiveSignal:
    """
    Signal produced by the intraday time-adaptive strategy.

    Attributes:
        symbol: Instrument identifier.
        direction: 1 = long, -1 = short, 0 = exit/flat.
        weight: Target portfolio weight for this leg.
        confidence: Signal quality score in [0, 1].
        stop_loss: Absolute stop loss price.
        take_profit: Absolute take profit price (if applicable).
        entry_price: Price at signal generation.
        trade_type: Sub-strategy that generated this signal.
        time_window: Current time window name.
    """

    symbol: str
    direction: int
    weight: float
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_price: Optional[float] = None
    trade_type: str = ""
    time_window: str = ""


# =============================================================================
# PER-SYMBOL INTRADAY STATE
# =============================================================================

@dataclass
class _IntradayState:
    """
    Internal per-symbol intraday tracking state.

    Reset at the start of each new trading day.
    """

    # Opening range tracking
    opening_range_high: float = -np.inf
    opening_range_low: float = np.inf
    opening_range_bars_seen: int = 0
    opening_range_set: bool = False

    # VWAP tracking
    cumulative_price_volume: float = 0.0
    cumulative_volume: float = 0.0
    vwap: float = np.nan
    vwap_sum_sq_dev: float = 0.0   # For rolling VWAP std
    vwap_n: int = 0

    # Price/volume history for the current day
    close_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    high_history: List[float] = field(default_factory=list)
    low_history: List[float] = field(default_factory=list)

    # Current day tracking
    current_date: Optional[object] = None  # date object
    bars_today: int = 0

    # Active position tracking (simplified)
    active_direction: int = 0        # 0 = flat, 1 = long, -1 = short
    active_entry_price: float = 0.0
    active_entry_time: Optional[datetime] = None
    active_trade_type: str = ""


# =============================================================================
# INTRADAY TIME-ADAPTIVE STRATEGY
# =============================================================================

class IntradayTimeAdaptive:
    """
    Intraday strategy that adapts sub-strategy selection to time of day.

    Usage::

        strategy = IntradayTimeAdaptive()

        for bar in intraday_5min_bars:
            strategy.update(
                symbol="AAPL",
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

        signals = strategy.generate_signals(
            symbols=["AAPL"],
            timestamp=current_timestamp,
        )
    """

    def __init__(self, config: Optional[TimeAdaptiveConfig] = None):
        """
        Initialize the intraday time-adaptive strategy.

        Args:
            config: Strategy configuration.  Uses defaults if not provided.
        """
        self._config = config or TimeAdaptiveConfig()

        # Per-symbol intraday state
        self._state: Dict[str, _IntradayState] = defaultdict(
            _IntradayState
        )

        # Global bar counter for warmup
        self._total_bars: int = 0

        # Pre-compute time boundaries as ``time`` objects for fast comparison
        self._morning_start = time(*self._config.morning_start)
        self._morning_end = time(*self._config.morning_end)
        self._midday_start = time(*self._config.midday_start)
        self._midday_end = time(*self._config.midday_end)
        self._afternoon_start = time(*self._config.afternoon_start)
        self._afternoon_end = time(*self._config.afternoon_end)
        self._close_start = time(*self._config.close_start)
        self._close_end = time(*self._config.close_end)
        self._force_close_time = time(15, 55)
        self._midday_close_time = time(11, 30)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

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
        """
        Feed a single 5-min bar to the strategy.

        Args:
            symbol: Instrument identifier.
            timestamp: Bar timestamp.
            open_price: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume.
        """
        if not np.isfinite(close) or close <= 0:
            return
        if not np.isfinite(volume) or volume < 0:
            volume = 0.0

        st = self._state[symbol]
        bar_date = timestamp.date()

        # Reset intraday state on new day
        if st.current_date is not None and bar_date != st.current_date:
            self._reset_intraday(symbol)

        st = self._state[symbol]
        st.current_date = bar_date
        st.bars_today += 1
        self._total_bars += 1

        # Update price/volume history
        st.close_history.append(close)
        st.volume_history.append(volume)
        st.high_history.append(high)
        st.low_history.append(low)

        # Update opening range
        bar_time = timestamp.time()
        if not st.opening_range_set and self._morning_start <= bar_time < self._morning_end:
            st.opening_range_bars_seen += 1
            st.opening_range_high = max(st.opening_range_high, high)
            st.opening_range_low = min(st.opening_range_low, low)
            if st.opening_range_bars_seen >= self._config.opening_range_bars:
                st.opening_range_set = True

        # Update VWAP
        if volume > 0:
            typical_price = (high + low + close) / 3.0
            st.cumulative_price_volume += typical_price * volume
            st.cumulative_volume += volume
            st.vwap = st.cumulative_price_volume / st.cumulative_volume

            # Update VWAP standard deviation (Welford-style)
            st.vwap_n += 1
            deviation = typical_price - st.vwap
            st.vwap_sum_sq_dev += deviation * deviation * volume
        elif st.cumulative_volume > 0:
            # No volume bar -- keep existing VWAP
            pass

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[TimeAdaptiveSignal]:
        """
        Generate time-adaptive signals for the current bar.

        Args:
            symbols: List of symbols to generate signals for.
            timestamp: Current bar timestamp.

        Returns:
            List of TimeAdaptiveSignal objects.  Empty if no signals
            are warranted for the current time window.
        """
        if self._total_bars < self._config.warmup:
            return []

        bar_time = timestamp.time()
        window = self._classify_time_window(bar_time)

        signals: List[TimeAdaptiveSignal] = []

        for symbol in symbols:
            st = self._state.get(symbol)
            if st is None or st.bars_today < 1:
                continue

            signal = self._generate_symbol_signal(symbol, st, timestamp, bar_time, window)
            if signal is not None:
                signals.append(signal)

        return signals

    def reset(self) -> None:
        """Reset all internal state for a new session or backtest run."""
        self._state.clear()
        self._total_bars = 0

    # ------------------------------------------------------------------
    # TIME WINDOW CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify_time_window(self, bar_time: time) -> str:
        """
        Determine which time-of-day regime the current bar falls into.

        Args:
            bar_time: Time component of the current bar timestamp.

        Returns:
            One of "morning", "midday", "afternoon", "close", or "outside".
        """
        if self._morning_start <= bar_time < self._morning_end:
            return "morning"
        elif self._midday_start <= bar_time < self._midday_end:
            return "midday"
        elif self._afternoon_start <= bar_time < self._afternoon_end:
            return "afternoon"
        elif self._close_start <= bar_time < self._close_end:
            return "close"
        else:
            return "outside"

    # ------------------------------------------------------------------
    # PER-SYMBOL SIGNAL GENERATION
    # ------------------------------------------------------------------

    def _generate_symbol_signal(
        self,
        symbol: str,
        st: _IntradayState,
        timestamp: datetime,
        bar_time: time,
        window: str,
    ) -> Optional[TimeAdaptiveSignal]:
        """
        Generate a signal for a single symbol based on the current time window.

        Args:
            symbol: Instrument identifier.
            st: Per-symbol intraday state.
            timestamp: Current bar timestamp.
            bar_time: Time component of timestamp.
            window: Current time window classification.

        Returns:
            TimeAdaptiveSignal or None.
        """
        # Force close all positions by 15:55 regardless of window
        if bar_time >= self._force_close_time and st.active_direction != 0:
            return self._create_exit_signal(symbol, st, "force_close_eod", window)

        if window == "morning":
            return self._morning_momentum(symbol, st, timestamp, bar_time)
        elif window == "midday":
            return self._midday_manage(symbol, st, timestamp, bar_time)
        elif window == "afternoon":
            return self._afternoon_mean_reversion(symbol, st, timestamp)
        elif window == "close":
            return self._close_momentum(symbol, st, timestamp, bar_time)
        else:
            return None

    # ------------------------------------------------------------------
    # MORNING MOMENTUM (9:30-10:30)
    # ------------------------------------------------------------------

    def _morning_momentum(
        self,
        symbol: str,
        st: _IntradayState,
        timestamp: datetime,
        bar_time: time,
    ) -> Optional[TimeAdaptiveSignal]:
        """
        Opening range breakout strategy with VWAP confirmation.

        Waits for the opening range to be established, then trades
        breakouts above/below with VWAP as a directional filter.
        """
        if not st.opening_range_set:
            return None

        if len(st.close_history) < 1:
            return None

        current_price = st.close_history[-1]
        range_high = st.opening_range_high
        range_low = st.opening_range_low

        # Guard against degenerate range
        if range_high <= range_low or not np.isfinite(range_high) or not np.isfinite(range_low):
            return None

        vwap = st.vwap
        has_vwap = np.isfinite(vwap)

        # Already in a position -- no new entries during morning
        if st.active_direction != 0:
            return None

        direction = 0
        confidence = 0.0

        # Breakout above opening range
        if current_price > range_high:
            # VWAP confirmation: price should be above VWAP for long
            if has_vwap and current_price > vwap:
                direction = 1
                # Confidence based on breakout magnitude relative to range
                range_size = range_high - range_low
                breakout_pct = (current_price - range_high) / range_size if range_size > 0 else 0.0
                confidence = min(1.0, 0.5 + breakout_pct * 2.0)
            elif not has_vwap:
                direction = 1
                confidence = 0.4

        # Breakdown below opening range
        elif current_price < range_low:
            # VWAP confirmation: price should be below VWAP for short
            if has_vwap and current_price < vwap:
                direction = -1
                range_size = range_high - range_low
                breakout_pct = (range_low - current_price) / range_size if range_size > 0 else 0.0
                confidence = min(1.0, 0.5 + breakout_pct * 2.0)
            elif not has_vwap:
                direction = -1
                confidence = 0.4

        if direction == 0:
            return None

        # Stop loss: opposite end of opening range
        if direction == 1:
            stop_loss = range_low
        else:
            stop_loss = range_high

        # Update active position tracking
        st.active_direction = direction
        st.active_entry_price = current_price
        st.active_entry_time = timestamp
        st.active_trade_type = "morning_momentum"

        return TimeAdaptiveSignal(
            symbol=symbol,
            direction=direction,
            weight=self._config.max_weight * confidence,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=None,
            entry_price=current_price,
            trade_type="morning_momentum",
            time_window="morning",
        )

    # ------------------------------------------------------------------
    # MIDDAY MANAGEMENT (10:30-14:00)
    # ------------------------------------------------------------------

    def _midday_manage(
        self,
        symbol: str,
        st: _IntradayState,
        timestamp: datetime,
        bar_time: time,
    ) -> Optional[TimeAdaptiveSignal]:
        """
        No new entries during the midday random walk period.

        Manages existing positions:
        - Close unprofitable morning positions by 11:30
        - Check stop losses on existing positions
        """
        if st.active_direction == 0:
            return None

        current_price = st.close_history[-1] if st.close_history else None
        if current_price is None or not np.isfinite(current_price):
            return None

        # Close morning positions by 11:30 if not profitable
        if bar_time >= self._midday_close_time and st.active_trade_type == "morning_momentum":
            if st.active_entry_price > 0:
                pnl = (current_price - st.active_entry_price) * st.active_direction
                if pnl <= 0:
                    return self._create_exit_signal(
                        symbol, st, "midday_close_unprofitable", "midday",
                    )

        # Check stop loss for any active position
        return self._check_stop_loss(symbol, st, current_price, "midday")

    # ------------------------------------------------------------------
    # AFTERNOON MEAN-REVERSION (14:00-15:30)
    # ------------------------------------------------------------------

    def _afternoon_mean_reversion(
        self,
        symbol: str,
        st: _IntradayState,
        timestamp: datetime,
    ) -> Optional[TimeAdaptiveSignal]:
        """
        Fade extreme deviations from intraday VWAP.

        Entry: price > vwap_std_entry * std above/below VWAP
        Target: return to VWAP
        Stop: vwap_std_stop * std from VWAP
        """
        if not np.isfinite(st.vwap) or st.cumulative_volume <= 0:
            return None

        if len(st.close_history) < 2:
            return None

        current_price = st.close_history[-1]
        vwap = st.vwap

        # Compute VWAP standard deviation
        vwap_std = self._compute_vwap_std(st)
        if vwap_std <= 0 or not np.isfinite(vwap_std):
            return None

        deviation = (current_price - vwap) / vwap_std

        # If already in a position, check for mean-reversion target or stop
        if st.active_direction != 0:
            # Target: return to VWAP
            if st.active_trade_type == "afternoon_mean_reversion":
                if st.active_direction == -1 and current_price <= vwap:
                    return self._create_exit_signal(
                        symbol, st, "mean_reversion_target", "afternoon",
                    )
                elif st.active_direction == 1 and current_price >= vwap:
                    return self._create_exit_signal(
                        symbol, st, "mean_reversion_target", "afternoon",
                    )
            # Check stop
            return self._check_stop_loss(symbol, st, current_price, "afternoon")

        direction = 0
        confidence = 0.0

        # Price > entry_threshold std above VWAP -> short (fade)
        if deviation > self._config.vwap_std_entry:
            direction = -1
            confidence = min(1.0, 0.4 + (deviation - self._config.vwap_std_entry) * 0.3)

        # Price < entry_threshold std below VWAP -> long (fade)
        elif deviation < -self._config.vwap_std_entry:
            direction = 1
            confidence = min(1.0, 0.4 + (abs(deviation) - self._config.vwap_std_entry) * 0.3)

        if direction == 0:
            return None

        # Stop loss: vwap_std_stop from VWAP
        if direction == -1:
            stop_loss = vwap + self._config.vwap_std_stop * vwap_std
        else:
            stop_loss = vwap - self._config.vwap_std_stop * vwap_std

        # Take profit: VWAP
        take_profit = vwap

        # Update active position tracking
        st.active_direction = direction
        st.active_entry_price = current_price
        st.active_entry_time = timestamp
        st.active_trade_type = "afternoon_mean_reversion"

        return TimeAdaptiveSignal(
            symbol=symbol,
            direction=direction,
            weight=self._config.max_weight * confidence,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=current_price,
            trade_type="afternoon_mean_reversion",
            time_window="afternoon",
        )

    # ------------------------------------------------------------------
    # CLOSE MOMENTUM (15:30-16:00)
    # ------------------------------------------------------------------

    def _close_momentum(
        self,
        symbol: str,
        st: _IntradayState,
        timestamp: datetime,
        bar_time: time,
    ) -> Optional[TimeAdaptiveSignal]:
        """
        Follow the last 30-min trend direction using 6-bar momentum.

        Close all positions by 15:55.
        """
        # Force close at 15:55 (handled in parent, but double-check)
        if bar_time >= self._force_close_time:
            if st.active_direction != 0:
                return self._create_exit_signal(symbol, st, "force_close_eod", "close")
            return None

        n_momentum = self._config.close_momentum_bars
        if len(st.close_history) < n_momentum + 1:
            return None

        current_price = st.close_history[-1]

        # Already in a close momentum position -- let it ride until 15:55
        if st.active_direction != 0 and st.active_trade_type == "close_momentum":
            return None

        # If in a different position type, close it first
        if st.active_direction != 0:
            return self._create_exit_signal(
                symbol, st, "close_window_transition", "close",
            )

        # Compute 6-bar (30-min) momentum
        past_price = st.close_history[-(n_momentum + 1)]
        if past_price <= 0 or not np.isfinite(past_price):
            return None

        momentum = (current_price - past_price) / past_price

        # Require a minimum momentum threshold to avoid noise
        min_momentum = 0.001  # 10 bps
        if abs(momentum) < min_momentum:
            return None

        direction = 1 if momentum > 0 else -1
        confidence = min(1.0, abs(momentum) / 0.005)  # Scale by 50 bps

        # No stop loss for close momentum -- tight time window
        st.active_direction = direction
        st.active_entry_price = current_price
        st.active_entry_time = timestamp
        st.active_trade_type = "close_momentum"

        return TimeAdaptiveSignal(
            symbol=symbol,
            direction=direction,
            weight=self._config.max_weight * confidence,
            confidence=confidence,
            stop_loss=None,
            take_profit=None,
            entry_price=current_price,
            trade_type="close_momentum",
            time_window="close",
        )

    # ------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------

    def _compute_vwap_std(self, st: _IntradayState) -> float:
        """
        Compute the standard deviation of prices around VWAP.

        Uses the volume-weighted sum of squared deviations from VWAP
        tracked incrementally via Welford's method.

        Args:
            st: Per-symbol intraday state.

        Returns:
            VWAP standard deviation.  Returns 0.0 if insufficient data.
        """
        if st.cumulative_volume <= 0 or st.vwap_n < 2:
            return 0.0

        variance = st.vwap_sum_sq_dev / st.cumulative_volume
        if variance <= 0:
            return 0.0

        return float(math.sqrt(variance))

    def _check_stop_loss(
        self,
        symbol: str,
        st: _IntradayState,
        current_price: float,
        window: str,
    ) -> Optional[TimeAdaptiveSignal]:
        """
        Check if the active position has hit its stop loss.

        For morning_momentum: stop is at opposite end of opening range.
        For afternoon_mean_reversion: stop is at vwap_std_stop * std.

        Args:
            symbol: Instrument identifier.
            st: Per-symbol intraday state.
            current_price: Current close price.
            window: Current time window name.

        Returns:
            Exit signal if stop is hit, None otherwise.
        """
        if st.active_direction == 0:
            return None

        if st.active_trade_type == "morning_momentum":
            if st.active_direction == 1 and current_price <= st.opening_range_low:
                return self._create_exit_signal(symbol, st, "stop_loss", window)
            elif st.active_direction == -1 and current_price >= st.opening_range_high:
                return self._create_exit_signal(symbol, st, "stop_loss", window)

        elif st.active_trade_type == "afternoon_mean_reversion":
            vwap_std = self._compute_vwap_std(st)
            if vwap_std > 0 and np.isfinite(st.vwap):
                if st.active_direction == 1:
                    stop_price = st.vwap - self._config.vwap_std_stop * vwap_std
                    if current_price <= stop_price:
                        return self._create_exit_signal(symbol, st, "stop_loss", window)
                elif st.active_direction == -1:
                    stop_price = st.vwap + self._config.vwap_std_stop * vwap_std
                    if current_price >= stop_price:
                        return self._create_exit_signal(symbol, st, "stop_loss", window)

        return None

    def _create_exit_signal(
        self,
        symbol: str,
        st: _IntradayState,
        reason: str,
        window: str,
    ) -> TimeAdaptiveSignal:
        """
        Create an exit signal and reset the active position state.

        Args:
            symbol: Instrument identifier.
            st: Per-symbol intraday state.
            reason: Exit reason string for metadata.
            window: Current time window name.

        Returns:
            TimeAdaptiveSignal with direction=0 (exit).
        """
        entry_price = st.active_entry_price
        trade_type = st.active_trade_type

        # Reset active position
        st.active_direction = 0
        st.active_entry_price = 0.0
        st.active_entry_time = None
        st.active_trade_type = ""

        return TimeAdaptiveSignal(
            symbol=symbol,
            direction=0,
            weight=0.0,
            confidence=1.0,
            stop_loss=None,
            take_profit=None,
            entry_price=entry_price,
            trade_type=f"exit_{trade_type}_{reason}",
            time_window=window,
        )

    def _reset_intraday(self, symbol: str) -> None:
        """
        Reset all intraday state for a symbol at the start of a new day.

        Preserves the symbol key in the state dict but clears all
        accumulated intraday data (opening range, VWAP, positions, etc.).

        Args:
            symbol: Instrument identifier.
        """
        self._state[symbol] = _IntradayState()
