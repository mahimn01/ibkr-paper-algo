"""
Chameleon Day Trader - Aggressive Intraday Trading

A more aggressive version of the Chameleon Strategy optimized for day trading:
- Fast regime detection (5/15/30 bar lookbacks on 5-min data)
- Lower entry thresholds (more trades)
- Momentum-based entries
- Intraday risk management

For use with 5-minute bars during market hours.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple


class DayTradeMode(Enum):
    """Intraday regime classification."""
    STRONG_MOMENTUM_UP = auto()    # Strong upward momentum - aggressive long
    MOMENTUM_UP = auto()           # Upward momentum - long bias
    CHOPPY_BULLISH = auto()        # Choppy but slightly bullish
    NEUTRAL = auto()               # No clear direction
    CHOPPY_BEARISH = auto()        # Choppy but slightly bearish
    MOMENTUM_DOWN = auto()         # Downward momentum - short bias
    STRONG_MOMENTUM_DOWN = auto()  # Strong downward momentum - aggressive short
    VOLATILITY_SPIKE = auto()      # Extreme volatility - be careful


@dataclass
class DayTradeSignal:
    """Day trading signal."""
    symbol: str
    timestamp: datetime
    action: str  # 'buy', 'short', 'sell', 'cover', 'hold'
    size: float  # Position size as fraction (0.1 = 10%)
    mode: DayTradeMode
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class FastMomentumAnalyzer:
    """
    Fast momentum analysis for day trading.

    Uses short lookbacks appropriate for 5-minute bars:
    - Fast: 5 bars = 25 minutes
    - Medium: 15 bars = 75 minutes
    - Slow: 30 bars = 2.5 hours
    """

    def __init__(self):
        self._prices: Dict[str, Deque[float]] = {}
        self._volumes: Dict[str, Deque[float]] = {}
        self._highs: Dict[str, Deque[float]] = {}
        self._lows: Dict[str, Deque[float]] = {}

        # Fast lookbacks for day trading
        self.fast_period = 5      # 25 minutes
        self.medium_period = 15   # 75 minutes
        self.slow_period = 30     # 2.5 hours
        self.max_history = 100    # ~8 hours

    def update(self, symbol: str, price: float, high: float, low: float, volume: float):
        """Update with new bar."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.max_history)
            self._volumes[symbol] = deque(maxlen=self.max_history)
            self._highs[symbol] = deque(maxlen=self.max_history)
            self._lows[symbol] = deque(maxlen=self.max_history)

        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)
        self._highs[symbol].append(high)
        self._lows[symbol].append(low)

    def get_momentum_signals(self, symbol: str) -> Optional[Dict]:
        """Get momentum signals for day trading."""
        if symbol not in self._prices or len(self._prices[symbol]) < self.slow_period:
            return None

        prices = list(self._prices[symbol])
        volumes = list(self._volumes[symbol])
        highs = list(self._highs[symbol])
        lows = list(self._lows[symbol])

        current_price = prices[-1]

        # Fast momentum (5 bars)
        fast_return = (prices[-1] - prices[-self.fast_period]) / prices[-self.fast_period]
        fast_sma = sum(prices[-self.fast_period:]) / self.fast_period

        # Medium momentum (15 bars)
        medium_return = (prices[-1] - prices[-self.medium_period]) / prices[-self.medium_period]
        medium_sma = sum(prices[-self.medium_period:]) / self.medium_period

        # Slow momentum (30 bars)
        slow_return = (prices[-1] - prices[-self.slow_period]) / prices[-self.slow_period]
        slow_sma = sum(prices[-self.slow_period:]) / self.slow_period

        # Price vs SMAs
        above_fast = current_price > fast_sma
        above_medium = current_price > medium_sma
        above_slow = current_price > slow_sma

        # Volatility (ATR-like)
        recent_ranges = [highs[i] - lows[i] for i in range(-self.fast_period, 0)]
        avg_range = sum(recent_ranges) / len(recent_ranges)
        volatility = avg_range / current_price  # As percentage

        # Volume analysis
        recent_volume = sum(volumes[-self.fast_period:]) / self.fast_period
        older_volume = sum(volumes[-self.medium_period:-self.fast_period]) / (self.medium_period - self.fast_period)
        volume_surge = recent_volume / older_volume if older_volume > 0 else 1.0

        # Trend strength (are all timeframes aligned?)
        bullish_alignment = sum([above_fast, above_medium, above_slow])
        bearish_alignment = sum([not above_fast, not above_medium, not above_slow])

        # RSI-like momentum
        gains = []
        losses = []
        for i in range(-self.medium_period + 1, 0):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {
            'fast_return': fast_return,
            'medium_return': medium_return,
            'slow_return': slow_return,
            'above_fast': above_fast,
            'above_medium': above_medium,
            'above_slow': above_slow,
            'bullish_alignment': bullish_alignment,
            'bearish_alignment': bearish_alignment,
            'volatility': volatility,
            'volume_surge': volume_surge,
            'rsi': rsi,
            'current_price': current_price,
            'fast_sma': fast_sma,
            'medium_sma': medium_sma,
            'slow_sma': slow_sma,
        }


class ChameleonDayTrader:
    """
    Aggressive day trading strategy.

    Key features:
    - Fast regime detection using momentum
    - Low entry thresholds (more trades)
    - Momentum chasing with volume confirmation
    - Tight stop losses for risk management
    - Takes profits quickly
    """

    def __init__(
        self,
        position_pct: float = 0.02,         # 2% of account per trade
        max_position_pct: float = 0.05,     # 5% max for strong signals
        max_position_dollars: float = 10000,  # Hard cap at $10k per position
        stop_loss_pct: float = 0.005,       # 0.5% stop loss (tight for day trading)
        take_profit_pct: float = 0.01,      # 1% take profit
        min_volume_surge: float = 1.2,      # 20% volume increase to confirm
    ):
        self.analyzer = FastMomentumAnalyzer()

        # Position sizing (percentage with dollar cap)
        self.position_pct = position_pct
        self.max_position_pct = max_position_pct
        self.max_position_dollars = max_position_dollars

        # Risk management
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_volume_surge = min_volume_surge

        # State
        self.positions: Dict[str, dict] = {}
        self._current_mode: Dict[str, DayTradeMode] = {}

    def clear_positions(self):
        """Clear all positions (call after warmup)."""
        self.positions.clear()

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> Optional[DayTradeSignal]:
        """Process new bar and generate trading signal."""

        # Update analyzer
        self.analyzer.update(symbol, close, high, low, volume)

        # Get momentum signals
        signals = self.analyzer.get_momentum_signals(symbol)
        if signals is None:
            return None

        # Determine current mode
        mode = self._determine_mode(signals)
        self._current_mode[symbol] = mode

        # Generate trading decision
        return self._generate_signal(
            symbol=symbol,
            timestamp=timestamp,
            price=close,
            signals=signals,
            mode=mode,
        )

    def _determine_mode(self, signals: Dict) -> DayTradeMode:
        """Determine day trading mode from signals."""
        fast_ret = signals['fast_return']
        medium_ret = signals['medium_return']
        bullish = signals['bullish_alignment']
        bearish = signals['bearish_alignment']
        volatility = signals['volatility']
        rsi = signals['rsi']

        # Volatility spike - be careful
        if volatility > 0.02:  # 2% range in 5 min is huge
            return DayTradeMode.VOLATILITY_SPIKE

        # Strong momentum up
        if fast_ret > 0.003 and medium_ret > 0.002 and bullish >= 2:
            return DayTradeMode.STRONG_MOMENTUM_UP

        # Strong momentum down
        if fast_ret < -0.003 and medium_ret < -0.002 and bearish >= 2:
            return DayTradeMode.STRONG_MOMENTUM_DOWN

        # Regular momentum
        if fast_ret > 0.001 and bullish >= 2:
            return DayTradeMode.MOMENTUM_UP

        if fast_ret < -0.001 and bearish >= 2:
            return DayTradeMode.MOMENTUM_DOWN

        # Choppy with bias
        if rsi > 55 or bullish >= 2:
            return DayTradeMode.CHOPPY_BULLISH

        if rsi < 45 or bearish >= 2:
            return DayTradeMode.CHOPPY_BEARISH

        return DayTradeMode.NEUTRAL

    def _generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        signals: Dict,
        mode: DayTradeMode,
    ) -> DayTradeSignal:
        """Generate trading signal based on mode."""

        has_position = symbol in self.positions
        position = self.positions.get(symbol)

        # Check existing position for exit
        if has_position:
            return self._check_exit(symbol, timestamp, price, signals, mode, position)

        # No position - check for entry
        return self._check_entry(symbol, timestamp, price, signals, mode)

    def _check_entry(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        signals: Dict,
        mode: DayTradeMode,
    ) -> DayTradeSignal:
        """Check for entry opportunities."""

        volume_confirmed = signals['volume_surge'] >= self.min_volume_surge
        rsi = signals['rsi']

        # LONG entries
        if mode == DayTradeMode.STRONG_MOMENTUM_UP:
            # Strong momentum - enter aggressively
            size_pct = self.max_position_pct if volume_confirmed else self.position_pct
            stop = price * (1 - self.stop_loss_pct * 2)  # Wider stop for momentum
            take_profit = price * (1 + self.take_profit_pct * 2)

            self.positions[symbol] = {
                'direction': 1,
                'entry_price': price,
                'entry_time': timestamp,
                'stop_loss': stop,
                'take_profit': take_profit,
            }

            return DayTradeSignal(
                symbol=symbol,
                timestamp=timestamp,
                action='buy',
                size=size_pct,
                mode=mode,
                confidence=0.8 if volume_confirmed else 0.6,
                entry_price=price,
                stop_loss=stop,
                take_profit=take_profit,
                reason=f"Strong momentum UP, RSI={rsi:.0f}, Vol surge={signals['volume_surge']:.1f}x",
            )

        elif mode == DayTradeMode.MOMENTUM_UP and rsi < 70:
            # Regular momentum - moderate position
            size_pct = self.position_pct
            stop = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)

            self.positions[symbol] = {
                'direction': 1,
                'entry_price': price,
                'entry_time': timestamp,
                'stop_loss': stop,
                'take_profit': take_profit,
            }

            return DayTradeSignal(
                symbol=symbol,
                timestamp=timestamp,
                action='buy',
                size=size_pct,
                mode=mode,
                confidence=0.6,
                entry_price=price,
                stop_loss=stop,
                take_profit=take_profit,
                reason=f"Momentum UP, RSI={rsi:.0f}",
            )

        # SHORT entries
        elif mode == DayTradeMode.STRONG_MOMENTUM_DOWN:
            size_pct = self.max_position_pct if volume_confirmed else self.position_pct
            stop = price * (1 + self.stop_loss_pct * 2)
            take_profit = price * (1 - self.take_profit_pct * 2)

            self.positions[symbol] = {
                'direction': -1,
                'entry_price': price,
                'entry_time': timestamp,
                'stop_loss': stop,
                'take_profit': take_profit,
            }

            return DayTradeSignal(
                symbol=symbol,
                timestamp=timestamp,
                action='short',
                size=size_pct,
                mode=mode,
                confidence=0.8 if volume_confirmed else 0.6,
                entry_price=price,
                stop_loss=stop,
                take_profit=take_profit,
                reason=f"Strong momentum DOWN, RSI={rsi:.0f}, Vol surge={signals['volume_surge']:.1f}x",
            )

        elif mode == DayTradeMode.MOMENTUM_DOWN and rsi > 30:
            size_pct = self.position_pct
            stop = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)

            self.positions[symbol] = {
                'direction': -1,
                'entry_price': price,
                'entry_time': timestamp,
                'stop_loss': stop,
                'take_profit': take_profit,
            }

            return DayTradeSignal(
                symbol=symbol,
                timestamp=timestamp,
                action='short',
                size=size_pct,
                mode=mode,
                confidence=0.6,
                entry_price=price,
                stop_loss=stop,
                take_profit=take_profit,
                reason=f"Momentum DOWN, RSI={rsi:.0f}",
            )

        # CHOPPY_BULLISH - ALWAYS BUY (MAXIMUM AGGRESSION)
        elif mode == DayTradeMode.CHOPPY_BULLISH:
            size_pct = self.position_pct * 0.75
            stop = price * (1 - self.stop_loss_pct * 1.5)
            take_profit = price * (1 + self.take_profit_pct * 0.75)

            self.positions[symbol] = {
                'direction': 1,
                'entry_price': price,
                'entry_time': timestamp,
                'stop_loss': stop,
                'take_profit': take_profit,
            }

            return DayTradeSignal(
                symbol=symbol,
                timestamp=timestamp,
                action='buy',
                size=size_pct,
                mode=mode,
                confidence=0.5,
                entry_price=price,
                stop_loss=stop,
                take_profit=take_profit,
                reason=f"Bullish bias LONG, RSI={rsi:.0f}",
            )

        # CHOPPY_BEARISH - ALWAYS SHORT (MAXIMUM AGGRESSION)
        elif mode == DayTradeMode.CHOPPY_BEARISH:
            size_pct = self.position_pct * 0.75
            stop = price * (1 + self.stop_loss_pct * 1.5)
            take_profit = price * (1 - self.take_profit_pct * 0.75)

            self.positions[symbol] = {
                'direction': -1,
                'entry_price': price,
                'entry_time': timestamp,
                'stop_loss': stop,
                'take_profit': take_profit,
            }

            return DayTradeSignal(
                symbol=symbol,
                timestamp=timestamp,
                action='short',
                size=size_pct,
                mode=mode,
                confidence=0.5,
                entry_price=price,
                stop_loss=stop,
                take_profit=take_profit,
                reason=f"Bearish bias SHORT, RSI={rsi:.0f}",
            )

        # NEUTRAL mode - trade based on micro momentum (SUPER AGGRESSIVE)
        elif mode == DayTradeMode.NEUTRAL:
            fast_ret = signals['fast_return']
            # Tiny positive momentum - go long
            if fast_ret > 0.0005 and rsi < 60:
                size_pct = self.position_pct * 0.5
                stop = price * (1 - self.stop_loss_pct)
                take_profit = price * (1 + self.take_profit_pct * 0.5)

                self.positions[symbol] = {
                    'direction': 1,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'stop_loss': stop,
                    'take_profit': take_profit,
                }

                return DayTradeSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    action='buy',
                    size=size_pct,
                    mode=mode,
                    confidence=0.35,
                    entry_price=price,
                    stop_loss=stop,
                    take_profit=take_profit,
                    reason=f"Micro momentum long, RSI={rsi:.0f}, ret={fast_ret*100:.2f}%",
                )
            # Tiny negative momentum - go short
            elif fast_ret < -0.0005 and rsi > 40:
                size_pct = self.position_pct * 0.5
                stop = price * (1 + self.stop_loss_pct)
                take_profit = price * (1 - self.take_profit_pct * 0.5)

                self.positions[symbol] = {
                    'direction': -1,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'stop_loss': stop,
                    'take_profit': take_profit,
                }

                return DayTradeSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    action='short',
                    size=size_pct,
                    mode=mode,
                    confidence=0.35,
                    entry_price=price,
                    stop_loss=stop,
                    take_profit=take_profit,
                    reason=f"Micro momentum short, RSI={rsi:.0f}, ret={fast_ret*100:.2f}%",
                )

        # No entry
        return DayTradeSignal(
            symbol=symbol,
            timestamp=timestamp,
            action='hold',
            size=0,
            mode=mode,
            confidence=0,
            entry_price=price,
            reason="No entry signal",
        )

    def _check_exit(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        signals: Dict,
        mode: DayTradeMode,
        position: dict,
    ) -> DayTradeSignal:
        """Check for exit signals."""

        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        action = 'hold'
        reason = ""

        if direction > 0:  # Long position
            # Stop loss
            if price <= stop_loss:
                action = 'sell'
                reason = f"Stop loss hit at ${price:.2f}"

            # Take profit
            elif price >= take_profit:
                action = 'sell'
                reason = f"Take profit hit at ${price:.2f}"

            # Momentum reversal
            elif mode in (DayTradeMode.STRONG_MOMENTUM_DOWN, DayTradeMode.MOMENTUM_DOWN):
                action = 'sell'
                reason = f"Momentum reversed to {mode.name}"

        else:  # Short position
            # Stop loss
            if price >= stop_loss:
                action = 'cover'
                reason = f"Stop loss hit at ${price:.2f}"

            # Take profit
            elif price <= take_profit:
                action = 'cover'
                reason = f"Take profit hit at ${price:.2f}"

            # Momentum reversal
            elif mode in (DayTradeMode.STRONG_MOMENTUM_UP, DayTradeMode.MOMENTUM_UP):
                action = 'cover'
                reason = f"Momentum reversed to {mode.name}"

        if action in ('sell', 'cover'):
            pnl = (price - entry_price) * direction
            pnl_pct = pnl / entry_price * 100
            del self.positions[symbol]
            reason += f" | P&L: {pnl_pct:+.2f}%"

        return DayTradeSignal(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            size=0,
            mode=mode,
            confidence=0.8 if action != 'hold' else 0,
            entry_price=price,
            reason=reason,
        )


def create_daytrader(
    aggressive: bool = True,
    max_position_dollars: float = 10000,
) -> ChameleonDayTrader:
    """Create a day trader instance.

    Args:
        aggressive: Use aggressive settings (tighter stops, lower thresholds)
        max_position_dollars: Hard cap on position size in dollars (default $10k)
    """
    if aggressive:
        return ChameleonDayTrader(
            position_pct=0.02,         # 2% per trade
            max_position_pct=0.05,     # 5% for strong signals
            max_position_dollars=max_position_dollars,
            stop_loss_pct=0.005,       # 0.5% stop
            take_profit_pct=0.01,      # 1% take profit
            min_volume_surge=1.1,      # Lower volume threshold
        )
    else:
        return ChameleonDayTrader(
            position_pct=0.01,         # 1% per trade
            max_position_pct=0.03,     # 3% for strong signals
            max_position_dollars=max_position_dollars,
            stop_loss_pct=0.007,
            take_profit_pct=0.015,
            min_volume_surge=1.3,
        )
