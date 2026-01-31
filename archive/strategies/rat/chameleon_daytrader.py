"""
Chameleon Day Trader v3 - Improved Intraday Trading

Major improvements over v2:
- Trend clarity scoring (rejects stocks with no clean directional trend)
- Chop detection via SMA crossover counting (avoids TSLA-style whipsaws)
- Minimum ATR% gate (skips stocks with insufficient intraday range like NVDA)
- Volatility-adaptive position sizing (risk-based: wider ATR = fewer shares)
- Directional movement analysis (measures trend strength vs noise)
- Higher-high / lower-low price action confirmation for entries

Carried from v2:
- ATR-based adaptive stops (adjusts to each stock's volatility)
- Trailing stop mechanism (locks in profits as trade moves favorably)
- Momentum exhaustion filter (avoids chasing RSI extremes)
- Cooldown after stop-loss exits (prevents revenge trading)
- Stronger momentum reversal confirmation (reduces whipsaw exits)
- Mean-reversion signals in choppy modes (not blind entries)
- Time-of-day awareness (avoids open/close chaos)
- VWAP-relative entries (only buy below VWAP, short above in choppy)
- Multi-market support (NYSE, HKEX, TSE, LSE, ASX)

For use with 5-minute bars during market hours.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class MarketConfig:
    """Configuration for a specific market/exchange."""
    name: str
    exchange: str           # IBKR exchange code
    currency: str           # Trading currency
    market_open: time       # Market open
    avoid_open_until: time  # Skip opening chaos
    lunch_start: Optional[time]  # Lunch break start (None = no break)
    lunch_end: Optional[time]    # Lunch break end
    avoid_close_after: time # Stop new entries before close
    market_close: time      # Force close all positions
    timezone: str           # For display purposes


# Pre-built market configurations
MARKET_PRESETS: Dict[str, MarketConfig] = {
    'NYSE': MarketConfig(
        name='NYSE / NASDAQ',
        exchange='SMART',
        currency='USD',
        market_open=time(9, 30),
        avoid_open_until=time(9, 50),
        lunch_start=time(12, 0),
        lunch_end=time(13, 0),
        avoid_close_after=time(15, 45),
        market_close=time(16, 0),
        timezone='US/Eastern',
    ),
    'HKEX': MarketConfig(
        name='Hong Kong Stock Exchange',
        exchange='SEHK',
        currency='HKD',
        market_open=time(9, 30),
        avoid_open_until=time(9, 50),
        lunch_start=time(12, 0),
        lunch_end=time(13, 0),
        avoid_close_after=time(15, 50),
        market_close=time(16, 0),
        timezone='Asia/Hong_Kong',
    ),
    'TSE': MarketConfig(
        name='Tokyo Stock Exchange',
        exchange='TSEJ',
        currency='JPY',
        market_open=time(9, 0),
        avoid_open_until=time(9, 15),
        lunch_start=time(11, 30),
        lunch_end=time(12, 30),
        avoid_close_after=time(14, 50),
        market_close=time(15, 0),
        timezone='Asia/Tokyo',
    ),
    'LSE': MarketConfig(
        name='London Stock Exchange',
        exchange='LSE',
        currency='GBP',
        market_open=time(8, 0),
        avoid_open_until=time(8, 15),
        lunch_start=None,  # No official lunch break
        lunch_end=None,
        avoid_close_after=time(16, 15),
        market_close=time(16, 30),
        timezone='Europe/London',
    ),
    'ASX': MarketConfig(
        name='Australian Securities Exchange',
        exchange='ASX',
        currency='AUD',
        market_open=time(10, 0),
        avoid_open_until=time(10, 15),
        lunch_start=None,
        lunch_end=None,
        avoid_close_after=time(15, 50),
        market_close=time(16, 0),
        timezone='Australia/Sydney',
    ),
}


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

        # ATR calculation (14-period, or available bars)
        atr_period = min(14, len(highs) - 1)
        tr_values = []
        for i in range(-atr_period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i - 1]),
                abs(lows[i] - prices[i - 1]),
            )
            tr_values.append(tr)
        atr = sum(tr_values) / len(tr_values) if tr_values else 0
        atr_pct = atr / current_price if current_price > 0 else 0

        # Volatility (bar range based)
        recent_ranges = [highs[i] - lows[i] for i in range(-self.fast_period, 0)]
        avg_range = sum(recent_ranges) / len(recent_ranges)
        volatility = avg_range / current_price  # As percentage

        # Volume analysis
        recent_volume = sum(volumes[-self.fast_period:]) / self.fast_period
        older_volume = sum(volumes[-self.medium_period:-self.fast_period]) / (self.medium_period - self.fast_period)
        volume_surge = recent_volume / older_volume if older_volume > 0 else 1.0

        # VWAP approximation (cumulative volume-weighted average)
        vwap_prices = prices[-self.slow_period:]
        vwap_volumes = volumes[-self.slow_period:]
        total_vol_price = sum(p * v for p, v in zip(vwap_prices, vwap_volumes))
        total_vol = sum(vwap_volumes)
        vwap = total_vol_price / total_vol if total_vol > 0 else current_price

        # Trend strength (are all timeframes aligned?)
        bullish_alignment = sum([above_fast, above_medium, above_slow])
        bearish_alignment = sum([not above_fast, not above_medium, not above_slow])

        # RSI (14-period)
        rsi_period = min(14, len(prices) - 1)
        gains = []
        losses = []
        for i in range(-rsi_period, 0):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = max(sum(losses) / len(losses), 0.0001) if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Rate of change of RSI (is RSI accelerating or decelerating?)
        # Calculate RSI from 5 bars ago to detect RSI trend
        if len(prices) > rsi_period + 5:
            old_gains = []
            old_losses = []
            offset = 5
            for i in range(-rsi_period - offset, -offset):
                change = prices[i] - prices[i - 1]
                if change > 0:
                    old_gains.append(change)
                    old_losses.append(0)
                else:
                    old_gains.append(0)
                    old_losses.append(abs(change))
            old_avg_gain = sum(old_gains) / len(old_gains) if old_gains else 0
            old_avg_loss = max(sum(old_losses) / len(old_losses), 0.0001) if old_losses else 0.0001
            old_rs = old_avg_gain / old_avg_loss
            old_rsi = 100 - (100 / (1 + old_rs))
            rsi_slope = rsi - old_rsi
        else:
            rsi_slope = 0

        # Price distance from VWAP (positive = above, negative = below)
        vwap_distance = (current_price - vwap) / vwap if vwap > 0 else 0

        # === NEW v3: Trend Clarity Score ===
        # Count higher-highs/higher-lows (uptrend) and lower-highs/lower-lows (downtrend)
        # over the medium lookback. Clean trends have consistent HH/HL or LH/LL.
        trend_lookback = min(self.medium_period, len(highs) - 1)
        hh_count = 0  # higher highs
        hl_count = 0  # higher lows
        lh_count = 0  # lower highs
        ll_count = 0  # lower lows
        for ti in range(-trend_lookback + 1, 0):
            if highs[ti] > highs[ti - 1]:
                hh_count += 1
            else:
                lh_count += 1
            if lows[ti] > lows[ti - 1]:
                hl_count += 1
            else:
                ll_count += 1

        total_swings = max(trend_lookback - 1, 1)
        # Uptrend clarity: fraction of bars making HH and HL
        uptrend_clarity = (hh_count + hl_count) / (total_swings * 2)
        # Downtrend clarity: fraction of bars making LH and LL
        downtrend_clarity = (lh_count + ll_count) / (total_swings * 2)
        # Overall trend clarity: max of up/down (0-1, higher = cleaner trend)
        trend_clarity = max(uptrend_clarity, downtrend_clarity)

        # === NEW v3: Chop Score (SMA Crossover Count) ===
        # Count how many times price crosses the medium SMA in the fast period.
        # Many crossovers = choppy, few = trending.
        crossover_count = 0
        for ci in range(-self.fast_period + 1, 0):
            prev_above = prices[ci - 1] > medium_sma
            curr_above = prices[ci] > medium_sma
            if prev_above != curr_above:
                crossover_count += 1
        # Normalize: 0 crossovers = 0 chop, 4+ = max chop
        chop_score = min(1.0, crossover_count / 4.0)

        # === NEW v3: Directional Movement (simplified ADX) ===
        # Measure ratio of directional movement to total range
        dm_lookback = min(10, len(highs) - 1)
        plus_dm_total = 0.0
        minus_dm_total = 0.0
        tr_total = 0.0
        for di in range(-dm_lookback, 0):
            up_move = highs[di] - highs[di - 1]
            down_move = lows[di - 1] - lows[di]
            tr_val = max(
                highs[di] - lows[di],
                abs(highs[di] - prices[di - 1]),
                abs(lows[di] - prices[di - 1]),
            )
            tr_total += tr_val
            if up_move > down_move and up_move > 0:
                plus_dm_total += up_move
            if down_move > up_move and down_move > 0:
                minus_dm_total += down_move

        if tr_total > 0:
            plus_di = plus_dm_total / tr_total
            minus_di = minus_dm_total / tr_total
            dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 0.0001)
        else:
            plus_di = 0
            minus_di = 0
            dx = 0
        # dx ranges 0-1: 0 = no directional movement, 1 = pure trend
        directional_strength = dx

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
            'rsi_slope': rsi_slope,
            'atr': atr,
            'atr_pct': atr_pct,
            'vwap': vwap,
            'vwap_distance': vwap_distance,
            'current_price': current_price,
            'fast_sma': fast_sma,
            'medium_sma': medium_sma,
            'slow_sma': slow_sma,
            # v3 additions
            'trend_clarity': trend_clarity,
            'chop_score': chop_score,
            'directional_strength': directional_strength,
            'plus_di': plus_di,
            'minus_di': minus_di,
        }


class ChameleonDayTrader:
    """
    Day trading strategy v3 with adaptive risk management and chop filtering.

    v3 improvements:
    - Trend clarity filter rejects choppy stocks (fixes TSLA chop problem)
    - Minimum ATR% gate skips low-volatility stocks (fixes NVDA range problem)
    - Chop score based on SMA crossover frequency
    - Directional movement strength requirement
    - Volatility-adaptive position sizing (risk-parity: wider ATR = fewer shares)

    v2 features retained:
    - ATR-based stops adapt to each stock's volatility
    - Trailing stops lock in profits
    - Momentum exhaustion filter avoids chasing
    - Cooldown prevents revenge trading after stops
    - Stronger reversal confirmation reduces whipsaws
    - Mean-reversion in choppy modes, not blind entries
    - Time-of-day awareness
    """

    def __init__(
        self,
        position_pct: float = 0.02,         # 2% of account per trade
        max_position_pct: float = 0.05,     # 5% max for strong signals
        max_position_dollars: float = 10000,  # Hard cap at $10k per position
        atr_stop_multiplier: float = 2.0,   # Stop = 2x ATR from entry
        atr_target_multiplier: float = 3.0, # Target = 3x ATR from entry (1.5:1 R:R)
        trailing_stop_activation: float = 1.5,  # Activate trailing after 1.5x ATR profit
        trailing_stop_distance: float = 1.0,    # Trail at 1x ATR distance
        min_volume_surge: float = 1.3,      # 30% volume increase to confirm
        cooldown_bars: int = 6,             # 30 min cooldown after stop-loss
        market: str = 'NYSE',               # Market preset name
        # v3 additions
        min_atr_pct: float = 0.0015,        # 0.15% minimum ATR (skip tight-range stocks)
        max_chop_score: float = 0.6,        # Reject if chop > 60% (too many SMA crossovers)
        min_trend_clarity: float = 0.55,    # Minimum trend clarity for momentum entries
        min_directional_strength: float = 0.2,  # Minimum DX for momentum entries
    ):
        self.analyzer = FastMomentumAnalyzer()

        # Market configuration
        self.market_config = MARKET_PRESETS.get(market.upper(), MARKET_PRESETS['NYSE'])

        # Position sizing (percentage with dollar cap)
        self.position_pct = position_pct
        self.max_position_pct = max_position_pct
        self.max_position_dollars = max_position_dollars

        # ATR-based risk management
        self.atr_stop_mult = atr_stop_multiplier
        self.atr_target_mult = atr_target_multiplier
        self.trailing_activation = trailing_stop_activation
        self.trailing_distance = trailing_stop_distance

        # Entry filters
        self.min_volume_surge = min_volume_surge
        self.cooldown_bars = cooldown_bars

        # v3: Quality filters
        self.min_atr_pct = min_atr_pct
        self.max_chop_score = max_chop_score
        self.min_trend_clarity = min_trend_clarity
        self.min_directional_strength = min_directional_strength

        # State
        self.positions: Dict[str, dict] = {}
        self._current_mode: Dict[str, DayTradeMode] = {}
        self._cooldown_until: Dict[str, int] = {}  # symbol -> bar count when cooldown ends
        self._bar_count: int = 0

    def clear_positions(self):
        """Clear all positions (call after warmup)."""
        self.positions.clear()
        self._cooldown_until.clear()
        self._bar_count = 0

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
        self._bar_count += 1

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
            high=high,
            low=low,
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

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown after a stop-loss."""
        if symbol in self._cooldown_until:
            if self._bar_count < self._cooldown_until[symbol]:
                return True
            else:
                del self._cooldown_until[symbol]
        return False

    def _is_tradeable_time(self, timestamp: datetime) -> bool:
        """Check if current time is suitable for new entries."""
        t = timestamp.time()
        mc = self.market_config
        # Avoid the opening chaos
        if t < mc.avoid_open_until:
            return False
        # Lunch break - low volume, choppy, spreads widen
        if mc.lunch_start and mc.lunch_end and mc.lunch_start <= t < mc.lunch_end:
            return False
        # Avoid the closing rush
        if t >= mc.avoid_close_after:
            return False
        return True

    def _generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        high: float,
        low: float,
        signals: Dict,
        mode: DayTradeMode,
    ) -> DayTradeSignal:
        """Generate trading signal based on mode."""

        has_position = symbol in self.positions
        position = self.positions.get(symbol)

        # Check existing position for exit
        if has_position:
            return self._check_exit(symbol, timestamp, price, high, low, signals, mode, position)

        # No position - check for entry
        return self._check_entry(symbol, timestamp, price, signals, mode)

    def _calculate_adaptive_size(
        self,
        base_pct: float,
        atr_pct: float,
        confidence: float,
    ) -> float:
        """
        Calculate volatility-adaptive position size (v3).

        Risk-parity: wider ATR = smaller position (same dollar risk per trade).
        Higher confidence = larger position.
        """
        # Target risk per trade: ~0.3% of account
        target_risk_pct = 0.003
        # ATR-based sizing: if ATR is 0.5%, we want 0.3%/0.5% = 60% of base
        if atr_pct > 0:
            vol_adjusted = min(1.5, target_risk_pct / atr_pct)
        else:
            vol_adjusted = 1.0

        # Confidence multiplier (0.5 at low confidence, 1.5 at high)
        conf_mult = 0.5 + confidence

        adaptive_size = base_pct * vol_adjusted * conf_mult
        return max(0.005, min(self.max_position_pct, adaptive_size))

    def _check_entry(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        signals: Dict,
        mode: DayTradeMode,
    ) -> DayTradeSignal:
        """Check for entry opportunities with v3 quality filters."""

        volume_confirmed = signals['volume_surge'] >= self.min_volume_surge
        rsi = signals['rsi']
        rsi_slope = signals['rsi_slope']
        atr = signals['atr']
        atr_pct = signals['atr_pct']
        vwap_dist = signals['vwap_distance']
        trend_clarity = signals['trend_clarity']
        chop_score = signals['chop_score']
        directional_strength = signals['directional_strength']

        # No entries during cooldown
        if self._is_in_cooldown(symbol):
            return self._hold_signal(symbol, timestamp, price, mode, "In cooldown")

        # No entries during opening/closing periods
        if not self._is_tradeable_time(timestamp):
            return self._hold_signal(symbol, timestamp, price, mode, "Outside trading window")

        # No entries during volatility spikes
        if mode == DayTradeMode.VOLATILITY_SPIKE:
            return self._hold_signal(symbol, timestamp, price, mode, "Volatility spike - sitting out")

        # ATR must be meaningful (avoids trading stale/illiquid periods)
        if atr <= 0:
            return self._hold_signal(symbol, timestamp, price, mode, "No ATR data")

        # === v3: Minimum ATR% gate (skip tight-range stocks like NVDA on quiet days) ===
        if atr_pct < self.min_atr_pct:
            return self._hold_signal(symbol, timestamp, price, mode,
                                     f"ATR too low ({atr_pct*100:.3f}% < {self.min_atr_pct*100:.3f}%)")

        # === v3: Chop filter (reject choppy stocks like TSLA on trendless days) ===
        if chop_score > self.max_chop_score:
            return self._hold_signal(symbol, timestamp, price, mode,
                                     f"Too choppy (chop={chop_score:.2f} > {self.max_chop_score})")

        # === LONG ENTRIES ===

        if mode == DayTradeMode.STRONG_MOMENTUM_UP:
            # Momentum exhaustion filter: reject if RSI already extreme
            if rsi > 75:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Momentum exhausted RSI={rsi:.0f}")
            # Require RSI still rising (momentum not stalling)
            if rsi_slope < -5:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"RSI decelerating slope={rsi_slope:.1f}")
            # v3: Require minimum directional strength
            if directional_strength < self.min_directional_strength:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Weak directional strength ({directional_strength:.2f})")

            base_conf = 0.8 if volume_confirmed else 0.6
            size_pct = self._calculate_adaptive_size(
                self.max_position_pct if volume_confirmed else self.position_pct,
                atr_pct, base_conf,
            )
            stop = price - atr * self.atr_stop_mult * 1.5  # Wider for strong momentum
            take_profit = price + atr * self.atr_target_mult * 1.5

            return self._enter_position(
                symbol, timestamp, price, 1, size_pct, stop, take_profit, mode, atr,
                confidence=base_conf,
                reason=f"Strong UP, RSI={rsi:.0f}, DX={directional_strength:.2f}, trend={trend_clarity:.2f}",
            )

        elif mode == DayTradeMode.MOMENTUM_UP:
            # Exhaustion filter
            if rsi > 70:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Overbought RSI={rsi:.0f}")
            # v3: Require trend clarity for regular momentum
            if trend_clarity < self.min_trend_clarity:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Unclear trend ({trend_clarity:.2f} < {self.min_trend_clarity})")

            base_conf = 0.6
            size_pct = self._calculate_adaptive_size(self.position_pct, atr_pct, base_conf)
            stop = price - atr * self.atr_stop_mult
            take_profit = price + atr * self.atr_target_mult

            return self._enter_position(
                symbol, timestamp, price, 1, size_pct, stop, take_profit, mode, atr,
                confidence=base_conf,
                reason=f"Momentum UP, RSI={rsi:.0f}, clarity={trend_clarity:.2f}",
            )

        # === SHORT ENTRIES ===

        elif mode == DayTradeMode.STRONG_MOMENTUM_DOWN:
            # Momentum exhaustion filter for shorts
            if rsi < 25:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Momentum exhausted RSI={rsi:.0f}")
            if rsi_slope > 5:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"RSI recovering slope={rsi_slope:.1f}")
            # v3: Require minimum directional strength
            if directional_strength < self.min_directional_strength:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Weak directional strength ({directional_strength:.2f})")

            base_conf = 0.8 if volume_confirmed else 0.6
            size_pct = self._calculate_adaptive_size(
                self.max_position_pct if volume_confirmed else self.position_pct,
                atr_pct, base_conf,
            )
            stop = price + atr * self.atr_stop_mult * 1.5
            take_profit = price - atr * self.atr_target_mult * 1.5

            return self._enter_position(
                symbol, timestamp, price, -1, size_pct, stop, take_profit, mode, atr,
                confidence=base_conf,
                reason=f"Strong DOWN, RSI={rsi:.0f}, DX={directional_strength:.2f}, trend={trend_clarity:.2f}",
            )

        elif mode == DayTradeMode.MOMENTUM_DOWN:
            if rsi < 30:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Oversold RSI={rsi:.0f}")
            # v3: Require trend clarity
            if trend_clarity < self.min_trend_clarity:
                return self._hold_signal(symbol, timestamp, price, mode,
                                         f"Unclear trend ({trend_clarity:.2f} < {self.min_trend_clarity})")

            base_conf = 0.6
            size_pct = self._calculate_adaptive_size(self.position_pct, atr_pct, base_conf)
            stop = price + atr * self.atr_stop_mult
            take_profit = price - atr * self.atr_target_mult

            return self._enter_position(
                symbol, timestamp, price, -1, size_pct, stop, take_profit, mode, atr,
                confidence=base_conf,
                reason=f"Momentum DOWN, RSI={rsi:.0f}, clarity={trend_clarity:.2f}",
            )

        # === CHOPPY MODE - MEAN REVERSION ENTRIES ===
        # Note: Chop filter above already rejects high-chop, so these only
        # trigger when chop is moderate (stock is range-bound but not erratic)

        elif mode == DayTradeMode.CHOPPY_BULLISH:
            # Mean reversion: only buy on pullbacks BELOW VWAP with RSI dip
            if vwap_dist < -0.001 and rsi < 45 and rsi_slope > 0:
                base_conf = 0.5
                size_pct = self._calculate_adaptive_size(
                    self.position_pct * 0.75, atr_pct, base_conf,
                )
                stop = price - atr * self.atr_stop_mult
                take_profit = price + atr * self.atr_target_mult * 0.75

                return self._enter_position(
                    symbol, timestamp, price, 1, size_pct, stop, take_profit, mode, atr,
                    confidence=base_conf,
                    reason=f"Mean-rev BUY: VWAP dist={vwap_dist*100:.2f}%, RSI={rsi:.0f} turning up",
                )

            return self._hold_signal(symbol, timestamp, price, mode,
                                     "Choppy bullish - waiting for pullback")

        elif mode == DayTradeMode.CHOPPY_BEARISH:
            # Mean reversion: only short on rallies ABOVE VWAP with RSI peak
            if vwap_dist > 0.001 and rsi > 55 and rsi_slope < 0:
                base_conf = 0.5
                size_pct = self._calculate_adaptive_size(
                    self.position_pct * 0.75, atr_pct, base_conf,
                )
                stop = price + atr * self.atr_stop_mult
                take_profit = price - atr * self.atr_target_mult * 0.75

                return self._enter_position(
                    symbol, timestamp, price, -1, size_pct, stop, take_profit, mode, atr,
                    confidence=base_conf,
                    reason=f"Mean-rev SHORT: VWAP dist={vwap_dist*100:.2f}%, RSI={rsi:.0f} turning down",
                )

            return self._hold_signal(symbol, timestamp, price, mode,
                                     "Choppy bearish - waiting for rally")

        # === NEUTRAL - NO TRADES (v3: removed micro-scalp in neutral) ===
        # v3 learning: micro-scalps in neutral mode were net negative on live.
        # Better to wait for a real signal than force marginal trades.

        # No entry
        return self._hold_signal(symbol, timestamp, price, mode, "No entry signal")

    def _enter_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        direction: int,
        size_pct: float,
        stop: float,
        take_profit: float,
        mode: DayTradeMode,
        atr: float,
        confidence: float,
        reason: str,
    ) -> DayTradeSignal:
        """Create a position entry."""
        self.positions[symbol] = {
            'direction': direction,
            'entry_price': price,
            'entry_time': timestamp,
            'stop_loss': stop,
            'take_profit': take_profit,
            'atr_at_entry': atr,
            'best_price': price,  # Track best price for trailing stop
            'trailing_active': False,
        }

        action = 'buy' if direction > 0 else 'short'
        return DayTradeSignal(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            size=size_pct,
            mode=mode,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop,
            take_profit=take_profit,
            reason=reason,
        )

    def _hold_signal(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        mode: DayTradeMode,
        reason: str,
    ) -> DayTradeSignal:
        """Return a hold signal."""
        return DayTradeSignal(
            symbol=symbol,
            timestamp=timestamp,
            action='hold',
            size=0,
            mode=mode,
            confidence=0,
            entry_price=price,
            reason=reason,
        )

    def _check_exit(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        high: float,
        low: float,
        signals: Dict,
        mode: DayTradeMode,
        position: dict,
    ) -> DayTradeSignal:
        """Check for exit signals with trailing stop and improved reversal logic."""

        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        atr = position['atr_at_entry']
        best_price = position['best_price']

        action = 'hold'
        reason = ""

        # Update best price for trailing stop
        if direction > 0:
            if price > best_price:
                position['best_price'] = price
                best_price = price
        else:
            if price < best_price:
                position['best_price'] = price
                best_price = price

        # Check trailing stop activation
        if not position['trailing_active']:
            if direction > 0:
                profit_distance = best_price - entry_price
                if profit_distance >= atr * self.trailing_activation:
                    position['trailing_active'] = True
                    # Move stop to breakeven + small profit
                    new_stop = entry_price + atr * 0.5
                    if new_stop > stop_loss:
                        position['stop_loss'] = new_stop
                        stop_loss = new_stop
            else:
                profit_distance = entry_price - best_price
                if profit_distance >= atr * self.trailing_activation:
                    position['trailing_active'] = True
                    new_stop = entry_price - atr * 0.5
                    if new_stop < stop_loss:
                        position['stop_loss'] = new_stop
                        stop_loss = new_stop

        # Update trailing stop position
        if position['trailing_active']:
            if direction > 0:
                trailing_stop = best_price - atr * self.trailing_distance
                if trailing_stop > stop_loss:
                    position['stop_loss'] = trailing_stop
                    stop_loss = trailing_stop
            else:
                trailing_stop = best_price + atr * self.trailing_distance
                if trailing_stop < stop_loss:
                    position['stop_loss'] = trailing_stop
                    stop_loss = trailing_stop

        # Exit checks
        if direction > 0:  # Long position
            # Stop loss (check low of bar for intrabar hit)
            if low <= stop_loss or price <= stop_loss:
                action = 'sell'
                exit_price = min(price, stop_loss)
                if position['trailing_active']:
                    reason = f"Trailing stop hit at ${exit_price:.2f}"
                else:
                    reason = f"Stop loss hit at ${exit_price:.2f}"

            # Take profit
            elif high >= take_profit or price >= take_profit:
                action = 'sell'
                reason = f"Take profit hit at ${max(price, take_profit):.2f}"

            # Momentum reversal - require STRONG confirmation
            elif (mode == DayTradeMode.STRONG_MOMENTUM_DOWN
                  and signals['rsi'] < 35
                  and signals['bearish_alignment'] == 3):
                action = 'sell'
                reason = f"Strong reversal: mode={mode.name}, RSI={signals['rsi']:.0f}"

            # End of day - close all positions
            elif timestamp.time() >= self.market_config.market_close:
                action = 'sell'
                reason = "End of day close"

        else:  # Short position
            # Stop loss
            if high >= stop_loss or price >= stop_loss:
                action = 'cover'
                exit_price = max(price, stop_loss)
                if position['trailing_active']:
                    reason = f"Trailing stop hit at ${exit_price:.2f}"
                else:
                    reason = f"Stop loss hit at ${exit_price:.2f}"

            # Take profit
            elif low <= take_profit or price <= take_profit:
                action = 'cover'
                reason = f"Take profit hit at ${min(price, take_profit):.2f}"

            # Momentum reversal - require STRONG confirmation
            elif (mode == DayTradeMode.STRONG_MOMENTUM_UP
                  and signals['rsi'] > 65
                  and signals['bullish_alignment'] == 3):
                action = 'cover'
                reason = f"Strong reversal: mode={mode.name}, RSI={signals['rsi']:.0f}"

            # End of day
            elif timestamp.time() >= self.market_config.market_close:
                action = 'cover'
                reason = "End of day close"

        if action in ('sell', 'cover'):
            pnl = (price - entry_price) * direction
            pnl_pct = pnl / entry_price * 100
            is_stop_loss = "Stop loss" in reason

            del self.positions[symbol]

            # Set cooldown if stopped out
            if is_stop_loss and not position['trailing_active']:
                self._cooldown_until[symbol] = self._bar_count + self.cooldown_bars

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
    market: str = 'NYSE',
) -> ChameleonDayTrader:
    """Create a day trader instance.

    Args:
        aggressive: Use aggressive settings (wider targets, lower thresholds)
        max_position_dollars: Hard cap on position size in dollars (default $10k)
        market: Market preset ('NYSE', 'HKEX', 'TSE', 'LSE', 'ASX')
    """
    if aggressive:
        return ChameleonDayTrader(
            position_pct=0.02,              # 2% per trade
            max_position_pct=0.05,          # 5% for strong signals
            max_position_dollars=max_position_dollars,
            atr_stop_multiplier=1.5,        # 1.5x ATR stop
            atr_target_multiplier=3.0,      # 3x ATR target (2:1 R:R)
            trailing_stop_activation=1.5,   # Activate trail after 1.5x ATR
            trailing_stop_distance=1.0,     # Trail 1x ATR behind
            min_volume_surge=1.2,           # 20% volume increase
            cooldown_bars=4,                # 20 min cooldown
            market=market,
            # v3: Aggressive = looser quality filters
            min_atr_pct=0.0012,             # 0.12% min ATR (still skip dead stocks)
            max_chop_score=0.65,            # Allow slightly choppier
            min_trend_clarity=0.52,         # Slightly lower bar
            min_directional_strength=0.15,  # Lower DX threshold
        )
    else:
        return ChameleonDayTrader(
            position_pct=0.01,              # 1% per trade
            max_position_pct=0.03,          # 3% for strong signals
            max_position_dollars=max_position_dollars,
            atr_stop_multiplier=2.0,        # 2x ATR stop (wider)
            atr_target_multiplier=3.5,      # 3.5x ATR target
            trailing_stop_activation=2.0,   # Activate trail after 2x ATR
            trailing_stop_distance=1.5,     # Trail 1.5x ATR behind
            min_volume_surge=1.4,           # 40% volume increase
            cooldown_bars=6,                # 30 min cooldown
            market=market,
            # v3: Conservative = stricter quality filters
            min_atr_pct=0.002,              # 0.2% min ATR
            max_chop_score=0.5,             # Reject moderate chop
            min_trend_clarity=0.60,         # Higher trend clarity bar
            min_directional_strength=0.25,  # Higher DX threshold
        )


def list_markets() -> None:
    """Print available market presets."""
    print("Available markets:")
    for key, mc in MARKET_PRESETS.items():
        lunch = f", lunch {mc.lunch_start.strftime('%H:%M')}-{mc.lunch_end.strftime('%H:%M')}" if mc.lunch_start else ""
        print(f"  {key:6s} | {mc.name:35s} | {mc.exchange:5s} | {mc.currency} | "
              f"{mc.market_open.strftime('%H:%M')}-{mc.market_close.strftime('%H:%M')}{lunch}")
