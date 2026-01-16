"""
Enhanced Signal Generation for RAT Framework

This module provides research-backed enhancements for generating trading signals
from daily OHLCV data when tick-level microstructure data isn't available.

Academic References:
- Lee & Ready (1991): Trade classification algorithm
- Jegadeesh & Titman (1993): Momentum strategies
- Fama & French (1992): Factor models
- Kelly (1956): Optimal position sizing
- Bollinger (1992): Volatility bands for mean reversion
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


# =============================================================================
# RESEARCH-BACKED PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class ResearchParameters:
    """
    Parameters derived from academic research and industry practice.

    All parameters have citations or justification.
    """

    # Momentum Parameters (Jegadeesh & Titman 1993)
    momentum_short_window: int = 5      # 1-week momentum
    momentum_long_window: int = 20      # 1-month momentum
    momentum_signal_threshold: float = 0.02  # 2% threshold for signal

    # Mean Reversion Parameters (Bollinger 1992)
    bollinger_period: int = 20          # Standard Bollinger period
    bollinger_std: float = 2.0          # Standard 2 std deviation
    mean_reversion_entry: float = 2.0   # Enter at 2 std
    mean_reversion_exit: float = 0.5    # Exit at 0.5 std

    # RSI Parameters (Wilder 1978)
    rsi_period: int = 14                # Standard RSI period
    rsi_overbought: float = 70.0        # Overbought threshold
    rsi_oversold: float = 30.0          # Oversold threshold
    rsi_extreme_overbought: float = 80.0  # Strong overbought
    rsi_extreme_oversold: float = 20.0    # Strong oversold

    # Volume Parameters
    volume_ma_period: int = 20          # Volume moving average
    volume_spike_threshold: float = 2.0  # 2x average = spike

    # ATR Parameters (Wilder 1978)
    atr_period: int = 14                # Standard ATR period

    # Fair Value Estimation (Graham/Dodd)
    fair_value_sma_period: int = 50     # 50-day SMA as fair value
    deviation_threshold: float = 0.05   # 5% deviation significant

    # Position Sizing (Kelly 1956, with fractional adjustment)
    kelly_fraction: float = 0.25        # Quarter Kelly (conservative)
    max_position_pct: float = 0.05      # Max 5% per position
    min_position_pct: float = 0.01      # Min 1% per position

    # Risk Parameters (Industry standard)
    max_daily_loss: float = 0.02        # 2% max daily loss
    max_drawdown: float = 0.15          # 15% max drawdown
    correlation_limit: float = 0.6      # Max 0.6 correlation

    # Regime Detection
    trend_threshold: float = 0.03       # 3% for trend classification
    volatility_lookback: int = 20       # 20-day volatility


PARAMS = ResearchParameters()


# =============================================================================
# ORDER FLOW ESTIMATION (Lee-Ready Algorithm Adaptation)
# =============================================================================

class OrderFlowEstimator:
    """
    Estimate order flow (buy/sell pressure) from OHLCV data.

    Based on Lee & Ready (1991) tick test, adapted for daily bars:
    - Close > Open: Buying pressure
    - Close < Open: Selling pressure
    - (Close - Low) / (High - Low): Buy pressure ratio
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._history: Dict[str, Deque[Dict]] = {}

    def update(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: datetime,
    ) -> Tuple[float, float]:
        """
        Update with new bar and return estimated (buy_volume, sell_volume).

        Uses multiple methods combined:
        1. Close location value (CLV): (close - low) / (high - low)
        2. Price direction: close > open = buy, close < open = sell
        3. True range position: Where close is in the day's range
        """
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.window_size)

        # Prevent division by zero
        range_size = high - low
        if range_size == 0:
            range_size = 0.0001

        # Method 1: Close Location Value (CLV)
        # Ranges from 0 (at low) to 1 (at high)
        clv = (close - low) / range_size

        # Method 2: Price direction
        direction = 1.0 if close >= open_price else -1.0

        # Method 3: Body position in range
        body_mid = (open_price + close) / 2
        body_position = (body_mid - low) / range_size

        # Combine methods (weighted average)
        buy_pressure = (0.5 * clv + 0.3 * ((direction + 1) / 2) + 0.2 * body_position)
        buy_pressure = max(0, min(1, buy_pressure))

        # Allocate volume
        buy_volume = volume * buy_pressure
        sell_volume = volume * (1 - buy_pressure)

        # Store for analysis
        self._history[symbol].append({
            'timestamp': timestamp,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'clv': clv,
            'direction': direction,
            'volume': volume,
        })

        return buy_volume, sell_volume

    def get_flow_imbalance(self, symbol: str, periods: int = 5) -> float:
        """
        Compute flow imbalance over recent periods.

        Returns: Value in [-1, 1], positive = buying pressure
        """
        if symbol not in self._history or len(self._history[symbol]) < periods:
            return 0.0

        recent = list(self._history[symbol])[-periods:]
        total_buy = sum(d['buy_volume'] for d in recent)
        total_sell = sum(d['sell_volume'] for d in recent)

        total = total_buy + total_sell
        if total == 0:
            return 0.0

        return (total_buy - total_sell) / total

    def get_volume_momentum(self, symbol: str) -> float:
        """
        Compute volume momentum (is volume increasing or decreasing).

        Returns: Ratio of recent to older volume, normalized
        """
        if symbol not in self._history or len(self._history[symbol]) < 10:
            return 0.0

        data = list(self._history[symbol])
        recent = sum(d['volume'] for d in data[-5:])
        older = sum(d['volume'] for d in data[-10:-5])

        if older == 0:
            return 0.0

        ratio = recent / older
        # Normalize to [-1, 1], 1.0 ratio = 0, 2.0 ratio = 1, 0.5 ratio = -1
        return max(-1, min(1, (ratio - 1)))


# =============================================================================
# FAIR VALUE ESTIMATION (for Reflexivity Module)
# =============================================================================

class FairValueEstimator:
    """
    Estimate fair value using multiple methods.

    Inspired by Graham & Dodd's Security Analysis.
    Uses SMA as a proxy for "intrinsic value" when fundamentals unavailable.
    """

    def __init__(
        self,
        short_period: int = 20,
        long_period: int = 50,
        very_long_period: int = 200,
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.very_long_period = very_long_period
        self._prices: Dict[str, Deque[float]] = {}

    def update(self, symbol: str, price: float) -> None:
        """Add a new price."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.very_long_period)
        self._prices[symbol].append(price)

    def get_fair_value(self, symbol: str) -> Optional[float]:
        """
        Estimate fair value as weighted average of SMAs.

        Short-term SMA: More responsive but noisy
        Long-term SMA: More stable, represents "true" fair value
        """
        if symbol not in self._prices:
            return None

        prices = list(self._prices[symbol])
        n = len(prices)

        if n < self.short_period:
            return None

        # Calculate SMAs
        sma_short = sum(prices[-self.short_period:]) / self.short_period

        if n >= self.long_period:
            sma_long = sum(prices[-self.long_period:]) / self.long_period
        else:
            sma_long = sma_short

        if n >= self.very_long_period:
            sma_very_long = sum(prices[-self.very_long_period:]) / self.very_long_period
        else:
            sma_very_long = sma_long

        # Weighted average (more weight to longer-term = more stable fair value)
        fair_value = 0.2 * sma_short + 0.3 * sma_long + 0.5 * sma_very_long

        return fair_value

    def get_deviation(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Get deviation from fair value as percentage.

        Positive = price above fair value (overvalued)
        Negative = price below fair value (undervalued)
        """
        fair_value = self.get_fair_value(symbol)
        if fair_value is None or fair_value == 0:
            return None

        return (current_price - fair_value) / fair_value


# =============================================================================
# TECHNICAL INDICATORS (Research-backed implementations)
# =============================================================================

class TechnicalIndicators:
    """
    Research-backed technical indicators.

    All implementations use standard parameters from academic literature.
    """

    def __init__(self, max_history: int = 252):
        self._prices: Dict[str, Deque[float]] = {}
        self._highs: Dict[str, Deque[float]] = {}
        self._lows: Dict[str, Deque[float]] = {}
        self._volumes: Dict[str, Deque[float]] = {}
        self.max_history = max_history

    def update(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Update with new bar data."""
        for store, val in [
            (self._prices, close),
            (self._highs, high),
            (self._lows, low),
            (self._volumes, volume),
        ]:
            if symbol not in store:
                store[symbol] = deque(maxlen=self.max_history)
            store[symbol].append(val)

    def rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Relative Strength Index (Wilder 1978).

        RSI = 100 - 100 / (1 + RS)
        RS = Average Gain / Average Loss
        """
        if symbol not in self._prices or len(self._prices[symbol]) < period + 1:
            return None

        prices = list(self._prices[symbol])
        changes = [prices[i] - prices[i-1] for i in range(-period, 0)]

        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]

        avg_gain = sum(gains) / period if gains else 0.0001
        avg_loss = sum(losses) / period if losses else 0.0001

        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)

        return rsi

    def sma(self, symbol: str, period: int) -> Optional[float]:
        """Simple Moving Average."""
        if symbol not in self._prices or len(self._prices[symbol]) < period:
            return None

        prices = list(self._prices[symbol])[-period:]
        return sum(prices) / period

    def ema(self, symbol: str, period: int) -> Optional[float]:
        """Exponential Moving Average."""
        if symbol not in self._prices or len(self._prices[symbol]) < period:
            return None

        prices = list(self._prices[symbol])
        multiplier = 2 / (period + 1)

        # Initialize with SMA
        ema = sum(prices[:period]) / period

        # Apply EMA formula
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Bollinger Bands (Bollinger 1992).

        Returns: (lower_band, middle_band, upper_band)
        """
        if symbol not in self._prices or len(self._prices[symbol]) < period:
            return None

        prices = list(self._prices[symbol])[-period:]
        middle = sum(prices) / period

        variance = sum((p - middle) ** 2 for p in prices) / period
        std = math.sqrt(variance)

        lower = middle - std_dev * std
        upper = middle + std_dev * std

        return (lower, middle, upper)

    def atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Average True Range (Wilder 1978).

        Measures volatility.
        """
        if symbol not in self._prices or len(self._prices[symbol]) < period + 1:
            return None

        prices = list(self._prices[symbol])
        highs = list(self._highs[symbol])
        lows = list(self._lows[symbol])

        n = min(len(prices), len(highs), len(lows))
        if n < period + 1:
            return None

        true_ranges = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i-1]),
                abs(lows[i] - prices[i-1])
            )
            true_ranges.append(tr)

        return sum(true_ranges) / period

    def momentum(
        self,
        symbol: str,
        short_period: int = 5,
        long_period: int = 20,
    ) -> Optional[float]:
        """
        Momentum as ratio of short-term to long-term returns.

        Based on Jegadeesh & Titman (1993).
        """
        if symbol not in self._prices or len(self._prices[symbol]) < long_period:
            return None

        prices = list(self._prices[symbol])

        current = prices[-1]
        short_ago = prices[-short_period]
        long_ago = prices[-long_period]

        if short_ago == 0 or long_ago == 0:
            return None

        short_return = (current - short_ago) / short_ago
        long_return = (current - long_ago) / long_ago

        return long_return  # Use longer-term momentum for signal

    def volume_ratio(self, symbol: str, period: int = 20) -> Optional[float]:
        """Volume ratio (current vs average)."""
        if symbol not in self._volumes or len(self._volumes[symbol]) < period:
            return None

        volumes = list(self._volumes[symbol])
        avg_volume = sum(volumes[-period:-1]) / (period - 1) if period > 1 else volumes[-1]

        if avg_volume == 0:
            return None

        return volumes[-1] / avg_volume


# =============================================================================
# REGIME DETECTOR (Enhanced for daily data)
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    UNKNOWN = auto()
    STRONG_UPTREND = auto()
    UPTREND = auto()
    CONSOLIDATION = auto()
    DOWNTREND = auto()
    STRONG_DOWNTREND = auto()
    HIGH_VOLATILITY = auto()


class RegimeDetector:
    """
    Detect market regime from price action.

    Uses trend, volatility, and mean reversion characteristics.
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self._regime_history: Dict[str, Deque[MarketRegime]] = {}

    def update(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Update indicators."""
        self.indicators.update(symbol, high, low, close, volume)

    def detect(self, symbol: str) -> Tuple[MarketRegime, float]:
        """
        Detect current regime.

        Returns: (regime, confidence)
        """
        # Get indicators
        sma_20 = self.indicators.sma(symbol, 20)
        sma_50 = self.indicators.sma(symbol, 50)
        atr = self.indicators.atr(symbol, 14)
        momentum = self.indicators.momentum(symbol, 5, 20)
        rsi = self.indicators.rsi(symbol, 14)

        if symbol not in self.indicators._prices:
            return (MarketRegime.UNKNOWN, 0.0)

        current_price = self.indicators._prices[symbol][-1]

        if sma_20 is None or sma_50 is None:
            return (MarketRegime.UNKNOWN, 0.0)

        # Calculate trend strength
        price_vs_sma20 = (current_price - sma_20) / sma_20 if sma_20 else 0
        price_vs_sma50 = (current_price - sma_50) / sma_50 if sma_50 else 0
        sma_alignment = (sma_20 - sma_50) / sma_50 if sma_50 else 0

        # Calculate volatility
        if atr and sma_20:
            volatility_ratio = atr / sma_20
            is_high_volatility = volatility_ratio > 0.03  # 3% daily ATR
        else:
            is_high_volatility = False

        # Regime classification
        # Thresholds relaxed for daily data to capture more regime diversity
        # Research (Ang & Bekaert 2002) shows regime persistence even with smaller deviations
        if is_high_volatility and abs(price_vs_sma20) < 0.015:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.7
        elif price_vs_sma20 > 0.03 and sma_alignment > 0.01:
            regime = MarketRegime.STRONG_UPTREND
            confidence = 0.8
        elif price_vs_sma20 > 0.01 or (price_vs_sma20 > 0 and sma_alignment > 0.005):
            regime = MarketRegime.UPTREND
            confidence = 0.7
        elif price_vs_sma20 < -0.03 and sma_alignment < -0.01:
            regime = MarketRegime.STRONG_DOWNTREND
            confidence = 0.8
        elif price_vs_sma20 < -0.01 or (price_vs_sma20 < 0 and sma_alignment < -0.005):
            regime = MarketRegime.DOWNTREND
            confidence = 0.7
        else:
            regime = MarketRegime.CONSOLIDATION
            confidence = 0.6

        # Track regime persistence
        if symbol not in self._regime_history:
            self._regime_history[symbol] = deque(maxlen=10)
        self._regime_history[symbol].append(regime)

        # Adjust confidence based on persistence
        recent = list(self._regime_history[symbol])[-5:]
        persistence = sum(1 for r in recent if r == regime) / len(recent)
        confidence *= (0.5 + 0.5 * persistence)

        return (regime, min(0.95, confidence))


# =============================================================================
# ENHANCED SIGNAL GENERATOR
# =============================================================================

@dataclass
class EnhancedSignal:
    """Enhanced signal with additional metadata."""
    signal: Signal
    regime: MarketRegime
    indicators: Dict[str, float]
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]


class EnhancedSignalGenerator:
    """
    Generate trading signals using research-backed methods.

    Combines multiple signal sources with regime-aware weighting.
    """

    def __init__(self):
        self.flow_estimator = OrderFlowEstimator()
        self.fair_value = FairValueEstimator()
        self.indicators = TechnicalIndicators()
        self.regime_detector = RegimeDetector()
        self._last_signals: Dict[str, EnhancedSignal] = {}

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> Optional[EnhancedSignal]:
        """
        Update with new bar and generate signal if appropriate.
        """
        # Update all components
        self.flow_estimator.update(symbol, open_price, high, low, close, volume, timestamp)
        self.fair_value.update(symbol, close)
        self.indicators.update(symbol, high, low, close, volume)
        self.regime_detector.update(symbol, high, low, close, volume)

        # Detect regime
        regime, regime_confidence = self.regime_detector.detect(symbol)

        # Generate signals based on regime
        signal = self._generate_regime_signal(symbol, close, regime, regime_confidence)

        if signal:
            self._last_signals[symbol] = signal

        return signal

    def _generate_regime_signal(
        self,
        symbol: str,
        current_price: float,
        regime: MarketRegime,
        regime_confidence: float,
    ) -> Optional[EnhancedSignal]:
        """Generate signal appropriate for current regime."""

        # Gather indicators
        rsi = self.indicators.rsi(symbol, PARAMS.rsi_period)
        momentum = self.indicators.momentum(symbol, 5, 20)
        bollinger = self.indicators.bollinger_bands(symbol, 20, 2.0)
        atr = self.indicators.atr(symbol, 14)
        flow_imbalance = self.flow_estimator.get_flow_imbalance(symbol, 5)
        fair_value_deviation = self.fair_value.get_deviation(symbol, current_price)

        if rsi is None or momentum is None:
            return None

        indicators = {
            'rsi': rsi,
            'momentum': momentum,
            'flow_imbalance': flow_imbalance,
            'fair_value_deviation': fair_value_deviation or 0,
        }

        # Regime-specific signal logic
        if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND):
            # Trend following: Buy on pullbacks
            signal = self._trend_following_signal(
                symbol, current_price, rsi, momentum, flow_imbalance,
                direction=1, regime_confidence=regime_confidence
            )

        elif regime in (MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND):
            # Trend following: Sell on rallies
            signal = self._trend_following_signal(
                symbol, current_price, rsi, momentum, flow_imbalance,
                direction=-1, regime_confidence=regime_confidence
            )

        elif regime == MarketRegime.CONSOLIDATION:
            # Mean reversion
            signal = self._mean_reversion_signal(
                symbol, current_price, rsi, bollinger, fair_value_deviation,
                regime_confidence=regime_confidence
            )

        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Volatility breakout
            signal = self._volatility_signal(
                symbol, current_price, atr, bollinger, flow_imbalance,
                regime_confidence=regime_confidence
            )

        else:
            return None

        if signal is None:
            return None

        # Calculate position size and stops
        position_size = self._calculate_position_size(signal, atr, current_price)
        stop_loss, take_profit = self._calculate_stops(
            signal, current_price, atr, regime
        )

        return EnhancedSignal(
            signal=signal,
            regime=regime,
            indicators=indicators,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _trend_following_signal(
        self,
        symbol: str,
        price: float,
        rsi: float,
        momentum: float,
        flow_imbalance: float,
        direction: int,
        regime_confidence: float,
    ) -> Optional[Signal]:
        """
        Trend following signal.

        Buy on pullbacks in uptrend, sell on rallies in downtrend.

        Based on Jegadeesh & Titman (1993) momentum research - trends persist
        and pullbacks in trends offer favorable entry points.
        """
        if direction > 0:  # Uptrend
            # Look for pullback in uptrend - RSI below 60 (not extremely overbought)
            # and flow imbalance not strongly negative
            if rsi < 60 and flow_imbalance > -0.5:
                # Confidence based on how oversold we are within the uptrend
                rsi_score = (60 - rsi) / 60  # Higher score for lower RSI
                flow_score = (flow_imbalance + 0.5) / 1.0  # Normalized flow
                confidence = regime_confidence * 0.6 * (0.6 * rsi_score + 0.4 * flow_score)
                confidence = max(0.3, min(0.9, confidence))

                return Signal(
                    source=SignalSource.TOPOLOGY,  # Using topology as "regime" source
                    signal_type=SignalType.LONG,
                    symbol=symbol,
                    direction=1.0,
                    confidence=confidence,
                    urgency=0.6,
                    metadata={'strategy': 'trend_pullback', 'rsi': rsi, 'momentum': momentum},
                )
        else:  # Downtrend
            # Look for rally in downtrend - RSI above 40 (not extremely oversold)
            if rsi > 40 and flow_imbalance < 0.5:
                rsi_score = (rsi - 40) / 60  # Higher score for higher RSI
                flow_score = (0.5 - flow_imbalance) / 1.0
                confidence = regime_confidence * 0.6 * (0.6 * rsi_score + 0.4 * flow_score)
                confidence = max(0.3, min(0.9, confidence))

                return Signal(
                    source=SignalSource.TOPOLOGY,
                    signal_type=SignalType.SHORT,
                    symbol=symbol,
                    direction=-1.0,
                    confidence=confidence,
                    urgency=0.6,
                    metadata={'strategy': 'trend_rally', 'rsi': rsi, 'momentum': momentum},
                )

        return None

    def _mean_reversion_signal(
        self,
        symbol: str,
        price: float,
        rsi: float,
        bollinger: Optional[Tuple[float, float, float]],
        deviation: Optional[float],
        regime_confidence: float,
    ) -> Optional[Signal]:
        """
        Mean reversion signal for consolidation regime.

        Buy at lower band / oversold, sell at upper band / overbought.

        Thresholds relaxed for daily data where extreme conditions are rare.
        Academic research shows mean reversion works even with moderate deviations
        in consolidating markets (Poterba & Summers 1988).
        """
        if bollinger is None:
            return None

        lower, middle, upper = bollinger

        # Normalize position in Bollinger bands
        band_width = upper - lower
        if band_width == 0:
            return None

        position = (price - lower) / band_width  # 0 = at lower, 1 = at upper

        # Relaxed thresholds for daily data:
        # - Bollinger position: 0.35/0.65 instead of 0.2/0.8
        # - RSI: 45/55 instead of 30/70 (moderate overbought/oversold)
        # This generates more signals while still capturing mean reversion opportunities

        rsi_oversold_threshold = 45.0  # More lenient than standard 30
        rsi_overbought_threshold = 55.0  # More lenient than standard 70
        bb_lower_threshold = 0.35  # Bottom 35% of bands
        bb_upper_threshold = 0.65  # Top 35% of bands

        # Buy signal: price in lower portion of bands + RSI not overbought
        if position < bb_lower_threshold and rsi < rsi_oversold_threshold:
            # Confidence increases as position and RSI get more extreme
            position_score = (bb_lower_threshold - position) / bb_lower_threshold
            rsi_score = (rsi_oversold_threshold - rsi) / rsi_oversold_threshold
            confidence = regime_confidence * 0.6 * (0.5 * position_score + 0.5 * rsi_score)
            confidence = max(0.3, min(0.85, confidence))  # Floor at 0.3 for daily data

            return Signal(
                source=SignalSource.REFLEXIVITY,  # Using reflexivity for mean reversion
                signal_type=SignalType.LONG,
                symbol=symbol,
                direction=1.0,
                confidence=confidence,
                urgency=0.5,
                metadata={'strategy': 'mean_reversion_buy', 'bb_position': position, 'rsi': rsi},
            )

        # Sell signal: price in upper portion of bands + RSI not oversold
        elif position > bb_upper_threshold and rsi > rsi_overbought_threshold:
            position_score = (position - bb_upper_threshold) / (1 - bb_upper_threshold)
            rsi_score = (rsi - rsi_overbought_threshold) / (100 - rsi_overbought_threshold)
            confidence = regime_confidence * 0.6 * (0.5 * position_score + 0.5 * rsi_score)
            confidence = max(0.3, min(0.85, confidence))

            return Signal(
                source=SignalSource.REFLEXIVITY,
                signal_type=SignalType.SHORT,
                symbol=symbol,
                direction=-1.0,
                confidence=confidence,
                urgency=0.5,
                metadata={'strategy': 'mean_reversion_sell', 'bb_position': position, 'rsi': rsi},
            )

        # Fallback: Moderate mean reversion when price deviates from middle band
        # This ensures we generate some signals even in mild consolidation
        elif deviation is not None:
            if deviation < -0.02 and rsi < 50:  # 2% below fair value
                confidence = regime_confidence * 0.4 * (1 + abs(deviation) * 5)
                confidence = max(0.25, min(0.6, confidence))
                return Signal(
                    source=SignalSource.REFLEXIVITY,
                    signal_type=SignalType.LONG,
                    symbol=symbol,
                    direction=1.0,
                    confidence=confidence,
                    urgency=0.4,
                    metadata={'strategy': 'fair_value_buy', 'deviation': deviation},
                )
            elif deviation > 0.02 and rsi > 50:  # 2% above fair value
                confidence = regime_confidence * 0.4 * (1 + abs(deviation) * 5)
                confidence = max(0.25, min(0.6, confidence))
                return Signal(
                    source=SignalSource.REFLEXIVITY,
                    signal_type=SignalType.SHORT,
                    symbol=symbol,
                    direction=-1.0,
                    confidence=confidence,
                    urgency=0.4,
                    metadata={'strategy': 'fair_value_sell', 'deviation': deviation},
                )

        return None

    def _volatility_signal(
        self,
        symbol: str,
        price: float,
        atr: Optional[float],
        bollinger: Optional[Tuple[float, float, float]],
        flow_imbalance: float,
        regime_confidence: float,
    ) -> Optional[Signal]:
        """
        Volatility breakout signal.

        Trade breakouts in high volatility with flow confirmation.
        Research (Chou 1988, Engle & Ng 1993) shows volatility clustering
        creates tradeable breakout opportunities.
        """
        if bollinger is None or atr is None:
            return None

        lower, middle, upper = bollinger
        band_width = upper - lower
        if band_width == 0:
            return None

        # Calculate how far outside bands price is
        if price > upper:
            breakout_strength = (price - upper) / band_width
        elif price < lower:
            breakout_strength = (lower - price) / band_width
        else:
            breakout_strength = 0

        # Relaxed flow thresholds for daily data
        flow_threshold = 0.1  # More lenient than 0.3

        # Strong breakout above upper band with positive flow
        if price > upper and flow_imbalance > flow_threshold:
            # Confidence based on breakout strength and flow
            confidence = regime_confidence * 0.5 * (1 + breakout_strength) * (1 + flow_imbalance)
            confidence = max(0.35, min(0.75, confidence))

            return Signal(
                source=SignalSource.ATTENTION,  # Attention for breakouts
                signal_type=SignalType.LONG,
                symbol=symbol,
                direction=1.0,
                confidence=confidence,
                urgency=0.8,  # High urgency for breakouts
                metadata={'strategy': 'volatility_breakout_long', 'breakout_strength': breakout_strength},
            )

        # Strong breakdown below lower band with negative flow
        elif price < lower and flow_imbalance < -flow_threshold:
            confidence = regime_confidence * 0.5 * (1 + breakout_strength) * (1 - flow_imbalance)
            confidence = max(0.35, min(0.75, confidence))

            return Signal(
                source=SignalSource.ATTENTION,
                signal_type=SignalType.SHORT,
                symbol=symbol,
                direction=-1.0,
                confidence=confidence,
                urgency=0.8,
                metadata={'strategy': 'volatility_breakout_short', 'breakout_strength': breakout_strength},
            )

        # Fallback: Trade in direction of flow during high volatility
        # Even without breakout, strong flow during volatility can be a signal
        elif abs(flow_imbalance) > 0.25:
            direction = 1.0 if flow_imbalance > 0 else -1.0
            signal_type = SignalType.LONG if direction > 0 else SignalType.SHORT
            confidence = regime_confidence * 0.4 * abs(flow_imbalance)
            confidence = max(0.25, min(0.55, confidence))

            return Signal(
                source=SignalSource.ATTENTION,
                signal_type=signal_type,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                urgency=0.6,
                metadata={'strategy': 'volatility_flow', 'flow_imbalance': flow_imbalance},
            )

        return None

    def _calculate_position_size(
        self,
        signal: Signal,
        atr: Optional[float],
        price: float,
    ) -> float:
        """
        Calculate position size using Kelly-inspired formula.

        Adjusts for volatility and confidence.
        """
        # Base size from confidence
        base_size = signal.confidence * PARAMS.kelly_fraction

        # Adjust for volatility (reduce size when volatility high)
        if atr and price > 0:
            volatility = atr / price
            volatility_adjustment = max(0.5, 1 - volatility * 10)
            base_size *= volatility_adjustment

        # Apply min/max constraints
        return max(PARAMS.min_position_pct, min(PARAMS.max_position_pct, base_size))

    def _calculate_stops(
        self,
        signal: Signal,
        price: float,
        atr: Optional[float],
        regime: MarketRegime,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels.

        Uses ATR-based stops adjusted for regime.
        """
        if atr is None:
            return (None, None)

        # Regime-based ATR multipliers
        if regime in (MarketRegime.HIGH_VOLATILITY,):
            stop_mult = 3.0  # Wider stops in high vol
            target_mult = 2.0
        elif regime in (MarketRegime.CONSOLIDATION,):
            stop_mult = 1.5  # Tighter stops in consolidation
            target_mult = 1.5
        else:
            stop_mult = 2.0  # Standard
            target_mult = 3.0  # Let trends run

        if signal.direction > 0:  # Long
            stop_loss = price - stop_mult * atr
            take_profit = price + target_mult * atr
        else:  # Short
            stop_loss = price + stop_mult * atr
            take_profit = price - target_mult * atr

        return (stop_loss, take_profit)
