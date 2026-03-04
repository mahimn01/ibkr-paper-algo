"""
Edge 9: Intermarket Momentum Cascade (IMC)

When BTC breaks a major level, altcoins follow with a 5-30 minute
delay. The delay varies by market cap:
    - Large caps (ETH, SOL): 5-10 minutes (~1-2 bars on 5m)
    - Mid caps:              10-20 minutes (~2-4 bars)
    - Small caps:            15-30 minutes (~3-6 bars)

This is lead-lag arbitrage accessible to retail because crypto
delays are MINUTES (not microseconds like in equities).

Implementation:
    1. Detect BTC breakout (20-period high/low + volume confirmation)
    2. Estimate optimal lag per alt using rolling cross-correlation
    3. If breakout detected AND within lag window -> trade alt
    4. Exit after estimated propagation delay + buffer

Expected SR: 1.2-1.6
Correlation with others: Moderate with RADL (both BTC-driven)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from crypto_alpha.edges.base_edge import CryptoEdge
from crypto_alpha.types import CryptoAssetState, CryptoEdgeVote, EdgeSignal

logger = logging.getLogger(__name__)

# Reference symbol that leads
BTC_SYMBOL = "BTC/USDT"


class IntermarketCascade(CryptoEdge):
    """
    Trade altcoin delayed reactions to BTC breakouts.

    BTC is the information leader in crypto. When BTC breaks
    significant levels, alts follow with predictable delays.
    """

    def __init__(
        self,
        breakout_lookback: int = 20,     # Bars for breakout detection
        volume_confirm_mult: float = 1.8, # Volume must exceed this * avg
        max_lag_bars: int = 12,           # Maximum propagation lag to consider
        min_correlation: float = 0.3,     # Minimum BTC-alt correlation to trade
        correlation_window: int = 100,    # Rolling window for correlation
        exit_after_bars: int = 6,         # Exit N bars after entry
        breakout_strength_min: float = 0.005,  # Min % move for breakout
    ):
        self._breakout_lookback = breakout_lookback
        self._volume_confirm_mult = volume_confirm_mult
        self._max_lag_bars = max_lag_bars
        self._min_correlation = min_correlation
        self._correlation_window = correlation_window
        self._exit_after_bars = exit_after_bars
        self._breakout_strength_min = breakout_strength_min

        # BTC state
        self._btc_prices: deque = deque(maxlen=500)
        self._btc_volumes: deque = deque(maxlen=500)
        self._btc_returns: deque = deque(maxlen=500)
        self._btc_highs: deque = deque(maxlen=breakout_lookback + 5)
        self._btc_lows: deque = deque(maxlen=breakout_lookback + 5)

        # Alt state
        self._alt_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._alt_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._bar_count: Dict[str, int] = defaultdict(int)

        # Computed state
        self._optimal_lag: Dict[str, int] = {}  # symbol -> estimated lag in bars
        self._btc_alt_corr: Dict[str, float] = {}  # BTC-alt correlation
        self._active_breakout: Optional[Dict] = None  # Current breakout event
        self._bars_since_breakout: int = 0
        self._entry_bars: Dict[str, int] = defaultdict(int)  # Track bars since entry

    @property
    def name(self) -> str:
        return "IntermarketCascade"

    @property
    def warmup_bars(self) -> int:
        return self._correlation_window + self._breakout_lookback

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        self._bar_count[symbol] += 1

        if symbol == BTC_SYMBOL:
            # Update BTC state
            if self._btc_prices:
                ret = price / self._btc_prices[-1] - 1 if self._btc_prices[-1] > 0 else 0
                self._btc_returns.append(ret)

            self._btc_prices.append(price)
            self._btc_volumes.append(volume)

            high = kwargs.get('high', price)
            low = kwargs.get('low', price)
            self._btc_highs.append(high)
            self._btc_lows.append(low)

            # Detect breakout
            self._detect_breakout(timestamp)

            # Increment breakout timer
            if self._active_breakout is not None:
                self._bars_since_breakout += 1
                # Expire breakout after max_lag_bars * 2
                if self._bars_since_breakout > self._max_lag_bars * 2:
                    self._active_breakout = None

        else:
            # Update alt state
            if self._alt_prices[symbol]:
                prev = self._alt_prices[symbol][-1]
                ret = price / prev - 1 if prev > 0 else 0
                self._alt_returns[symbol].append(ret)

            self._alt_prices[symbol].append(price)

            # Update lag and correlation estimates periodically
            if self._bar_count[symbol] % 50 == 0:
                self._estimate_lag(symbol)

            # Track entry duration
            if symbol in self._entry_bars and self._entry_bars[symbol] > 0:
                self._entry_bars[symbol] += 1

    def _detect_breakout(self, timestamp: datetime) -> None:
        """Detect BTC breaking N-period high or low with volume confirmation."""
        if len(self._btc_prices) < self._breakout_lookback + 1:
            return
        if len(self._btc_volumes) < self._breakout_lookback:
            return

        current_price = self._btc_prices[-1]
        prices_window = list(self._btc_prices)[-self._breakout_lookback - 1:-1]

        if not prices_window:
            return

        period_high = max(prices_window)
        period_low = min(prices_window)

        # Volume confirmation
        recent_volumes = list(self._btc_volumes)
        avg_vol = np.mean(recent_volumes[-self._breakout_lookback:-1]) if len(recent_volumes) > self._breakout_lookback else np.mean(recent_volumes)
        current_vol = recent_volumes[-1] if recent_volumes else 0
        vol_confirmed = current_vol > avg_vol * self._volume_confirm_mult

        # Breakout strength (% above/below level)
        upside_break = (current_price / period_high - 1) if period_high > 0 else 0
        downside_break = (1 - current_price / period_low) if period_low > 0 else 0

        # Detect upside breakout
        if (upside_break > self._breakout_strength_min and vol_confirmed
                and self._active_breakout is None):
            self._active_breakout = {
                'direction': 1,  # Bullish
                'strength': upside_break,
                'timestamp': timestamp,
                'btc_price': current_price,
                'level_broken': period_high,
                'volume_ratio': current_vol / avg_vol if avg_vol > 0 else 1,
            }
            self._bars_since_breakout = 0
            logger.debug(f"BTC UPSIDE breakout: {upside_break:.3%} above {period_high:.0f}")

        # Detect downside breakout
        elif (downside_break > self._breakout_strength_min and vol_confirmed
                and self._active_breakout is None):
            self._active_breakout = {
                'direction': -1,  # Bearish
                'strength': downside_break,
                'timestamp': timestamp,
                'btc_price': current_price,
                'level_broken': period_low,
                'volume_ratio': current_vol / avg_vol if avg_vol > 0 else 1,
            }
            self._bars_since_breakout = 0
            logger.debug(f"BTC DOWNSIDE breakout: {downside_break:.3%} below {period_low:.0f}")

    def _estimate_lag(self, symbol: str) -> None:
        """
        Estimate the propagation lag from BTC to alt using cross-correlation.

        For each lag k (0 to max_lag):
            corr(BTC_returns[t-k], ALT_returns[t])
        Optimal lag = argmax(corr)
        """
        btc_rets = np.array(self._btc_returns, dtype=np.float64)
        alt_rets = np.array(self._alt_returns[symbol], dtype=np.float64)

        # Align lengths
        min_len = min(len(btc_rets), len(alt_rets), self._correlation_window)
        if min_len < 30:
            return

        btc_r = btc_rets[-min_len:]
        alt_r = alt_rets[-min_len:]

        best_corr = -1.0
        best_lag = 0

        for lag in range(0, min(self._max_lag_bars + 1, min_len - 10)):
            if lag == 0:
                btc_slice = btc_r
                alt_slice = alt_r
            else:
                btc_slice = btc_r[:-lag]
                alt_slice = alt_r[lag:]

            if len(btc_slice) < 10:
                break

            corr = np.corrcoef(btc_slice, alt_slice)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag

        self._optimal_lag[symbol] = best_lag
        self._btc_alt_corr[symbol] = best_corr

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate signal for an alt based on BTC breakout cascade."""
        # Don't trade BTC against itself
        if symbol == BTC_SYMBOL:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="BTC is the leader, not a follower",
            )

        # Check if we should exit an existing position
        if self._entry_bars.get(symbol, 0) > self._exit_after_bars:
            self._entry_bars[symbol] = 0
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Exit: propagation window expired",
                data={'action': 'exit'},
            )

        # No active breakout
        if self._active_breakout is None:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="No BTC breakout detected",
            )

        # Check correlation (only trade correlated alts)
        corr = self._btc_alt_corr.get(symbol, 0.0)
        if corr < self._min_correlation:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Low BTC-{symbol} correlation: {corr:.2f}",
            )

        # Check if we're within the propagation window
        optimal_lag = self._optimal_lag.get(symbol, 2)
        breakout = self._active_breakout

        # Signal window: [optimal_lag - 1, optimal_lag + 2] bars after breakout
        window_start = max(0, optimal_lag - 1)
        window_end = optimal_lag + 3

        if not (window_start <= self._bars_since_breakout <= window_end):
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Outside propagation window: bars={self._bars_since_breakout}, "
                       f"window=[{window_start},{window_end}]",
            )

        # Compute confidence
        direction = breakout['direction']
        strength = breakout['strength']
        vol_ratio = breakout['volume_ratio']

        confidence = min(1.0, (
            0.3 * min(1.0, strength / 0.02)  # Breakout magnitude
            + 0.3 * min(1.0, corr)             # Correlation strength
            + 0.2 * min(1.0, vol_ratio / 3.0)  # Volume confirmation
            + 0.2 * (1 - abs(self._bars_since_breakout - optimal_lag) / max(window_end, 1))
        ))

        data = {
            'btc_breakout_direction': direction,
            'breakout_strength': strength,
            'btc_alt_correlation': corr,
            'optimal_lag_bars': optimal_lag,
            'bars_since_breakout': self._bars_since_breakout,
            'volume_ratio': vol_ratio,
        }

        # Mark entry
        self._entry_bars[symbol] = 1

        if direction > 0:
            # BTC bullish breakout -> alts will follow up
            if confidence > 0.6:
                return EdgeSignal(
                    edge_name=self.name,
                    vote=CryptoEdgeVote.STRONG_LONG,
                    confidence=confidence,
                    reason=f"BTC upside cascade -> {symbol} (lag={optimal_lag}, corr={corr:.2f})",
                    data=data,
                )
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.LONG,
                confidence=confidence,
                reason=f"BTC upside cascade -> {symbol} (lag={optimal_lag}, corr={corr:.2f})",
                data=data,
            )
        else:
            # BTC bearish breakout -> alts will follow down
            if confidence > 0.6:
                return EdgeSignal(
                    edge_name=self.name,
                    vote=CryptoEdgeVote.STRONG_SHORT,
                    confidence=confidence,
                    reason=f"BTC downside cascade -> {symbol} (lag={optimal_lag}, corr={corr:.2f})",
                    data=data,
                )
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.SHORT,
                confidence=confidence,
                reason=f"BTC downside cascade -> {symbol} (lag={optimal_lag}, corr={corr:.2f})",
                data=data,
            )

    def reset(self) -> None:
        self._btc_prices.clear()
        self._btc_volumes.clear()
        self._btc_returns.clear()
        self._btc_highs.clear()
        self._btc_lows.clear()
        self._alt_prices.clear()
        self._alt_returns.clear()
        self._bar_count.clear()
        self._optimal_lag.clear()
        self._btc_alt_corr.clear()
        self._active_breakout = None
        self._bars_since_breakout = 0
        self._entry_bars.clear()
