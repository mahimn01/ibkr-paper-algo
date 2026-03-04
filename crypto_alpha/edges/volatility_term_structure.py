"""
Edge 4: Volatility Term Structure (VTS)

The relationship between short-term and long-term realized volatility contains
regime information that is DIFFERENT from what price-based HMMs capture.

Key insight: When short-term vol spikes above long-term vol (inverted term
structure), markets are stressed and mean-reversion strategies work. When the
term structure is normal (short < long), trends are more likely to continue.

This is a proxy for the options implied volatility surface that works with
just price data.

Vol Regime Classification:
    NORMAL:     slope > 0.10, long_vol < 0.60 ann -> trend following mode
    INVERTED:   slope < -0.15 -> mean reversion mode (short-term stress)
    CRISIS:     ultra_short_vol > 2 * long_vol -> capital preservation
    COMPRESSED: abs(slope) < 0.08 -> breakout imminent

Signal Logic:
    NORMAL     -> LONG bias (trends tend to continue)
    INVERTED   -> NEUTRAL/SHORT bias (expect mean reversion)
    CRISIS     -> STRONG exit signals (reduce positions)
    COMPRESSED -> directional breakout (follow price trend)

Expected SR: 0.8-1.2
Correlation with others: Low (pure vol-regime signal, orthogonal to price/flow edges)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from crypto_alpha.edges.base_edge import CryptoEdge
from crypto_alpha.types import CryptoAssetState, CryptoEdgeVote, EdgeSignal

logger = logging.getLogger(__name__)

# Annualization factor for hourly crypto data: sqrt(8760 hours/year)
ANNUALIZE_HOURLY = math.sqrt(8760)


class VolatilityTermStructure(CryptoEdge):
    """
    Trade vol regime changes using the realized volatility term structure.

    Computes realized volatility at four horizons (ultra-short, short, medium,
    long) and classifies the current vol regime from the slope and convexity
    of the term structure curve.
    """

    def __init__(
        self,
        ultra_short_window: int = 6,       # 6h
        short_window: int = 24,            # 24h (1 day)
        medium_window: int = 168,          # 168h (7 days)
        long_window: int = 720,            # 720h (30 days)
        slope_threshold_normal: float = 0.10,
        slope_threshold_inverted: float = -0.15,
        crisis_ratio: float = 2.0,         # ultra_short/long > 2.0 = crisis
        compressed_range: float = 0.08,    # max abs(slope) for compressed
        trend_lookback: int = 48,          # bars to assess price trend for breakout direction
        slope_history_len: int = 120,      # bars of slope history for z-score
    ):
        self._ultra_short_window = ultra_short_window
        self._short_window = short_window
        self._medium_window = medium_window
        self._long_window = long_window
        self._slope_threshold_normal = slope_threshold_normal
        self._slope_threshold_inverted = slope_threshold_inverted
        self._crisis_ratio = crisis_ratio
        self._compressed_range = compressed_range
        self._trend_lookback = trend_lookback
        self._slope_history_len = slope_history_len

        # Per-symbol state
        self._prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=long_window + 50)
        )
        self._bar_count: Dict[str, int] = defaultdict(int)

        # Computed vol state per symbol
        self._slope_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=slope_history_len)
        )
        self._vol_state: Dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "VolTermStructure"

    @property
    def warmup_bars(self) -> int:
        return self._long_window

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """Feed a single hourly bar to the edge."""
        self._prices[symbol].append(price)
        self._bar_count[symbol] += 1

        # Only compute vol state once we have enough data
        if self._bar_count[symbol] >= self._long_window:
            self._compute_vol_state(symbol)

    def _realized_vol(self, prices: np.ndarray, window: int) -> float:
        """
        Compute annualized realized volatility from the last `window` prices.

        Uses log returns and annualizes with sqrt(8760) for hourly crypto data.
        Returns 0.0 if insufficient data.
        """
        if len(prices) < window + 1:
            return 0.0

        p = prices[-(window + 1):]
        log_returns = np.diff(np.log(p))

        if len(log_returns) < 2:
            return 0.0

        return float(np.std(log_returns, ddof=1) * ANNUALIZE_HOURLY)

    def _compute_vol_state(self, symbol: str) -> None:
        """Compute the full vol term structure state for a symbol."""
        prices = np.array(self._prices[symbol], dtype=np.float64)

        ultra_short_vol = self._realized_vol(prices, self._ultra_short_window)
        short_vol = self._realized_vol(prices, self._short_window)
        medium_vol = self._realized_vol(prices, self._medium_window)
        long_vol = self._realized_vol(prices, self._long_window)

        # Guard against zero/tiny long_vol
        if long_vol < 1e-8:
            return

        # Term structure slope: (long - short) / long
        # Positive = normal (contango-like), Negative = inverted (backwardation-like)
        slope = (long_vol - short_vol) / long_vol

        # Term structure convexity: ultra_short - 2*short + medium
        # Positive = vol acceleration (crash incoming)
        # Negative = vol deceleration (recovery)
        convexity = ultra_short_vol - 2 * short_vol + medium_vol

        # Track slope history for z-score
        self._slope_history[symbol].append(slope)

        # Slope z-score
        slope_z = 0.0
        if len(self._slope_history[symbol]) >= 20:
            slope_arr = np.array(self._slope_history[symbol], dtype=np.float64)
            slope_std = np.std(slope_arr, ddof=1)
            if slope_std > 1e-8:
                slope_z = (slope - np.mean(slope_arr)) / slope_std

        # Classify regime
        ultra_long_ratio = ultra_short_vol / long_vol if long_vol > 1e-8 else 1.0
        regime = self._classify_regime(slope, ultra_long_ratio, long_vol)

        # Price trend for breakout direction (simple: compare current vs lookback)
        trend_direction = 0
        if len(prices) >= self._trend_lookback + 1:
            lookback_price = prices[-(self._trend_lookback + 1)]
            current_price = prices[-1]
            if lookback_price > 0:
                pct_change = (current_price - lookback_price) / lookback_price
                if pct_change > 0.01:
                    trend_direction = 1
                elif pct_change < -0.01:
                    trend_direction = -1

        self._vol_state[symbol] = {
            'ultra_short_vol': ultra_short_vol,
            'short_vol': short_vol,
            'medium_vol': medium_vol,
            'long_vol': long_vol,
            'slope': slope,
            'convexity': convexity,
            'slope_z': slope_z,
            'ultra_long_ratio': ultra_long_ratio,
            'regime': regime,
            'trend_direction': trend_direction,
        }

    def _classify_regime(
        self, slope: float, ultra_long_ratio: float, long_vol: float
    ) -> str:
        """
        Classify the current vol regime.

        Priority order matters: CRISIS > INVERTED > COMPRESSED > NORMAL
        """
        # CRISIS: ultra-short vol is more than crisis_ratio * long vol
        if ultra_long_ratio > self._crisis_ratio:
            return "CRISIS"

        # INVERTED: slope strongly negative (short vol >> long vol)
        if slope < self._slope_threshold_inverted:
            return "INVERTED"

        # COMPRESSED: slope near zero, all vols similar
        if abs(slope) < self._compressed_range:
            return "COMPRESSED"

        # NORMAL: slope positive and long vol not extreme
        if slope > self._slope_threshold_normal and long_vol < 0.60:
            return "NORMAL"

        # Default: NORMAL if slope is positive, COMPRESSED otherwise
        if slope > 0:
            return "NORMAL"
        return "COMPRESSED"

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate trading signal from vol term structure regime."""
        vol = self._vol_state.get(symbol)

        if vol is None:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Insufficient data for vol computation",
            )

        regime = vol['regime']
        slope = vol['slope']
        slope_z = vol['slope_z']
        convexity = vol['convexity']
        long_vol = vol['long_vol']
        ultra_short_vol = vol['ultra_short_vol']
        trend_direction = vol['trend_direction']

        # Build common data dict for all signals
        data = {
            'regime': regime,
            'slope': round(slope, 4),
            'slope_z': round(slope_z, 2),
            'convexity': round(convexity, 6),
            'ultra_short_vol': round(ultra_short_vol, 4),
            'short_vol': round(vol['short_vol'], 4),
            'medium_vol': round(vol['medium_vol'], 4),
            'long_vol': round(long_vol, 4),
            'ultra_long_ratio': round(vol['ultra_long_ratio'], 3),
            'trend_direction': trend_direction,
        }

        # --- CRISIS regime: strong exit signals ---
        if regime == "CRISIS":
            confidence = min(1.0, vol['ultra_long_ratio'] / (self._crisis_ratio * 1.5))
            # In a crisis, reduce all exposure
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.STRONG_SHORT,
                confidence=confidence,
                reason=(
                    f"CRISIS: ultra/long ratio={vol['ultra_long_ratio']:.2f}, "
                    f"ultra_vol={ultra_short_vol:.3f}, long_vol={long_vol:.3f}"
                ),
                data={**data, 'action': 'crisis_exit'},
            )

        # --- INVERTED regime: mean reversion mode ---
        if regime == "INVERTED":
            # Confidence scales with how inverted the slope is
            inversion_strength = abs(slope) / abs(self._slope_threshold_inverted)
            slope_z_factor = min(1.0, abs(slope_z) / 2.0) if slope_z < 0 else 0.3
            confidence = min(1.0, 0.4 * inversion_strength + 0.3 * slope_z_factor)

            # Positive convexity during inversion = accelerating stress
            if convexity > 0.01:
                vote = CryptoEdgeVote.SHORT
                reason = (
                    f"INVERTED+accelerating: slope={slope:.3f}, "
                    f"convexity={convexity:.4f}"
                )
            else:
                # Negative convexity = stress may be easing, just reduce exposure
                vote = CryptoEdgeVote.NEUTRAL
                confidence *= 0.5
                reason = (
                    f"INVERTED+decelerating: slope={slope:.3f}, "
                    f"convexity={convexity:.4f}"
                )

            return EdgeSignal(
                edge_name=self.name,
                vote=vote,
                confidence=confidence,
                reason=reason,
                data={**data, 'action': 'mean_reversion'},
            )

        # --- COMPRESSED regime: breakout imminent ---
        if regime == "COMPRESSED":
            # Direction from price trend
            if trend_direction > 0:
                vote = CryptoEdgeVote.LONG
                action = 'breakout_long'
            elif trend_direction < 0:
                vote = CryptoEdgeVote.SHORT
                action = 'breakout_short'
            else:
                vote = CryptoEdgeVote.NEUTRAL
                action = 'breakout_wait'

            # Confidence: lower vol = more compressed = higher breakout probability
            compression = 1.0 - abs(slope) / self._compressed_range
            vol_compression = 1.0 - min(1.0, long_vol / 0.50)  # Lower vol = more compressed
            confidence = min(1.0, 0.5 * compression + 0.3 * vol_compression)

            # Reduce confidence if no clear trend direction
            if trend_direction == 0:
                confidence *= 0.2

            return EdgeSignal(
                edge_name=self.name,
                vote=vote,
                confidence=confidence,
                reason=(
                    f"COMPRESSED: slope={slope:.3f}, "
                    f"trend={'UP' if trend_direction > 0 else 'DOWN' if trend_direction < 0 else 'FLAT'}"
                ),
                data={**data, 'action': action},
            )

        # --- NORMAL regime: trend following mode ---
        # slope > 0.10, long_vol < 0.60 -> trends tend to continue
        slope_strength = slope / self._slope_threshold_normal
        slope_z_factor = min(1.0, abs(slope_z) / 2.0) if slope_z > 0 else 0.3
        vol_quality = 1.0 - min(1.0, long_vol / 0.60)  # Lower vol = cleaner trends
        confidence = min(1.0, 0.3 * min(slope_strength, 3.0) / 3.0 + 0.3 * slope_z_factor + 0.2 * vol_quality)

        # LONG bias in normal regime (trends continue)
        vote = CryptoEdgeVote.LONG

        # Upgrade to STRONG_LONG if slope is very healthy and convexity is negative (stable)
        if slope > self._slope_threshold_normal * 2 and convexity < 0:
            vote = CryptoEdgeVote.STRONG_LONG
            confidence = min(1.0, confidence * 1.2)
            reason = (
                f"NORMAL+stable: slope={slope:.3f}, "
                f"convexity={convexity:.4f}, long_vol={long_vol:.3f}"
            )
        else:
            reason = (
                f"NORMAL: slope={slope:.3f}, "
                f"long_vol={long_vol:.3f}, slope_z={slope_z:.2f}"
            )

        return EdgeSignal(
            edge_name=self.name,
            vote=vote,
            confidence=confidence,
            reason=reason,
            data={**data, 'action': 'trend_follow'},
        )

    def reset(self) -> None:
        self._prices.clear()
        self._bar_count.clear()
        self._slope_history.clear()
        self._vol_state.clear()
