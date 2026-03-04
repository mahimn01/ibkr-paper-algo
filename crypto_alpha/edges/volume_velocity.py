"""
Edge 6: Volume Velocity Breakout (VVB)

Detects explosive moves before they happen by tracking the rate of change
(velocity) and acceleration of volume. Sharp volume acceleration predicts
breakouts — this is a proxy for social sentiment velocity, as volume surges
reflect real money flowing in response to news/social catalysts.

Key mechanisms:
  - Volume Velocity: EMA(short) / EMA(long) - 1 measures relative volume surge.
  - Volume Acceleration: Change in velocity over a window (second derivative).
  - Price-Volume Confirmation: Volume surge + directional price move = breakout.
  - Volume Climax Detection: Extreme volume (>3x avg) signals exhaustion,
    expect mean reversion after climax.

Signal logic:
  - velocity > threshold AND price momentum same direction -> breakout LONG/SHORT
  - acceleration positive AND no climax -> enter breakout early
  - volume > climax_threshold -> contrarian signal (expect reversal)
  - Confidence = f(velocity_z, acceleration, price_confirmation)

Expected SR: 0.6-1.0
Correlation with others: Low-moderate (volume dynamics, slight overlap with VFD)
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


class VolumeVelocityBreakout(CryptoEdge):
    """
    Trade volume-acceleration-driven breakouts and climax reversals.

    Uses the first and second derivatives of volume (exponentially smoothed)
    together with price momentum to identify breakout entries and exhaustion
    reversals before they become obvious on price charts.
    """

    def __init__(
        self,
        short_ema: int = 6,                    # Short EMA span (6h)
        long_ema: int = 72,                    # Long EMA span (3 days)
        velocity_threshold: float = 0.5,       # 50% above long EMA = surge
        acceleration_window: int = 12,         # Window for measuring acceleration
        climax_multiplier: float = 3.0,        # 3x avg volume = climax
        price_momentum_window: int = 12,       # Window for price momentum (12h)
        confirmation_weight: float = 0.3,      # Weight for price confirmation
        velocity_weight: float = 0.4,          # Weight for velocity component
        acceleration_weight: float = 0.3,      # Weight for acceleration component
        min_warmup: int = 100,                 # Bars before producing signals
    ):
        self._short_ema_span = short_ema
        self._long_ema_span = long_ema
        self._velocity_threshold = velocity_threshold
        self._acceleration_window = acceleration_window
        self._climax_multiplier = climax_multiplier
        self._price_momentum_window = price_momentum_window
        self._confirmation_weight = confirmation_weight
        self._velocity_weight = velocity_weight
        self._acceleration_weight = acceleration_weight
        self._min_warmup = min_warmup

        # EMA smoothing factors
        self._short_alpha = 2.0 / (short_ema + 1)
        self._long_alpha = 2.0 / (long_ema + 1)

        maxlen = max(long_ema, price_momentum_window) + acceleration_window + 60

        # Per-symbol state
        self._volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._closes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._bar_count: Dict[str, int] = defaultdict(int)

        # EMA state (running)
        self._ema_short: Dict[str, float] = {}
        self._ema_long: Dict[str, float] = {}

        # Velocity history for acceleration calculation
        self._velocity_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=acceleration_window + 20)
        )

        # Track recent climax events for cooldown
        self._last_climax_bar: Dict[str, int] = defaultdict(lambda: -999)
        self._climax_cooldown: int = 6  # Bars to wait after climax before new breakout

    @property
    def name(self) -> str:
        return "VolumeVelocityBreakout"

    @property
    def warmup_bars(self) -> int:
        return self._min_warmup

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """Feed a single bar of data."""
        self._volumes[symbol].append(volume)
        self._closes[symbol].append(price)
        self._bar_count[symbol] += 1

        # ---- Update EMAs ----
        if symbol not in self._ema_short:
            # Initialize EMAs with first volume value
            self._ema_short[symbol] = volume
            self._ema_long[symbol] = volume
        else:
            self._ema_short[symbol] = (
                self._short_alpha * volume
                + (1 - self._short_alpha) * self._ema_short[symbol]
            )
            self._ema_long[symbol] = (
                self._long_alpha * volume
                + (1 - self._long_alpha) * self._ema_long[symbol]
            )

        # ---- Compute and store velocity ----
        long_ema = self._ema_long[symbol]
        if long_ema > 1e-10:
            velocity = self._ema_short[symbol] / long_ema - 1.0
        else:
            velocity = 0.0
        self._velocity_history[symbol].append(velocity)

        # ---- Detect climax ----
        if long_ema > 1e-10 and volume > self._climax_multiplier * long_ema:
            self._last_climax_bar[symbol] = self._bar_count[symbol]

    def _compute_velocity_zscore(self, symbol: str) -> float:
        """Z-score of current velocity relative to recent velocity distribution."""
        vh = self._velocity_history[symbol]
        if len(vh) < self._acceleration_window:
            return 0.0

        arr = np.array(list(vh), dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std < 1e-10:
            return 0.0
        return (arr[-1] - mean) / std

    def _compute_acceleration(self, symbol: str) -> float:
        """
        Volume acceleration: change in velocity over the acceleration window.
        Positive = volume surge is intensifying (early breakout signal).
        """
        vh = self._velocity_history[symbol]
        window = self._acceleration_window

        if len(vh) < window:
            return 0.0

        recent = list(vh)[-window:]
        # Acceleration = slope of velocity over window (linear regression)
        x = np.arange(window, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        x_var = np.var(x, ddof=0)

        if x_var < 1e-12:
            return 0.0

        slope = np.sum((x - x_mean) * (y - y_mean)) / (x_var * window)
        return float(slope)

    def _compute_price_momentum(self, symbol: str) -> float:
        """
        Normalized price momentum over the momentum window.
        Returns value in [-1, 1] range approximately.
        """
        closes = self._closes[symbol]
        window = self._price_momentum_window

        if len(closes) < window + 1:
            return 0.0

        prices = list(closes)[-(window + 1):]
        start_price = prices[0]
        end_price = prices[-1]

        if abs(start_price) < 1e-10:
            return 0.0

        raw_return = (end_price - start_price) / start_price

        # Normalize by recent volatility
        returns = np.diff(np.log(np.array(prices, dtype=np.float64)))
        vol = np.std(returns, ddof=1)

        if vol < 1e-10:
            return np.clip(raw_return * 100, -3.0, 3.0)

        # Return as z-score of momentum
        return np.clip(raw_return / vol, -3.0, 3.0)

    def _is_in_climax(self, symbol: str) -> bool:
        """Check if we are in or just after a volume climax event."""
        bars_since = self._bar_count[symbol] - self._last_climax_bar[symbol]
        return bars_since <= self._climax_cooldown

    def _is_climax_current(self, symbol: str) -> bool:
        """Check if the CURRENT bar is a climax bar."""
        return self._bar_count[symbol] == self._last_climax_bar[symbol]

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate trading signal from volume velocity analysis."""
        if self._bar_count[symbol] < self._min_warmup:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Warming up",
            )

        # Current velocity
        vh = self._velocity_history[symbol]
        if not vh:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="No velocity data",
            )

        current_velocity = vh[-1]
        velocity_z = self._compute_velocity_zscore(symbol)
        acceleration = self._compute_acceleration(symbol)
        price_mom = self._compute_price_momentum(symbol)
        is_climax = self._is_climax_current(symbol)
        in_climax_cooldown = self._is_in_climax(symbol)

        # Volume ratio for diagnostics
        long_ema = self._ema_long.get(symbol, 1.0)
        current_vol = self._volumes[symbol][-1] if self._volumes[symbol] else 0.0
        vol_ratio = current_vol / long_ema if long_ema > 1e-10 else 0.0

        data = {
            'velocity': round(current_velocity, 4),
            'velocity_z': round(velocity_z, 3),
            'acceleration': round(acceleration, 6),
            'price_momentum': round(price_mom, 4),
            'vol_ratio': round(vol_ratio, 3),
            'is_climax': is_climax,
            'in_cooldown': in_climax_cooldown,
        }

        # ---- Climax Reversal Signal ----
        # Volume climax = exhaustion. After extreme volume, expect mean reversion.
        # Trade AGAINST the price direction.
        if is_climax:
            # Contrarian: fade the move
            if price_mom > 0.5:
                confidence = min(1.0, vol_ratio / (self._climax_multiplier * 2))
                data['action'] = 'climax_reversal_short'
                return EdgeSignal(
                    edge_name=self.name,
                    vote=CryptoEdgeVote.SHORT,
                    confidence=confidence,
                    reason=f"Volume climax reversal: vol_ratio={vol_ratio:.1f}x, mom={price_mom:.2f}",
                    data=data,
                )
            elif price_mom < -0.5:
                confidence = min(1.0, vol_ratio / (self._climax_multiplier * 2))
                data['action'] = 'climax_reversal_long'
                return EdgeSignal(
                    edge_name=self.name,
                    vote=CryptoEdgeVote.LONG,
                    confidence=confidence,
                    reason=f"Volume climax reversal: vol_ratio={vol_ratio:.1f}x, mom={price_mom:.2f}",
                    data=data,
                )

        # ---- Suppress breakout signals during climax cooldown ----
        if in_climax_cooldown:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Post-climax cooldown",
                data=data,
            )

        # ---- Breakout Signal ----
        # Volume velocity exceeds threshold AND price confirms direction
        velocity_surging = current_velocity > self._velocity_threshold
        acceleration_positive = acceleration > 0

        if velocity_surging and acceleration_positive:
            # Compute composite confidence
            vel_conf = min(1.0, abs(velocity_z) / 3.0) * self._velocity_weight
            acc_conf = min(1.0, abs(acceleration) * 20) * self._acceleration_weight
            price_conf = min(1.0, abs(price_mom) / 2.0) * self._confirmation_weight

            confidence = min(1.0, vel_conf + acc_conf + price_conf)

            # Direction from price momentum
            if price_mom > 0.3:
                # Volume surge + price going up = bullish breakout
                if velocity_z > 2.0 and price_mom > 1.0:
                    vote = CryptoEdgeVote.STRONG_LONG
                    data['action'] = 'strong_breakout_long'
                else:
                    vote = CryptoEdgeVote.LONG
                    data['action'] = 'breakout_long'

                return EdgeSignal(
                    edge_name=self.name,
                    vote=vote,
                    confidence=confidence,
                    reason=(
                        f"Bullish breakout: vel={current_velocity:.2f}, "
                        f"acc={acceleration:.4f}, mom={price_mom:.2f}"
                    ),
                    data=data,
                )

            elif price_mom < -0.3:
                # Volume surge + price going down = bearish breakout
                if velocity_z > 2.0 and price_mom < -1.0:
                    vote = CryptoEdgeVote.STRONG_SHORT
                    data['action'] = 'strong_breakout_short'
                else:
                    vote = CryptoEdgeVote.SHORT
                    data['action'] = 'breakout_short'

                return EdgeSignal(
                    edge_name=self.name,
                    vote=vote,
                    confidence=confidence,
                    reason=(
                        f"Bearish breakout: vel={current_velocity:.2f}, "
                        f"acc={acceleration:.4f}, mom={price_mom:.2f}"
                    ),
                    data=data,
                )

            else:
                # Volume surging but price flat — pre-breakout accumulation
                # Weak signal: acceleration is positive but no price confirmation yet
                confidence *= 0.3  # Reduce confidence significantly
                data['action'] = 'pre_breakout'
                return EdgeSignal(
                    edge_name=self.name,
                    vote=CryptoEdgeVote.NEUTRAL,
                    confidence=0.0,
                    reason=f"Pre-breakout: vol surge but no price confirmation",
                    data=data,
                )

        # ---- Early Acceleration Signal ----
        # Acceleration is positive and strong, but velocity hasn't crossed threshold yet
        # This catches the beginning of surges
        if acceleration > 0.02 and abs(price_mom) > 0.5 and not velocity_surging:
            weak_conf = min(0.4, abs(acceleration) * 10)
            direction = 1 if price_mom > 0 else -1
            vote = CryptoEdgeVote.LONG if direction > 0 else CryptoEdgeVote.SHORT
            data['action'] = 'early_acceleration'

            return EdgeSignal(
                edge_name=self.name,
                vote=vote,
                confidence=weak_conf,
                reason=(
                    f"Early acceleration: acc={acceleration:.4f}, "
                    f"vel={current_velocity:.2f}, mom={price_mom:.2f}"
                ),
                data=data,
            )

        # ---- No signal ----
        return EdgeSignal(
            edge_name=self.name,
            vote=CryptoEdgeVote.NEUTRAL,
            confidence=0.0,
            reason=f"No signal: vel={current_velocity:.2f}, acc={acceleration:.4f}",
            data=data,
        )

    def reset(self) -> None:
        self._volumes.clear()
        self._closes.clear()
        self._bar_count.clear()
        self._ema_short.clear()
        self._ema_long.clear()
        self._velocity_history.clear()
        self._last_climax_bar.clear()
