"""
Edge 1: Liquidation Cascade Predictor (LCP)

Predicts forced liquidation cascades on crypto exchanges by analyzing
the relationship between open interest buildup and price movement.

Key insight: When OI accumulates without corresponding price movement,
leveraged positions are building up. These positions have liquidation
levels that, when hit, create mechanical forced selling/buying cascades.

Signal logic:
    - OI rising while price consolidates = leverage coiling (cascade fuel)
    - OI_imbalance = OI_change_rate / price_change_rate (high = coiled spring)
    - Cascade direction = opposite of OI buildup direction:
        * OI built during uptrend -> longs overextended -> cascade DOWN
        * OI built during downtrend -> shorts overextended -> cascade UP
    - Cascade proximity: vol picking up but OI hasn't flushed yet = imminent
    - Confidence = f(OI_imbalance, proximity_to_extremes, vol_ratio)

Expected SR: 1.0-1.5
Correlation with others: Low (unique OI signal, uncorrelated with basis/funding)
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


class LiquidationCascade(CryptoEdge):
    """
    Predict liquidation cascades from open interest buildup patterns.

    When open interest accumulates disproportionately to price movement,
    leveraged positions are building. The eventual unwind creates
    mechanical flow in the opposite direction of the buildup.
    """

    def __init__(
        self,
        lookback: int = 60,                   # OI history for rate estimation
        oi_imbalance_threshold: float = 2.0,   # OI change rate / price change rate ratio
        cascade_vol_trigger: float = 1.5,      # Current vol / avg vol ratio to trigger
        min_oi_change: float = 0.05,           # Minimum 5% OI change over lookback to signal
        warmup: int = 100,                     # Bars before producing signals
    ):
        self._lookback = lookback
        self._oi_imbalance_threshold = oi_imbalance_threshold
        self._cascade_vol_trigger = cascade_vol_trigger
        self._min_oi_change = min_oi_change
        self._warmup = warmup

        # Per-symbol state: prices, volumes, OI
        self._prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )
        self._volumes: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )
        self._oi_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )
        self._bar_count: Dict[str, int] = defaultdict(int)

        # Cached analytics per symbol (updated each bar)
        self._oi_imbalance: Dict[str, float] = defaultdict(float)
        self._buildup_direction: Dict[str, int] = defaultdict(int)  # 1=up, -1=down
        self._vol_ratio: Dict[str, float] = defaultdict(lambda: 1.0)
        self._oi_change_pct: Dict[str, float] = defaultdict(float)

    @property
    def name(self) -> str:
        return "LiquidationCascade"

    @property
    def warmup_bars(self) -> int:
        return self._warmup

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """
        Feed a single bar of data. Expects open_interest via kwargs.
        """
        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)
        self._bar_count[symbol] += 1

        open_interest = kwargs.get('open_interest')
        if open_interest is not None and open_interest > 0:
            self._oi_history[symbol].append(open_interest)

        # Recompute analytics once we have enough data
        if (len(self._prices[symbol]) >= self._lookback
                and len(self._oi_history[symbol]) >= self._lookback):
            self._compute_analytics(symbol)

    def _compute_analytics(self, symbol: str) -> None:
        """Compute OI imbalance, buildup direction, and vol ratio."""
        prices = np.array(self._prices[symbol], dtype=np.float64)
        oi = np.array(self._oi_history[symbol], dtype=np.float64)
        volumes = np.array(self._volumes[symbol], dtype=np.float64)

        # Use the last `lookback` bars
        lb = self._lookback
        prices_lb = prices[-lb:]
        oi_lb = oi[-lb:]
        volumes_lb = volumes[-lb:]

        # --- OI change rate ---
        oi_start = oi_lb[0]
        oi_end = oi_lb[-1]
        if oi_start > 0:
            oi_change_pct = (oi_end - oi_start) / oi_start
        else:
            oi_change_pct = 0.0
        self._oi_change_pct[symbol] = oi_change_pct

        # --- Price change rate (absolute, to measure movement magnitude) ---
        price_start = prices_lb[0]
        price_end = prices_lb[-1]
        if price_start > 0:
            price_change_pct = abs(price_end - price_start) / price_start
        else:
            price_change_pct = 0.0

        # --- OI Imbalance Score ---
        # High ratio = OI accumulating without price moving (coiled spring)
        if price_change_pct > 1e-8:
            self._oi_imbalance[symbol] = abs(oi_change_pct) / price_change_pct
        else:
            # Price flat but OI changing = extremely coiled
            self._oi_imbalance[symbol] = abs(oi_change_pct) * 100.0 if abs(oi_change_pct) > 0.01 else 0.0

        # --- Buildup Direction ---
        # Determine the trend during which OI accumulated
        # Use linear regression slope of prices over lookback
        x = np.arange(lb, dtype=np.float64)
        x_mean = x.mean()
        p_mean = prices_lb.mean()
        slope_num = np.sum((x - x_mean) * (prices_lb - p_mean))
        slope_den = np.sum((x - x_mean) ** 2)
        price_slope = slope_num / slope_den if slope_den > 1e-12 else 0.0

        if oi_change_pct > 0:
            # OI was building up
            if price_slope > 0:
                # OI built during uptrend -> longs accumulated -> cascade DOWN
                self._buildup_direction[symbol] = -1
            else:
                # OI built during downtrend -> shorts accumulated -> cascade UP
                self._buildup_direction[symbol] = 1
        else:
            # OI declining = positions unwinding, no cascade fuel
            self._buildup_direction[symbol] = 0

        # --- Vol Ratio (cascade proximity indicator) ---
        # Compare recent vol to average vol
        recent_window = max(5, lb // 6)
        recent_vol = np.std(np.diff(np.log(prices_lb[-recent_window:]))) if len(prices_lb) > recent_window else 0.0
        avg_vol = np.std(np.diff(np.log(prices_lb))) if len(prices_lb) > 1 else 0.0

        if avg_vol > 1e-10:
            self._vol_ratio[symbol] = recent_vol / avg_vol
        else:
            self._vol_ratio[symbol] = 1.0

    def _price_near_extreme(self, symbol: str) -> float:
        """
        Check if price is near N-period high or low.
        Returns a proximity score 0-1 (1 = at the extreme).
        """
        prices = self._prices[symbol]
        if len(prices) < self._lookback:
            return 0.0

        prices_arr = np.array(prices, dtype=np.float64)
        lb_prices = prices_arr[-self._lookback:]
        current = lb_prices[-1]
        hi = lb_prices.max()
        lo = lb_prices.min()
        span = hi - lo

        if span < 1e-10:
            return 0.0

        # How close to either extreme (0 = middle, 1 = at high or low)
        dist_to_high = abs(current - hi) / span
        dist_to_low = abs(current - lo) / span
        proximity = 1.0 - min(dist_to_high, dist_to_low)

        return max(0.0, min(1.0, proximity))

    def _volume_increasing(self, symbol: str) -> bool:
        """Check if recent volume is increasing (confirming cascade setup)."""
        volumes = self._volumes[symbol]
        if len(volumes) < self._lookback:
            return False

        vol_arr = np.array(volumes, dtype=np.float64)
        recent = vol_arr[-10:]
        earlier = vol_arr[-self._lookback:-10]

        if len(earlier) == 0 or np.mean(earlier) < 1e-10:
            return False

        return float(np.mean(recent)) > float(np.mean(earlier)) * 1.1

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate trading signal from liquidation cascade analysis."""
        # Check we have enough data
        if (len(self._prices[symbol]) < self._lookback
                or len(self._oi_history[symbol]) < self._lookback):
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Insufficient OI or price history",
            )

        oi_imbalance = self._oi_imbalance[symbol]
        buildup_dir = self._buildup_direction[symbol]
        vol_ratio = self._vol_ratio[symbol]
        oi_change = self._oi_change_pct[symbol]
        proximity = self._price_near_extreme(symbol)
        vol_increasing = self._volume_increasing(symbol)

        # --- No signal conditions ---

        # OI hasn't changed enough to indicate meaningful buildup
        if abs(oi_change) < self._min_oi_change:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"OI change too small: {oi_change:.3f}",
                data={'oi_change': oi_change, 'oi_imbalance': oi_imbalance},
            )

        # OI declining = positions flushing, no cascade fuel
        if buildup_dir == 0:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="OI declining, no cascade fuel",
                data={'oi_change': oi_change, 'oi_imbalance': oi_imbalance},
            )

        # Imbalance below threshold = OI and price moving together (normal)
        if oi_imbalance < self._oi_imbalance_threshold:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"OI imbalance below threshold: {oi_imbalance:.2f}",
                data={'oi_imbalance': oi_imbalance, 'threshold': self._oi_imbalance_threshold},
            )

        # --- Signal generation ---

        # Compute confidence from multiple factors
        # 1. OI imbalance strength (capped contribution)
        imbalance_score = min(1.0, (oi_imbalance - self._oi_imbalance_threshold) /
                              (self._oi_imbalance_threshold * 2.0))

        # 2. Proximity to price extreme (cascade more likely near extremes)
        proximity_score = proximity

        # 3. Vol ratio: rising vol = cascade more imminent
        vol_score = min(1.0, max(0.0, (vol_ratio - 1.0) / (self._cascade_vol_trigger - 1.0)))

        # 4. Volume trend bonus
        vol_trend_bonus = 0.15 if vol_increasing else 0.0

        # Composite confidence
        confidence = (
            0.35 * imbalance_score
            + 0.30 * proximity_score
            + 0.25 * vol_score
            + 0.10 * min(1.0, abs(oi_change) / (self._min_oi_change * 4.0))
        ) + vol_trend_bonus

        confidence = max(0.0, min(1.0, confidence))

        # --- Determine vote strength ---

        # HIGH probability cascade: imbalance extreme + vol spiking + near price extreme
        high_prob = (
            oi_imbalance > self._oi_imbalance_threshold * 2.0
            and vol_ratio >= self._cascade_vol_trigger
            and proximity > 0.7
            and vol_increasing
        )

        # Cascade direction is OPPOSITE of buildup direction
        # buildup_dir = -1 means longs built up, cascade will be DOWN, so we SHORT
        # buildup_dir = +1 means shorts built up, cascade will be UP, so we LONG
        cascade_dir = buildup_dir  # Already set as opposite in _compute_analytics

        if cascade_dir > 0:
            # Expecting short squeeze (cascade UP) -> go LONG
            if high_prob:
                vote = CryptoEdgeVote.STRONG_LONG
                confidence = min(1.0, confidence * 1.3)
                reason = (f"HIGH prob short squeeze: imbalance={oi_imbalance:.2f}, "
                          f"vol_ratio={vol_ratio:.2f}, proximity={proximity:.2f}")
            else:
                vote = CryptoEdgeVote.LONG
                reason = (f"Short squeeze setup: imbalance={oi_imbalance:.2f}, "
                          f"vol_ratio={vol_ratio:.2f}")
        else:
            # Expecting long liquidation cascade (cascade DOWN) -> go SHORT
            if high_prob:
                vote = CryptoEdgeVote.STRONG_SHORT
                confidence = min(1.0, confidence * 1.3)
                reason = (f"HIGH prob long liquidation: imbalance={oi_imbalance:.2f}, "
                          f"vol_ratio={vol_ratio:.2f}, proximity={proximity:.2f}")
            else:
                vote = CryptoEdgeVote.SHORT
                reason = (f"Long liquidation setup: imbalance={oi_imbalance:.2f}, "
                          f"vol_ratio={vol_ratio:.2f}")

        return EdgeSignal(
            edge_name=self.name,
            vote=vote,
            confidence=confidence,
            reason=reason,
            data={
                'oi_imbalance': oi_imbalance,
                'oi_change_pct': oi_change,
                'buildup_direction': buildup_dir,
                'cascade_direction': cascade_dir,
                'vol_ratio': vol_ratio,
                'proximity': proximity,
                'vol_increasing': vol_increasing,
                'high_prob': high_prob,
            },
        )

    def reset(self) -> None:
        self._prices.clear()
        self._volumes.clear()
        self._oi_history.clear()
        self._bar_count.clear()
        self._oi_imbalance.clear()
        self._buildup_direction.clear()
        self._vol_ratio.clear()
        self._oi_change_pct.clear()
