"""
Edge 5: Volume Flow Detector (VFD)

Detects institutional/whale accumulation and distribution patterns by
analyzing the relationship between volume and price movement.

Key mechanisms:
  - Volume-Price Ratio (VPR): High volume absorbed without moving price
    signals "smart money" accumulation.
  - Accumulation/Distribution Line: Classic Chaikin A/D using intra-bar
    position of close relative to high-low range.
  - On-Balance Volume (OBV) Divergence: OBV trending opposite to price
    signals hidden accumulation or distribution.
  - Flow Imbalance: Z-scored ratio of buying volume (close > open) vs
    selling volume (close < open).

Signal logic:
  - OBV divergence + high VPR + accumulation -> LONG (smart money buying)
  - OBV divergence + low VPR + distribution -> SHORT (smart money selling)
  - Confidence = f(obv_divergence_magnitude, vpr_z, flow_imbalance_z)

Expected SR: 0.8-1.2
Correlation with others: Low (volume-only signal, orthogonal to price-based edges)
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


class VolumeFlowDetector(CryptoEdge):
    """
    Detect smart-money accumulation/distribution via volume-price analysis.

    Combines OBV divergence, volume-price ratio, and flow imbalance to
    identify when large players are quietly building or unwinding positions
    before price reflects their activity.
    """

    def __init__(
        self,
        lookback: int = 72,                    # 3 days of hourly bars
        obv_divergence_window: int = 48,       # 2 days for OBV trend
        vpr_window: int = 24,                  # 1 day for volume-price ratio
        flow_imbalance_threshold: float = 0.3, # 30% imbalance triggers signal
        ad_weight: float = 0.3,                # Weight for A/D component
        obv_weight: float = 0.4,               # Weight for OBV divergence
        flow_weight: float = 0.3,              # Weight for flow imbalance
        min_warmup: int = 120,                 # Bars before producing signals
    ):
        self._lookback = lookback
        self._obv_divergence_window = obv_divergence_window
        self._vpr_window = vpr_window
        self._flow_imbalance_threshold = flow_imbalance_threshold
        self._ad_weight = ad_weight
        self._obv_weight = obv_weight
        self._flow_weight = flow_weight
        self._min_warmup = min_warmup

        maxlen = lookback + 60

        # Per-symbol price/volume history
        self._closes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._highs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._lows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._opens: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

        # Derived indicators
        self._obv: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._ad_line: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._vpr: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._flow_imbalance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

        # Running state
        self._bar_count: Dict[str, int] = defaultdict(int)
        self._buy_vol_sum: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback))
        self._sell_vol_sum: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback))

    @property
    def name(self) -> str:
        return "VolumeFlowDetector"

    @property
    def warmup_bars(self) -> int:
        return self._min_warmup

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """
        Feed a single bar. Expects kwargs with 'high' and 'low'.
        Uses previous close as open proxy if 'open' not supplied.
        """
        high = kwargs.get('high', price)
        low = kwargs.get('low', price)

        # Infer open from previous close if not available
        if self._closes[symbol]:
            open_price = self._closes[symbol][-1]
        else:
            open_price = price

        self._closes[symbol].append(price)
        self._highs[symbol].append(high)
        self._lows[symbol].append(low)
        self._volumes[symbol].append(volume)
        self._opens[symbol].append(open_price)
        self._bar_count[symbol] += 1

        # ---- Update OBV ----
        if len(self._closes[symbol]) >= 2:
            prev_close = self._closes[symbol][-2]
            prev_obv = self._obv[symbol][-1] if self._obv[symbol] else 0.0
            if price > prev_close:
                new_obv = prev_obv + volume
            elif price < prev_close:
                new_obv = prev_obv - volume
            else:
                new_obv = prev_obv
            self._obv[symbol].append(new_obv)
        else:
            self._obv[symbol].append(volume)

        # ---- Update Accumulation/Distribution Line ----
        hl_range = high - low
        if hl_range > 1e-10:
            clv = ((price - low) - (high - price)) / hl_range  # Close Location Value
            ad_volume = clv * volume
        else:
            ad_volume = 0.0

        prev_ad = self._ad_line[symbol][-1] if self._ad_line[symbol] else 0.0
        self._ad_line[symbol].append(prev_ad + ad_volume)

        # ---- Update Volume-Price Ratio (VPR) ----
        if len(self._closes[symbol]) >= 2:
            prev_close = self._closes[symbol][-2]
            abs_change = abs(price - prev_close)
            if abs_change > 1e-10:
                vpr = volume / abs_change
            else:
                # Price didn't move but volume occurred — strong absorption
                vpr = volume * 10.0  # High VPR indicates absorption
        else:
            vpr = 0.0
        self._vpr[symbol].append(vpr)

        # ---- Update Flow Imbalance ----
        # Buying bar: close > open, Selling bar: close < open
        if price > open_price:
            self._buy_vol_sum[symbol].append(volume)
            self._sell_vol_sum[symbol].append(0.0)
        elif price < open_price:
            self._buy_vol_sum[symbol].append(0.0)
            self._sell_vol_sum[symbol].append(volume)
        else:
            # Neutral bar — split volume
            self._buy_vol_sum[symbol].append(volume * 0.5)
            self._sell_vol_sum[symbol].append(volume * 0.5)

        total_buy = sum(self._buy_vol_sum[symbol])
        total_sell = sum(self._sell_vol_sum[symbol])
        total = total_buy + total_sell
        if total > 0:
            imbalance = (total_buy - total_sell) / total
        else:
            imbalance = 0.0
        self._flow_imbalance[symbol].append(imbalance)

    def _compute_obv_divergence(self, symbol: str) -> float:
        """
        Compute OBV-price divergence.

        Returns positive value when OBV trends up while price trends down
        (bullish divergence / accumulation), and negative value when OBV
        trends down while price trends up (bearish divergence / distribution).
        """
        window = self._obv_divergence_window
        obv = self._obv[symbol]
        closes = self._closes[symbol]

        if len(obv) < window or len(closes) < window:
            return 0.0

        obv_arr = np.array(list(obv)[-window:], dtype=np.float64)
        price_arr = np.array(list(closes)[-window:], dtype=np.float64)

        # Compute linear regression slopes (normalized)
        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.var(x, ddof=0)
        if x_var < 1e-12:
            return 0.0

        # OBV slope (normalized by mean OBV)
        obv_mean = np.mean(obv_arr)
        if abs(obv_mean) > 1e-10:
            obv_slope = np.sum((x - x_mean) * (obv_arr - np.mean(obv_arr))) / (x_var * window)
            obv_slope_norm = obv_slope / abs(obv_mean)
        else:
            obv_slope_norm = 0.0

        # Price slope (normalized by mean price)
        price_mean = np.mean(price_arr)
        if abs(price_mean) > 1e-10:
            price_slope = np.sum((x - x_mean) * (price_arr - np.mean(price_arr))) / (x_var * window)
            price_slope_norm = price_slope / abs(price_mean)
        else:
            price_slope_norm = 0.0

        # Divergence = OBV direction minus price direction
        # Positive = OBV up / price down (bullish accumulation)
        # Negative = OBV down / price up (bearish distribution)
        divergence = obv_slope_norm - price_slope_norm
        return divergence

    def _compute_vpr_zscore(self, symbol: str) -> float:
        """Compute z-score of current VPR relative to recent history."""
        window = self._vpr_window
        vpr = self._vpr[symbol]

        if len(vpr) < window:
            return 0.0

        vpr_arr = np.array(list(vpr)[-window:], dtype=np.float64)
        mean = np.mean(vpr_arr)
        std = np.std(vpr_arr, ddof=1)

        if std < 1e-10:
            return 0.0

        return (vpr_arr[-1] - mean) / std

    def _compute_flow_imbalance_zscore(self, symbol: str) -> float:
        """Compute z-score of current flow imbalance."""
        fi = self._flow_imbalance[symbol]
        if len(fi) < self._lookback:
            window = len(fi)
        else:
            window = self._lookback

        if window < 10:
            return 0.0

        fi_arr = np.array(list(fi)[-window:], dtype=np.float64)
        mean = np.mean(fi_arr)
        std = np.std(fi_arr, ddof=1)

        if std < 1e-10:
            return 0.0

        return (fi_arr[-1] - mean) / std

    def _compute_ad_trend(self, symbol: str) -> float:
        """Compute A/D line trend direction and magnitude (normalized slope)."""
        ad = self._ad_line[symbol]
        window = min(self._obv_divergence_window, len(ad))

        if window < 10:
            return 0.0

        ad_arr = np.array(list(ad)[-window:], dtype=np.float64)
        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.var(x, ddof=0)

        if x_var < 1e-12:
            return 0.0

        ad_mean = np.mean(ad_arr)
        slope = np.sum((x - x_mean) * (ad_arr - ad_mean)) / (x_var * window)

        # Normalize by mean absolute A/D to get comparable magnitude
        abs_mean = np.mean(np.abs(ad_arr))
        if abs_mean > 1e-10:
            return slope / abs_mean
        return 0.0

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate trading signal from volume flow analysis."""
        if self._bar_count[symbol] < self._min_warmup:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Warming up",
            )

        # Compute all components
        obv_div = self._compute_obv_divergence(symbol)
        vpr_z = self._compute_vpr_zscore(symbol)
        flow_z = self._compute_flow_imbalance_zscore(symbol)
        ad_trend = self._compute_ad_trend(symbol)

        # Current flow imbalance (raw)
        current_imbalance = self._flow_imbalance[symbol][-1] if self._flow_imbalance[symbol] else 0.0

        # ---- Signal Logic ----
        # Accumulation signal: OBV diverges bullishly, high VPR (absorption),
        # flow imbalance tilts positive, A/D trending up
        accumulation_score = (
            self._obv_weight * max(0, obv_div * 10)
            + self._ad_weight * max(0, ad_trend * 5)
            + self._flow_weight * max(0, flow_z)
        )

        # Distribution signal: OBV diverges bearishly, low VPR,
        # flow imbalance tilts negative, A/D trending down
        distribution_score = (
            self._obv_weight * max(0, -obv_div * 10)
            + self._ad_weight * max(0, -ad_trend * 5)
            + self._flow_weight * max(0, -flow_z)
        )

        # VPR boost: High VPR means volume absorbed without price move
        # This amplifies accumulation signals (smart money buying quietly)
        vpr_boost = 1.0 + max(0, vpr_z * 0.2)

        # Apply VPR boost asymmetrically
        # High VPR + accumulation = smart money buying quietly -> stronger LONG
        # Low VPR + distribution = noisy selling -> stronger SHORT
        if vpr_z > 1.0:
            accumulation_score *= vpr_boost
        elif vpr_z < -1.0:
            distribution_score *= (1.0 + abs(vpr_z) * 0.2)

        # Compute net score and confidence
        net_score = accumulation_score - distribution_score
        raw_confidence = min(1.0, (abs(net_score) / 3.0))

        # Additional confidence from flow imbalance exceeding threshold
        imbalance_bonus = 0.0
        if abs(current_imbalance) > self._flow_imbalance_threshold:
            imbalance_bonus = 0.15

        confidence = min(1.0, raw_confidence + imbalance_bonus)

        data = {
            'obv_divergence': round(obv_div, 6),
            'vpr_z': round(vpr_z, 3),
            'flow_imbalance_z': round(flow_z, 3),
            'flow_imbalance_raw': round(current_imbalance, 4),
            'ad_trend': round(ad_trend, 6),
            'accumulation_score': round(accumulation_score, 4),
            'distribution_score': round(distribution_score, 4),
            'net_score': round(net_score, 4),
        }

        # Determine vote
        if net_score > 1.5 and obv_div > 0 and confidence > 0.4:
            vote = CryptoEdgeVote.STRONG_LONG
            reason = (
                f"Strong accumulation: OBV_div={obv_div:.4f}, "
                f"VPR_z={vpr_z:.2f}, flow_z={flow_z:.2f}"
            )
        elif net_score > 0.5 and confidence > 0.2:
            vote = CryptoEdgeVote.LONG
            reason = (
                f"Accumulation detected: OBV_div={obv_div:.4f}, "
                f"flow_imb={current_imbalance:.3f}"
            )
        elif net_score < -1.5 and obv_div < 0 and confidence > 0.4:
            vote = CryptoEdgeVote.STRONG_SHORT
            reason = (
                f"Strong distribution: OBV_div={obv_div:.4f}, "
                f"VPR_z={vpr_z:.2f}, flow_z={flow_z:.2f}"
            )
        elif net_score < -0.5 and confidence > 0.2:
            vote = CryptoEdgeVote.SHORT
            reason = (
                f"Distribution detected: OBV_div={obv_div:.4f}, "
                f"flow_imb={current_imbalance:.3f}"
            )
        else:
            vote = CryptoEdgeVote.NEUTRAL
            confidence = 0.0
            reason = f"No clear flow signal: net={net_score:.3f}"

        return EdgeSignal(
            edge_name=self.name,
            vote=vote,
            confidence=confidence,
            reason=reason,
            data=data,
        )

    def reset(self) -> None:
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()
        self._volumes.clear()
        self._opens.clear()
        self._obv.clear()
        self._ad_line.clear()
        self._vpr.clear()
        self._flow_imbalance.clear()
        self._bar_count.clear()
        self._buy_vol_sum.clear()
        self._sell_vol_sum.clear()
