"""
Edge 3: Cross-Exchange Divergence (CED)

Detects price divergence between perp and spot (or across exchanges)
and trades convergence/catch-up.

Key insight: Different exchanges have different user bases and liquidity.
When one venue's price diverges from another, information is propagating
and the lagging venue will catch up. On hourly bars this captures cases
where one venue moves 1-2 hours before others during volatile periods.

Works in two modes:
1. Spot-Perp Divergence (always available):
   Perp and spot on Binance can diverge beyond just the basis.
   Uses perp price (price arg) vs spot price (kwargs['spot_price']).
2. Cross-Exchange Divergence (optional):
   When kwargs['alt_exchange_price'] is available, compares primary
   exchange price against an alternative exchange.

Signal components:
- Price divergence z-score (mean reversion when extreme)
- Momentum divergence (ROC comparison between venues)
- Volume-weighted divergence confirmation

Expected SR: 1.0-1.5
Correlation with PBMR: Moderate (both use spot/perp but CED focuses on
    short-term divergence dynamics, not OU mean reversion of the basis level)
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


class CrossExchangeDivergence(CryptoEdge):
    """
    Trade divergence between perp/spot (or cross-exchange) prices.

    Two signal types:
    1. Ratio Z-Score: When the price ratio deviates from its rolling
       mean, expect convergence (mean reversion).
    2. Momentum Divergence: When one venue's ROC significantly exceeds
       the other, the lagging venue will catch up (momentum).

    Both signals are volume-weighted for confirmation.
    """

    def __init__(
        self,
        lookback: int = 72,                       # 3 days of hourly bars for ratio stats
        entry_z: float = 1.8,                      # Z-score to enter convergence trade
        exit_z: float = 0.5,                       # Z-score to exit
        momentum_window: int = 12,                 # 12 hours for ROC
        momentum_divergence_threshold: float = 0.02,  # 2% difference in ROC
        warmup: int = 72,
    ):
        self._lookback = lookback
        self._entry_z = entry_z
        self._exit_z = exit_z
        self._momentum_window = momentum_window
        self._momentum_divergence_threshold = momentum_divergence_threshold
        self._warmup = warmup

        # Per-symbol price series
        self._perp_prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )
        self._spot_prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )
        self._alt_prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )

        # Per-symbol volume series (perp and spot)
        self._perp_volumes: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )
        self._spot_volumes: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )

        # Per-symbol ratio history (perp/spot or primary/alt)
        self._ratio_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback + 50)
        )

        self._bar_count: Dict[str, int] = defaultdict(int)
        self._positions: Dict[str, int] = defaultdict(int)  # 1=long, -1=short, 0=flat

    @property
    def name(self) -> str:
        return "CrossExchangeDivergence"

    @property
    def warmup_bars(self) -> int:
        return self._warmup

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """
        Feed a single bar of data.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timestamp: Bar timestamp
            price: Perp price (close)
            volume: Perp volume
            **kwargs:
                spot_price: Spot price on same exchange (required)
                spot_volume: Spot volume (optional, defaults to perp volume)
                alt_exchange_price: Price on alternative exchange (optional)
        """
        spot_price = kwargs.get('spot_price')
        spot_volume = kwargs.get('spot_volume', volume)
        alt_price = kwargs.get('alt_exchange_price')

        self._perp_prices[symbol].append(price)
        self._perp_volumes[symbol].append(volume)

        if spot_price and spot_price > 0:
            self._spot_prices[symbol].append(spot_price)
            self._spot_volumes[symbol].append(spot_volume)

            # Compute price ratio (perp / spot)
            ratio = price / spot_price
            self._ratio_history[symbol].append(ratio)

        if alt_price and alt_price > 0:
            self._alt_prices[symbol].append(alt_price)

        self._bar_count[symbol] += 1

    def _compute_ratio_zscore(self, symbol: str) -> Optional[float]:
        """Compute z-score of the perp/spot price ratio vs rolling stats."""
        ratios = self._ratio_history[symbol]
        if len(ratios) < self._lookback:
            return None

        arr = np.array(ratios, dtype=np.float64)
        window = arr[-self._lookback:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)

        if std < 1e-10:
            return None

        current = arr[-1]
        return float((current - mean) / std)

    def _compute_momentum_divergence(self, symbol: str) -> Optional[float]:
        """
        Compute difference in rate-of-change (ROC) between perp and spot.

        Returns perp_roc - spot_roc:
            Positive = perp outpacing spot (perp overextended)
            Negative = spot outpacing perp (perp will catch up)
        """
        perp = self._perp_prices[symbol]
        spot = self._spot_prices[symbol]

        if (len(perp) < self._momentum_window + 1
                or len(spot) < self._momentum_window + 1):
            return None

        perp_arr = np.array(perp, dtype=np.float64)
        spot_arr = np.array(spot, dtype=np.float64)

        # ROC = (current - past) / past
        perp_current = perp_arr[-1]
        perp_past = perp_arr[-(self._momentum_window + 1)]
        spot_current = spot_arr[-1]
        spot_past = spot_arr[-(self._momentum_window + 1)]

        if perp_past <= 0 or spot_past <= 0:
            return None

        perp_roc = (perp_current - perp_past) / perp_past
        spot_roc = (spot_current - spot_past) / spot_past

        return float(perp_roc - spot_roc)

    def _compute_volume_ratio(self, symbol: str) -> float:
        """
        Compute relative volume ratio (perp vs spot) for weighting.

        Returns a value centered around 1.0. Higher means perp volume
        dominates, which gives more weight to perp-driven signals.
        """
        perp_vols = self._perp_volumes[symbol]
        spot_vols = self._spot_volumes[symbol]

        if len(perp_vols) < 2 or len(spot_vols) < 2:
            return 1.0

        recent_perp = float(np.mean(list(perp_vols)[-12:])) if len(perp_vols) >= 12 else float(np.mean(list(perp_vols)))
        recent_spot = float(np.mean(list(spot_vols)[-12:])) if len(spot_vols) >= 12 else float(np.mean(list(spot_vols)))

        if recent_spot <= 0:
            return 1.0

        return recent_perp / recent_spot

    def _compute_alt_exchange_zscore(self, symbol: str) -> Optional[float]:
        """
        If alt exchange data is available, compute z-score of
        primary/alt price ratio.
        """
        perp = self._perp_prices[symbol]
        alt = self._alt_prices[symbol]

        if len(perp) < self._lookback or len(alt) < self._lookback:
            return None

        perp_arr = np.array(perp, dtype=np.float64)[-self._lookback:]
        alt_arr = np.array(alt, dtype=np.float64)[-self._lookback:]

        ratios = perp_arr / alt_arr
        mean = np.mean(ratios)
        std = np.std(ratios, ddof=1)

        if std < 1e-10:
            return None

        return float((ratios[-1] - mean) / std)

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate trading signal from cross-exchange/venue divergence."""
        if self._bar_count[symbol] < self._warmup:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Warming up",
            )

        # --- Component 1: Price ratio z-score ---
        z_score = self._compute_ratio_zscore(symbol)
        if z_score is None:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Insufficient ratio data",
            )

        # --- Component 2: Momentum divergence ---
        mom_div = self._compute_momentum_divergence(symbol)

        # --- Component 3: Volume ratio ---
        vol_ratio = self._compute_volume_ratio(symbol)
        # Normalize volume ratio to a weight between 0.5 and 1.5
        vol_weight = max(0.5, min(1.5, vol_ratio))

        # --- Component 4: Alt exchange z-score (if available) ---
        alt_z = self._compute_alt_exchange_zscore(symbol)

        # --- Position management ---
        current_pos = self._positions[symbol]

        # Exit check: if z-score has reverted near mean
        if current_pos != 0 and abs(z_score) < self._exit_z:
            self._positions[symbol] = 0
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Exit: z={z_score:.2f} reverted to mean",
                data={'z_score': z_score, 'action': 'exit'},
            )

        # --- Signal Generation ---
        # Signal A: Z-score convergence trade (mean reversion)
        z_signal = 0
        z_confidence = 0.0
        if abs(z_score) > self._entry_z:
            # Ratio too high (perp premium) -> short perp (expect convergence)
            # Ratio too low (perp discount) -> long perp (expect convergence)
            z_signal = -1 if z_score > 0 else 1
            z_confidence = min(1.0, abs(z_score) / 3.5) * vol_weight

        # Signal B: Momentum divergence trade (catch-up)
        mom_signal = 0
        mom_confidence = 0.0
        if mom_div is not None and abs(mom_div) > self._momentum_divergence_threshold:
            # perp ROC >> spot ROC -> perp overextended -> short perp
            # spot ROC >> perp ROC -> perp will catch up -> long perp
            mom_signal = -1 if mom_div > 0 else 1
            mom_confidence = min(1.0, abs(mom_div) / 0.05) * vol_weight * 0.7

        # Signal C: Alt exchange z-score (additional confirmation)
        alt_signal = 0
        alt_confidence = 0.0
        if alt_z is not None and abs(alt_z) > self._entry_z:
            alt_signal = -1 if alt_z > 0 else 1
            alt_confidence = min(1.0, abs(alt_z) / 3.5) * 0.5

        # --- Combine signals ---
        # Weighted combination of all active signal components
        total_weight = 0.0
        weighted_direction = 0.0

        if z_signal != 0:
            weighted_direction += z_signal * z_confidence * 1.0  # Primary weight
            total_weight += z_confidence * 1.0

        if mom_signal != 0:
            weighted_direction += mom_signal * mom_confidence * 0.7  # Secondary weight
            total_weight += mom_confidence * 0.7

        if alt_signal != 0:
            weighted_direction += alt_signal * alt_confidence * 0.5  # Tertiary weight
            total_weight += alt_confidence * 0.5

        if total_weight < 1e-6:
            # No active signals
            if current_pos != 0:
                vote = CryptoEdgeVote.LONG if current_pos > 0 else CryptoEdgeVote.SHORT
                return EdgeSignal(
                    edge_name=self.name,
                    vote=vote,
                    confidence=0.1,
                    reason=f"Holding: z={z_score:.2f}",
                    data={'z_score': z_score, 'action': 'hold'},
                )
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"No signal: z={z_score:.2f}",
                data={'z_score': z_score},
            )

        # Final direction and confidence
        combined_direction = weighted_direction / total_weight
        combined_confidence = min(1.0, total_weight / 1.5)

        # Determine vote
        if abs(combined_direction) < 0.3:
            # Conflicting signals, stay out
            if current_pos != 0:
                vote = CryptoEdgeVote.LONG if current_pos > 0 else CryptoEdgeVote.SHORT
                return EdgeSignal(
                    edge_name=self.name,
                    vote=vote,
                    confidence=combined_confidence * 0.3,
                    reason=f"Mixed signals: z={z_score:.2f}, mom_div={mom_div}",
                    data={
                        'z_score': z_score,
                        'momentum_divergence': mom_div,
                        'alt_z': alt_z,
                        'action': 'hold_reduced',
                    },
                )
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Conflicting: z={z_score:.2f}, mom_div={mom_div}",
                data={'z_score': z_score, 'momentum_divergence': mom_div, 'alt_z': alt_z},
            )

        # Strong or normal signal
        direction = 1 if combined_direction > 0 else -1
        is_strong = abs(combined_direction) > 0.7 and combined_confidence > 0.5

        if direction == 1:
            # Concordance: z-score AND momentum both say go long
            concordant = (z_signal == 1 and mom_signal == 1)

            if is_strong or concordant:
                vote = CryptoEdgeVote.STRONG_LONG
                final_confidence = min(1.0, combined_confidence * 1.3)
            else:
                vote = CryptoEdgeVote.LONG
                final_confidence = combined_confidence

            self._positions[symbol] = 1
            action = 'entry_long' if current_pos <= 0 else 'hold_long'

        else:
            concordant = (z_signal == -1 and mom_signal == -1)

            if is_strong or concordant:
                vote = CryptoEdgeVote.STRONG_SHORT
                final_confidence = min(1.0, combined_confidence * 1.3)
            else:
                vote = CryptoEdgeVote.SHORT
                final_confidence = combined_confidence

            self._positions[symbol] = -1
            action = 'entry_short' if current_pos >= 0 else 'hold_short'

        return EdgeSignal(
            edge_name=self.name,
            vote=vote,
            confidence=final_confidence,
            reason=(
                f"{'Strong ' if is_strong else ''}"
                f"{'Long' if direction == 1 else 'Short'}: "
                f"z={z_score:.2f}, mom_div={f'{mom_div:.4f}' if mom_div else 'N/A'}, "
                f"vol_w={vol_weight:.2f}"
            ),
            data={
                'z_score': z_score,
                'momentum_divergence': mom_div,
                'alt_z': alt_z,
                'volume_ratio': vol_ratio,
                'volume_weight': vol_weight,
                'combined_direction': combined_direction,
                'combined_confidence': combined_confidence,
                'z_signal': z_signal,
                'mom_signal': mom_signal,
                'alt_signal': alt_signal,
                'action': action,
            },
        )

    def reset(self) -> None:
        self._perp_prices.clear()
        self._spot_prices.clear()
        self._alt_prices.clear()
        self._perp_volumes.clear()
        self._spot_volumes.clear()
        self._ratio_history.clear()
        self._bar_count.clear()
        self._positions.clear()
