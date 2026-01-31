"""
Volume Analyzer for Day Trading Stock Selection.

Analyzes volume patterns to identify:
- Liquidity (can we trade efficiently?)
- Volume anomalies (unusual activity = potential move)
- Accumulation/Distribution (smart money flows)
- Volume momentum (increasing interest)

Key Concepts:
- Relative volume (RVOL) - Today's volume vs average
- Dollar volume - Price × Volume (true liquidity measure)
- Volume-price divergences - Price up on low vol = weak
- Pre-market volume - Early indicator of day's activity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from trading_algo.stock_selector.models import VolumeScore


@dataclass
class PriceBar:
    """OHLCV price bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class VolumeAnalyzer:
    """
    Comprehensive volume analysis for day trading.

    Volume is critical for day trading:
    - High volume = tight spreads, easy entry/exit
    - Volume spikes often precede price moves
    - Accumulation/distribution reveals smart money
    - Pre-market volume predicts day's volatility

    Minimum requirements:
    - 500K average daily volume
    - $10M average dollar volume
    """

    def __init__(
        self,
        min_avg_volume: float = 500_000,        # Minimum shares/day
        min_dollar_volume: float = 10_000_000,  # Minimum $/day
        volume_spike_threshold: float = 2.0,    # 2x normal = spike
        lookback_short: int = 5,
        lookback_long: int = 20,
    ):
        self.min_avg_volume = min_avg_volume
        self.min_dollar_volume = min_dollar_volume
        self.volume_spike_threshold = volume_spike_threshold
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def analyze(
        self,
        bars: List[PriceBar],
        current_volume: Optional[float] = None,
        premarket_volume: Optional[float] = None,
    ) -> Optional[VolumeScore]:
        """
        Analyze volume characteristics.

        Args:
            bars: List of daily OHLCV bars (most recent last)
            current_volume: Today's volume so far (for live analysis)
            premarket_volume: Pre-market volume if available

        Returns:
            VolumeScore with all metrics
        """
        if len(bars) < self.lookback_long:
            return None

        # Average volumes
        avg_vol_20d = self._calculate_avg_volume(bars, self.lookback_long)
        avg_vol_5d = self._calculate_avg_volume(bars, self.lookback_short)

        # Dollar volume
        current_price = bars[-1].close
        avg_dollar_volume = avg_vol_20d * current_price

        # Relative volume
        today_volume = current_volume if current_volume else bars[-1].volume
        relative_volume = today_volume / avg_vol_20d if avg_vol_20d > 0 else 1.0

        # Pre-market ratio
        if premarket_volume and avg_vol_20d > 0:
            # Estimate typical pre-market as 5% of daily
            typical_premarket = avg_vol_20d * 0.05
            premarket_ratio = premarket_volume / typical_premarket if typical_premarket > 0 else 1.0
        else:
            premarket_ratio = 1.0

        # Volume spike detection
        spike_detected, spike_magnitude = self._detect_volume_spike(
            bars, today_volume, avg_vol_20d
        )

        # Accumulation/Distribution
        acc_detected, dist_detected = self._detect_accumulation_distribution(bars)

        # Volume momentum
        vol_momentum = self._calculate_volume_momentum(bars)

        # Smart money indicators
        smart_money = self._analyze_smart_money_patterns(bars)

        # Calculate derived scores
        liquidity_score = self._score_liquidity(avg_vol_20d, avg_dollar_volume)
        volume_momentum_score = self._score_volume_momentum(vol_momentum, relative_volume)
        smart_money_score = self._score_smart_money(smart_money, acc_detected, dist_detected)

        return VolumeScore(
            avg_volume_20d=avg_vol_20d,
            avg_dollar_volume=avg_dollar_volume,
            relative_volume=relative_volume,
            premarket_volume_ratio=premarket_ratio,
            volume_spike_detected=spike_detected,
            volume_spike_magnitude=spike_magnitude,
            accumulation_detected=acc_detected,
            distribution_detected=dist_detected,
            liquidity_score=liquidity_score,
            volume_momentum_score=volume_momentum_score,
            smart_money_score=smart_money_score,
        )

    def _calculate_avg_volume(
        self,
        bars: List[PriceBar],
        lookback: int,
    ) -> float:
        """Calculate average daily volume."""
        recent_bars = bars[-lookback:]
        volumes = [b.volume for b in recent_bars]
        return sum(volumes) / len(volumes) if volumes else 0

    def _detect_volume_spike(
        self,
        bars: List[PriceBar],
        current_volume: float,
        avg_volume: float,
    ) -> Tuple[bool, float]:
        """
        Detect unusual volume spike.

        Uses z-score approach for statistical significance.
        """
        if avg_volume == 0:
            return False, 0.0

        # Calculate volume standard deviation
        volumes = [b.volume for b in bars[-20:]]
        mean_vol = sum(volumes) / len(volumes)
        variance = sum((v - mean_vol) ** 2 for v in volumes) / len(volumes)
        std_vol = math.sqrt(variance) if variance > 0 else 1

        # Z-score of current volume
        z_score = (current_volume - mean_vol) / std_vol if std_vol > 0 else 0

        # Also check simple ratio
        ratio = current_volume / avg_volume

        # Spike if z-score > 2 OR ratio > threshold
        spike_detected = z_score > 2.0 or ratio > self.volume_spike_threshold

        # Magnitude is the larger of z-score or ratio-1
        magnitude = max(z_score, ratio - 1)

        return spike_detected, magnitude

    def _detect_accumulation_distribution(
        self,
        bars: List[PriceBar],
        lookback: int = 10,
    ) -> Tuple[bool, bool]:
        """
        Detect accumulation (buying) or distribution (selling) patterns.

        Accumulation: Price up on above-average volume
        Distribution: Price down on above-average volume
        """
        if len(bars) < lookback + 5:
            return False, False

        recent_bars = bars[-lookback:]
        avg_vol = sum(b.volume for b in bars[-(lookback+20):-lookback]) / 20

        # Count up days and down days with volume
        up_volume = 0
        down_volume = 0
        total_high_vol_days = 0

        for i in range(1, len(recent_bars)):
            vol_ratio = recent_bars[i].volume / avg_vol if avg_vol > 0 else 1
            price_change = recent_bars[i].close - recent_bars[i-1].close

            if vol_ratio > 1.2:  # Above average volume
                total_high_vol_days += 1
                if price_change > 0:
                    up_volume += recent_bars[i].volume
                else:
                    down_volume += recent_bars[i].volume

        # Accumulation: More up-volume than down-volume on high vol days
        total_volume = up_volume + down_volume
        if total_volume > 0:
            up_ratio = up_volume / total_volume
            accumulation = up_ratio > 0.65 and total_high_vol_days >= 3
            distribution = up_ratio < 0.35 and total_high_vol_days >= 3
        else:
            accumulation = False
            distribution = False

        return accumulation, distribution

    def _calculate_volume_momentum(
        self,
        bars: List[PriceBar],
    ) -> float:
        """
        Calculate volume momentum (-1 to 1).

        Positive = volume increasing
        Negative = volume decreasing
        """
        if len(bars) < 20:
            return 0.0

        # Compare recent volume to earlier volume
        recent_vol = sum(b.volume for b in bars[-5:]) / 5
        earlier_vol = sum(b.volume for b in bars[-20:-5]) / 15

        if earlier_vol == 0:
            return 0.0

        ratio = recent_vol / earlier_vol

        # Convert to -1 to 1 scale
        # ratio of 2 = strong positive momentum
        # ratio of 0.5 = strong negative momentum
        momentum = (ratio - 1) / 1  # ±100% change = ±1
        momentum = max(-1, min(1, momentum))

        return momentum

    def _analyze_smart_money_patterns(
        self,
        bars: List[PriceBar],
        lookback: int = 10,
    ) -> dict:
        """
        Analyze patterns that suggest institutional activity.

        Smart money indicators:
        - Large volume on small price moves (accumulation without moving price)
        - Volume clusters at specific price levels
        - End-of-day volume spikes (institutional rebalancing)
        """
        if len(bars) < lookback:
            return {'score': 0.5}

        recent_bars = bars[-lookback:]

        # Calculate average metrics
        avg_vol = sum(b.volume for b in recent_bars) / len(recent_bars)
        avg_range = sum((b.high - b.low) for b in recent_bars) / len(recent_bars)

        # Look for high volume, low range days (stealth accumulation)
        stealth_days = 0
        for bar in recent_bars:
            range_pct = (bar.high - bar.low) / bar.close if bar.close > 0 else 0
            vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1

            # High volume (>1.5x) but low range (<0.5x average)
            if vol_ratio > 1.5 and range_pct < avg_range * 0.5:
                stealth_days += 1

        # Volume consistency (institutions spread orders)
        vol_std = math.sqrt(
            sum((b.volume - avg_vol) ** 2 for b in recent_bars) / len(recent_bars)
        )
        vol_cv = vol_std / avg_vol if avg_vol > 0 else 1

        # Lower CV = more consistent volume = possible institutional
        consistency_score = max(0, 1 - vol_cv)

        # Combined smart money score
        stealth_score = stealth_days / lookback
        smart_money_score = (stealth_score * 0.6 + consistency_score * 0.4)

        return {
            'score': smart_money_score,
            'stealth_days': stealth_days,
            'volume_consistency': consistency_score,
        }

    def _score_liquidity(
        self,
        avg_volume: float,
        avg_dollar_volume: float,
    ) -> float:
        """
        Score liquidity for day trading (0-100).

        Requirements:
        - Minimum 500K shares/day
        - Minimum $10M dollar volume/day
        - More is better up to a point
        """
        # Volume score
        if avg_volume < self.min_avg_volume * 0.5:
            vol_score = 0
        elif avg_volume < self.min_avg_volume:
            vol_score = (avg_volume / self.min_avg_volume) * 50
        elif avg_volume < self.min_avg_volume * 5:
            # Good volume
            vol_score = 50 + (avg_volume - self.min_avg_volume) / (self.min_avg_volume * 4) * 30
        else:
            # Excellent volume
            vol_score = 80 + min(20, (avg_volume / (self.min_avg_volume * 10)) * 20)

        # Dollar volume score
        if avg_dollar_volume < self.min_dollar_volume * 0.5:
            dv_score = 0
        elif avg_dollar_volume < self.min_dollar_volume:
            dv_score = (avg_dollar_volume / self.min_dollar_volume) * 50
        elif avg_dollar_volume < self.min_dollar_volume * 10:
            dv_score = 50 + (avg_dollar_volume - self.min_dollar_volume) / (self.min_dollar_volume * 9) * 30
        else:
            dv_score = 80 + min(20, (avg_dollar_volume / (self.min_dollar_volume * 50)) * 20)

        # Combined score (both matter)
        return min(100, (vol_score * 0.4 + dv_score * 0.6))

    def _score_volume_momentum(
        self,
        vol_momentum: float,
        relative_volume: float,
    ) -> float:
        """
        Score volume momentum (0-100).

        Higher relative volume and positive momentum = better
        """
        # Relative volume component (0-50 points)
        if relative_volume < 0.5:
            rvol_score = relative_volume * 20  # Low volume = bad
        elif relative_volume < 1.0:
            rvol_score = 10 + (relative_volume - 0.5) * 40
        elif relative_volume < 2.0:
            rvol_score = 30 + (relative_volume - 1.0) * 15
        else:
            rvol_score = 45 + min(5, (relative_volume - 2.0) * 2.5)

        # Momentum component (0-50 points)
        # Positive momentum is good, but not too extreme
        if vol_momentum < -0.5:
            mom_score = 0  # Declining interest
        elif vol_momentum < 0:
            mom_score = 10 + (vol_momentum + 0.5) * 20
        elif vol_momentum < 0.5:
            mom_score = 20 + vol_momentum * 50  # Increasing interest
        else:
            mom_score = 45 + min(5, (vol_momentum - 0.5) * 10)

        return min(100, rvol_score + mom_score)

    def _score_smart_money(
        self,
        smart_money: dict,
        accumulation: bool,
        distribution: bool,
    ) -> float:
        """
        Score smart money indicators (0-100).
        """
        base_score = smart_money.get('score', 0.5) * 60

        # Bonus for accumulation pattern
        if accumulation:
            base_score += 30
        elif distribution:
            base_score += 10  # Distribution can also be tradeable

        return min(100, base_score)
