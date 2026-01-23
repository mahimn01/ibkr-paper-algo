"""
Technical Analyzer for Day Trading Stock Selection.

Analyzes chart patterns and technical setups:
- Support/Resistance levels
- Breakout/Breakdown potential
- Consolidation patterns
- Price action quality

Key Concepts:
- Key levels act as magnets and barriers
- Consolidation precedes expansion
- Volume confirms technical moves
- Clean charts are easier to trade
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from trading_algo.stock_selector.models import TechnicalScore


@dataclass
class PriceBar:
    """OHLCV price bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalAnalyzer:
    """
    Technical analysis for day trading setups.

    Identifies:
    - Key support/resistance levels
    - Breakout setups (consolidation + pressure)
    - Reversal patterns
    - Price action quality
    """

    def __init__(
        self,
        support_resistance_lookback: int = 60,
        consolidation_threshold: float = 0.03,  # 3% range = consolidation
    ):
        self.sr_lookback = support_resistance_lookback
        self.consolidation_threshold = consolidation_threshold

    def analyze(self, bars: List[PriceBar]) -> Optional[TechnicalScore]:
        """
        Analyze technical setup.

        Args:
            bars: Daily OHLCV bars (most recent last)

        Returns:
            TechnicalScore with all metrics
        """
        if len(bars) < 30:
            return None

        current_price = bars[-1].close

        # Find support/resistance levels
        supports, resistances = self._find_support_resistance(bars)

        # Distance to nearest levels
        dist_to_resistance = self._distance_to_level(current_price, resistances, 'above')
        dist_to_support = self._distance_to_level(current_price, supports, 'below')

        # At key level?
        at_key_level = dist_to_resistance < 0.01 or dist_to_support < 0.01

        # Consolidation analysis
        consol_days, range_pct = self._analyze_consolidation(bars)

        # Breakout/breakdown potential
        breakout_pot = self._calculate_breakout_potential(bars, resistances)
        breakdown_pot = self._calculate_breakdown_potential(bars, supports)

        # Price action patterns
        higher_highs = self._check_higher_highs(bars)
        lower_lows = self._check_lower_lows(bars)
        inside_day = self._is_inside_day(bars)

        # Derived scores
        setup_score = self._score_technical_setup(
            dist_to_resistance, dist_to_support, at_key_level, consol_days
        )
        breakout_score = self._score_breakout(breakout_pot, consol_days, range_pct)
        reversal_score = self._score_reversal(bars, dist_to_support, dist_to_resistance)

        return TechnicalScore(
            distance_to_resistance=dist_to_resistance,
            distance_to_support=dist_to_support,
            at_key_level=at_key_level,
            consolidation_days=consol_days,
            breakout_potential=breakout_pot,
            breakdown_potential=breakdown_pot,
            higher_highs=higher_highs,
            lower_lows=lower_lows,
            inside_day=inside_day,
            technical_setup_score=setup_score,
            breakout_score=breakout_score,
            reversal_score=reversal_score,
        )

    def _find_support_resistance(
        self,
        bars: List[PriceBar],
    ) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels using swing points.

        Returns:
            (supports, resistances) - Lists of price levels
        """
        if len(bars) < 10:
            return [], []

        lookback_bars = bars[-self.sr_lookback:]

        # Find swing highs (local maxima)
        swing_highs = []
        for i in range(2, len(lookback_bars) - 2):
            if (lookback_bars[i].high > lookback_bars[i-1].high and
                lookback_bars[i].high > lookback_bars[i-2].high and
                lookback_bars[i].high > lookback_bars[i+1].high and
                lookback_bars[i].high > lookback_bars[i+2].high):
                swing_highs.append(lookback_bars[i].high)

        # Find swing lows (local minima)
        swing_lows = []
        for i in range(2, len(lookback_bars) - 2):
            if (lookback_bars[i].low < lookback_bars[i-1].low and
                lookback_bars[i].low < lookback_bars[i-2].low and
                lookback_bars[i].low < lookback_bars[i+1].low and
                lookback_bars[i].low < lookback_bars[i+2].low):
                swing_lows.append(lookback_bars[i].low)

        # Cluster nearby levels (within 1%)
        supports = self._cluster_levels(swing_lows)
        resistances = self._cluster_levels(swing_highs)

        # Also add round numbers near current price
        current = bars[-1].close
        round_levels = self._find_round_numbers(current)
        supports.extend([l for l in round_levels if l < current])
        resistances.extend([l for l in round_levels if l > current])

        return sorted(set(supports)), sorted(set(resistances))

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.01) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                # Save cluster average
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Don't forget last cluster
        clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

    def _find_round_numbers(self, price: float) -> List[float]:
        """Find psychologically significant round numbers near price."""
        rounds = []

        # Determine scale
        if price > 500:
            step = 50
        elif price > 100:
            step = 10
        elif price > 50:
            step = 5
        elif price > 10:
            step = 1
        else:
            step = 0.5

        # Find rounds within 10%
        base = int(price / step) * step
        for i in range(-3, 4):
            level = base + i * step
            if 0.9 * price < level < 1.1 * price:
                rounds.append(level)

        return rounds

    def _distance_to_level(
        self,
        price: float,
        levels: List[float],
        direction: str,
    ) -> float:
        """Calculate distance to nearest level in direction."""
        if not levels:
            return 0.10  # Default 10% if no levels found

        if direction == 'above':
            above_levels = [l for l in levels if l > price]
            if above_levels:
                nearest = min(above_levels)
                return (nearest - price) / price
        else:  # below
            below_levels = [l for l in levels if l < price]
            if below_levels:
                nearest = max(below_levels)
                return (price - nearest) / price

        return 0.10

    def _analyze_consolidation(
        self,
        bars: List[PriceBar],
    ) -> Tuple[int, float]:
        """
        Analyze consolidation pattern.

        Returns:
            (days_in_consolidation, range_percent)
        """
        if len(bars) < 5:
            return 0, 0.0

        # Look back for consolidation
        consol_days = 0
        current_high = bars[-1].high
        current_low = bars[-1].low

        for i in range(2, min(30, len(bars))):
            bar = bars[-i]
            range_high = max(current_high, bar.high)
            range_low = min(current_low, bar.low)
            range_pct = (range_high - range_low) / range_low if range_low > 0 else 0

            if range_pct < self.consolidation_threshold:
                consol_days = i
                current_high = range_high
                current_low = range_low
            else:
                break

        # Final range
        if consol_days > 0:
            range_pct = (current_high - current_low) / current_low
        else:
            range_pct = (bars[-1].high - bars[-1].low) / bars[-1].low if bars[-1].low > 0 else 0

        return consol_days, range_pct

    def _calculate_breakout_potential(
        self,
        bars: List[PriceBar],
        resistances: List[float],
    ) -> float:
        """
        Calculate probability of upside breakout.

        Factors:
        - Close to resistance
        - Volume building
        - Higher lows pattern
        """
        current_price = bars[-1].close

        # Distance to resistance factor
        above_resistances = [r for r in resistances if r > current_price]
        if above_resistances:
            nearest_resistance = min(above_resistances)
            dist_pct = (nearest_resistance - current_price) / current_price
            dist_factor = max(0, 1 - dist_pct * 20)  # Closer = higher score
        else:
            dist_factor = 0.5

        # Volume trend factor
        if len(bars) >= 10:
            recent_vol = sum(b.volume for b in bars[-5:]) / 5
            older_vol = sum(b.volume for b in bars[-10:-5]) / 5
            vol_factor = min(1, (recent_vol / older_vol) - 0.5) if older_vol > 0 else 0.5
        else:
            vol_factor = 0.5

        # Higher lows factor
        hl_factor = 0.7 if self._check_higher_lows(bars[-10:]) else 0.3

        # Combined
        potential = dist_factor * 0.4 + vol_factor * 0.3 + hl_factor * 0.3
        return min(1, potential)

    def _calculate_breakdown_potential(
        self,
        bars: List[PriceBar],
        supports: List[float],
    ) -> float:
        """Calculate probability of downside breakdown."""
        current_price = bars[-1].close

        # Distance to support
        below_supports = [s for s in supports if s < current_price]
        if below_supports:
            nearest_support = max(below_supports)
            dist_pct = (current_price - nearest_support) / current_price
            dist_factor = max(0, 1 - dist_pct * 20)
        else:
            dist_factor = 0.5

        # Lower highs factor
        lh_factor = 0.7 if self._check_lower_highs(bars[-10:]) else 0.3

        potential = dist_factor * 0.5 + lh_factor * 0.5
        return min(1, potential)

    def _check_higher_highs(self, bars: List[PriceBar], lookback: int = 10) -> bool:
        """Check if making higher highs."""
        if len(bars) < lookback:
            return False

        recent = bars[-lookback:]
        highs = [b.high for b in recent]

        # Check if recent highs are higher than earlier highs
        mid = len(highs) // 2
        recent_max = max(highs[mid:])
        earlier_max = max(highs[:mid])

        return recent_max > earlier_max

    def _check_lower_lows(self, bars: List[PriceBar], lookback: int = 10) -> bool:
        """Check if making lower lows."""
        if len(bars) < lookback:
            return False

        recent = bars[-lookback:]
        lows = [b.low for b in recent]

        mid = len(lows) // 2
        recent_min = min(lows[mid:])
        earlier_min = min(lows[:mid])

        return recent_min < earlier_min

    def _check_higher_lows(self, bars: List[PriceBar]) -> bool:
        """Check if making higher lows (bullish)."""
        if len(bars) < 5:
            return False

        lows = [b.low for b in bars]
        mid = len(lows) // 2
        return min(lows[mid:]) > min(lows[:mid])

    def _check_lower_highs(self, bars: List[PriceBar]) -> bool:
        """Check if making lower highs (bearish)."""
        if len(bars) < 5:
            return False

        highs = [b.high for b in bars]
        mid = len(highs) // 2
        return max(highs[mid:]) < max(highs[:mid])

    def _is_inside_day(self, bars: List[PriceBar]) -> bool:
        """Check if today is an inside day (range within yesterday's range)."""
        if len(bars) < 2:
            return False

        today = bars[-1]
        yesterday = bars[-2]

        return today.high < yesterday.high and today.low > yesterday.low

    def _score_technical_setup(
        self,
        dist_resistance: float,
        dist_support: float,
        at_key_level: bool,
        consol_days: int,
    ) -> float:
        """Score overall technical setup for day trading."""
        score = 50  # Base

        # Near key level is good (tradeable)
        if at_key_level:
            score += 20
        elif dist_resistance < 0.03 or dist_support < 0.03:
            score += 10

        # Consolidation building energy
        if consol_days > 3:
            score += min(20, consol_days * 3)

        # Good risk/reward setup (closer to support than resistance)
        if dist_support < dist_resistance * 0.5:
            score += 10  # Good long setup
        elif dist_resistance < dist_support * 0.5:
            score += 10  # Good short setup

        return min(100, score)

    def _score_breakout(
        self,
        breakout_potential: float,
        consol_days: int,
        range_pct: float,
    ) -> float:
        """Score breakout probability."""
        # Base from breakout potential
        score = breakout_potential * 60

        # Consolidation increases breakout energy
        if consol_days >= 5:
            score += 20
        elif consol_days >= 3:
            score += 10

        # Tight range = more explosive breakout
        if range_pct < 0.02:
            score += 15
        elif range_pct < 0.03:
            score += 10

        return min(100, score)

    def _score_reversal(
        self,
        bars: List[PriceBar],
        dist_support: float,
        dist_resistance: float,
    ) -> float:
        """Score reversal probability."""
        score = 30  # Base

        # Near support with bullish candle = potential bounce
        if dist_support < 0.02:
            today = bars[-1]
            if today.close > today.open:  # Green candle
                score += 30
            score += 15

        # Near resistance with bearish candle = potential rejection
        if dist_resistance < 0.02:
            today = bars[-1]
            if today.close < today.open:  # Red candle
                score += 30
            score += 15

        # Look for reversal candle patterns
        if self._is_hammer(bars[-1]):
            score += 20
        elif self._is_shooting_star(bars[-1]):
            score += 20

        return min(100, score)

    def _is_hammer(self, bar: PriceBar) -> bool:
        """Check for hammer pattern (bullish reversal)."""
        body = abs(bar.close - bar.open)
        lower_wick = min(bar.open, bar.close) - bar.low
        upper_wick = bar.high - max(bar.open, bar.close)
        total_range = bar.high - bar.low

        if total_range == 0:
            return False

        # Hammer: small body, long lower wick, small upper wick
        return (lower_wick > body * 2 and
                upper_wick < body * 0.5 and
                body < total_range * 0.3)

    def _is_shooting_star(self, bar: PriceBar) -> bool:
        """Check for shooting star pattern (bearish reversal)."""
        body = abs(bar.close - bar.open)
        lower_wick = min(bar.open, bar.close) - bar.low
        upper_wick = bar.high - max(bar.open, bar.close)
        total_range = bar.high - bar.low

        if total_range == 0:
            return False

        # Shooting star: small body, long upper wick, small lower wick
        return (upper_wick > body * 2 and
                lower_wick < body * 0.5 and
                body < total_range * 0.3)
