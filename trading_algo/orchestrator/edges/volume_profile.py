"""
Edge 4: Volume Profile / Auction Theory

Analyzes where volume occurs at each price level.

Key concepts from auction market theory:
- Value Area: Where 70% of volume occurs (fair value zone)
- Point of Control (POC): Price with most volume (strongest level)
- Poor Structure: Low volume areas (price moves fast through these)
"""

import statistics
from typing import Dict

from ..types import AssetState, EdgeSignal, EdgeVote, MarketRegime


class VolumeProfileEngine:
    """
    Analyzes where volume occurs at each price level.

    Key concepts from auction market theory:
    - Value Area: Where 70% of volume occurs (fair value zone)
    - Point of Control (POC): Price with most volume (strongest level)
    - Poor Structure: Low volume areas (price moves fast through these)

    Trading implications:
    - Price above VA: Bullish, look for pullbacks to VA high
    - Price below VA: Bearish, look for rallies to VA low
    - At POC: Consolidation, wait for breakout
    - In poor structure: Expect rapid moves
    """

    def __init__(self, price_increment: float = 0.10):
        self.price_increment = price_increment

    def build_profile(self, state: AssetState) -> Dict[str, float]:
        """
        Build volume profile from intraday data.

        Returns dict with VA high, VA low, POC, and structure quality.
        """
        result = {
            "value_area_high": 0.0,
            "value_area_low": 0.0,
            "poc": 0.0,
            "current_vs_va": "inside",  # 'above', 'below', 'inside'
            "current_vs_poc": 0.0,  # Distance from POC in ATR
            "structure_quality": 0.5,  # 0-1, higher = better structure
        }

        prices = list(state.prices)
        volumes = list(state.volumes)

        if len(prices) < 20 or len(volumes) < 20:
            return result

        # Build volume at price histogram
        vol_by_price: Dict[float, float] = {}
        for price, volume in zip(prices, volumes):
            # Round price to increment
            rounded = round(price / self.price_increment) * self.price_increment
            vol_by_price[rounded] = vol_by_price.get(rounded, 0) + volume

        if not vol_by_price:
            return result

        # Find Point of Control (highest volume price)
        poc_price = max(vol_by_price.keys(), key=lambda p: vol_by_price[p])
        result["poc"] = poc_price

        # Calculate Value Area (70% of volume)
        total_volume = sum(vol_by_price.values())
        target_volume = total_volume * 0.70

        # Start from POC and expand outward
        sorted_prices = sorted(vol_by_price.keys())
        poc_idx = sorted_prices.index(poc_price)

        va_volume = vol_by_price[poc_price]
        low_idx = poc_idx
        high_idx = poc_idx

        while va_volume < target_volume and (low_idx > 0 or high_idx < len(sorted_prices) - 1):
            # Expand to whichever side has more volume
            low_vol = vol_by_price[sorted_prices[low_idx - 1]] if low_idx > 0 else 0
            high_vol = vol_by_price[sorted_prices[high_idx + 1]] if high_idx < len(sorted_prices) - 1 else 0

            if low_vol >= high_vol and low_idx > 0:
                low_idx -= 1
                va_volume += low_vol
            elif high_idx < len(sorted_prices) - 1:
                high_idx += 1
                va_volume += high_vol
            else:
                break

        result["value_area_low"] = sorted_prices[low_idx]
        result["value_area_high"] = sorted_prices[high_idx]

        # Current price vs value area
        current_price = prices[-1]
        if current_price > result["value_area_high"]:
            result["current_vs_va"] = "above"
        elif current_price < result["value_area_low"]:
            result["current_vs_va"] = "below"
        else:
            result["current_vs_va"] = "inside"

        # Distance from POC in ATR units
        if state.atr > 0:
            result["current_vs_poc"] = (current_price - poc_price) / state.atr

        # Structure quality: How evenly distributed is volume?
        volumes_list = list(vol_by_price.values())
        if len(volumes_list) > 2:
            mean_vol = statistics.mean(volumes_list)
            stdev_vol = statistics.stdev(volumes_list)
            # Lower coefficient of variation = more even distribution = better structure
            cv = stdev_vol / mean_vol if mean_vol > 0 else 1
            result["structure_quality"] = max(0, min(1, 1 - (cv / 2)))

        return result

    def get_vote(self, state: AssetState, regime: MarketRegime) -> EdgeSignal:
        """Get voting signal based on volume profile."""
        profile = self.build_profile(state)

        va_position = profile["current_vs_va"]
        poc_distance = profile["current_vs_poc"]
        structure = profile["structure_quality"]

        # Above value area = bullish, below = bearish
        if va_position == "above":
            if regime in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP]:
                return EdgeSignal("VolumeProfile", EdgeVote.LONG, 0.6,
                                f"Price above value area, in uptrend. POC dist: {poc_distance:.1f} ATR",
                                profile)
            else:
                # Above VA but not trending up - potential reversal
                return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.4,
                                f"Above VA but no trend support", profile)

        elif va_position == "below":
            if regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN]:
                return EdgeSignal("VolumeProfile", EdgeVote.SHORT, 0.6,
                                f"Price below value area, in downtrend. POC dist: {poc_distance:.1f} ATR",
                                profile)
            else:
                return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.4,
                                f"Below VA but no trend support", profile)

        else:  # Inside value area
            if structure < 0.3:
                # Poor structure = expect breakout
                return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.3,
                                f"Inside VA with poor structure - breakout likely", profile)
            else:
                # Good structure = range trade
                if poc_distance > 1.0:
                    return EdgeSignal("VolumeProfile", EdgeVote.SHORT, 0.5,
                                    f"At VA high, fade to POC", profile)
                elif poc_distance < -1.0:
                    return EdgeSignal("VolumeProfile", EdgeVote.LONG, 0.5,
                                    f"At VA low, fade to POC", profile)

        return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.3,
                        f"No clear volume profile signal", profile)
