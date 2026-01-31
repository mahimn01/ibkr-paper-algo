"""
Edge 3: Statistical Extreme Detector

Identifies when price, volume, or momentum are at statistical extremes.

Key insight: Trade only at extremes where mean reversion OR breakout
is statistically likely. Avoid the middle where outcomes are random.

Uses z-scores to measure distance from mean in standard deviations.
"""

import statistics
from typing import Dict, List

from ..types import AssetState, EdgeSignal, EdgeVote, MarketRegime


class StatisticalExtremeDetector:
    """
    Identifies when price, volume, or momentum are at statistical extremes.

    Key insight: Trade only at extremes where mean reversion OR breakout
    is statistically likely. Avoid the middle where outcomes are random.

    Uses z-scores to measure distance from mean in standard deviations.
    |z| > 2.0 is significant (95% confidence)
    |z| > 2.5 is very significant (99% confidence)
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def calculate_zscore(self, values: List[float]) -> float:
        """Calculate z-score of most recent value."""
        if len(values) < 10:
            return 0.0
        try:
            mean = statistics.mean(values[:-1])
            stdev = statistics.stdev(values[:-1])
            if stdev < 0.0001:
                return 0.0
            return (values[-1] - mean) / stdev
        except:
            return 0.0

    def calculate_percentile(self, values: List[float]) -> float:
        """Calculate percentile rank of most recent value."""
        if len(values) < 10:
            return 50.0
        sorted_vals = sorted(values[:-1])
        current = values[-1]
        rank = sum(1 for v in sorted_vals if v <= current)
        return (rank / len(sorted_vals)) * 100

    def analyze(self, state: AssetState) -> Dict[str, float]:
        """
        Calculate statistical metrics for the asset.

        Returns dict with:
        - price_zscore: How far from recent mean
        - volume_zscore: Is volume unusually high/low
        - momentum_zscore: Is momentum extreme
        - range_percentile: Where in today's range (0=low, 100=high)
        - extension_from_vwap: Distance from VWAP in ATR units
        """
        result = {
            "price_zscore": 0.0,
            "volume_zscore": 0.0,
            "momentum_zscore": 0.0,
            "range_percentile": 50.0,
            "extension_from_vwap": 0.0,
            "is_extreme": False,
            "extreme_type": "none",
        }

        prices = list(state.prices)
        volumes = list(state.volumes)

        if len(prices) < 20:
            return result

        # Price z-score (vs recent prices)
        result["price_zscore"] = self.calculate_zscore(prices[-self.lookback:])

        # Volume z-score
        if volumes:
            result["volume_zscore"] = self.calculate_zscore(volumes[-self.lookback:])

        # Momentum z-score (rate of change)
        returns = [(prices[i] - prices[i-1]) / prices[i-1]
                   for i in range(-min(self.lookback, len(prices)-1), 0)]
        if returns:
            result["momentum_zscore"] = self.calculate_zscore(returns)

        # Range percentile (where in today's high-low range)
        if state.day_high > state.day_low:
            result["range_percentile"] = ((prices[-1] - state.day_low) /
                                          (state.day_high - state.day_low)) * 100

        # Extension from VWAP
        if state.vwap > 0 and state.atr > 0:
            result["extension_from_vwap"] = (prices[-1] - state.vwap) / state.atr

        # Determine if we're at an extreme
        if abs(result["price_zscore"]) > 2.0 or abs(result["momentum_zscore"]) > 2.0:
            result["is_extreme"] = True
            if result["price_zscore"] > 2.0:
                result["extreme_type"] = "extended_high"
            elif result["price_zscore"] < -2.0:
                result["extreme_type"] = "extended_low"
            elif result["momentum_zscore"] > 2.0:
                result["extreme_type"] = "momentum_surge"
            elif result["momentum_zscore"] < -2.0:
                result["extreme_type"] = "momentum_crash"

        return result

    def get_vote(self, state: AssetState, regime: MarketRegime) -> EdgeSignal:
        """Get voting signal based on statistical analysis."""
        stats = self.analyze(state)

        price_z = stats["price_zscore"]
        mom_z = stats["momentum_zscore"]
        vol_z = stats["volume_zscore"]
        vwap_ext = stats["extension_from_vwap"]

        # In range-bound markets, fade extremes
        if regime == MarketRegime.RANGE_BOUND:
            if price_z > 2.0 and mom_z > 1.5:
                return EdgeSignal("Statistics", EdgeVote.SHORT, 0.7,
                                f"Mean reversion: price z={price_z:.1f}, extended from range",
                                stats)
            if price_z < -2.0 and mom_z < -1.5:
                return EdgeSignal("Statistics", EdgeVote.LONG, 0.7,
                                f"Mean reversion: price z={price_z:.1f}, oversold in range",
                                stats)

        # In trend markets, trade with momentum at moderate extremes
        if regime in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP]:
            if mom_z > 1.5 and price_z < 2.0:  # Momentum strong but not overextended
                return EdgeSignal("Statistics", EdgeVote.LONG, 0.6,
                                f"Momentum confirmation: mom z={mom_z:.1f} in uptrend",
                                stats)
            if price_z > 2.5:  # Too extended even in uptrend
                return EdgeSignal("Statistics", EdgeVote.VETO_LONG, 0.6,
                                f"Overextended: price z={price_z:.1f}, avoid longs",
                                stats)

        if regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN]:
            if mom_z < -1.5 and price_z > -2.0:
                return EdgeSignal("Statistics", EdgeVote.SHORT, 0.6,
                                f"Momentum confirmation: mom z={mom_z:.1f} in downtrend",
                                stats)
            if price_z < -2.5:
                return EdgeSignal("Statistics", EdgeVote.VETO_SHORT, 0.6,
                                f"Overextended: price z={price_z:.1f}, avoid shorts",
                                stats)

        # Volume confirmation
        if vol_z > 2.0 and abs(mom_z) > 1.0:
            # High volume confirms the move
            if mom_z > 1.0:
                return EdgeSignal("Statistics", EdgeVote.LONG, 0.5,
                                f"Volume surge ({vol_z:.1f}σ) confirms momentum",
                                stats)
            else:
                return EdgeSignal("Statistics", EdgeVote.SHORT, 0.5,
                                f"Volume surge ({vol_z:.1f}σ) confirms selling",
                                stats)

        return EdgeSignal("Statistics", EdgeVote.NEUTRAL, 0.4,
                        f"No statistical extreme: z={price_z:.1f}", stats)
