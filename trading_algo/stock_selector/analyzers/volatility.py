"""
Volatility Analyzer for Day Trading Stock Selection.

Analyzes multiple dimensions of volatility to identify stocks with
tradeable price movement characteristics.

Key Concepts:
- ATR% (Average True Range as % of price) - Core day trading metric
- Volatility regime - Is current vol high/low vs historical?
- Volatility clustering - Vol tends to persist (GARCH effect)
- Gap analysis - Overnight gaps affect day trading risk/opportunity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from trading_algo.stock_selector.models import VolatilityScore


@dataclass
class PriceBar:
    """OHLCV price bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis for day trading suitability.

    Day trading requires:
    - Sufficient volatility to profit from intraday moves
    - Not so volatile that stops get blown constantly
    - Consistent volatility (predictable behavior)
    - Appropriate gap behavior

    Ideal ATR% range for day trading: 1.5% - 5%
    Below 1%: Not enough movement
    Above 8%: Too erratic, hard to manage risk
    """

    def __init__(
        self,
        ideal_atr_min: float = 0.015,      # 1.5% minimum ATR
        ideal_atr_max: float = 0.05,       # 5% maximum ideal ATR
        max_acceptable_atr: float = 0.08,  # 8% absolute max
        lookback_short: int = 5,           # Short-term vol
        lookback_long: int = 20,           # Long-term vol
    ):
        self.ideal_atr_min = ideal_atr_min
        self.ideal_atr_max = ideal_atr_max
        self.max_acceptable_atr = max_acceptable_atr
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def analyze(self, bars: List[PriceBar]) -> Optional[VolatilityScore]:
        """
        Analyze volatility characteristics from price bars.

        Args:
            bars: List of daily OHLCV bars (most recent last)

        Returns:
            VolatilityScore with all metrics and derived scores
        """
        if len(bars) < self.lookback_long + 5:
            return None

        # Calculate returns for volatility
        returns = self._calculate_returns(bars)

        # Historical volatility (annualized)
        vol_20d = self._calculate_historical_vol(returns, self.lookback_long)
        vol_5d = self._calculate_historical_vol(returns, self.lookback_short)

        # ATR calculation
        atr, atr_percent = self._calculate_atr(bars)

        # Average intraday range
        avg_range = self._calculate_avg_intraday_range(bars)

        # Gap analysis
        gap_freq, gap_avg = self._calculate_gap_metrics(bars)

        # Volatility percentile (where is current vol vs history)
        vol_percentile = self._calculate_vol_percentile(returns)

        # Volatility trend (is vol increasing or decreasing?)
        vol_trend = self._calculate_vol_trend(returns)

        # Now calculate derived scores
        volatility_score = self._score_volatility(atr_percent, vol_20d)
        tradeable_range_score = self._score_tradeable_range(atr_percent, avg_range)

        return VolatilityScore(
            historical_volatility_20d=vol_20d,
            historical_volatility_5d=vol_5d,
            atr_percent=atr_percent,
            average_intraday_range=avg_range,
            gap_frequency=gap_freq,
            gap_average_size=gap_avg,
            volatility_score=volatility_score,
            volatility_trend=vol_trend,
            tradeable_range_score=tradeable_range_score,
            vol_percentile=vol_percentile,
        )

    def _calculate_returns(self, bars: List[PriceBar]) -> List[float]:
        """Calculate daily log returns."""
        returns = []
        for i in range(1, len(bars)):
            if bars[i-1].close > 0:
                ret = math.log(bars[i].close / bars[i-1].close)
                returns.append(ret)
        return returns

    def _calculate_historical_vol(
        self,
        returns: List[float],
        lookback: int,
    ) -> float:
        """Calculate annualized historical volatility."""
        if len(returns) < lookback:
            lookback = len(returns)

        recent_returns = returns[-lookback:]
        if not recent_returns:
            return 0.0

        # Standard deviation of returns
        mean_ret = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_ret) ** 2 for r in recent_returns) / len(recent_returns)
        daily_vol = math.sqrt(variance)

        # Annualize (252 trading days)
        annualized_vol = daily_vol * math.sqrt(252)
        return annualized_vol

    def _calculate_atr(
        self,
        bars: List[PriceBar],
        period: int = 14,
    ) -> Tuple[float, float]:
        """
        Calculate Average True Range and ATR%.

        True Range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        """
        if len(bars) < period + 1:
            return 0.0, 0.0

        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Use last 'period' TRs for ATR
        recent_trs = true_ranges[-period:]
        atr = sum(recent_trs) / len(recent_trs)

        # ATR as percentage of current price
        current_price = bars[-1].close
        atr_percent = atr / current_price if current_price > 0 else 0

        return atr, atr_percent

    def _calculate_avg_intraday_range(
        self,
        bars: List[PriceBar],
        lookback: int = 20,
    ) -> float:
        """Calculate average (high-low)/close ratio."""
        recent_bars = bars[-lookback:]
        ranges = []

        for bar in recent_bars:
            if bar.close > 0:
                intraday_range = (bar.high - bar.low) / bar.close
                ranges.append(intraday_range)

        return sum(ranges) / len(ranges) if ranges else 0.0

    def _calculate_gap_metrics(
        self,
        bars: List[PriceBar],
        gap_threshold: float = 0.01,  # 1% gap
        lookback: int = 60,
    ) -> Tuple[float, float]:
        """
        Calculate gap frequency and average gap size.

        Gap = |open - prev_close| / prev_close
        """
        if len(bars) < 2:
            return 0.0, 0.0

        recent_bars = bars[-lookback:] if len(bars) > lookback else bars
        gaps = []

        for i in range(1, len(recent_bars)):
            prev_close = recent_bars[i-1].close
            open_price = recent_bars[i].open

            if prev_close > 0:
                gap = abs(open_price - prev_close) / prev_close
                gaps.append(gap)

        if not gaps:
            return 0.0, 0.0

        # Frequency of gaps above threshold
        significant_gaps = [g for g in gaps if g >= gap_threshold]
        gap_frequency = len(significant_gaps) / len(gaps)

        # Average gap size (of significant gaps)
        avg_gap = sum(significant_gaps) / len(significant_gaps) if significant_gaps else 0

        return gap_frequency, avg_gap

    def _calculate_vol_percentile(
        self,
        returns: List[float],
        lookback: int = 252,
    ) -> float:
        """
        Calculate where current volatility sits vs historical.

        Returns percentile (0-100).
        """
        if len(returns) < 30:
            return 50.0

        # Calculate rolling volatility
        window = 20
        rolling_vols = []

        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            mean_ret = sum(window_returns) / len(window_returns)
            variance = sum((r - mean_ret) ** 2 for r in window_returns) / len(window_returns)
            vol = math.sqrt(variance) * math.sqrt(252)
            rolling_vols.append(vol)

        if not rolling_vols:
            return 50.0

        current_vol = rolling_vols[-1]
        below_current = sum(1 for v in rolling_vols if v < current_vol)
        percentile = (below_current / len(rolling_vols)) * 100

        return percentile

    def _calculate_vol_trend(
        self,
        returns: List[float],
        short_window: int = 5,
        long_window: int = 20,
    ) -> float:
        """
        Calculate volatility trend (-1 to 1).

        Positive = volatility increasing
        Negative = volatility decreasing
        """
        if len(returns) < long_window + 5:
            return 0.0

        # Short-term vol
        short_returns = returns[-short_window:]
        short_mean = sum(short_returns) / len(short_returns)
        short_var = sum((r - short_mean) ** 2 for r in short_returns) / len(short_returns)
        short_vol = math.sqrt(short_var)

        # Long-term vol
        long_returns = returns[-long_window:]
        long_mean = sum(long_returns) / len(long_returns)
        long_var = sum((r - long_mean) ** 2 for r in long_returns) / len(long_returns)
        long_vol = math.sqrt(long_var)

        if long_vol == 0:
            return 0.0

        # Ratio indicates trend
        ratio = short_vol / long_vol

        # Convert to -1 to 1 scale
        # ratio > 1 means vol increasing, < 1 means decreasing
        trend = (ratio - 1) / 0.5  # Normalize: ±50% change = ±1
        trend = max(-1, min(1, trend))

        return trend

    def _score_volatility(
        self,
        atr_percent: float,
        historical_vol: float,
    ) -> float:
        """
        Score overall volatility for day trading (0-100).

        Ideal range: 1.5% - 5% ATR
        """
        # Score based on ATR% being in ideal range
        if atr_percent < self.ideal_atr_min:
            # Too low - penalize proportionally
            score = (atr_percent / self.ideal_atr_min) * 50
        elif atr_percent <= self.ideal_atr_max:
            # In ideal range - high score
            # Peak score at midpoint of ideal range
            midpoint = (self.ideal_atr_min + self.ideal_atr_max) / 2
            distance_from_mid = abs(atr_percent - midpoint)
            range_half = (self.ideal_atr_max - self.ideal_atr_min) / 2
            score = 100 - (distance_from_mid / range_half) * 20
        elif atr_percent <= self.max_acceptable_atr:
            # Above ideal but acceptable - declining score
            excess = atr_percent - self.ideal_atr_max
            max_excess = self.max_acceptable_atr - self.ideal_atr_max
            score = 80 - (excess / max_excess) * 40
        else:
            # Too volatile
            score = max(0, 40 - (atr_percent - self.max_acceptable_atr) * 500)

        return max(0, min(100, score))

    def _score_tradeable_range(
        self,
        atr_percent: float,
        avg_range: float,
    ) -> float:
        """
        Score the tradeable range quality (0-100).

        Good day trading stocks have:
        - Consistent intraday ranges
        - Range roughly matches ATR (not gappy)
        """
        if atr_percent == 0:
            return 0.0

        # Ratio of intraday range to ATR
        # Ideally close to 1.0 (consistent behavior)
        ratio = avg_range / atr_percent if atr_percent > 0 else 0

        # Score based on ratio being close to 1.0
        # Perfect = 100, deviation penalized
        deviation = abs(ratio - 1.0)
        score = max(0, 100 - deviation * 100)

        # Bonus for higher absolute range (more opportunity)
        if avg_range > 0.02:  # >2% average range
            score = min(100, score + 10)

        return score
