"""
Momentum Analyzer for Day Trading Stock Selection.

Analyzes price momentum and trend characteristics to identify:
- Trending stocks (easier to trade with momentum)
- Relative strength (outperformers/underperformers)
- Overbought/oversold conditions (mean reversion opportunities)
- Momentum persistence (how long moves last)

Key Concepts:
- Multi-timeframe momentum (1d, 5d, 20d alignment)
- Relative strength vs market and sector
- RSI and momentum oscillators
- Moving average relationships
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from trading_algo.stock_selector.models import MomentumScore


@dataclass
class PriceBar:
    """OHLCV price bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class MomentumAnalyzer:
    """
    Multi-timeframe momentum analysis for day trading.

    Momentum characteristics that matter:
    - Strong short-term momentum = day trading opportunity
    - Momentum aligned across timeframes = higher probability
    - Relative strength shows leadership/laggards
    - Extreme RSI = potential reversal or continuation
    """

    def __init__(
        self,
        rsi_period: int = 14,
        sma_short: int = 20,
        sma_long: int = 50,
    ):
        self.rsi_period = rsi_period
        self.sma_short = sma_short
        self.sma_long = sma_long

    def analyze(
        self,
        bars: List[PriceBar],
        market_bars: Optional[List[PriceBar]] = None,
        sector_bars: Optional[List[PriceBar]] = None,
    ) -> Optional[MomentumScore]:
        """
        Analyze momentum characteristics.

        Args:
            bars: Stock's daily OHLCV bars
            market_bars: SPY bars for relative strength
            sector_bars: Sector ETF bars for relative strength

        Returns:
            MomentumScore with all metrics
        """
        if len(bars) < 60:
            return None

        # Price momentum (returns)
        return_1d = self._calculate_return(bars, 1)
        return_5d = self._calculate_return(bars, 5)
        return_20d = self._calculate_return(bars, 20)
        return_intraday = (bars[-1].close - bars[-1].open) / bars[-1].open if bars[-1].open > 0 else 0

        # Moving averages
        sma_20 = self._calculate_sma(bars, self.sma_short)
        sma_50 = self._calculate_sma(bars, self.sma_long)
        current_price = bars[-1].close

        price_vs_sma_20 = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        price_vs_sma_50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
        sma_20_vs_50 = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0

        # RSI
        rsi = self._calculate_rsi(bars, self.rsi_period)
        rsi_trend = self._calculate_rsi_trend(bars)

        # Relative strength
        rs_sector = self._calculate_relative_strength(bars, sector_bars) if sector_bars else 0.0
        rs_market = self._calculate_relative_strength(bars, market_bars) if market_bars else 0.0

        # Derived scores
        momentum_score = self._score_momentum(return_1d, return_5d, return_20d, rsi)
        trend_strength = self._score_trend_strength(price_vs_sma_20, price_vs_sma_50, sma_20_vs_50)
        mean_reversion_score = self._score_mean_reversion(rsi, price_vs_sma_20, return_5d)

        return MomentumScore(
            return_1d=return_1d,
            return_5d=return_5d,
            return_20d=return_20d,
            return_intraday=return_intraday,
            price_vs_sma_20=price_vs_sma_20,
            price_vs_sma_50=price_vs_sma_50,
            sma_20_vs_50=sma_20_vs_50,
            rsi_14=rsi,
            rsi_trend=rsi_trend,
            relative_strength_sector=rs_sector,
            relative_strength_market=rs_market,
            momentum_score=momentum_score,
            trend_strength=trend_strength,
            mean_reversion_score=mean_reversion_score,
        )

    def _calculate_return(
        self,
        bars: List[PriceBar],
        days: int,
    ) -> float:
        """Calculate return over N days."""
        if len(bars) <= days:
            return 0.0

        end_price = bars[-1].close
        start_price = bars[-days-1].close

        if start_price == 0:
            return 0.0

        return (end_price - start_price) / start_price

    def _calculate_sma(
        self,
        bars: List[PriceBar],
        period: int,
    ) -> float:
        """Calculate simple moving average."""
        if len(bars) < period:
            return bars[-1].close if bars else 0.0

        recent_closes = [b.close for b in bars[-period:]]
        return sum(recent_closes) / len(recent_closes)

    def _calculate_rsi(
        self,
        bars: List[PriceBar],
        period: int,
    ) -> float:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(bars) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(-period, 0):
            change = bars[i].close - bars[i-1].close
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.0001

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_rsi_trend(
        self,
        bars: List[PriceBar],
        lookback: int = 5,
    ) -> float:
        """
        Calculate RSI trend (-1 to 1).

        Positive = RSI increasing
        Negative = RSI decreasing
        """
        if len(bars) < self.rsi_period + lookback + 5:
            return 0.0

        # Calculate RSI at different points
        current_rsi = self._calculate_rsi(bars, self.rsi_period)
        past_rsi = self._calculate_rsi(bars[:-lookback], self.rsi_period)

        # Normalize change
        change = current_rsi - past_rsi
        trend = change / 20  # ±20 RSI points = ±1
        trend = max(-1, min(1, trend))

        return trend

    def _calculate_relative_strength(
        self,
        stock_bars: List[PriceBar],
        benchmark_bars: Optional[List[PriceBar]],
        period: int = 20,
    ) -> float:
        """
        Calculate relative strength vs benchmark.

        RS = Stock Return / Benchmark Return
        Normalized to -1 to 1 scale.
        """
        if not benchmark_bars or len(benchmark_bars) < period:
            return 0.0

        stock_return = self._calculate_return(stock_bars, period)
        bench_return = self._calculate_return(benchmark_bars, period)

        # Avoid division by zero
        if abs(bench_return) < 0.001:
            # If benchmark flat, just use stock return
            return max(-1, min(1, stock_return * 10))

        # Relative strength ratio
        rs_ratio = stock_return / bench_return if bench_return != 0 else 1

        # Convert to -1 to 1 scale
        # rs_ratio > 1 means outperforming
        if rs_ratio >= 1:
            rs = min(1, (rs_ratio - 1) / 1)  # 2x outperformance = 1
        else:
            rs = max(-1, (rs_ratio - 1) / 1)  # 0.5x = -0.5

        return rs

    def _score_momentum(
        self,
        return_1d: float,
        return_5d: float,
        return_20d: float,
        rsi: float,
    ) -> float:
        """
        Score overall momentum (0-100).

        High score = strong momentum in a direction
        """
        # Calculate momentum alignment
        signs = [
            1 if return_1d > 0.005 else (-1 if return_1d < -0.005 else 0),
            1 if return_5d > 0.01 else (-1 if return_5d < -0.01 else 0),
            1 if return_20d > 0.02 else (-1 if return_20d < -0.02 else 0),
        ]

        # Check alignment (all same direction)
        if signs[0] == signs[1] == signs[2] and signs[0] != 0:
            alignment_bonus = 20
        elif signs[0] == signs[1] and signs[0] != 0:
            alignment_bonus = 10
        else:
            alignment_bonus = 0

        # Magnitude score
        total_magnitude = (abs(return_1d) * 3 + abs(return_5d) * 2 + abs(return_20d)) / 6
        magnitude_score = min(50, total_magnitude * 500)  # 10% move = 50 points

        # RSI contribution (extreme RSI = strong momentum)
        if rsi > 70 or rsi < 30:
            rsi_score = 20
        elif rsi > 60 or rsi < 40:
            rsi_score = 10
        else:
            rsi_score = 0

        return min(100, magnitude_score + alignment_bonus + rsi_score)

    def _score_trend_strength(
        self,
        price_vs_sma_20: float,
        price_vs_sma_50: float,
        sma_20_vs_50: float,
    ) -> float:
        """
        Score trend strength (0-100).

        Strong trend = price above MAs, MAs aligned
        """
        score = 50  # Neutral starting point

        # Price position relative to MAs
        if price_vs_sma_20 > 0.02 and price_vs_sma_50 > 0.05:
            score += 20  # Strong uptrend
        elif price_vs_sma_20 > 0 and price_vs_sma_50 > 0:
            score += 10  # Mild uptrend
        elif price_vs_sma_20 < -0.02 and price_vs_sma_50 < -0.05:
            score += 20  # Strong downtrend (also tradeable)
        elif price_vs_sma_20 < 0 and price_vs_sma_50 < 0:
            score += 10  # Mild downtrend

        # MA alignment
        if sma_20_vs_50 > 0.02:
            score += 15  # 20 MA well above 50 MA (bullish)
        elif sma_20_vs_50 < -0.02:
            score += 15  # 20 MA well below 50 MA (bearish)
        elif abs(sma_20_vs_50) < 0.01:
            score -= 10  # MAs converged (no clear trend)

        # Penalty for mixed signals
        if (price_vs_sma_20 > 0) != (price_vs_sma_50 > 0):
            score -= 15  # Price between MAs (choppy)

        return max(0, min(100, score))

    def _score_mean_reversion(
        self,
        rsi: float,
        price_vs_sma_20: float,
        return_5d: float,
    ) -> float:
        """
        Score mean reversion opportunity (0-100).

        High score = oversold bounce or overbought fade opportunity
        """
        score = 0

        # RSI extremes
        if rsi < 25:
            score += 40  # Very oversold
        elif rsi < 35:
            score += 25  # Oversold
        elif rsi > 75:
            score += 40  # Very overbought
        elif rsi > 65:
            score += 25  # Overbought

        # Extended from MA
        ma_extension = abs(price_vs_sma_20)
        if ma_extension > 0.08:
            score += 30  # Very extended
        elif ma_extension > 0.05:
            score += 20  # Extended
        elif ma_extension > 0.03:
            score += 10  # Somewhat extended

        # Sharp recent move (potential snapback)
        if abs(return_5d) > 0.10:
            score += 20  # >10% move in 5 days
        elif abs(return_5d) > 0.05:
            score += 10  # >5% move in 5 days

        return min(100, score)
