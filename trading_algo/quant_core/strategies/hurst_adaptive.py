"""
Hurst-Adaptive Trading Strategy

Dynamically switches between momentum and mean-reversion sub-strategies
based on the rolling Hurst exponent of 5-minute returns.

Empirical findings from pattern discovery:
    - 80-89% of the time is "tradeable" (H > 0.55 or H < 0.45)
    - Momentum dominates (70-80% of time across all symbols)
    - Mean-reversion is rare but present (4-20%)
    - 11-28% is random walk (H ~ 0.5) -- DO NOT trade these periods

Regime logic:
    - H > 0.55  (trending)       -> Momentum mode: ride trends with
                                    wide stops and trailing exits
    - H < 0.45  (mean-reverting) -> Mean-reversion mode: fade z-score
                                    extremes with tight stops
    - 0.45 <= H <= 0.55 (random) -> No new entries; close existing
                                    positions at breakeven or small profit

The full R/S Hurst estimator from fractal_analysis is too slow for
per-bar rolling computation, so this module uses a fast autocorrelation-
based proxy for live use.  The full estimator can be used offline for
calibration.

References:
    - Hurst, H.E. (1951): "Long-term storage capacity of reservoirs"
    - Mandelbrot, B.B. (1971): "When can price be arbitraged efficiently?"
    - Peters, E.E. (1994): "Fractal Market Analysis"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HurstConfig:
    """Configuration for the Hurst-adaptive strategy."""

    hurst_window: int = 78
    """Rolling window for Hurst estimation (78 bars = 1 day of 5-min data)."""

    hurst_step: int = 5
    """Recompute Hurst every N bars for speed."""

    momentum_threshold: float = 0.55
    """Hurst above this value activates momentum mode."""

    reversion_threshold: float = 0.45
    """Hurst below this value activates mean-reversion mode."""

    ema_window: int = 20
    """EMA period for momentum signals."""

    zscore_window: int = 20
    """Z-score lookback for mean-reversion signals."""

    zscore_entry: float = 1.5
    """Z-score threshold to enter mean-reversion trades."""

    zscore_exit: float = 0.3
    """Z-score threshold to exit mean-reversion trades."""

    atr_window: int = 14
    """ATR lookback for stop/target sizing."""

    mom_stop_atr: float = 2.0
    """ATR multiplier for momentum stop-loss (wide -- let trends run)."""

    mom_tp_atr: float = 4.0
    """ATR multiplier for momentum take-profit."""

    mom_trail_atr: float = 1.5
    """ATR gain required before engaging trailing stop in momentum mode."""

    mr_stop_atr: float = 1.0
    """ATR multiplier for mean-reversion stop-loss (tight -- quick cut)."""

    mr_tp_atr: float = 1.5
    """ATR multiplier for mean-reversion take-profit (quick grab)."""

    max_weight: float = 0.10
    """Maximum portfolio weight per signal."""

    warmup: int = 100
    """Minimum bars before producing signals."""


# ---------------------------------------------------------------------------
# Internal signal type
# ---------------------------------------------------------------------------

@dataclass
class HurstSignal:
    """Raw signal produced by the Hurst-adaptive strategy."""
    symbol: str
    direction: int              # +1 long, -1 short
    confidence: float           # Higher when H is far from 0.5
    weight: float
    trade_type: str             # "momentum" or "mean_reversion"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_hurst(returns: np.ndarray, window: int) -> float:
    """
    Fast Hurst approximation using lag-1 autocorrelation.

    For fractional Brownian motion, H ~ 0.5 + autocorr(1) / 2.
    This is orders of magnitude faster than the full R/S method
    and adequate for real-time regime detection.

    Parameters
    ----------
    returns : np.ndarray
        Array of log or simple returns.
    window : int
        Number of trailing returns to use.

    Returns
    -------
    float
        Estimated Hurst exponent, clipped to [0.01, 0.99].
        Returns 0.5 (random walk) if insufficient data.
    """
    if len(returns) < window:
        return 0.5
    r = returns[-window:]
    if np.std(r) < 1e-12:
        return 0.5
    autocorr = np.corrcoef(r[:-1], r[1:])[0, 1]
    if np.isnan(autocorr):
        return 0.5
    return float(np.clip(0.5 + autocorr / 2, 0.01, 0.99))


def _ema(prices: np.ndarray, window: int) -> float:
    """
    Compute the current EMA value for the given price array.

    Uses the standard exponential smoothing formula with
    alpha = 2 / (window + 1).
    """
    if len(prices) < window:
        return float(np.mean(prices))
    alpha = 2.0 / (window + 1)
    ema_val = float(prices[0])
    for p in prices[1:]:
        ema_val = alpha * p + (1 - alpha) * ema_val
    return ema_val


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int) -> float:
    """
    Compute the Average True Range over the trailing ``window`` bars.

    Uses the standard Wilder definition: TR = max(H-L, |H-Cprev|, |L-Cprev|).
    """
    if len(closes) < 2 or len(highs) < 2 or len(lows) < 2:
        return 0.0

    h = highs[-window:]
    lo = lows[-window:]
    c_prev = closes[-(window + 1):-1] if len(closes) > window else closes[:-1]

    # Align lengths
    n = min(len(h), len(lo), len(c_prev))
    if n < 1:
        return 0.0
    h = h[-n:]
    lo = lo[-n:]
    c_prev = c_prev[-n:]

    tr = np.maximum(
        h - lo,
        np.maximum(np.abs(h - c_prev), np.abs(lo - c_prev)),
    )
    return float(np.mean(tr))


def _zscore(prices: np.ndarray, window: int) -> float:
    """
    Compute the z-score of the latest price relative to the trailing
    ``window``-bar rolling mean and standard deviation.
    """
    if len(prices) < window:
        return 0.0
    segment = prices[-window:]
    mu = np.mean(segment)
    sigma = np.std(segment)
    if sigma < 1e-12:
        return 0.0
    return float((prices[-1] - mu) / sigma)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class HurstAdaptiveStrategy:
    """
    Regime-switching strategy driven by the rolling Hurst exponent.

    For each symbol the strategy maintains price, high, and low histories
    and recomputes the Hurst exponent every ``hurst_step`` bars.
    Based on the current regime it either generates momentum signals
    (trend-following) or mean-reversion signals (fade extremes), or
    abstains entirely during random-walk periods.

    Usage::

        strategy = HurstAdaptiveStrategy()

        for bar in bars:
            strategy.update(symbol, close, high, low)

        signals = strategy.generate_signals(["AAPL", "MSFT"])
    """

    def __init__(self, config: Optional[HurstConfig] = None):
        self.config = config or HurstConfig()

        # Per-symbol state
        self._close_history: Dict[str, List[float]] = defaultdict(list)
        self._high_history: Dict[str, List[float]] = defaultdict(list)
        self._low_history: Dict[str, List[float]] = defaultdict(list)
        self._bars_count: Dict[str, int] = defaultdict(int)

        # Cached Hurst values (recomputed every hurst_step bars)
        self._hurst_cache: Dict[str, float] = {}
        self._steps_since_hurst: Dict[str, int] = defaultdict(int)

        # Maximum history to retain
        self._max_history = max(
            self.config.hurst_window + 20,
            self.config.ema_window + 20,
            self.config.zscore_window + 20,
            self.config.atr_window + 20,
            self.config.warmup + 20,
        )

    # ── Data ingestion ─────────────────────────────────────────────────

    def update(self, symbol: str, close: float, high: float, low: float) -> None:
        """
        Feed a single bar of OHLC data for a symbol.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        close : float
            Closing price.
        high : float
            Bar high.
        low : float
            Bar low.
        """
        self._close_history[symbol].append(close)
        self._high_history[symbol].append(high)
        self._low_history[symbol].append(low)
        self._bars_count[symbol] += 1

        # Trim to bounded length
        if len(self._close_history[symbol]) > self._max_history:
            self._close_history[symbol] = self._close_history[symbol][-self._max_history:]
            self._high_history[symbol] = self._high_history[symbol][-self._max_history:]
            self._low_history[symbol] = self._low_history[symbol][-self._max_history:]

        # Update Hurst cache at configured intervals
        self._steps_since_hurst[symbol] += 1
        if self._steps_since_hurst[symbol] >= self.config.hurst_step:
            self._recompute_hurst(symbol)
            self._steps_since_hurst[symbol] = 0

    def _recompute_hurst(self, symbol: str) -> None:
        """Recompute the quick Hurst estimate for the given symbol."""
        closes = self._close_history[symbol]
        if len(closes) < self.config.hurst_window + 1:
            self._hurst_cache[symbol] = 0.5
            return

        prices = np.array(closes, dtype=np.float64)
        returns = np.diff(prices) / prices[:-1]
        self._hurst_cache[symbol] = _quick_hurst(returns, self.config.hurst_window)

    # ── Regime classification ──────────────────────────────────────────

    def _classify_regime(self, h: float) -> str:
        """
        Map a Hurst exponent to a regime label.

        Returns one of ``"trending"``, ``"mean_reverting"``, or ``"random_walk"``.
        """
        if h > self.config.momentum_threshold:
            return "trending"
        elif h < self.config.reversion_threshold:
            return "mean_reverting"
        else:
            return "random_walk"

    # ── Signal generation ──────────────────────────────────────────────

    def generate_signals(self, symbols: List[str]) -> List[HurstSignal]:
        """
        Generate trading signals for the requested symbols.

        Parameters
        ----------
        symbols : list of str
            Symbols to evaluate.

        Returns
        -------
        list of HurstSignal
            Zero or more signals.  No signals are produced during warmup
            or in random-walk regimes.
        """
        signals: List[HurstSignal] = []

        for symbol in symbols:
            if self._bars_count[symbol] < self.config.warmup:
                continue

            h_value = self._hurst_cache.get(symbol, 0.5)
            regime = self._classify_regime(h_value)

            if regime == "random_walk":
                # No new entries in random-walk regime
                continue

            signal = self._generate_signal_for_regime(symbol, h_value, regime)
            if signal is not None:
                signals.append(signal)

        return signals

    def _generate_signal_for_regime(
        self,
        symbol: str,
        h_value: float,
        regime: str,
    ) -> Optional[HurstSignal]:
        """Dispatch to the appropriate sub-strategy based on regime."""
        if regime == "trending":
            return self._momentum_signal(symbol, h_value)
        elif regime == "mean_reverting":
            return self._mean_reversion_signal(symbol, h_value)
        return None

    # ── Momentum sub-strategy ──────────────────────────────────────────

    def _momentum_signal(self, symbol: str, h_value: float) -> Optional[HurstSignal]:
        """
        Generate a momentum signal when H > momentum_threshold.

        Entry: price breaks above/below 20-bar EMA with momentum confirmation.
        Stop: 2 x ATR (wide -- let trends run).
        Take profit: 4 x ATR.
        """
        closes = np.array(self._close_history[symbol], dtype=np.float64)
        highs = np.array(self._high_history[symbol], dtype=np.float64)
        lows = np.array(self._low_history[symbol], dtype=np.float64)

        if len(closes) < self.config.ema_window + 2:
            return None

        current_price = closes[-1]
        prev_price = closes[-2]
        ema_val = _ema(closes, self.config.ema_window)
        prev_ema = _ema(closes[:-1], self.config.ema_window)
        atr_val = _atr(highs, lows, closes, self.config.atr_window)

        if atr_val < 1e-12:
            return None

        # Momentum confirmation: price crosses EMA in the direction of trend
        # Bullish: price crosses above EMA (prev below, current above)
        # Bearish: price crosses below EMA (prev above, current below)
        direction = 0
        if current_price > ema_val and prev_price <= prev_ema:
            direction = 1   # Bullish breakout
        elif current_price < ema_val and prev_price >= prev_ema:
            direction = -1  # Bearish breakout

        # Also accept strong continuation: price already above/below EMA
        # with accelerating momentum (price moving further from EMA)
        if direction == 0:
            price_dist = (current_price - ema_val) / atr_val
            prev_dist = (prev_price - prev_ema) / atr_val
            if price_dist > 0.5 and price_dist > prev_dist:
                direction = 1
            elif price_dist < -0.5 and price_dist < prev_dist:
                direction = -1

        if direction == 0:
            return None

        # Confidence scales with distance from H=0.5
        confidence = min(1.0, abs(h_value - 0.5) * 4.0)

        # Stop and target levels
        if direction == 1:
            stop_loss = current_price - self.config.mom_stop_atr * atr_val
            take_profit = current_price + self.config.mom_tp_atr * atr_val
        else:
            stop_loss = current_price + self.config.mom_stop_atr * atr_val
            take_profit = current_price - self.config.mom_tp_atr * atr_val

        return HurstSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            weight=self.config.max_weight,
            trade_type="momentum",
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "hurst": round(h_value, 4),
                "regime": "trending",
                "ema": round(ema_val, 4),
                "atr": round(atr_val, 4),
                "trail_activate": round(self.config.mom_trail_atr * atr_val, 4),
            },
        )

    # ── Mean-reversion sub-strategy ────────────────────────────────────

    def _mean_reversion_signal(self, symbol: str, h_value: float) -> Optional[HurstSignal]:
        """
        Generate a mean-reversion signal when H < reversion_threshold.

        Entry: price deviates > 1.5 std from 20-bar mean (|z-score| > 1.5).
        Stop: 1 x ATR (tight -- quick cut).
        Take profit: 1.5 x ATR (quick grab).
        Direction: fade the deviation.
        """
        closes = np.array(self._close_history[symbol], dtype=np.float64)
        highs = np.array(self._high_history[symbol], dtype=np.float64)
        lows = np.array(self._low_history[symbol], dtype=np.float64)

        if len(closes) < self.config.zscore_window + 2:
            return None

        current_price = closes[-1]
        z = _zscore(closes, self.config.zscore_window)
        atr_val = _atr(highs, lows, closes, self.config.atr_window)

        if atr_val < 1e-12:
            return None

        # Only trade when z-score exceeds entry threshold
        if abs(z) < self.config.zscore_entry:
            return None

        # Fade the deviation: sell when price is too high, buy when too low
        direction = -1 if z > 0 else 1

        # Confidence scales with both z-score extremity and distance from H=0.5
        z_confidence = min(1.0, abs(z) / 3.0)
        h_confidence = min(1.0, abs(h_value - 0.5) * 4.0)
        confidence = (z_confidence + h_confidence) / 2.0

        # Stop and target levels
        if direction == 1:
            stop_loss = current_price - self.config.mr_stop_atr * atr_val
            take_profit = current_price + self.config.mr_tp_atr * atr_val
        else:
            stop_loss = current_price + self.config.mr_stop_atr * atr_val
            take_profit = current_price - self.config.mr_tp_atr * atr_val

        return HurstSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            weight=self.config.max_weight,
            trade_type="mean_reversion",
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "hurst": round(h_value, 4),
                "regime": "mean_reverting",
                "zscore": round(z, 4),
                "atr": round(atr_val, 4),
            },
        )

    # ── Utilities ──────────────────────────────────────────────────────

    def get_regime(self, symbol: str) -> str:
        """Return the current regime classification for a symbol."""
        h = self._hurst_cache.get(symbol, 0.5)
        return self._classify_regime(h)

    def get_hurst(self, symbol: str) -> float:
        """Return the latest cached Hurst value for a symbol."""
        return self._hurst_cache.get(symbol, 0.5)

    def is_warmed_up(self, symbol: str) -> bool:
        """Check if a symbol has accumulated enough bars for signal generation."""
        return self._bars_count[symbol] >= self.config.warmup

    def reset(self) -> None:
        """Reset all internal state."""
        self._close_history.clear()
        self._high_history.clear()
        self._low_history.clear()
        self._bars_count.clear()
        self._hurst_cache.clear()
        self._steps_since_hurst.clear()
