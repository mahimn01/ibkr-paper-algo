"""
Signal Filters: Pre and post-processing filters for signals.

Types of filters:
1. Confidence threshold
2. Urgency threshold
3. Time-based (no trading in certain periods)
4. Regime-based (certain signals only in certain regimes)
5. Correlation filter (avoid redundant signals)
6. Rate limiter (max signals per time window)

Pure mathematical filtering - no AI.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalSource


class FilterType(Enum):
    """Types of signal filters."""

    CONFIDENCE_THRESHOLD = auto()
    URGENCY_THRESHOLD = auto()
    TIME_WINDOW = auto()
    REGIME_GATE = auto()
    CORRELATION_DEDUP = auto()
    RATE_LIMIT = auto()
    VOLATILITY_GATE = auto()
    DRAWDOWN_GATE = auto()


@dataclass
class FilterResult:
    """Result of applying a filter."""

    passed: bool
    filter_type: FilterType
    reason: Optional[str] = None
    original_signal: Optional[Signal] = None
    modified_signal: Optional[Signal] = None


class SignalFilter:
    """
    Apply filters to signals before trading.

    Supports chaining multiple filters.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        urgency_threshold: float = 0.3,
        trading_start: dt_time = dt_time(9, 30),
        trading_end: dt_time = dt_time(16, 0),
        max_signals_per_hour: int = 10,
        correlation_threshold: float = 0.8,
        max_drawdown_pct: float = 0.10,
    ):
        self.confidence_threshold = confidence_threshold
        self.urgency_threshold = urgency_threshold
        self.trading_start = trading_start
        self.trading_end = trading_end
        self.max_signals_per_hour = max_signals_per_hour
        self.correlation_threshold = correlation_threshold
        self.max_drawdown_pct = max_drawdown_pct

        # State tracking
        self._recent_signals: Deque[Tuple[datetime, Signal]] = deque(maxlen=1000)
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 100.0
        self._current_equity: float = 100.0
        self._current_regime: Optional[str] = None

        # Regime-specific allowed sources
        self._regime_sources: Dict[str, List[SignalSource]] = {
            "TRENDING": [SignalSource.ATTENTION, SignalSource.ADVERSARIAL],
            "CONSOLIDATION": [SignalSource.REFLEXIVITY, SignalSource.ALPHA],
            "BUBBLE": [SignalSource.TOPOLOGY, SignalSource.REFLEXIVITY],
            "ROTATION": [SignalSource.ALPHA, SignalSource.ADVERSARIAL],
        }

        # Active filters
        self._active_filters: List[FilterType] = [
            FilterType.CONFIDENCE_THRESHOLD,
            FilterType.TIME_WINDOW,
            FilterType.RATE_LIMIT,
        ]

    def add_filter(self, filter_type: FilterType) -> None:
        """Add a filter to the chain."""
        if filter_type not in self._active_filters:
            self._active_filters.append(filter_type)

    def remove_filter(self, filter_type: FilterType) -> None:
        """Remove a filter from the chain."""
        if filter_type in self._active_filters:
            self._active_filters.remove(filter_type)

    def filter(
        self,
        signal: Signal,
        timestamp: Optional[datetime] = None,
    ) -> FilterResult:
        """Apply all active filters to a signal."""
        ts = timestamp or datetime.now()

        for filter_type in self._active_filters:
            result = self._apply_filter(filter_type, signal, ts)
            if not result.passed:
                return result

        # All filters passed
        self._recent_signals.append((ts, signal))
        return FilterResult(
            passed=True,
            filter_type=FilterType.CONFIDENCE_THRESHOLD,  # Last filter
            original_signal=signal,
            modified_signal=signal,
        )

    def filter_batch(
        self,
        signals: List[Signal],
        timestamp: Optional[datetime] = None,
    ) -> List[Signal]:
        """Filter a batch of signals, returning only those that pass."""
        ts = timestamp or datetime.now()
        passed = []

        for signal in signals:
            result = self.filter(signal, ts)
            if result.passed:
                passed.append(result.modified_signal or signal)

        return passed

    def _apply_filter(
        self,
        filter_type: FilterType,
        signal: Signal,
        timestamp: datetime,
    ) -> FilterResult:
        """Apply a specific filter."""
        if filter_type == FilterType.CONFIDENCE_THRESHOLD:
            return self._filter_confidence(signal)

        elif filter_type == FilterType.URGENCY_THRESHOLD:
            return self._filter_urgency(signal)

        elif filter_type == FilterType.TIME_WINDOW:
            return self._filter_time_window(signal, timestamp)

        elif filter_type == FilterType.REGIME_GATE:
            return self._filter_regime_gate(signal)

        elif filter_type == FilterType.CORRELATION_DEDUP:
            return self._filter_correlation(signal, timestamp)

        elif filter_type == FilterType.RATE_LIMIT:
            return self._filter_rate_limit(signal, timestamp)

        elif filter_type == FilterType.VOLATILITY_GATE:
            return self._filter_volatility(signal)

        elif filter_type == FilterType.DRAWDOWN_GATE:
            return self._filter_drawdown(signal)

        return FilterResult(passed=True, filter_type=filter_type)

    def _filter_confidence(self, signal: Signal) -> FilterResult:
        """Filter by minimum confidence."""
        if signal.confidence < self.confidence_threshold:
            return FilterResult(
                passed=False,
                filter_type=FilterType.CONFIDENCE_THRESHOLD,
                reason=f"Confidence {signal.confidence:.2f} below threshold {self.confidence_threshold}",
                original_signal=signal,
            )
        return FilterResult(
            passed=True,
            filter_type=FilterType.CONFIDENCE_THRESHOLD,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_urgency(self, signal: Signal) -> FilterResult:
        """Filter by minimum urgency."""
        if signal.urgency < self.urgency_threshold:
            return FilterResult(
                passed=False,
                filter_type=FilterType.URGENCY_THRESHOLD,
                reason=f"Urgency {signal.urgency:.2f} below threshold {self.urgency_threshold}",
                original_signal=signal,
            )
        return FilterResult(
            passed=True,
            filter_type=FilterType.URGENCY_THRESHOLD,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_time_window(
        self, signal: Signal, timestamp: datetime
    ) -> FilterResult:
        """Filter by trading hours."""
        current_time = timestamp.time()

        if current_time < self.trading_start or current_time > self.trading_end:
            return FilterResult(
                passed=False,
                filter_type=FilterType.TIME_WINDOW,
                reason=f"Outside trading hours ({self.trading_start} - {self.trading_end})",
                original_signal=signal,
            )

        # Also filter first/last 5 minutes (high volatility)
        minutes_from_open = (
            (current_time.hour - self.trading_start.hour) * 60 +
            (current_time.minute - self.trading_start.minute)
        )
        minutes_to_close = (
            (self.trading_end.hour - current_time.hour) * 60 +
            (self.trading_end.minute - current_time.minute)
        )

        if minutes_from_open < 5 or minutes_to_close < 5:
            return FilterResult(
                passed=False,
                filter_type=FilterType.TIME_WINDOW,
                reason="Too close to market open/close",
                original_signal=signal,
            )

        return FilterResult(
            passed=True,
            filter_type=FilterType.TIME_WINDOW,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_regime_gate(self, signal: Signal) -> FilterResult:
        """Filter signals that don't match current regime."""
        if self._current_regime is None:
            return FilterResult(
                passed=True,
                filter_type=FilterType.REGIME_GATE,
                original_signal=signal,
                modified_signal=signal,
            )

        allowed = self._regime_sources.get(self._current_regime, list(SignalSource))

        if signal.source not in allowed:
            return FilterResult(
                passed=False,
                filter_type=FilterType.REGIME_GATE,
                reason=f"Source {signal.source.name} not allowed in regime {self._current_regime}",
                original_signal=signal,
            )

        return FilterResult(
            passed=True,
            filter_type=FilterType.REGIME_GATE,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_correlation(
        self, signal: Signal, timestamp: datetime
    ) -> FilterResult:
        """Filter signals highly correlated with recent signals."""
        # Look at signals in last 5 minutes
        cutoff = timestamp - timedelta(minutes=5)
        recent = [
            (ts, s) for ts, s in self._recent_signals
            if ts > cutoff and s.symbol == signal.symbol
        ]

        for ts, recent_signal in recent:
            # Check if same source and similar direction
            if recent_signal.source == signal.source:
                direction_similarity = (
                    signal.direction * recent_signal.direction
                )

                if direction_similarity > self.correlation_threshold:
                    return FilterResult(
                        passed=False,
                        filter_type=FilterType.CORRELATION_DEDUP,
                        reason=f"Too similar to signal from {ts}",
                        original_signal=signal,
                    )

        return FilterResult(
            passed=True,
            filter_type=FilterType.CORRELATION_DEDUP,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_rate_limit(
        self, signal: Signal, timestamp: datetime
    ) -> FilterResult:
        """Limit signals per hour."""
        cutoff = timestamp - timedelta(hours=1)
        recent_count = sum(
            1 for ts, s in self._recent_signals
            if ts > cutoff and s.symbol == signal.symbol
        )

        if recent_count >= self.max_signals_per_hour:
            return FilterResult(
                passed=False,
                filter_type=FilterType.RATE_LIMIT,
                reason=f"Rate limit exceeded ({recent_count} signals in last hour)",
                original_signal=signal,
            )

        return FilterResult(
            passed=True,
            filter_type=FilterType.RATE_LIMIT,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_volatility(self, signal: Signal) -> FilterResult:
        """
        Filter during extreme volatility.

        Reduces confidence during high volatility periods.
        """
        # This would need volatility data injected
        # For now, pass through with potential confidence reduction
        return FilterResult(
            passed=True,
            filter_type=FilterType.VOLATILITY_GATE,
            original_signal=signal,
            modified_signal=signal,
        )

    def _filter_drawdown(self, signal: Signal) -> FilterResult:
        """Filter when in significant drawdown."""
        if self._current_equity <= 0:
            return FilterResult(
                passed=True,
                filter_type=FilterType.DRAWDOWN_GATE,
                original_signal=signal,
                modified_signal=signal,
            )

        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity

        if drawdown > self.max_drawdown_pct:
            # In drawdown - only allow high confidence signals
            if signal.confidence < 0.8:
                return FilterResult(
                    passed=False,
                    filter_type=FilterType.DRAWDOWN_GATE,
                    reason=f"In drawdown ({drawdown:.1%}), need confidence > 0.8",
                    original_signal=signal,
                )

        return FilterResult(
            passed=True,
            filter_type=FilterType.DRAWDOWN_GATE,
            original_signal=signal,
            modified_signal=signal,
        )

    def set_regime(self, regime: str) -> None:
        """Set current market regime."""
        self._current_regime = regime

    def update_equity(self, pnl: float) -> None:
        """Update equity tracking."""
        self._current_equity += pnl
        self._daily_pnl += pnl

        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self._daily_pnl = 0.0

    def configure_regime_sources(
        self,
        regime: str,
        allowed_sources: List[SignalSource],
    ) -> None:
        """Configure which sources are allowed in a regime."""
        self._regime_sources[regime] = allowed_sources

    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics on filter activations."""
        # Would track filter hits in production
        return {
            "recent_signals": len(self._recent_signals),
            "current_drawdown_pct": (
                (self._peak_equity - self._current_equity) / self._peak_equity
                if self._peak_equity > 0 else 0
            ),
            "daily_pnl": self._daily_pnl,
        }

    def inject_backtest_state(
        self,
        equity: float,
        peak_equity: float,
        regime: Optional[str] = None,
    ) -> None:
        """Inject state for backtesting."""
        self._current_equity = equity
        self._peak_equity = peak_equity
        self._current_regime = regime
