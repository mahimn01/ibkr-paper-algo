"""
Signal Combiner: Optimally combine signals from all RAT modules.

Mathematical approaches:
1. Inverse-variance weighting
2. Kelly-optimal weighting
3. Regime-conditional weighting
4. Performance-adaptive weighting

No AI required - pure mathematical signal combination.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource, CombinedSignal


class WeightingMethod(Enum):
    """Signal weighting methods."""

    EQUAL = auto()                  # Equal weights
    INVERSE_VARIANCE = auto()       # Weight by 1/variance
    SHARPE_WEIGHTED = auto()        # Weight by recent Sharpe
    KELLY_OPTIMAL = auto()          # Kelly criterion weights
    REGIME_CONDITIONAL = auto()     # Different weights per regime
    ADAPTIVE = auto()               # Online learning of weights


@dataclass
class SourcePerformance:
    """Track performance of each signal source."""

    source: SignalSource
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    predictions: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    actuals: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def mean_return(self) -> float:
        if not self.returns:
            return 0.0
        return sum(self.returns) / len(self.returns)

    @property
    def variance(self) -> float:
        if len(self.returns) < 2:
            return 1.0
        mean = self.mean_return
        return sum((r - mean) ** 2 for r in self.returns) / len(self.returns)

    @property
    def sharpe(self) -> float:
        if not self.returns:
            return 0.0
        if self.variance == 0:
            # Zero variance with positive mean return implies a perfect signal.
            return 10.0 if self.mean_return > 0 else (-10.0 if self.mean_return < 0 else 0.0)
        return self.mean_return / math.sqrt(self.variance) * math.sqrt(252)

    @property
    def win_rate(self) -> float:
        if not self.returns:
            return 0.5
        wins = sum(1 for r in self.returns if r > 0)
        return wins / len(self.returns)


@dataclass
class CombinedDecision:
    """Final combined trading decision."""

    timestamp: datetime
    symbol: str
    action: str                     # "buy", "sell", "hold"
    signal_type: SignalType
    direction: float                # -1 to 1
    confidence: float               # 0 to 1
    urgency: float                  # 0 to 1
    position_size_pct: float        # Suggested position size
    contributing_sources: List[SignalSource]
    weights_used: Dict[SignalSource, float]
    raw_signals: Dict[SignalSource, float]

    def should_trade(self, min_confidence: float = 0.5) -> bool:
        """Check if decision warrants a trade."""
        return (
            self.action != "hold" and
            self.confidence >= min_confidence and
            abs(self.direction) > 0.1
        )


class SignalCombiner:
    """
    Combine signals from multiple RAT modules.

    Uses mathematical weighting based on:
    - Recent performance of each source
    - Current market regime
    - Signal agreement/divergence
    """

    def __init__(
        self,
        weighting_method: WeightingMethod = WeightingMethod.SHARPE_WEIGHTED,
        min_signals_required: int = 2,
        agreement_threshold: float = 0.6,
        max_position_pct: float = 0.25,
    ):
        self.weighting_method = weighting_method
        self.min_signals_required = min_signals_required
        self.agreement_threshold = agreement_threshold
        self.max_position_pct = max_position_pct

        # Performance tracking per source
        self._performance: Dict[SignalSource, SourcePerformance] = {
            source: SourcePerformance(source=source)
            for source in SignalSource
        }

        # Regime-conditional weights (populated if using regime weighting)
        self._regime_weights: Dict[str, Dict[SignalSource, float]] = {}

        # Online weight learning state
        self._adaptive_weights: Dict[SignalSource, float] = {
            source: 1.0 / len(SignalSource) for source in SignalSource
        }
        self._learning_rate = 0.1

        # Decision history
        self._decisions: Deque[CombinedDecision] = deque(maxlen=1000)

    def combine(
        self,
        signals: List[Signal],
        current_regime: Optional[str] = None,
    ) -> CombinedDecision:
        """Combine multiple signals into a single decision."""
        symbol = signals[0].symbol if signals else "UNKNOWN"
        timestamp = datetime.now()

        if len(signals) < self.min_signals_required:
            return self._create_hold_decision(symbol, timestamp, signals)

        # Calculate weights based on method
        weights = self._calculate_weights(signals, current_regime)

        # Combine signals
        weighted_direction = 0.0
        weighted_confidence = 0.0
        weighted_urgency = 0.0
        total_weight = 0.0

        raw_signals: Dict[SignalSource, float] = {}
        contributing_sources: List[SignalSource] = []

        for signal in signals:
            w = weights.get(signal.source, 0.0)
            if w > 0:
                weighted_direction += signal.direction * signal.confidence * w
                weighted_confidence += signal.confidence * w
                weighted_urgency += signal.urgency * w
                total_weight += w
                raw_signals[signal.source] = signal.direction
                contributing_sources.append(signal.source)

        if total_weight == 0:
            return self._create_hold_decision(symbol, timestamp, signals)

        # Normalize
        final_direction = weighted_direction / total_weight
        final_confidence = weighted_confidence / total_weight
        final_urgency = weighted_urgency / total_weight

        # Check agreement
        agreement = self._calculate_agreement(signals)
        if agreement < self.agreement_threshold:
            # Low agreement reduces confidence
            final_confidence *= agreement

        # Determine action
        if abs(final_direction) < 0.1:
            action = "hold"
            signal_type = SignalType.HOLD
        elif final_direction > 0:
            action = "buy"
            signal_type = SignalType.LONG
        else:
            action = "sell"
            signal_type = SignalType.SHORT

        # Calculate position size (Kelly-inspired)
        position_size = self._calculate_position_size(
            final_direction, final_confidence, signals
        )

        decision = CombinedDecision(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            signal_type=signal_type,
            direction=final_direction,
            confidence=final_confidence,
            urgency=final_urgency,
            position_size_pct=position_size,
            contributing_sources=contributing_sources,
            weights_used=weights,
            raw_signals=raw_signals,
        )

        self._decisions.append(decision)
        return decision

    def _calculate_weights(
        self,
        signals: List[Signal],
        current_regime: Optional[str] = None,
    ) -> Dict[SignalSource, float]:
        """Calculate weights for each signal source."""
        sources = {s.source for s in signals}

        if self.weighting_method == WeightingMethod.EQUAL:
            return {source: 1.0 / len(sources) for source in sources}

        elif self.weighting_method == WeightingMethod.INVERSE_VARIANCE:
            return self._inverse_variance_weights(sources)

        elif self.weighting_method == WeightingMethod.SHARPE_WEIGHTED:
            return self._sharpe_weights(sources)

        elif self.weighting_method == WeightingMethod.KELLY_OPTIMAL:
            return self._kelly_weights(sources)

        elif self.weighting_method == WeightingMethod.REGIME_CONDITIONAL:
            return self._regime_weights.get(
                current_regime or "default",
                {source: 1.0 / len(sources) for source in sources}
            )

        elif self.weighting_method == WeightingMethod.ADAPTIVE:
            return {
                source: self._adaptive_weights.get(source, 0.0)
                for source in sources
            }

        return {source: 1.0 / len(sources) for source in sources}

    def _inverse_variance_weights(
        self, sources: set[SignalSource]
    ) -> Dict[SignalSource, float]:
        """Weight inversely proportional to variance."""
        weights = {}
        total = 0.0

        for source in sources:
            perf = self._performance[source]
            var = max(perf.variance, 0.001)  # Avoid division by zero
            w = 1.0 / var
            weights[source] = w
            total += w

        # Normalize
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _sharpe_weights(self, sources: set[SignalSource]) -> Dict[SignalSource, float]:
        """Weight by recent Sharpe ratio."""
        weights = {}
        total = 0.0

        for source in sources:
            perf = self._performance[source]
            # Use max(0, sharpe) to avoid negative weights
            sharpe = max(0.0, perf.sharpe)
            # Add small constant to avoid zero weights for new sources
            w = sharpe + 0.1
            weights[source] = w
            total += w

        # Normalize
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _kelly_weights(self, sources: set[SignalSource]) -> Dict[SignalSource, float]:
        """
        Kelly criterion-inspired weights.

        Kelly fraction: f = (p * b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio
        """
        weights = {}
        total = 0.0

        for source in sources:
            perf = self._performance[source]

            if len(perf.returns) < 10:
                w = 0.1  # Default for new sources
            else:
                p = perf.win_rate
                q = 1 - p

                # Calculate average win and loss
                wins = [r for r in perf.returns if r > 0]
                losses = [abs(r) for r in perf.returns if r < 0]

                avg_win = sum(wins) / len(wins) if wins else 0.01
                avg_loss = sum(losses) / len(losses) if losses else 0.01

                b = avg_win / avg_loss  # Win/loss ratio

                # Kelly fraction
                kelly = (p * b - q) / b if b > 0 else 0

                # Fractional Kelly (more conservative)
                w = max(0, kelly * 0.5)

            weights[source] = w
            total += w

        # Normalize
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}
        else:
            # Fallback to equal
            weights = {s: 1.0 / len(sources) for s in sources}

        return weights

    def _calculate_agreement(self, signals: List[Signal]) -> float:
        """Calculate agreement among signals (0-1)."""
        if not signals:
            return 0.0

        directions = [s.direction for s in signals]
        n = len(directions)

        if n == 1:
            return 1.0

        # Count signals agreeing on direction
        positive = sum(1 for d in directions if d > 0)
        negative = n - positive

        # Agreement is max proportion in same direction
        agreement = max(positive, negative) / n

        return agreement

    def _calculate_position_size(
        self,
        direction: float,
        confidence: float,
        signals: List[Signal],
    ) -> float:
        """
        Calculate suggested position size.

        Uses a simplified Kelly-inspired approach:
        - Higher confidence = larger position
        - Higher agreement = larger position
        - Capped at max_position_pct
        """
        base_size = abs(direction) * confidence

        # Adjust for agreement
        agreement = self._calculate_agreement(signals)
        adjusted_size = base_size * agreement

        # Apply Kelly-style scaling based on edge estimate
        edge_estimate = confidence - 0.5  # Simple edge proxy
        if edge_estimate > 0:
            kelly_scale = min(2.0, 1.0 + edge_estimate)
            adjusted_size *= kelly_scale

        # Cap at maximum
        return min(adjusted_size, self.max_position_pct)

    def _create_hold_decision(
        self,
        symbol: str,
        timestamp: datetime,
        signals: List[Signal],
    ) -> CombinedDecision:
        """Create a hold decision when signals are insufficient."""
        return CombinedDecision(
            timestamp=timestamp,
            symbol=symbol,
            action="hold",
            signal_type=SignalType.HOLD,
            direction=0.0,
            confidence=0.0,
            urgency=0.0,
            position_size_pct=0.0,
            contributing_sources=[],
            weights_used={},
            raw_signals={s.source: s.direction for s in signals},
        )

    def update_performance(
        self,
        source: SignalSource,
        prediction: float,
        actual: float,
        pnl: float,
    ) -> None:
        """Update performance tracking for a source."""
        perf = self._performance[source]
        perf.predictions.append(prediction)
        perf.actuals.append(actual)
        perf.returns.append(pnl)

        # Update adaptive weights
        if self.weighting_method == WeightingMethod.ADAPTIVE:
            self._update_adaptive_weights(source, pnl)

    def _update_adaptive_weights(self, source: SignalSource, pnl: float) -> None:
        """Online weight update using exponential gradient."""
        # Reward sources with positive returns
        reward = 1.0 if pnl > 0 else -0.5

        # Update weight
        old_weight = self._adaptive_weights[source]
        new_weight = old_weight * math.exp(self._learning_rate * reward)
        self._adaptive_weights[source] = new_weight

        # Normalize all weights
        total = sum(self._adaptive_weights.values())
        if total > 0:
            self._adaptive_weights = {
                s: w / total for s, w in self._adaptive_weights.items()
            }

    def set_regime_weights(
        self,
        regime: str,
        weights: Dict[SignalSource, float],
    ) -> None:
        """Set weights for a specific market regime."""
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}
        self._regime_weights[regime] = weights

    def get_source_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all sources."""
        stats = {}
        for source, perf in self._performance.items():
            stats[source.name] = {
                "mean_return": perf.mean_return,
                "variance": perf.variance,
                "sharpe": perf.sharpe,
                "win_rate": perf.win_rate,
                "n_observations": len(perf.returns),
            }
        return stats

    def inject_backtest_performance(
        self,
        source: SignalSource,
        returns: List[float],
    ) -> None:
        """Inject historical performance for backtesting."""
        perf = self._performance[source]
        for r in returns:
            perf.returns.append(r)
