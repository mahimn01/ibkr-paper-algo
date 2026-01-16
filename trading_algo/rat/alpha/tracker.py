"""
Alpha Decay Tracker: Monitor factor performance and detect crowding.

Uses mathematical decay detection:
1. Rolling Sharpe degradation
2. Return autocorrelation collapse
3. Volume correlation increase (crowding signal)
4. Information coefficient decay

No AI required - pure statistical tracking.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


class DecayStage(Enum):
    """Alpha decay lifecycle stages."""

    FRESH = auto()          # New factor, untested
    ALPHA = auto()          # Generating excess returns
    MATURE = auto()         # Still working but growth slowing
    DECAYING = auto()       # Performance deteriorating
    CROWDED = auto()        # Too many using it, negative edge
    DEAD = auto()           # No longer viable


@dataclass
class AlphaFactor:
    """A tracked alpha factor."""

    name: str
    compute_fn: Callable[[Dict], float]     # Function to compute factor value
    creation_time: datetime = field(default_factory=datetime.now)

    # Performance tracking
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=252))
    predictions: Deque[float] = field(default_factory=lambda: deque(maxlen=252))
    actuals: Deque[float] = field(default_factory=lambda: deque(maxlen=252))

    # Decay metrics
    rolling_sharpe: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    rolling_ic: Deque[float] = field(default_factory=lambda: deque(maxlen=50))  # Info coef

    # Crowding metrics
    volume_correlation: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # State
    current_stage: DecayStage = DecayStage.FRESH
    last_update: Optional[datetime] = None
    total_return: float = 0.0
    trade_count: int = 0

    def is_viable(self) -> bool:
        """Check if factor is still viable for trading."""
        return self.current_stage in (DecayStage.FRESH, DecayStage.ALPHA, DecayStage.MATURE)


@dataclass
class AlphaState:
    """Current state of alpha analysis."""

    timestamp: datetime
    active_factors: List[str]
    decaying_factors: List[str]
    dead_factors: List[str]
    best_factor: Optional[str]
    needs_mutation: bool
    overall_alpha_health: float  # 0-1, aggregate health


class AlphaTracker:
    """
    Track alpha factor performance and detect decay.

    Mathematical approach:
    1. Rolling Sharpe ratio monitoring
    2. Information coefficient tracking
    3. Volume correlation for crowding
    4. Return autocorrelation analysis
    """

    def __init__(
        self,
        sharpe_window: int = 20,
        ic_window: int = 20,
        decay_threshold: float = 0.5,
        crowding_threshold: float = 0.7,
    ):
        self.sharpe_window = sharpe_window
        self.ic_window = ic_window
        self.decay_threshold = decay_threshold
        self.crowding_threshold = crowding_threshold

        # Registered factors
        self._factors: Dict[str, AlphaFactor] = {}

        # Market data for crowding detection
        self._market_volume: Deque[float] = deque(maxlen=252)

        # Historical factor signals for correlation
        self._factor_signals: Dict[str, Deque[float]] = {}

        # Last state
        self._last_state: Optional[AlphaState] = None

    def register_factor(
        self,
        name: str,
        compute_fn: Callable[[Dict], float],
    ) -> None:
        """Register a new alpha factor for tracking."""
        self._factors[name] = AlphaFactor(name=name, compute_fn=compute_fn)
        self._factor_signals[name] = deque(maxlen=252)

    def update_factor_performance(
        self,
        name: str,
        prediction: float,
        actual: float,
        pnl: float,
        market_volume: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update factor with new performance data."""
        if name not in self._factors:
            return

        ts = timestamp or datetime.now()
        factor = self._factors[name]

        # Record raw data
        factor.predictions.append(prediction)
        factor.actuals.append(actual)
        factor.returns.append(pnl)
        factor.total_return += pnl
        factor.trade_count += 1
        factor.last_update = ts

        # Update market volume
        self._market_volume.append(market_volume)

        # Store signal for correlation
        self._factor_signals[name].append(prediction)

        # Compute rolling metrics
        if len(factor.returns) >= self.sharpe_window:
            sharpe = self._compute_rolling_sharpe(list(factor.returns)[-self.sharpe_window:])
            factor.rolling_sharpe.append(sharpe)

        if len(factor.predictions) >= self.ic_window:
            ic = self._compute_information_coefficient(
                list(factor.predictions)[-self.ic_window:],
                list(factor.actuals)[-self.ic_window:],
            )
            factor.rolling_ic.append(ic)

        if len(self._factor_signals[name]) >= 20 and len(self._market_volume) >= 20:
            vol_corr = self._compute_volume_correlation(
                list(self._factor_signals[name])[-20:],
                list(self._market_volume)[-20:],
            )
            factor.volume_correlation.append(vol_corr)

        # Update stage
        factor.current_stage = self._determine_stage(factor)

    def _compute_rolling_sharpe(self, returns: List[float]) -> float:
        """Compute Sharpe ratio (assuming risk-free = 0)."""
        if not returns:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = math.sqrt(var_ret) if var_ret > 0 else 1e-10

        # Annualize assuming daily returns
        return (mean_ret / std_ret) * math.sqrt(252)

    def _compute_information_coefficient(
        self,
        predictions: List[float],
        actuals: List[float],
    ) -> float:
        """
        Compute Information Coefficient (rank correlation).

        IC = correlation between predicted and actual returns.
        """
        if len(predictions) != len(actuals) or len(predictions) < 3:
            return 0.0

        n = len(predictions)

        # Spearman rank correlation
        pred_ranks = self._rank(predictions)
        actual_ranks = self._rank(actuals)

        # Correlation of ranks
        mean_pred = sum(pred_ranks) / n
        mean_actual = sum(actual_ranks) / n

        numerator = sum(
            (pred_ranks[i] - mean_pred) * (actual_ranks[i] - mean_actual)
            for i in range(n)
        )
        denom_pred = math.sqrt(sum((r - mean_pred) ** 2 for r in pred_ranks))
        denom_actual = math.sqrt(sum((r - mean_actual) ** 2 for r in actual_ranks))

        if denom_pred * denom_actual == 0:
            return 0.0

        return numerator / (denom_pred * denom_actual)

    def _rank(self, values: List[float]) -> List[float]:
        """Compute ranks (1 = smallest)."""
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    def _compute_volume_correlation(
        self,
        signals: List[float],
        volumes: List[float],
    ) -> float:
        """
        Compute correlation between signal magnitude and market volume.

        High correlation = crowding (everyone trading same signal).
        """
        if len(signals) != len(volumes) or len(signals) < 3:
            return 0.0

        # Use absolute signal values
        abs_signals = [abs(s) for s in signals]

        n = len(signals)
        mean_sig = sum(abs_signals) / n
        mean_vol = sum(volumes) / n

        numerator = sum(
            (abs_signals[i] - mean_sig) * (volumes[i] - mean_vol)
            for i in range(n)
        )
        denom_sig = math.sqrt(sum((s - mean_sig) ** 2 for s in abs_signals))
        denom_vol = math.sqrt(sum((v - mean_vol) ** 2 for v in volumes))

        if denom_sig * denom_vol == 0:
            return 0.0

        return numerator / (denom_sig * denom_vol)

    def _determine_stage(self, factor: AlphaFactor) -> DecayStage:
        """Determine the current decay stage of a factor."""
        if factor.trade_count < 20:
            return DecayStage.FRESH

        # Check for crowding first (most severe)
        if factor.volume_correlation:
            recent_vol_corr = list(factor.volume_correlation)[-5:]
            avg_vol_corr = sum(recent_vol_corr) / len(recent_vol_corr)
            if avg_vol_corr > self.crowding_threshold:
                return DecayStage.CROWDED

        # Check Sharpe trajectory
        if len(factor.rolling_sharpe) >= 10:
            sharpes = list(factor.rolling_sharpe)
            early_sharpe = sum(sharpes[:5]) / 5
            recent_sharpe = sum(sharpes[-5:]) / 5

            if recent_sharpe < 0:
                return DecayStage.DEAD
            elif recent_sharpe < early_sharpe * self.decay_threshold:
                return DecayStage.DECAYING
            elif recent_sharpe < early_sharpe * 0.8:
                return DecayStage.MATURE
            else:
                return DecayStage.ALPHA

        # Check IC trajectory
        if len(factor.rolling_ic) >= 10:
            ics = list(factor.rolling_ic)
            early_ic = sum(ics[:5]) / 5
            recent_ic = sum(ics[-5:]) / 5

            if recent_ic < 0.02:
                return DecayStage.DEAD
            elif recent_ic < early_ic * self.decay_threshold:
                return DecayStage.DECAYING
            elif recent_ic < early_ic * 0.8:
                return DecayStage.MATURE
            else:
                return DecayStage.ALPHA

        return DecayStage.FRESH

    def analyze(self) -> AlphaState:
        """Analyze current state of all tracked factors."""
        active = []
        decaying = []
        dead = []
        best_factor = None
        best_sharpe = float('-inf')

        for name, factor in self._factors.items():
            if factor.current_stage in (DecayStage.FRESH, DecayStage.ALPHA, DecayStage.MATURE):
                active.append(name)

                # Track best by recent Sharpe
                if factor.rolling_sharpe:
                    recent_sharpe = factor.rolling_sharpe[-1]
                    if recent_sharpe > best_sharpe:
                        best_sharpe = recent_sharpe
                        best_factor = name

            elif factor.current_stage == DecayStage.DECAYING:
                decaying.append(name)

            else:
                dead.append(name)

        # Need mutation if most factors are decaying/dead
        total_factors = len(self._factors)
        viable_factors = len(active)
        needs_mutation = (
            total_factors > 0 and
            viable_factors / total_factors < 0.3
        )

        # Compute overall health
        if total_factors == 0:
            health = 0.5
        else:
            health = viable_factors / total_factors

        state = AlphaState(
            timestamp=datetime.now(),
            active_factors=active,
            decaying_factors=decaying,
            dead_factors=dead,
            best_factor=best_factor,
            needs_mutation=needs_mutation,
            overall_alpha_health=health,
        )

        self._last_state = state
        return state

    def get_factor_signal(self, name: str, data: Dict) -> Optional[float]:
        """Get signal from a specific factor."""
        if name not in self._factors:
            return None

        factor = self._factors[name]
        if not factor.is_viable():
            return None

        try:
            return factor.compute_fn(data)
        except Exception:
            return None

    def get_best_signal(self, data: Dict) -> Optional[Tuple[str, float]]:
        """Get signal from the best performing factor."""
        state = self.analyze()

        if not state.best_factor:
            return None

        signal = self.get_factor_signal(state.best_factor, data)
        if signal is None:
            return None

        return (state.best_factor, signal)

    def generate_signal(self, symbol: str, data: Dict) -> Optional[Signal]:
        """Generate trading signal from best factor."""
        result = self.get_best_signal(data)
        if result is None:
            return None

        factor_name, signal_value = result

        if abs(signal_value) < 0.1:  # Too weak
            return None

        factor = self._factors[factor_name]

        # Confidence based on factor health
        confidence = 0.5
        if factor.rolling_sharpe:
            confidence = min(0.95, max(0.3, factor.rolling_sharpe[-1] / 3.0))
        if factor.rolling_ic:
            confidence = min(0.95, confidence * (1 + factor.rolling_ic[-1]))

        signal_type = SignalType.LONG if signal_value > 0 else SignalType.SHORT

        return Signal(
            source=SignalSource.ALPHA,
            signal_type=signal_type,
            symbol=symbol,
            direction=1.0 if signal_value > 0 else -1.0,
            confidence=confidence,
            urgency=0.5,
            metadata={
                "factor_name": factor_name,
                "factor_stage": factor.current_stage.name,
                "signal_value": signal_value,
                "rolling_sharpe": factor.rolling_sharpe[-1] if factor.rolling_sharpe else None,
                "rolling_ic": factor.rolling_ic[-1] if factor.rolling_ic else None,
            },
        )

    def inject_backtest_data(
        self,
        factor_name: str,
        history: List[Dict],
    ) -> None:
        """
        Inject historical performance data for backtesting.

        Args:
            factor_name: Name of the factor
            history: List of dicts with keys:
                - timestamp: datetime
                - prediction: float
                - actual: float
                - pnl: float
                - market_volume: float
        """
        for record in history:
            self.update_factor_performance(
                name=factor_name,
                prediction=record["prediction"],
                actual=record["actual"],
                pnl=record["pnl"],
                market_volume=record["market_volume"],
                timestamp=record.get("timestamp"),
            )
