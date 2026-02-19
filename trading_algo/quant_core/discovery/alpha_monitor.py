"""
Alpha Decay Monitor -- Real-time signal health monitoring and lifecycle management.

Detects when alpha is decaying and automatically manages the signal lifecycle,
transitioning signals through HEALTHY -> WARNING -> DEGRADED -> RETIRED states.

Academic foundation:
    - McLean & Pontiff (2016): "Does Academic Research Destroy Stock Return
      Predictability?" -- Average 35% decay in published anomalies post-publication,
      58% decline in portfolio returns.
    - Chordia, Subrahmanyam & Tong (2014): "Have Capital Market Anomalies
      Attenuated?" -- Many anomalies have decayed to zero.

Key insight: alpha decays. The question is how fast, and when to stop trading.
This module provides the infrastructure to answer both questions in real time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from trading_algo.quant_core.utils.constants import (
    EPSILON,
    SQRT_252,
)
from trading_algo.quant_core.utils.math_utils import (
    rolling_mean,
    rolling_std,
    ewma_volatility,
    simple_returns,
)

logger = logging.getLogger(__name__)

# =============================================================================
# LIFECYCLE STATES
# =============================================================================

STATUS_HEALTHY: str = "healthy"
STATUS_WARNING: str = "warning"
STATUS_DEGRADED: str = "degraded"
STATUS_RETIRED: str = "retired"

_VALID_STATUSES = frozenset({STATUS_HEALTHY, STATUS_WARNING, STATUS_DEGRADED, STATUS_RETIRED})


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignalHealth:
    """
    Comprehensive health snapshot for a single alpha signal.

    Captures the current state of a signal across multiple dimensions:
    rolling information coefficient (IC), P&L trajectory, drawdown, and
    correlation drift relative to other monitored signals.

    Attributes:
        signal_name: Unique identifier for the signal.
        status: Lifecycle state -- one of "healthy", "warning", "degraded",
            or "retired".
        current_ic: Most recently computed IC value.
        rolling_ic_30d: 30-day rolling average IC.
        rolling_ic_90d: 90-day rolling average IC.
        rolling_ic_252d: 252-day rolling average IC.
        ic_trend: OLS slope of the IC series over the available lookback
            window. Negative values indicate decay.
        half_life_estimate: Estimated half-life of alpha decay in trading days.
        days_since_last_positive_ic: Number of observations since the last
            day with a positive IC reading.
        cumulative_pnl: Total P&L attributed to this signal since inception.
        recent_pnl_30d: P&L from the most recent 30 observations.
        sharpe_ratio_90d: Annualized Sharpe ratio over the last 90
            observations.
        drawdown_from_peak: Fractional drawdown of cumulative P&L from its
            peak value (0.0 = at peak, 1.0 = total loss).
        correlation_drift: Absolute change in average pairwise correlation
            between this signal and all other monitored signals, measured over
            the configured lookback window.
        last_updated: Timestamp of the most recent update.
    """

    signal_name: str
    status: str
    current_ic: float
    rolling_ic_30d: float
    rolling_ic_90d: float
    rolling_ic_252d: float
    ic_trend: float
    half_life_estimate: float
    days_since_last_positive_ic: int
    cumulative_pnl: float
    recent_pnl_30d: float
    sharpe_ratio_90d: float
    drawdown_from_peak: float
    correlation_drift: float
    last_updated: datetime


@dataclass
class AlphaDecayMetrics:
    """
    Quantitative assessment of alpha decay for a single signal.

    Combines exponential-decay modelling with structural-break detection
    to give an actionable picture of remaining alpha and projected
    exhaustion.

    Attributes:
        signal_name: Unique identifier for the signal.
        initial_ic: IC at signal inception (or the earliest available
            estimate).
        current_ic: Most recent IC value.
        decay_rate: Annualized exponential decay rate lambda, where
            IC(t) = IC_0 * exp(-lambda * t). Values in [0, 1] indicate
            gradual decay; values > 1 indicate rapid decay.
        half_life_days: Estimated number of trading days for the IC to
            halve, derived from the decay rate as ln(2) / lambda.
        structural_break: Whether a CUSUM test detected a structural
            break in the IC series.
        break_date: Timestamp of the detected structural break, or None
            if no break was detected.
        remaining_alpha_pct: Estimated percentage of original alpha that
            remains, in the range [0, 100].
        projected_zero_date: Projected date when the IC will reach zero
            under a linear extrapolation of the current trend, or None if
            the trend is non-negative.
    """

    signal_name: str
    initial_ic: float
    current_ic: float
    decay_rate: float
    half_life_days: float
    structural_break: bool
    break_date: Optional[datetime]
    remaining_alpha_pct: float
    projected_zero_date: Optional[datetime]


@dataclass
class MonitorConfig:
    """
    Configuration for the AlphaDecayMonitor.

    All window sizes are expressed in number of observations (typically
    trading days).  Thresholds govern the transitions between lifecycle
    states and the sensitivity of decay / break detection.

    Attributes:
        ic_window_short: Short-term rolling IC window.
        ic_window_medium: Medium-term rolling IC window.
        ic_window_long: Long-term rolling IC window.
        warning_ic_threshold: IC below this level triggers a WARNING.
        degraded_ic_threshold: IC below this level triggers DEGRADED.
        retire_after_degraded_days: Number of consecutive observations in
            DEGRADED before automatic retirement.
        decay_lookback: Number of observations used for fitting the
            exponential decay model (~2 years).
        structural_break_threshold: Z-score threshold for the CUSUM
            structural break test.
        min_observations: Minimum observations required before any
            health assessment is made.
        max_signal_age_days: Automatic retirement age (~3 years).
        max_drawdown_pct: Retire if cumulative P&L drawdown exceeds
            this fraction of peak.
        min_sharpe_90d: Retire if the 90-day Sharpe falls below this
            level.
        correlation_drift_threshold: Alert if the change in average
            pairwise correlation exceeds this value.
        correlation_lookback: Window for measuring correlation drift.
    """

    # IC monitoring
    ic_window_short: int = 30
    ic_window_medium: int = 90
    ic_window_long: int = 252

    # Health thresholds
    warning_ic_threshold: float = 0.005
    degraded_ic_threshold: float = 0.0
    retire_after_degraded_days: int = 60

    # Decay detection
    decay_lookback: int = 504
    structural_break_threshold: float = 2.0
    min_observations: int = 60

    # Signal lifecycle
    max_signal_age_days: int = 756
    max_drawdown_pct: float = 0.50
    min_sharpe_90d: float = -0.5

    # Correlation monitoring
    correlation_drift_threshold: float = 0.20
    correlation_lookback: int = 60


# =============================================================================
# INTERNAL STATE CONTAINER
# =============================================================================

@dataclass
class _SignalState:
    """
    Mutable internal bookkeeping for a single monitored signal.

    This is *not* part of the public API; callers interact with
    :class:`SignalHealth` and :class:`AlphaDecayMetrics`.
    """

    name: str
    inception_date: datetime
    initial_ic: float

    # Raw observation history (appended each update)
    signal_values: List[float] = field(default_factory=list)
    realized_returns: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Derived IC series (one entry per update)
    ic_series: List[float] = field(default_factory=list)

    # P&L tracking
    pnl_series: List[float] = field(default_factory=list)
    cumulative_pnl: float = 0.0
    peak_cumulative_pnl: float = 0.0

    # Lifecycle
    status: str = STATUS_HEALTHY
    degraded_since: Optional[datetime] = None
    retired_reason: Optional[str] = None

    # Correlation baseline (set after initial warm-up)
    baseline_correlations: Optional[Dict[str, float]] = None

    # Last computed health (cached between updates)
    last_health: Optional[SignalHealth] = None


# =============================================================================
# CUSUM TEST
# =============================================================================

def _cusum_test(
    series: NDArray[np.float64],
    threshold: float = 2.0,
) -> Tuple[bool, Optional[int]]:
    """
    CUSUM (Cumulative Sum Control Chart) test for structural breaks.

    Tests whether the mean of *series* has shifted at some unknown point.

    Hypotheses:
        H0: The series mean is constant over the entire sample.
        H1: The series mean has shifted at some interior point.

    The standardised CUSUM statistic is:

        S_k = (1 / (sigma * sqrt(n))) * sum_{i=1}^{k} (x_i - x_bar)

    A break is declared when max |S_k| exceeds the supplied threshold.

    Args:
        series: One-dimensional array of observations.
        threshold: Critical value (in standard-deviation units) above
            which a break is declared. The default of 2.0 corresponds
            roughly to a 5% significance level for moderate sample sizes.

    Returns:
        Tuple of (break_detected, break_index).
        *break_index* is the 0-based index of the maximum absolute CUSUM
        value, or ``None`` if no break was detected.

    Notes:
        Requires at least 3 observations. Returns ``(False, None)`` on
        degenerate inputs (constant series, insufficient data, etc.).
    """
    n = len(series)
    if n < 3:
        return False, None

    mean = np.mean(series)
    std = np.std(series, ddof=1)

    if std < EPSILON:
        return False, None

    # Cumulative sum of deviations from the overall mean, scaled
    cusum = np.cumsum(series - mean) / (std * np.sqrt(n))

    max_idx = int(np.argmax(np.abs(cusum)))
    max_cusum = float(np.abs(cusum[max_idx]))

    break_detected = max_cusum > threshold
    break_point = max_idx if break_detected else None

    return break_detected, break_point


# =============================================================================
# ALPHA DECAY MONITOR
# =============================================================================

class AlphaDecayMonitor:
    """
    Real-time alpha decay detection and signal lifecycle management.

    Monitors each registered signal's health through:

    1. **Rolling IC tracking** -- Spearman rank correlation between the
       signal forecast and the subsequent realized return, computed over
       configurable short / medium / long windows.
    2. **Exponential decay model fitting** -- Fits IC(t) = IC_0 * exp(-lambda*t)
       via log-linear regression to estimate half-life and remaining alpha.
    3. **Structural break detection (CUSUM)** -- Identifies regime changes
       in the IC series that may signal a permanent shift in alpha.
    4. **Correlation drift monitoring** -- Tracks pairwise signal correlations
       to detect crowding or diversification changes.
    5. **P&L-based health assessment** -- Monitors cumulative P&L drawdowns
       and recent Sharpe ratios for early warning.

    Lifecycle states
    ----------------
    HEALTHY
        IC is positive and stable; no degradation detected.
    WARNING
        IC is declining or has recently turned negative; the signal is
        still tradeable but warrants attention.
    DEGRADED
        IC is persistently negative or severe decay / drawdown has been
        detected. The signal should be scaled down.
    RETIRED
        The signal has been removed from trading, either automatically
        (exceeded degraded duration, age, or drawdown limits) or manually.

    Usage::

        monitor = AlphaDecayMonitor(config)

        # Register signals
        monitor.register_signal("momentum_20d")
        monitor.register_signal("reversal_5d")

        # Update with new data daily
        monitor.update(signal_name, signal_value, realized_return, timestamp)

        # Check health
        health = monitor.get_health("momentum_20d")
        if health.status == "retired":
            # Stop trading this signal
            ...

        # Get all signals needing attention
        alerts = monitor.get_alerts()
    """

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        """
        Initialise the monitor.

        Args:
            config: Monitoring configuration. Uses :class:`MonitorConfig`
                defaults when ``None``.
        """
        self._config: MonitorConfig = config or MonitorConfig()
        self._signals: Dict[str, _SignalState] = {}
        logger.info(
            "AlphaDecayMonitor initialised (ic_short=%d, ic_med=%d, ic_long=%d)",
            self._config.ic_window_short,
            self._config.ic_window_medium,
            self._config.ic_window_long,
        )

    # ------------------------------------------------------------------
    # Public API -- registration
    # ------------------------------------------------------------------

    def register_signal(
        self,
        name: str,
        initial_ic: float = 0.0,
        inception_date: Optional[datetime] = None,
    ) -> None:
        """
        Register a new signal for monitoring.

        Initialises all internal tracking state for the signal. The signal
        begins in the HEALTHY state and transitions automatically as data
        accumulates.

        Args:
            name: Unique identifier for the signal. Must not already be
                registered.
            initial_ic: IC value at signal inception, used as the baseline
                for decay estimation. Defaults to 0.0 (unknown).
            inception_date: Date the signal was first deployed. Defaults
                to the current UTC time.

        Raises:
            ValueError: If a signal with the given *name* is already
                registered.
        """
        if name in self._signals:
            raise ValueError(
                f"Signal '{name}' is already registered. Use retire_signal() "
                f"first if you wish to re-register it."
            )

        self._signals[name] = _SignalState(
            name=name,
            inception_date=inception_date or datetime.utcnow(),
            initial_ic=initial_ic,
        )
        logger.info("Registered signal '%s' (initial_ic=%.4f)", name, initial_ic)

    # ------------------------------------------------------------------
    # Public API -- updates
    # ------------------------------------------------------------------

    def update(
        self,
        signal_name: str,
        signal_value: float,
        realized_return: float,
        timestamp: datetime,
    ) -> SignalHealth:
        """
        Process a new observation for a signal.

        Appends the observation, recomputes rolling IC, updates P&L
        tracking, checks all health thresholds, and returns the current
        health snapshot.

        Args:
            signal_name: Name of a previously registered signal.
            signal_value: The signal's forecast value at *timestamp*.
            realized_return: The realised return corresponding to this
                forecast (i.e. the return that the signal was predicting).
            timestamp: Observation timestamp.

        Returns:
            The updated :class:`SignalHealth` for this signal.

        Raises:
            KeyError: If *signal_name* has not been registered.
        """
        state = self._get_state(signal_name)

        if state.status == STATUS_RETIRED:
            logger.debug(
                "Skipping update for retired signal '%s'", signal_name
            )
            return self._build_health(state, timestamp)

        # Append raw data
        state.signal_values.append(signal_value)
        state.realized_returns.append(realized_return)
        state.timestamps.append(timestamp)

        n = len(state.signal_values)

        # ----------------------------------------------------------
        # Compute point IC (rank correlation of the latest window)
        # ----------------------------------------------------------
        if n >= 2:
            window = min(n, self._config.ic_window_short)
            sv = np.array(state.signal_values[-window:], dtype=np.float64)
            rr = np.array(state.realized_returns[-window:], dtype=np.float64)
            ic = self._point_rank_correlation(sv, rr)
            state.ic_series.append(ic)
        else:
            state.ic_series.append(0.0)

        # ----------------------------------------------------------
        # Update P&L tracking (simple: signal_value * realized_return)
        # ----------------------------------------------------------
        period_pnl = signal_value * realized_return
        state.pnl_series.append(period_pnl)
        state.cumulative_pnl += period_pnl
        if state.cumulative_pnl > state.peak_cumulative_pnl:
            state.peak_cumulative_pnl = state.cumulative_pnl

        # ----------------------------------------------------------
        # Evaluate health and transition lifecycle
        # ----------------------------------------------------------
        self._evaluate_health(state, timestamp)

        health = self._build_health(state, timestamp)
        state.last_health = health
        return health

    def batch_update(
        self,
        signal_values: Dict[str, float],
        realized_return: float,
        timestamp: datetime,
    ) -> Dict[str, SignalHealth]:
        """
        Update multiple signals with a single realised return observation.

        This is a convenience wrapper around :meth:`update` that shares
        the *realized_return* across all signals. Useful when multiple
        signals forecast the same asset's return.

        Args:
            signal_values: Mapping of signal_name -> forecast value.
            realized_return: The realised return for this period.
            timestamp: Observation timestamp.

        Returns:
            Mapping of signal_name -> updated :class:`SignalHealth`.

        Raises:
            KeyError: If any signal name in *signal_values* has not been
                registered.
        """
        results: Dict[str, SignalHealth] = {}
        for name, value in signal_values.items():
            results[name] = self.update(name, value, realized_return, timestamp)
        return results

    # ------------------------------------------------------------------
    # Public API -- queries
    # ------------------------------------------------------------------

    def get_health(self, signal_name: str) -> SignalHealth:
        """
        Return the most recent health snapshot for a signal.

        If the signal has never been updated, a default health object
        with zero metrics is returned.

        Args:
            signal_name: Name of a registered signal.

        Returns:
            :class:`SignalHealth` for the signal.

        Raises:
            KeyError: If *signal_name* has not been registered.
        """
        state = self._get_state(signal_name)
        if state.last_health is not None:
            return state.last_health
        return self._build_health(state, state.inception_date)

    def get_all_health(self) -> Dict[str, SignalHealth]:
        """
        Return health snapshots for every registered signal.

        Returns:
            Mapping of signal_name -> :class:`SignalHealth`.
        """
        return {name: self.get_health(name) for name in self._signals}

    def get_alerts(self) -> List[Tuple[str, str, str]]:
        """
        Return alerts for all signals that currently need attention.

        Scans every registered (non-retired) signal and emits alerts for:

        - ``"ic_declining"``: Rolling short-term IC has a negative trend.
        - ``"structural_break"``: A CUSUM break was detected in the IC
          series.
        - ``"drawdown"``: Cumulative P&L drawdown exceeds 25% of peak.
        - ``"correlation_drift"``: Pairwise correlation change exceeds the
          configured threshold.
        - ``"approaching_retirement"``: Signal is in DEGRADED and has been
          for more than half of ``retire_after_degraded_days``.

        Returns:
            List of ``(signal_name, alert_type, message)`` tuples.
        """
        alerts: List[Tuple[str, str, str]] = []
        cfg = self._config

        for name, state in self._signals.items():
            if state.status == STATUS_RETIRED:
                continue

            n = len(state.ic_series)

            # --- IC declining ---
            if n >= cfg.min_observations:
                ic_arr = np.array(state.ic_series[-cfg.ic_window_short:], dtype=np.float64)
                trend = self._ic_trend(ic_arr)
                if trend < -EPSILON:
                    alerts.append((
                        name,
                        "ic_declining",
                        f"IC trend is negative ({trend:.6f}/day) over the last "
                        f"{min(n, cfg.ic_window_short)} observations.",
                    ))

            # --- Structural break ---
            if n >= cfg.min_observations:
                lookback = min(n, cfg.decay_lookback)
                ic_arr = np.array(state.ic_series[-lookback:], dtype=np.float64)
                broke, _ = _cusum_test(ic_arr, cfg.structural_break_threshold)
                if broke:
                    alerts.append((
                        name,
                        "structural_break",
                        "CUSUM test detected a structural break in the IC series.",
                    ))

            # --- Drawdown ---
            dd = self._drawdown_from_peak(state)
            if dd > 0.25:
                alerts.append((
                    name,
                    "drawdown",
                    f"Cumulative P&L drawdown is {dd:.1%} from peak.",
                ))

            # --- Correlation drift ---
            drift = self._monitor_correlation_drift(name)
            if drift > cfg.correlation_drift_threshold:
                alerts.append((
                    name,
                    "correlation_drift",
                    f"Average pairwise correlation has shifted by {drift:.3f} "
                    f"(threshold: {cfg.correlation_drift_threshold:.3f}).",
                ))

            # --- Approaching retirement ---
            if (
                state.status == STATUS_DEGRADED
                and state.degraded_since is not None
                and len(state.timestamps) > 0
            ):
                latest_ts = state.timestamps[-1]
                days_degraded = (latest_ts - state.degraded_since).days
                if days_degraded > cfg.retire_after_degraded_days // 2:
                    alerts.append((
                        name,
                        "approaching_retirement",
                        f"Signal has been DEGRADED for {days_degraded} days "
                        f"(retirement at {cfg.retire_after_degraded_days}).",
                    ))

        return alerts

    # ------------------------------------------------------------------
    # Public API -- decay estimation
    # ------------------------------------------------------------------

    def estimate_decay(self, signal_name: str) -> AlphaDecayMetrics:
        """
        Estimate the alpha decay profile for a signal.

        Performs three analyses:

        1. Fits an exponential decay model IC(t) = IC_0 * exp(-lambda*t)
           via log-linear OLS to estimate the decay rate and half-life.
        2. Runs a CUSUM structural-break test on the IC series.
        3. Projects the date at which alpha reaches zero under a linear
           extrapolation of the current IC trend.

        Args:
            signal_name: Name of a registered signal.

        Returns:
            :class:`AlphaDecayMetrics` with the estimated parameters.

        Raises:
            KeyError: If *signal_name* has not been registered.
        """
        state = self._get_state(signal_name)
        cfg = self._config
        n = len(state.ic_series)

        # Defaults for insufficient data
        if n < cfg.min_observations:
            return AlphaDecayMetrics(
                signal_name=signal_name,
                initial_ic=state.initial_ic,
                current_ic=state.ic_series[-1] if n > 0 else 0.0,
                decay_rate=0.0,
                half_life_days=float("inf"),
                structural_break=False,
                break_date=None,
                remaining_alpha_pct=100.0,
                projected_zero_date=None,
            )

        lookback = min(n, cfg.decay_lookback)
        ic_arr = np.array(state.ic_series[-lookback:], dtype=np.float64)
        ts_arr = state.timestamps[-lookback:]

        current_ic = float(ic_arr[-1])

        # --- Exponential decay fit ---
        decay_rate, half_life = self._fit_decay_model(ic_arr)

        # --- Structural break ---
        broke, break_idx = _cusum_test(ic_arr, cfg.structural_break_threshold)
        break_date: Optional[datetime] = None
        if broke and break_idx is not None and break_idx < len(ts_arr):
            break_date = ts_arr[break_idx]

        # --- Remaining alpha ---
        initial_ic_estimate = state.initial_ic if abs(state.initial_ic) > EPSILON else float(np.mean(ic_arr[:min(30, len(ic_arr))]))
        if abs(initial_ic_estimate) > EPSILON:
            remaining = max(0.0, min(100.0, (current_ic / initial_ic_estimate) * 100.0))
        else:
            remaining = 100.0 if current_ic > 0 else 0.0

        # --- Projected zero date (linear extrapolation) ---
        projected_zero: Optional[datetime] = None
        trend = self._ic_trend(ic_arr)
        if trend < -EPSILON and current_ic > EPSILON and len(ts_arr) > 0:
            days_to_zero = current_ic / abs(trend)
            projected_zero = ts_arr[-1] + timedelta(days=days_to_zero)

        return AlphaDecayMetrics(
            signal_name=signal_name,
            initial_ic=initial_ic_estimate,
            current_ic=current_ic,
            decay_rate=decay_rate,
            half_life_days=half_life,
            structural_break=broke,
            break_date=break_date,
            remaining_alpha_pct=remaining,
            projected_zero_date=projected_zero,
        )

    # ------------------------------------------------------------------
    # Public API -- lifecycle management
    # ------------------------------------------------------------------

    def retire_signal(self, signal_name: str, reason: str) -> None:
        """
        Manually retire a signal.

        The signal's status is set to RETIRED and the reason is recorded.
        Historical data is preserved for post-mortem analysis. Subsequent
        calls to :meth:`update` for this signal will be no-ops.

        Args:
            signal_name: Name of a registered signal.
            reason: Human-readable explanation for retirement.

        Raises:
            KeyError: If *signal_name* has not been registered.
        """
        state = self._get_state(signal_name)
        state.status = STATUS_RETIRED
        state.retired_reason = reason
        logger.warning(
            "Signal '%s' manually retired: %s", signal_name, reason
        )

    # ------------------------------------------------------------------
    # Public API -- portfolio-level assessment
    # ------------------------------------------------------------------

    def get_portfolio_health_score(self) -> float:
        """
        Compute an aggregate health score across all active signals.

        The score is in the range [0, 100], where 100 means all signals
        are perfectly healthy and 0 means all are retired or severely
        degraded.

        Scoring per signal:
            - HEALTHY: 100
            - WARNING: 60
            - DEGRADED: 20
            - RETIRED: 0

        Signals are weighted equally. If no signals are registered, the
        score is 0.0.

        Returns:
            Portfolio health score in [0, 100].
        """
        if not self._signals:
            return 0.0

        status_scores = {
            STATUS_HEALTHY: 100.0,
            STATUS_WARNING: 60.0,
            STATUS_DEGRADED: 20.0,
            STATUS_RETIRED: 0.0,
        }

        total = 0.0
        for state in self._signals.values():
            total += status_scores.get(state.status, 0.0)

        return total / len(self._signals)

    def get_decay_report(self) -> Dict:
        """
        Generate a comprehensive decay report for all signals.

        Returns a dictionary with three sections:

        - ``"signals"``: Per-signal decay metrics (dict of signal_name
          -> :class:`AlphaDecayMetrics` converted to dict).
        - ``"portfolio"``: Portfolio-level statistics including the
          health score and counts by lifecycle state.
        - ``"recommendations"``: List of human-readable recommendation
          strings for signals requiring action.

        Returns:
            Nested dictionary with the report data.
        """
        signal_metrics: Dict[str, Dict] = {}
        recommendations: List[str] = []

        status_counts: Dict[str, int] = {
            STATUS_HEALTHY: 0,
            STATUS_WARNING: 0,
            STATUS_DEGRADED: 0,
            STATUS_RETIRED: 0,
        }

        for name, state in self._signals.items():
            status_counts[state.status] = status_counts.get(state.status, 0) + 1

            metrics = self.estimate_decay(name)
            signal_metrics[name] = {
                "initial_ic": metrics.initial_ic,
                "current_ic": metrics.current_ic,
                "decay_rate": metrics.decay_rate,
                "half_life_days": metrics.half_life_days,
                "structural_break": metrics.structural_break,
                "break_date": (
                    metrics.break_date.isoformat() if metrics.break_date else None
                ),
                "remaining_alpha_pct": metrics.remaining_alpha_pct,
                "projected_zero_date": (
                    metrics.projected_zero_date.isoformat()
                    if metrics.projected_zero_date
                    else None
                ),
                "status": state.status,
            }

            # Recommendations
            if state.status == STATUS_DEGRADED:
                recommendations.append(
                    f"DEGRADED: '{name}' -- consider reducing allocation or "
                    f"retiring. Remaining alpha: {metrics.remaining_alpha_pct:.1f}%."
                )
            elif state.status == STATUS_WARNING:
                if metrics.half_life_days < 90:
                    recommendations.append(
                        f"WARNING: '{name}' has a short half-life "
                        f"({metrics.half_life_days:.0f} days). Monitor closely."
                    )
                if metrics.structural_break:
                    recommendations.append(
                        f"WARNING: '{name}' -- structural break detected. "
                        f"Investigate regime change."
                    )
            elif state.status == STATUS_HEALTHY and metrics.structural_break:
                recommendations.append(
                    f"NOTE: '{name}' is healthy but a structural break was "
                    f"detected. Verify signal integrity."
                )

        return {
            "signals": signal_metrics,
            "portfolio": {
                "health_score": self.get_portfolio_health_score(),
                "total_signals": len(self._signals),
                "status_counts": status_counts,
            },
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal -- state access
    # ------------------------------------------------------------------

    def _get_state(self, signal_name: str) -> _SignalState:
        """
        Retrieve internal state for a signal, raising on unknown names.

        Args:
            signal_name: Registered signal identifier.

        Returns:
            The mutable :class:`_SignalState`.

        Raises:
            KeyError: If *signal_name* is not registered.
        """
        try:
            return self._signals[signal_name]
        except KeyError:
            raise KeyError(
                f"Signal '{signal_name}' is not registered. Call "
                f"register_signal('{signal_name}') first."
            ) from None

    # ------------------------------------------------------------------
    # Internal -- health evaluation
    # ------------------------------------------------------------------

    def _evaluate_health(self, state: _SignalState, timestamp: datetime) -> None:
        """
        Evaluate a signal's health and perform lifecycle transitions.

        The evaluation proceeds through the following checks (in order):

        1. Minimum-data guard: if fewer than ``min_observations``
           observations are available, the signal remains HEALTHY.
        2. Rolling IC check against ``warning_ic_threshold`` and
           ``degraded_ic_threshold``.
        3. Degraded-duration check: if the signal has been DEGRADED for
           longer than ``retire_after_degraded_days``, it is retired.
        4. Age check: if the signal has exceeded ``max_signal_age_days``,
           it is retired.
        5. Drawdown check: if cumulative P&L drawdown exceeds
           ``max_drawdown_pct``, the signal is retired.
        6. Sharpe check: if the 90-day Sharpe is below ``min_sharpe_90d``,
           the signal is retired.

        Args:
            state: Internal state object for the signal.
            timestamp: Current observation timestamp.
        """
        cfg = self._config
        n = len(state.ic_series)

        # Guard: not enough data to assess
        if n < cfg.min_observations:
            return

        # Current short-window rolling IC
        short_ic = self._rolling_ic_mean(state, cfg.ic_window_short)

        # --- Determine base status from IC ---
        if short_ic > cfg.warning_ic_threshold:
            new_status = STATUS_HEALTHY
        elif short_ic > cfg.degraded_ic_threshold:
            new_status = STATUS_WARNING
        else:
            new_status = STATUS_DEGRADED

        # --- Track degraded duration ---
        if new_status == STATUS_DEGRADED:
            if state.degraded_since is None:
                state.degraded_since = timestamp
            else:
                days_degraded = (timestamp - state.degraded_since).days
                if days_degraded >= cfg.retire_after_degraded_days:
                    self._auto_retire(
                        state,
                        f"Degraded for {days_degraded} days "
                        f"(limit: {cfg.retire_after_degraded_days}).",
                    )
                    return
        else:
            # Reset degraded timer if signal recovers
            state.degraded_since = None

        # --- Age check ---
        signal_age = (timestamp - state.inception_date).days
        if signal_age > cfg.max_signal_age_days:
            self._auto_retire(
                state,
                f"Signal age ({signal_age} days) exceeds maximum "
                f"({cfg.max_signal_age_days}).",
            )
            return

        # --- Drawdown check ---
        dd = self._drawdown_from_peak(state)
        if dd > cfg.max_drawdown_pct:
            self._auto_retire(
                state,
                f"Cumulative P&L drawdown ({dd:.1%}) exceeds maximum "
                f"({cfg.max_drawdown_pct:.1%}).",
            )
            return

        # --- Sharpe check ---
        sharpe_90 = self._recent_sharpe(state, cfg.ic_window_medium)
        if sharpe_90 < cfg.min_sharpe_90d and n >= cfg.ic_window_medium:
            self._auto_retire(
                state,
                f"90-day Sharpe ({sharpe_90:.2f}) below minimum "
                f"({cfg.min_sharpe_90d:.2f}).",
            )
            return

        # --- Apply status ---
        if state.status != new_status:
            logger.info(
                "Signal '%s' status: %s -> %s",
                state.name,
                state.status,
                new_status,
            )
        state.status = new_status

    def _auto_retire(self, state: _SignalState, reason: str) -> None:
        """
        Retire a signal automatically with the given reason.

        Args:
            state: Internal signal state.
            reason: Reason string recorded on the state.
        """
        state.status = STATUS_RETIRED
        state.retired_reason = reason
        logger.warning("Signal '%s' auto-retired: %s", state.name, reason)

    # ------------------------------------------------------------------
    # Internal -- IC computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rolling_ic(
        signal_values: NDArray[np.float64],
        returns: NDArray[np.float64],
        window: int,
    ) -> NDArray[np.float64]:
        """
        Compute rolling Spearman rank IC between signal values and returns.

        For each position ``i >= window - 1``, computes the Spearman rank
        correlation between ``signal_values[i-window+1 : i+1]`` and
        ``returns[i-window+1 : i+1]``.

        Args:
            signal_values: Array of signal forecasts.
            returns: Array of realised returns (same length as
                *signal_values*).
            window: Rolling window size.

        Returns:
            Array of rolling IC values. Positions ``0 .. window-2`` are
            filled with ``NaN``.

        Raises:
            ValueError: If *signal_values* and *returns* have different
                lengths.
        """
        n = len(signal_values)
        if n != len(returns):
            raise ValueError(
                f"signal_values length ({n}) != returns length ({len(returns)})"
            )

        result = np.full(n, np.nan, dtype=np.float64)

        if n < window or window < 2:
            return result

        for i in range(window - 1, n):
            sv_window = signal_values[i - window + 1: i + 1]
            rr_window = returns[i - window + 1: i + 1]

            # Skip if either window is constant (rank correlation undefined)
            if np.std(sv_window) < EPSILON or np.std(rr_window) < EPSILON:
                result[i] = 0.0
                continue

            corr, _ = scipy_stats.spearmanr(sv_window, rr_window)

            # scipy may return NaN for degenerate inputs
            result[i] = corr if np.isfinite(corr) else 0.0

        return result

    @staticmethod
    def _point_rank_correlation(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """
        Compute a single Spearman rank correlation between *x* and *y*.

        Handles degenerate cases (constant arrays, NaN values) by
        returning 0.0.

        Args:
            x: First array.
            y: Second array (same length as *x*).

        Returns:
            Spearman rank correlation in [-1, 1], or 0.0 on failure.
        """
        if len(x) < 2 or len(x) != len(y):
            return 0.0

        # Remove NaN pairs
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return 0.0
        if np.std(x_clean) < EPSILON or np.std(y_clean) < EPSILON:
            return 0.0

        corr, _ = scipy_stats.spearmanr(x_clean, y_clean)
        return float(corr) if np.isfinite(corr) else 0.0

    def _rolling_ic_mean(self, state: _SignalState, window: int) -> float:
        """
        Compute the mean of the last *window* IC observations.

        Returns 0.0 if fewer than ``min_observations`` IC values are
        available.

        Args:
            state: Internal signal state.
            window: Number of recent IC values to average.

        Returns:
            Mean IC over the window.
        """
        n = len(state.ic_series)
        if n == 0:
            return 0.0

        effective_window = min(n, window)
        ic_slice = state.ic_series[-effective_window:]
        return float(np.nanmean(ic_slice))

    # ------------------------------------------------------------------
    # Internal -- structural break & decay fitting
    # ------------------------------------------------------------------

    def _detect_structural_break(
        self,
        ic_series: NDArray[np.float64],
    ) -> Tuple[bool, Optional[int]]:
        """
        Run a CUSUM structural-break test on an IC series.

        Delegates to the module-level :func:`_cusum_test` with the
        configured threshold.

        Args:
            ic_series: Array of IC values.

        Returns:
            Tuple of ``(break_detected, break_index)``.
        """
        return _cusum_test(ic_series, self._config.structural_break_threshold)

    @staticmethod
    def _fit_decay_model(
        ic_series: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """
        Fit an exponential decay model to the IC series.

        Model: IC(t) = IC_0 * exp(-lambda * t)

        Fitting procedure:
            1. Take only positive IC values (log-transform requires > 0).
            2. Compute log(IC), regress against time index t.
            3. Slope of the regression gives -lambda.
            4. Half-life = ln(2) / lambda.

        If the IC series contains no positive values, or the estimated
        decay rate is non-positive (i.e. IC is growing), returns a
        decay rate of 0.0 and infinite half-life.

        Args:
            ic_series: Array of IC values over time.

        Returns:
            Tuple of ``(decay_rate, half_life_days)``.
            *decay_rate* is the annualised lambda; *half_life_days* is
            in trading days.
        """
        n = len(ic_series)
        if n < 3:
            return 0.0, float("inf")

        # Use only positive IC values for log-linear fit
        positive_mask = ic_series > EPSILON
        if np.sum(positive_mask) < 3:
            return 0.0, float("inf")

        # Build time index for positive entries
        time_indices = np.arange(n, dtype=np.float64)
        t_pos = time_indices[positive_mask]
        ic_pos = ic_series[positive_mask]

        log_ic = np.log(ic_pos)

        # OLS: log(IC) = a - lambda * t
        # Using numpy polyfit (degree 1)
        try:
            coeffs = np.polyfit(t_pos, log_ic, 1)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, float("inf")

        slope = coeffs[0]  # This is -lambda (daily)

        # Decay requires negative slope
        if slope >= 0:
            return 0.0, float("inf")

        daily_lambda = -slope
        annualised_lambda = daily_lambda * 252.0

        # Half-life in trading days
        half_life = np.log(2) / daily_lambda

        # Clamp to reasonable bounds
        half_life = float(np.clip(half_life, 1.0, 10000.0))
        annualised_lambda = float(np.clip(annualised_lambda, 0.0, 100.0))

        return annualised_lambda, half_life

    # ------------------------------------------------------------------
    # Internal -- correlation drift
    # ------------------------------------------------------------------

    def _monitor_correlation_drift(self, signal_name: str) -> float:
        """
        Measure how much a signal's pairwise correlations have drifted.

        For each other active signal, computes the Spearman rank
        correlation of the two signals' forecast series over the most
        recent ``correlation_lookback`` observations, and compares it
        to the baseline correlation (established after the initial
        ``min_observations`` warm-up period).

        The returned value is the mean absolute change across all pairs.

        Args:
            signal_name: Name of the signal to evaluate.

        Returns:
            Mean absolute change in pairwise correlation.  Returns 0.0
            if there is only one signal or insufficient data.
        """
        state = self._get_state(signal_name)
        cfg = self._config
        n = len(state.signal_values)

        if n < cfg.correlation_lookback:
            return 0.0

        other_names = [
            name
            for name in self._signals
            if name != signal_name and self._signals[name].status != STATUS_RETIRED
        ]
        if not other_names:
            return 0.0

        # Current correlations
        sv = np.array(state.signal_values[-cfg.correlation_lookback:], dtype=np.float64)
        current_corrs: Dict[str, float] = {}

        for other_name in other_names:
            other_state = self._signals[other_name]
            if len(other_state.signal_values) < cfg.correlation_lookback:
                continue

            ov = np.array(
                other_state.signal_values[-cfg.correlation_lookback:],
                dtype=np.float64,
            )

            if np.std(sv) < EPSILON or np.std(ov) < EPSILON:
                current_corrs[other_name] = 0.0
            else:
                corr, _ = scipy_stats.spearmanr(sv, ov)
                current_corrs[other_name] = float(corr) if np.isfinite(corr) else 0.0

        if not current_corrs:
            return 0.0

        # Establish baseline if not yet set
        if state.baseline_correlations is None:
            if n >= cfg.min_observations + cfg.correlation_lookback:
                # Use the earliest window as baseline
                sv_base = np.array(
                    state.signal_values[
                        cfg.min_observations: cfg.min_observations + cfg.correlation_lookback
                    ],
                    dtype=np.float64,
                )
                baseline: Dict[str, float] = {}
                for other_name in other_names:
                    other_state = self._signals[other_name]
                    if len(other_state.signal_values) < cfg.min_observations + cfg.correlation_lookback:
                        continue
                    ov_base = np.array(
                        other_state.signal_values[
                            cfg.min_observations: cfg.min_observations + cfg.correlation_lookback
                        ],
                        dtype=np.float64,
                    )
                    if np.std(sv_base) < EPSILON or np.std(ov_base) < EPSILON:
                        baseline[other_name] = 0.0
                    else:
                        corr_b, _ = scipy_stats.spearmanr(sv_base, ov_base)
                        baseline[other_name] = float(corr_b) if np.isfinite(corr_b) else 0.0

                if baseline:
                    state.baseline_correlations = baseline
            else:
                # Not enough data for a baseline yet
                return 0.0

        if state.baseline_correlations is None:
            return 0.0

        # Compute mean absolute drift
        drifts: List[float] = []
        for other_name, current_corr in current_corrs.items():
            if other_name in state.baseline_correlations:
                drifts.append(abs(current_corr - state.baseline_correlations[other_name]))

        if not drifts:
            return 0.0

        return float(np.mean(drifts))

    # ------------------------------------------------------------------
    # Internal -- P&L and Sharpe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drawdown_from_peak(state: _SignalState) -> float:
        """
        Compute the current drawdown of cumulative P&L from its peak.

        Returns:
            Drawdown as a fraction in [0, 1]. Returns 0.0 if the signal
            has never had positive cumulative P&L.
        """
        if state.peak_cumulative_pnl <= EPSILON:
            # No positive P&L to draw down from; report 0 or measure
            # absolute loss relative to zero.
            if state.cumulative_pnl < -EPSILON:
                # Use absolute cumulative loss as a pseudo-drawdown measure.
                # Bound at 1.0 to avoid unbounded values.
                return min(1.0, abs(state.cumulative_pnl))
            return 0.0

        dd = (state.peak_cumulative_pnl - state.cumulative_pnl) / state.peak_cumulative_pnl
        return float(np.clip(dd, 0.0, 1.0))

    @staticmethod
    def _recent_sharpe(state: _SignalState, window: int) -> float:
        """
        Compute an annualised Sharpe ratio over the most recent *window*
        P&L observations.

        Args:
            state: Internal signal state.
            window: Number of recent observations.

        Returns:
            Annualised Sharpe ratio. Returns 0.0 on insufficient data.
        """
        n = len(state.pnl_series)
        if n < 2:
            return 0.0

        effective_window = min(n, window)
        pnl_slice = np.array(state.pnl_series[-effective_window:], dtype=np.float64)

        mean_pnl = np.mean(pnl_slice)
        std_pnl = np.std(pnl_slice, ddof=1)

        if std_pnl < EPSILON:
            return 0.0

        daily_sharpe = mean_pnl / std_pnl
        return float(daily_sharpe * SQRT_252)

    # ------------------------------------------------------------------
    # Internal -- trend estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _ic_trend(ic_arr: NDArray[np.float64]) -> float:
        """
        Estimate the linear trend (slope) of an IC series.

        Uses OLS regression of IC against a time index. A negative
        slope indicates decaying alpha.

        Args:
            ic_arr: Array of IC values.

        Returns:
            OLS slope (IC units per observation).
        """
        n = len(ic_arr)
        if n < 3:
            return 0.0

        # Filter out NaN
        valid_mask = np.isfinite(ic_arr)
        if np.sum(valid_mask) < 3:
            return 0.0

        t = np.arange(n, dtype=np.float64)[valid_mask]
        ic_valid = ic_arr[valid_mask]

        try:
            coeffs = np.polyfit(t, ic_valid, 1)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

        return float(coeffs[0])

    # ------------------------------------------------------------------
    # Internal -- build health snapshot
    # ------------------------------------------------------------------

    def _build_health(self, state: _SignalState, timestamp: datetime) -> SignalHealth:
        """
        Construct a :class:`SignalHealth` from the current internal state.

        Computes all rolling metrics on demand. This is called at the end
        of every :meth:`update` and by :meth:`get_health`.

        Args:
            state: Internal signal state.
            timestamp: Timestamp for the snapshot.

        Returns:
            Fully populated :class:`SignalHealth`.
        """
        cfg = self._config
        n = len(state.ic_series)

        # Current IC
        current_ic = state.ic_series[-1] if n > 0 else 0.0

        # Rolling ICs
        rolling_ic_30d = self._rolling_ic_mean(state, cfg.ic_window_short)
        rolling_ic_90d = self._rolling_ic_mean(state, cfg.ic_window_medium)
        rolling_ic_252d = self._rolling_ic_mean(state, cfg.ic_window_long)

        # IC trend
        if n >= cfg.min_observations:
            lookback = min(n, cfg.decay_lookback)
            ic_arr = np.array(state.ic_series[-lookback:], dtype=np.float64)
            ic_trend = self._ic_trend(ic_arr)
        else:
            ic_trend = 0.0

        # Half-life
        if n >= cfg.min_observations:
            lookback = min(n, cfg.decay_lookback)
            ic_arr = np.array(state.ic_series[-lookback:], dtype=np.float64)
            _, half_life = self._fit_decay_model(ic_arr)
        else:
            half_life = float("inf")

        # Days since last positive IC
        days_since_positive = 0
        if n > 0:
            for i in range(n - 1, -1, -1):
                if state.ic_series[i] > EPSILON:
                    break
                days_since_positive += 1
            # If the loop completes without breaking, all ICs are non-positive
            if days_since_positive == n and (n == 0 or state.ic_series[0] <= EPSILON):
                days_since_positive = n

        # P&L metrics
        cumulative_pnl = state.cumulative_pnl
        pnl_n = len(state.pnl_series)
        recent_window = min(pnl_n, cfg.ic_window_short)
        recent_pnl_30d = float(np.sum(state.pnl_series[-recent_window:])) if recent_window > 0 else 0.0

        # Sharpe
        sharpe_90d = self._recent_sharpe(state, cfg.ic_window_medium)

        # Drawdown
        drawdown = self._drawdown_from_peak(state)

        # Correlation drift
        try:
            corr_drift = self._monitor_correlation_drift(state.name)
        except (KeyError, ValueError):
            corr_drift = 0.0

        return SignalHealth(
            signal_name=state.name,
            status=state.status,
            current_ic=current_ic,
            rolling_ic_30d=rolling_ic_30d,
            rolling_ic_90d=rolling_ic_90d,
            rolling_ic_252d=rolling_ic_252d,
            ic_trend=ic_trend,
            half_life_estimate=half_life,
            days_since_last_positive_ic=days_since_positive,
            cumulative_pnl=cumulative_pnl,
            recent_pnl_30d=recent_pnl_30d,
            sharpe_ratio_90d=sharpe_90d,
            drawdown_from_peak=drawdown,
            correlation_drift=corr_drift,
            last_updated=timestamp,
        )
