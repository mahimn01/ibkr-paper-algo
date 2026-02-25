"""
Entropy Regime Meta-Filter

Gates ALL other strategy signals based on market entropy measured on a
reference symbol (SPY by default).  The core discovery is that markets
with low sample entropy are far more predictable and therefore far more
profitable for systematic strategies:

    SPY:  Low entropy 5-day forward Sharpe = +4.77  vs  High entropy = -0.15
    QQQ:  Low entropy Sharpe = +3.17
    NVDA: Low entropy Sharpe = +2.80

The filter computes rolling sample entropy on the reference symbol's
returns, classifies the current value into quartiles based on a trailing
calibration window, and outputs a scaling factor that downstream
strategies multiply against their raw signal weights.

Quartile scaling:
    Q1 (bottom 25 %, low entropy, most predictable):  100 % passthrough
    Q2 (25th-50th percentile):                          75 % scaling
    Q3 (50th-75th percentile):                          40 % scaling
    Q4 (top 25 %, high entropy, most random):           10 % scaling

In high-entropy regimes the filter acts as a near-block on new entries
while still allowing exits to proceed unimpeded.

References:
    - Richman & Moorman (2000): Sample Entropy
    - Pincus (1991): Approximate Entropy as a measure of system complexity
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from trading_algo.quant_core.ml.information_theory import sample_entropy

logger = logging.getLogger(__name__)

__all__ = [
    "EntropyFilterConfig",
    "EntropyRegimeFilter",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EntropyFilterConfig:
    """
    Configuration for the entropy regime meta-filter.

    Attributes:
        reference_symbol: Symbol whose returns are used for entropy
            computation (typically a broad index like SPY).
        entropy_window: Rolling window of daily returns fed into the
            sample entropy calculation.
        m: Embedding dimension for sample entropy.
        r: Tolerance for sample entropy, expressed as a fraction of the
            rolling window standard deviation.
        calibration_window: Number of historical entropy values used to
            compute quartile boundaries.  252 ~ 1 year of daily bars.
        q1_scale: Signal scaling for the bottom quartile (low entropy,
            most predictable market conditions).
        q2_scale: Signal scaling for the 25th-50th percentile.
        q3_scale: Signal scaling for the 50th-75th percentile.
        q4_scale: Signal scaling for the top quartile (high entropy,
            most random, near-block on entries).
        warmup: Minimum number of return observations before the filter
            can produce a valid scaling factor.  Must be >= entropy_window.
    """

    reference_symbol: str = "SPY"
    entropy_window: int = 42       # Rolling sample entropy window (daily bars)
    m: int = 2                     # Embedding dimension for SampEn
    r: float = 0.2                 # Tolerance (fraction of std)
    calibration_window: int = 252  # Window for quartile calibration
    # Scaling factors by quartile
    q1_scale: float = 1.0         # Low entropy (most predictable)
    q2_scale: float = 0.75
    q3_scale: float = 0.40
    q4_scale: float = 0.10        # High entropy (most random)
    warmup: int = 60


# =============================================================================
# ENTROPY REGIME FILTER
# =============================================================================

class EntropyRegimeFilter:
    """
    Meta-filter that gates strategy signals by market entropy.

    Usage::

        filt = EntropyRegimeFilter()

        # Feed daily returns for the reference symbol
        for daily_return in spy_returns:
            filt.update(daily_return)

        # Query the current signal scaling factor
        scale = filt.get_scaling_factor()    # float in [0.10, 1.0]
        regime = filt.get_entropy_regime()   # "LOW", "MEDIUM_LOW", ...

        # Downstream usage
        adjusted_weight = raw_signal_weight * scale

    The filter must receive at least ``config.warmup`` return observations
    before ``get_scaling_factor`` returns a meaningful value.  During the
    warmup period it returns ``config.q4_scale`` (most conservative).
    """

    # Regime label constants
    REGIME_LOW = "LOW"
    REGIME_MEDIUM_LOW = "MEDIUM_LOW"
    REGIME_MEDIUM_HIGH = "MEDIUM_HIGH"
    REGIME_HIGH = "HIGH"

    def __init__(self, config: Optional[EntropyFilterConfig] = None):
        """
        Initialize the entropy regime filter.

        Args:
            config: Filter configuration.  Uses defaults if not provided.
        """
        self._config = config or EntropyFilterConfig()

        # Validate warmup vs entropy_window
        if self._config.warmup < self._config.entropy_window:
            self._config.warmup = self._config.entropy_window
            logger.warning(
                "warmup (%d) was less than entropy_window (%d); "
                "adjusted warmup to %d.",
                self._config.warmup,
                self._config.entropy_window,
                self._config.warmup,
            )

        # Rolling buffer of reference-symbol daily returns
        max_buffer = self._config.calibration_window + self._config.entropy_window + 10
        self._returns_buffer: deque[float] = deque(maxlen=max_buffer)

        # Rolling buffer of computed entropy values (for quartile calibration)
        self._entropy_history: deque[float] = deque(
            maxlen=self._config.calibration_window
        )

        # Cached state
        self._current_entropy: float = np.nan
        self._current_regime: str = self.REGIME_HIGH
        self._current_scale: float = self._config.q4_scale
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def update(self, returns_value: float) -> None:
        """
        Feed a single daily return for the reference symbol.

        Args:
            returns_value: Daily simple return (e.g. 0.01 for +1 %).
                NaN values are silently ignored to handle market holidays
                and missing data.
        """
        if not np.isfinite(returns_value):
            return

        self._returns_buffer.append(float(returns_value))
        self._n_updates += 1

        # Recompute entropy once we have enough data
        if self._n_updates >= self._config.entropy_window:
            entropy_val = self._compute_entropy()

            if np.isfinite(entropy_val):
                self._current_entropy = entropy_val
                self._entropy_history.append(entropy_val)

                # Recompute regime and scaling factor
                self._update_regime()

    def get_scaling_factor(self) -> float:
        """
        Return the current signal scaling factor.

        Returns:
            Float in [config.q4_scale, config.q1_scale].  During warmup
            this returns ``config.q4_scale`` (most conservative).
        """
        if self._n_updates < self._config.warmup:
            return self._config.q4_scale
        return self._current_scale

    def get_entropy_regime(self) -> str:
        """
        Return the current entropy regime label.

        Returns:
            One of ``"LOW"``, ``"MEDIUM_LOW"``, ``"MEDIUM_HIGH"``, ``"HIGH"``.
            During warmup returns ``"HIGH"`` (conservative).
        """
        if self._n_updates < self._config.warmup:
            return self.REGIME_HIGH
        return self._current_regime

    def get_current_entropy(self) -> float:
        """
        Return the most recently computed sample entropy value.

        Returns:
            Sample entropy in nats.  ``NaN`` if not yet computed.
        """
        return self._current_entropy

    @property
    def is_warmed_up(self) -> bool:
        """True once enough data has been ingested for valid outputs."""
        return self._n_updates >= self._config.warmup

    def reset(self) -> None:
        """Reset all internal state for a new session or backtest run."""
        self._returns_buffer.clear()
        self._entropy_history.clear()
        self._current_entropy = np.nan
        self._current_regime = self.REGIME_HIGH
        self._current_scale = self._config.q4_scale
        self._n_updates = 0

    # ------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------

    def _compute_entropy(self) -> float:
        """
        Compute sample entropy on the most recent ``entropy_window``
        return observations.

        Returns:
            Sample entropy in nats.  May return ``NaN`` or ``inf`` for
            degenerate inputs (constant returns, insufficient matches).
        """
        window = self._config.entropy_window
        if len(self._returns_buffer) < window:
            return np.nan

        # Extract the most recent window of returns
        recent = list(self._returns_buffer)[-window:]
        x = np.array(recent, dtype=np.float64)

        # Guard: all-zero or constant returns
        if np.ptp(x) == 0.0:
            return 0.0

        entropy_val = sample_entropy(
            x,
            m=self._config.m,
            r=self._config.r,
        )

        # Treat inf as NaN for downstream quartile calibration
        if np.isinf(entropy_val):
            return np.nan

        return float(entropy_val)

    def _update_regime(self) -> None:
        """
        Classify the current entropy value into a quartile-based regime
        and set the corresponding scaling factor.

        Uses the trailing ``calibration_window`` entropy values to
        compute the 25th, 50th, and 75th percentile boundaries.  If
        fewer than 10 historical values are available, defaults to
        the most conservative regime (HIGH).
        """
        if not np.isfinite(self._current_entropy):
            self._current_regime = self.REGIME_HIGH
            self._current_scale = self._config.q4_scale
            return

        # Need a minimum number of entropy values for stable percentiles
        min_calibration = 10
        if len(self._entropy_history) < min_calibration:
            self._current_regime = self.REGIME_HIGH
            self._current_scale = self._config.q4_scale
            return

        # Compute quartile boundaries from history
        history_array = np.array(self._entropy_history, dtype=np.float64)
        # Filter out any NaN values that slipped through
        history_array = history_array[np.isfinite(history_array)]

        if len(history_array) < min_calibration:
            self._current_regime = self.REGIME_HIGH
            self._current_scale = self._config.q4_scale
            return

        q25 = float(np.percentile(history_array, 25))
        q50 = float(np.percentile(history_array, 50))
        q75 = float(np.percentile(history_array, 75))

        entropy = self._current_entropy

        if entropy <= q25:
            # Bottom quartile: low entropy, most predictable
            self._current_regime = self.REGIME_LOW
            self._current_scale = self._config.q1_scale
        elif entropy <= q50:
            # 25th-50th percentile
            self._current_regime = self.REGIME_MEDIUM_LOW
            self._current_scale = self._config.q2_scale
        elif entropy <= q75:
            # 50th-75th percentile
            self._current_regime = self.REGIME_MEDIUM_HIGH
            self._current_scale = self._config.q3_scale
        else:
            # Top quartile: high entropy, most random
            self._current_regime = self.REGIME_HIGH
            self._current_scale = self._config.q4_scale

        logger.debug(
            "Entropy regime updated: entropy=%.4f, q25=%.4f, q50=%.4f, "
            "q75=%.4f -> regime=%s, scale=%.2f",
            entropy, q25, q50, q75,
            self._current_regime,
            self._current_scale,
        )
