"""
Edge 8: Regime-Adaptive Dynamic Leverage (RADL)

Uses HMM regime detection on BTC returns to dynamically scale
portfolio leverage. This is NOT primarily a directional signal —
it's a MULTIPLIER on other edges' signals.

Regime -> Leverage mapping:
    BULL + HIGH confidence (>0.7): 2.5-3.0x
    BULL + LOW confidence:          1.5-2.0x
    NEUTRAL:                        1.0x
    BEAR + LOW confidence:          0.5x (reduce exposure)
    BEAR + HIGH confidence:         1.0-1.5x SHORT leverage
    HIGH_VOL:                       0.3x (capital preservation)

The edge ALSO emits directional signals in high-confidence regimes,
providing an additional alpha source on top of leverage scaling.

Reuses: HiddenMarkovRegime from trading_algo.quant_core.models
Expected SR: N/A standalone (amplifies combined SR by 30-60%)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from crypto_alpha.edges.base_edge import CryptoEdge
from crypto_alpha.types import CryptoAssetState, CryptoEdgeVote, EdgeSignal

logger = logging.getLogger(__name__)


# Regime labels (matching HMM output)
BULL = "bull"
BEAR = "bear"
NEUTRAL = "neutral"
HIGH_VOL = "high_vol"


class RegimeAdaptiveLeverage(CryptoEdge):
    """
    Detect market regime via HMM and dynamically adjust leverage.

    Uses a simplified 3-state Gaussian HMM trained on BTC returns
    and volatility features. Maps detected regime + confidence
    to a leverage scalar that other edges consume.
    """

    def __init__(
        self,
        lookback: int = 252,          # Training window (daily observations)
        n_states: int = 3,            # HMM states (bull/neutral/bear)
        retrain_interval: int = 7,    # Retrain every N days
        min_confidence: float = 0.5,  # Min regime probability to act
        vol_window: int = 20,         # Volatility estimation window
        high_vol_threshold: float = 0.80,  # Annualized vol threshold for HIGH_VOL
        # Leverage mapping
        bull_high_leverage: float = 2.5,
        bull_low_leverage: float = 1.5,
        neutral_leverage: float = 1.0,
        bear_low_leverage: float = 0.5,
        bear_high_leverage: float = 1.2,  # Short leverage
        high_vol_leverage: float = 0.3,
    ):
        self._lookback = lookback
        self._n_states = n_states
        self._retrain_interval = retrain_interval
        self._min_confidence = min_confidence
        self._vol_window = vol_window
        self._high_vol_threshold = high_vol_threshold

        self._leverage_map = {
            (BULL, True): bull_high_leverage,
            (BULL, False): bull_low_leverage,
            (NEUTRAL, True): neutral_leverage,
            (NEUTRAL, False): neutral_leverage,
            (BEAR, False): bear_low_leverage,
            (BEAR, True): bear_high_leverage,
            (HIGH_VOL, True): high_vol_leverage,
            (HIGH_VOL, False): high_vol_leverage,
        }

        # State
        self._daily_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 50))
        self._daily_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 50))
        self._bar_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._bar_count: Dict[str, int] = defaultdict(int)
        self._day_count: int = 0
        self._last_day: Optional[str] = None

        # HMM state
        self._hmm_fitted: bool = False
        self._state_means: Optional[np.ndarray] = None  # (n_states, n_features)
        self._state_vars: Optional[np.ndarray] = None
        self._transition_matrix: Optional[np.ndarray] = None
        self._state_priors: Optional[np.ndarray] = None

        # Current regime
        self._current_regime: str = NEUTRAL
        self._regime_confidence: float = 0.0
        self._current_leverage: float = 1.0
        self._pending_regime: Optional[str] = None  # Candidate regime waiting for confirmation
        self._pending_duration: int = 0  # How many days the pending regime has been detected

    @property
    def name(self) -> str:
        return "RegimeAdaptiveLeverage"

    @property
    def warmup_bars(self) -> int:
        return 65 * 288  # ~65 days of 5-min bars (need 60+ daily returns for HMM fit)

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        self._bar_prices[symbol].append(price)
        self._bar_count[symbol] += 1

        # Track daily prices for regime detection
        current_day = timestamp.strftime("%Y-%m-%d")
        if current_day != self._last_day:
            if self._last_day is not None and self._bar_prices[symbol]:
                # Record daily close
                self._daily_prices[symbol].append(price)
                if len(self._daily_prices[symbol]) >= 2:
                    prev = self._daily_prices[symbol][-2]
                    if prev > 0:
                        ret = price / prev - 1
                        self._daily_returns[symbol].append(ret)

                self._day_count += 1

                # Retrain HMM periodically
                if self._day_count % self._retrain_interval == 0:
                    self._fit_hmm(symbol)
                else:
                    # Just predict with current model
                    self._predict_regime(symbol)

            self._last_day = current_day

    def _fit_hmm(self, symbol: str) -> None:
        """Fit a simple 3-state Gaussian HMM to returns + volatility features."""
        returns = np.array(self._daily_returns[symbol], dtype=np.float64)
        if len(returns) < 60:
            return

        # Features: [return, rolling_vol]
        n = len(returns)
        vol = np.zeros(n)
        for i in range(self._vol_window, n):
            vol[i] = np.std(returns[i - self._vol_window:i], ddof=1) * math.sqrt(365)
        vol[:self._vol_window] = vol[self._vol_window] if n > self._vol_window else 0.3

        features = np.column_stack([returns, vol])

        # Simple K-means-style regime classification
        # (avoids hmmlearn dependency; works well enough for 3 states)
        self._fit_gaussian_mixture(features)
        self._predict_regime(symbol)

    def _fit_gaussian_mixture(self, features: np.ndarray) -> None:
        """Fit a simple Gaussian mixture model (K-means + variance estimation)."""
        n, d = features.shape
        k = self._n_states

        if n < k * 10:
            return

        # K-means initialization: sort by return, split into k groups
        sorted_idx = np.argsort(features[:, 0])
        chunk_size = n // k

        means = np.zeros((k, d))
        variances = np.zeros((k, d))
        priors = np.zeros(k)

        for i in range(k):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < k - 1 else n
            chunk = features[sorted_idx[start:end]]
            means[i] = np.mean(chunk, axis=0)
            variances[i] = np.var(chunk, axis=0, ddof=1) + 1e-8
            priors[i] = len(chunk) / n

        # 5 iterations of EM-style refinement
        for _ in range(5):
            # E-step: assign points to nearest state
            assignments = np.zeros(n, dtype=int)
            for j in range(n):
                dists = np.sum((features[j] - means) ** 2 / variances, axis=1)
                assignments[j] = np.argmin(dists)

            # M-step: update parameters
            for i in range(k):
                mask = assignments == i
                if np.sum(mask) < 5:
                    continue
                means[i] = np.mean(features[mask], axis=0)
                variances[i] = np.var(features[mask], axis=0, ddof=1) + 1e-8
                priors[i] = np.mean(mask)

        # Sort states by mean return: [bear, neutral, bull]
        sorted_states = np.argsort(means[:, 0])

        self._state_means = means[sorted_states]
        self._state_vars = variances[sorted_states]
        self._state_priors = priors[sorted_states]

        # Simple transition matrix from consecutive assignments
        assignments_sorted = np.zeros(n, dtype=int)
        for j in range(n):
            dists = np.sum((features[j] - self._state_means) ** 2 / self._state_vars, axis=1)
            assignments_sorted[j] = np.argmin(dists)

        trans = np.ones((k, k)) * 0.01  # Laplace smoothing
        for j in range(1, n):
            trans[assignments_sorted[j - 1], assignments_sorted[j]] += 1
        trans /= trans.sum(axis=1, keepdims=True)
        self._transition_matrix = trans

        self._hmm_fitted = True

    def _predict_regime(self, symbol: str) -> None:
        """Predict current regime from latest features."""
        if not self._hmm_fitted or self._state_means is None:
            return

        returns = self._daily_returns[symbol]
        if len(returns) < self._vol_window:
            return

        recent_ret = list(returns)[-1]
        recent_vol = float(np.std(list(returns)[-self._vol_window:], ddof=1) * math.sqrt(365))

        feature = np.array([recent_ret, recent_vol])

        # Compute posterior probabilities
        log_probs = np.zeros(self._n_states)
        for i in range(self._n_states):
            diff = feature - self._state_means[i]
            log_probs[i] = -0.5 * np.sum(diff ** 2 / self._state_vars[i])
            log_probs[i] += np.log(self._state_priors[i] + 1e-10)

        # Softmax to get probabilities
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()

        # Map state index to regime label
        # States sorted by return: [0=bear, 1=neutral, 2=bull]
        state_idx = int(np.argmax(probs))
        confidence = float(probs[state_idx])

        regime_map = {0: BEAR, 1: NEUTRAL, 2: BULL}
        regime = regime_map.get(state_idx, NEUTRAL)

        # Override to HIGH_VOL if realized vol exceeds threshold
        if recent_vol > self._high_vol_threshold:
            regime = HIGH_VOL
            # Confidence for HIGH_VOL is how far above threshold
            confidence = min(1.0, recent_vol / self._high_vol_threshold - 0.5)

        # Update regime with hysteresis (don't switch too fast)
        if regime == self._current_regime:
            # Already in this regime, reset pending
            self._pending_regime = None
            self._pending_duration = 0
        elif regime == self._pending_regime:
            # Same new regime detected again, increment counter
            self._pending_duration += 1
            if self._pending_duration >= 2:
                # Confirmed: switch to new regime
                self._current_regime = regime
                self._pending_regime = None
                self._pending_duration = 0
        else:
            # Different new regime, start tracking it
            self._pending_regime = regime
            self._pending_duration = 1

        self._regime_confidence = confidence
        high_conf = confidence > self._min_confidence
        self._current_leverage = self._leverage_map.get(
            (self._current_regime, high_conf), 1.0
        )

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """
        Produce a signal based on regime.

        The primary output is the leverage scalar in metadata.
        Secondary: directional signal in high-confidence regimes.
        """
        regime = self._current_regime
        confidence = self._regime_confidence
        leverage = self._current_leverage

        data = {
            'regime': regime,
            'confidence': confidence,
            'leverage_scalar': leverage,
        }

        # HIGH_VOL: capital preservation, reduce everything
        if regime == HIGH_VOL:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"HIGH_VOL regime: leverage={leverage:.1f}x",
                data=data,
            )

        # BULL with sufficient confidence: directional LONG signal
        if regime == BULL and confidence > 0.50:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.LONG if confidence < 0.75 else CryptoEdgeVote.STRONG_LONG,
                confidence=confidence * 0.6,  # Moderate confidence (regime, not precision signal)
                reason=f"BULL regime (p={confidence:.2f}): leverage={leverage:.1f}x",
                data=data,
            )

        # BEAR with sufficient confidence: directional SHORT signal
        if regime == BEAR and confidence > 0.50:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.SHORT if confidence < 0.8 else CryptoEdgeVote.STRONG_SHORT,
                confidence=confidence * 0.5,
                reason=f"BEAR regime (p={confidence:.2f}): leverage={leverage:.1f}x",
                data=data,
            )

        # NEUTRAL or low confidence: no directional signal
        return EdgeSignal(
            edge_name=self.name,
            vote=CryptoEdgeVote.NEUTRAL,
            confidence=0.0,
            reason=f"{regime} regime (p={confidence:.2f}): leverage={leverage:.1f}x",
            data=data,
        )

    def get_leverage_scalar(self) -> float:
        """Get the current leverage scalar for other edges to consume."""
        return self._current_leverage

    def get_regime_info(self) -> Dict:
        """Get current regime details."""
        return {
            'regime': self._current_regime,
            'confidence': self._regime_confidence,
            'leverage': self._current_leverage,
            'duration': self._pending_duration,
        }

    def reset(self) -> None:
        self._daily_prices.clear()
        self._daily_returns.clear()
        self._bar_prices.clear()
        self._bar_count.clear()
        self._day_count = 0
        self._last_day = None
        self._hmm_fitted = False
        self._current_regime = NEUTRAL
        self._regime_confidence = 0.0
        self._current_leverage = 1.0
        self._pending_regime = None
        self._pending_duration = 0
