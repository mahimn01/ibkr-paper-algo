"""
Edge 2: Funding Rate Momentum (FRM)

Predicts funding rate spikes using ARIMA(1,0,0) + GARCH(1,1)
on the funding rate time series, and enters BEFORE the spike.

Key insight: Funding rates are autoregressive. High funding today
strongly predicts high funding tomorrow. GARCH forecasts the
volatility of funding rates themselves.

Trade logic:
    1. Positive funding > 2σ -> market overleveraged long
       -> Short perp (earn high funding) + Long spot (hedge)
    2. Negative funding < -2σ -> market overleveraged short
       -> Long perp (earn funding) + Short spot (hedge)
    3. Funding momentum rising -> enter BEFORE next funding payment

Reuses: GARCH from trading_algo.quant_core.models
Expected SR: 1.5-2.0
Correlation with others: Low-moderate with PBMR (both involve basis)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from crypto_alpha.edges.base_edge import CryptoEdge
from crypto_alpha.types import CryptoAssetState, CryptoEdgeVote, EdgeSignal

logger = logging.getLogger(__name__)


class FundingRateMomentum(CryptoEdge):
    """
    Predict and trade funding rate momentum.

    Instead of passively harvesting current rates, this edge
    PREDICTS when rates will spike and positions accordingly.
    """

    def __init__(
        self,
        lookback: int = 90,           # Funding observations for model fitting
        entry_z: float = 0.85,         # Z-score threshold for entry (tuned for ±0.3% funding range)
        strong_z: float = 1.0,         # Z-score for strong conviction
        ar_persistence_min: float = 0.30,  # Min AR(1) coefficient to confirm momentum
        garch_alpha: float = 0.10,     # GARCH shock sensitivity
        garch_beta: float = 0.85,      # GARCH persistence
        prediction_horizon: int = 3,   # Predict N funding periods ahead
    ):
        self._lookback = lookback
        self._entry_z = entry_z
        self._strong_z = strong_z
        self._ar_persistence_min = ar_persistence_min
        self._garch_alpha = garch_alpha
        self._garch_beta = garch_beta
        self._prediction_horizon = prediction_horizon

        # Per-symbol state
        self._funding_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 50))
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._bar_count: Dict[str, int] = defaultdict(int)
        self._last_funding_rate: Dict[str, float] = defaultdict(float)

        # Model state
        self._ar_coeff: Dict[str, float] = {}  # AR(1) coefficient per symbol
        self._funding_mean: Dict[str, float] = {}
        self._funding_vol: Dict[str, float] = {}
        self._garch_conditional_var: Dict[str, float] = {}
        self._positions: Dict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "FundingRateMomentum"

    @property
    def warmup_bars(self) -> int:
        return 50  # Need ~50 funding rate observations (400 hours = ~17 days at 8h)

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        self._price_history[symbol].append(price)
        self._bar_count[symbol] += 1

        funding_rate = kwargs.get('funding_rate')
        if funding_rate is not None:
            prev_rate = self._last_funding_rate[symbol]
            # Only record if funding rate actually changed (new observation)
            if funding_rate != prev_rate or not self._funding_history[symbol]:
                self._funding_history[symbol].append(funding_rate)
                self._last_funding_rate[symbol] = funding_rate

                # Re-fit models every 10 new funding observations
                if len(self._funding_history[symbol]) % 10 == 0:
                    self._fit_models(symbol)

    def _fit_models(self, symbol: str) -> None:
        """Fit AR(1) and GARCH(1,1) to funding rate series."""
        rates = np.array(self._funding_history[symbol], dtype=np.float64)
        if len(rates) < 30:
            return

        # ── AR(1) coefficient estimation ──
        # r_t = c + phi * r_{t-1} + epsilon
        r_lag = rates[:-1]
        r_curr = rates[1:]

        r_lag_mean = np.mean(r_lag)
        r_curr_mean = np.mean(r_curr)

        cov = np.mean((r_lag - r_lag_mean) * (r_curr - r_curr_mean))
        var_lag = np.var(r_lag, ddof=1)

        if var_lag > 1e-16:
            phi = cov / var_lag
        else:
            phi = 0.0

        self._ar_coeff[symbol] = float(np.clip(phi, -0.99, 0.99))
        self._funding_mean[symbol] = float(np.mean(rates))
        self._funding_vol[symbol] = float(np.std(rates, ddof=1))

        # ── GARCH(1,1) on funding rate innovations ──
        # Compute innovations (residuals from AR model)
        c = r_curr_mean - phi * r_lag_mean
        innovations = r_curr - (c + phi * r_lag)
        innovations_sq = innovations ** 2

        # GARCH recursion for conditional variance
        n = len(innovations_sq)
        cond_var = np.zeros(n)
        omega = self._garch_alpha * float(np.mean(innovations_sq)) * 0.1
        cond_var[0] = float(np.var(innovations))

        for t in range(1, n):
            cond_var[t] = (
                omega
                + self._garch_alpha * innovations_sq[t - 1]
                + self._garch_beta * cond_var[t - 1]
            )

        self._garch_conditional_var[symbol] = float(cond_var[-1])

    def _forecast_funding(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Forecast next funding rate using AR(1) + GARCH.

        Returns dict with 'point_forecast', 'forecast_std', 'z_score',
        'ar_momentum' or None if not enough data.
        """
        if symbol not in self._ar_coeff:
            return None

        rates = self._funding_history[symbol]
        if len(rates) < 10:
            return None

        current_rate = rates[-1]
        phi = self._ar_coeff[symbol]
        mu = self._funding_mean[symbol]
        vol = self._funding_vol[symbol]

        # AR(1) point forecast: E[r_{t+1}] = c + phi * r_t
        c = mu * (1 - phi)
        point_forecast = c + phi * current_rate

        # Multi-step forecast for prediction_horizon
        multi_forecast = point_forecast
        for _ in range(self._prediction_horizon - 1):
            multi_forecast = c + phi * multi_forecast

        # Forecast uncertainty from GARCH
        cond_var = self._garch_conditional_var.get(symbol, vol ** 2)
        forecast_std = math.sqrt(max(cond_var, 1e-16))

        # Z-score: how extreme is the current rate relative to history?
        z_score = (current_rate - mu) / vol if vol > 1e-12 else 0.0

        # AR momentum: is the rate accelerating?
        if len(rates) >= 3:
            recent_rates = list(rates)[-3:]
            ar_momentum = recent_rates[-1] - recent_rates[-3]
        else:
            ar_momentum = 0.0

        return {
            'point_forecast': float(point_forecast),
            'multi_forecast': float(multi_forecast),
            'forecast_std': float(forecast_std),
            'z_score': float(z_score),
            'ar_momentum': float(ar_momentum),
            'phi': float(phi),
            'current_rate': float(current_rate),
        }

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate signal from funding rate prediction."""
        forecast = self._forecast_funding(symbol)
        if forecast is None:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="Not enough funding data",
            )

        z = forecast['z_score']
        phi = forecast['phi']
        current_rate = forecast['current_rate']
        momentum = forecast['ar_momentum']

        # Confidence based on z-score magnitude and AR persistence
        base_confidence = min(1.0, abs(z) / 4.0)
        ar_confidence = max(0.0, min(1.0, abs(phi) / 0.8))
        confidence = base_confidence * (0.5 + 0.5 * ar_confidence)

        # Only trade if AR persistence is meaningful
        if abs(phi) < self._ar_persistence_min:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Low AR persistence: phi={phi:.3f}",
                data=forecast,
            )

        # HIGH POSITIVE FUNDING -> short perp (collect funding)
        # Funding rate > 0 means longs pay shorts.
        # If z > threshold, rate is abnormally high -> will mean-revert
        # but if momentum is positive, rate is still rising -> keep collecting
        if z > self._strong_z and momentum >= 0:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.STRONG_SHORT,
                confidence=min(1.0, confidence * 1.2),
                reason=f"Strong positive funding: z={z:.2f}, rate={current_rate:.5f}, momentum rising",
                data=forecast,
            )

        if z > self._entry_z:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.SHORT,
                confidence=confidence,
                reason=f"High positive funding: z={z:.2f}, rate={current_rate:.5f}",
                data=forecast,
            )

        # HIGH NEGATIVE FUNDING -> long perp (collect funding from shorts)
        if z < -self._strong_z and momentum <= 0:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.STRONG_LONG,
                confidence=min(1.0, confidence * 1.2),
                reason=f"Strong negative funding: z={z:.2f}, rate={current_rate:.5f}, momentum falling",
                data=forecast,
            )

        if z < -self._entry_z:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.LONG,
                confidence=confidence,
                reason=f"High negative funding: z={z:.2f}, rate={current_rate:.5f}",
                data=forecast,
            )

        return EdgeSignal(
            edge_name=self.name,
            vote=CryptoEdgeVote.NEUTRAL,
            confidence=0.0,
            reason=f"Normal funding: z={z:.2f}, rate={current_rate:.5f}",
            data=forecast,
        )

    def reset(self) -> None:
        self._funding_history.clear()
        self._price_history.clear()
        self._bar_count.clear()
        self._last_funding_rate.clear()
        self._ar_coeff.clear()
        self._funding_mean.clear()
        self._funding_vol.clear()
        self._garch_conditional_var.clear()
        self._positions.clear()
