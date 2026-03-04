"""
Edge 7: Perpetual Basis Mean Reversion (PBMR)

Applies the Ornstein-Uhlenbeck model to the perpetual futures basis:
    basis = ln(perp_price) - ln(spot_price)

The basis mean-reverts because arbitrageurs enforce convergence.
When basis is extreme, we trade the convergence:
    - Basis too positive (perp premium) -> short perp, long spot
    - Basis too negative (perp discount) -> long perp, short spot

Enhanced with OU half-life for optimal entry/exit timing.

Reuses: OrnsteinUhlenbeck from trading_algo.quant_core.models
Expected SR: 1.5-2.0
Correlation with others: Low-moderate (shares some basis dynamics with FRM)
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


class PerpetualBasisMeanReversion(CryptoEdge):
    """
    Trade mean reversion of the perpetual-spot basis using OU model.

    The basis = ln(perp) - ln(spot) exhibits strong mean reversion
    because arbitrageurs enforce the no-arbitrage relationship.
    When the basis deviates significantly, we trade convergence.
    """

    def __init__(
        self,
        lookback: int = 120,        # Bars for OU parameter estimation
        entry_threshold: float = 1.5,  # S-score to enter (more conservative than equity default 1.25)
        exit_threshold: float = 0.4,
        stop_threshold: float = 4.0,
        refit_interval: int = 60,    # Re-estimate OU params every N bars
        min_basis_history: int = 60,  # Minimum basis observations before trading
    ):
        self._lookback = lookback
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._stop_threshold = stop_threshold
        self._refit_interval = refit_interval
        self._min_basis_history = min_basis_history

        # Per-symbol state
        self._perp_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 50))
        self._spot_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 50))
        self._basis_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback + 50))
        self._bar_count: Dict[str, int] = defaultdict(int)

        # OU model state per symbol
        self._ou_params: Dict[str, dict] = {}  # {kappa, theta, sigma, sigma_eq, half_life, r_sq, valid}
        self._positions: Dict[str, int] = defaultdict(int)  # 1=long basis, -1=short basis, 0=flat

    @property
    def name(self) -> str:
        return "PerpBasisMeanReversion"

    @property
    def warmup_bars(self) -> int:
        return self._min_basis_history

    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """
        Update with a new bar. Expects either:
        - kwargs['spot_price'] for perp bars, OR
        - separate calls for spot and perp (symbol suffixed)
        """
        spot_price = kwargs.get('spot_price')
        funding_rate = kwargs.get('funding_rate')

        self._perp_prices[symbol].append(price)
        if spot_price and spot_price > 0:
            self._spot_prices[symbol].append(spot_price)

            # Compute log basis
            basis = math.log(price) - math.log(spot_price)
            self._basis_history[symbol].append(basis)

        self._bar_count[symbol] += 1

        # Re-fit OU model periodically
        if (self._bar_count[symbol] % self._refit_interval == 0
                and len(self._basis_history[symbol]) >= self._min_basis_history):
            self._fit_ou(symbol)

    def _fit_ou(self, symbol: str) -> None:
        """Fit OU parameters to the basis series using regression."""
        basis = np.array(self._basis_history[symbol], dtype=np.float64)
        if len(basis) < self._lookback:
            return

        y = basis[-self._lookback:]
        delta_y = np.diff(y)
        y_lag = y[:-1]

        # OLS: delta_y = a + b * y_lag
        n = len(delta_y)
        if n < 20:
            return

        x_mean = np.mean(y_lag)
        y_mean = np.mean(delta_y)
        xy_cov = np.mean((y_lag - x_mean) * (delta_y - y_mean))
        x_var = np.var(y_lag, ddof=1)

        if x_var < 1e-12:
            return

        slope = xy_cov / x_var
        intercept = y_mean - slope * x_mean

        # For mean reversion, slope must be negative
        if slope >= 0:
            self._ou_params[symbol] = {'valid': False}
            return

        kappa = -slope
        theta = intercept / kappa if kappa > 1e-10 else 0.0

        # Sigma from residuals
        predicted = intercept + slope * y_lag
        residuals = delta_y - predicted
        sigma = float(np.std(residuals, ddof=2))

        # Half-life
        half_life = -math.log(2) / slope

        # Equilibrium std dev
        sigma_eq = sigma / math.sqrt(2 * kappa) if kappa > 1e-10 else sigma

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((delta_y - y_mean) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        is_valid = (
            kappa > 0.01
            and 1.0 < half_life < 500.0
            and sigma_eq > 1e-8
            and r_sq > 0.005
        )

        self._ou_params[symbol] = {
            'kappa': float(kappa),
            'theta': float(theta),
            'sigma': float(sigma),
            'sigma_eq': float(sigma_eq),
            'half_life': float(half_life),
            'r_sq': float(r_sq),
            'valid': is_valid,
        }

    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Generate trading signal from basis s-score."""
        params = self._ou_params.get(symbol, {})
        if not params.get('valid', False):
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="OU params invalid or not fitted",
            )

        if not self._basis_history[symbol]:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason="No basis data",
            )

        current_basis = self._basis_history[symbol][-1]
        theta = params['theta']
        sigma_eq = params['sigma_eq']

        if sigma_eq < 1e-10:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
            )

        s_score = (current_basis - theta) / sigma_eq
        confidence = min(1.0, abs(s_score) / 3.0) * params['r_sq']

        # Position management
        current_pos = self._positions[symbol]

        # Stop loss check
        if current_pos != 0 and abs(s_score) > self._stop_threshold:
            self._positions[symbol] = 0
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Stop triggered: s={s_score:.2f}",
                data={'s_score': s_score, 'action': 'stop'},
            )

        # Exit check
        if current_pos != 0 and abs(s_score) < self._exit_threshold:
            self._positions[symbol] = 0
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.NEUTRAL,
                confidence=0.0,
                reason=f"Exit: s={s_score:.2f} near mean",
                data={'s_score': s_score, 'action': 'exit'},
            )

        # Entry signals
        # Basis too HIGH (perp premium) -> short perp (perp will fall toward spot)
        # In the controller, this maps to: short the perp symbol
        if s_score > self._entry_threshold and current_pos >= 0:
            self._positions[symbol] = -1
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.SHORT,
                confidence=confidence,
                reason=f"Basis too high: s={s_score:.2f}, half_life={params['half_life']:.1f}",
                data={
                    's_score': s_score,
                    'basis': current_basis,
                    'theta': theta,
                    'half_life': params['half_life'],
                    'action': 'entry_short',
                },
            )

        # Basis too LOW (perp discount) -> long perp (perp will rise toward spot)
        if s_score < -self._entry_threshold and current_pos <= 0:
            self._positions[symbol] = 1
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.LONG,
                confidence=confidence,
                reason=f"Basis too low: s={s_score:.2f}, half_life={params['half_life']:.1f}",
                data={
                    's_score': s_score,
                    'basis': current_basis,
                    'theta': theta,
                    'half_life': params['half_life'],
                    'action': 'entry_long',
                },
            )

        # Strong signals at extreme s-scores
        if s_score > self._entry_threshold * 1.5 and current_pos == -1:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.STRONG_SHORT,
                confidence=min(1.0, confidence * 1.3),
                reason=f"Strong basis premium: s={s_score:.2f}",
                data={'s_score': s_score, 'action': 'hold_short'},
            )

        if s_score < -self._entry_threshold * 1.5 and current_pos == 1:
            return EdgeSignal(
                edge_name=self.name,
                vote=CryptoEdgeVote.STRONG_LONG,
                confidence=min(1.0, confidence * 1.3),
                reason=f"Strong basis discount: s={s_score:.2f}",
                data={'s_score': s_score, 'action': 'hold_long'},
            )

        # Hold existing position or neutral
        if current_pos != 0:
            vote = CryptoEdgeVote.LONG if current_pos > 0 else CryptoEdgeVote.SHORT
            return EdgeSignal(
                edge_name=self.name,
                vote=vote,
                confidence=confidence * 0.5,
                reason=f"Holding: s={s_score:.2f}",
                data={'s_score': s_score, 'action': 'hold'},
            )

        return EdgeSignal(
            edge_name=self.name,
            vote=CryptoEdgeVote.NEUTRAL,
            confidence=0.0,
            reason=f"No signal: s={s_score:.2f}",
            data={'s_score': s_score},
        )

    def reset(self) -> None:
        self._perp_prices.clear()
        self._spot_prices.clear()
        self._basis_history.clear()
        self._bar_count.clear()
        self._ou_params.clear()
        self._positions.clear()
