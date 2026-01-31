"""
Reflexivity Meter: Detect and measure Soros-style reflexivity.

Reflexivity occurs when:
1. Price movements affect fundamentals (self-fulfilling prophecy)
2. Fundamentals affect prices (traditional causality)
3. Both create feedback loops

Uses Granger causality tests to detect bidirectional causation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


class ReflexivityStage(Enum):
    """Stages of reflexivity cycle."""

    EFFICIENT = auto()      # No reflexivity, efficient market
    NASCENT = auto()        # Beginning of feedback loop
    ACCELERATING = auto()   # Feedback amplifying
    PEAK = auto()           # Maximum reflexivity
    UNWINDING = auto()      # Feedback reversing


@dataclass
class ReflexivityState:
    """Current reflexivity state."""

    timestamp: datetime
    symbol: str
    stage: ReflexivityStage
    reflexivity_coefficient: float  # -1 to 1
    granger_price_to_fund: float    # p-value for price -> fundamentals
    granger_fund_to_price: float    # p-value for fundamentals -> price
    is_bidirectional: bool

    @property
    def is_reflexive(self) -> bool:
        """Check if reflexivity is present."""
        return self.stage != ReflexivityStage.EFFICIENT


class ReflexivityMeter:
    """
    Detect and measure reflexivity in price-fundamental dynamics.

    Uses Granger causality to test:
    - Does past price predict future fundamentals?
    - Do past fundamentals predict future price?

    Bidirectional causality = reflexivity.
    """

    def __init__(
        self,
        lookback: int = 50,
        lag_order: int = 5,
        significance_level: float = 0.05,
        min_data_points: int = 30,
    ):
        self.lookback = lookback
        self.lag_order = lag_order
        self.significance_level = significance_level
        self.min_data_points = min_data_points

        # Data storage per symbol
        self._price_history: Dict[str, Deque[Tuple[datetime, float]]] = {}
        self._fundamental_history: Dict[str, Deque[Tuple[datetime, float]]] = {}

        # State cache
        self._last_state: Dict[str, ReflexivityState] = {}

    def update(
        self,
        symbol: str,
        price: float,
        fundamental: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update with new price and fundamental data."""
        ts = timestamp or datetime.now()

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.lookback)
            self._fundamental_history[symbol] = deque(maxlen=self.lookback)

        self._price_history[symbol].append((ts, price))
        self._fundamental_history[symbol].append((ts, fundamental))

    def compute_state(self, symbol: str) -> ReflexivityState:
        """Compute current reflexivity state."""
        if symbol not in self._price_history:
            return self._create_default_state(symbol)

        prices = [p for _, p in self._price_history[symbol]]
        fundamentals = [f for _, f in self._fundamental_history[symbol]]

        if len(prices) < self.min_data_points:
            return self._create_default_state(symbol)

        # Compute returns/changes
        price_returns = self._compute_returns(prices)
        fund_returns = self._compute_returns(fundamentals)

        # Granger causality tests
        gc_price_to_fund = self._granger_causality_test(
            price_returns, fund_returns, self.lag_order
        )
        gc_fund_to_price = self._granger_causality_test(
            fund_returns, price_returns, self.lag_order
        )

        # Determine reflexivity
        price_causes_fund = gc_price_to_fund < self.significance_level
        fund_causes_price = gc_fund_to_price < self.significance_level
        is_bidirectional = price_causes_fund and fund_causes_price

        # Compute coefficient
        coef = self._compute_coefficient(prices, fundamentals)

        # Determine stage
        stage = self._detect_stage(is_bidirectional, coef, price_causes_fund)

        state = ReflexivityState(
            timestamp=datetime.now(),
            symbol=symbol,
            stage=stage,
            reflexivity_coefficient=coef,
            granger_price_to_fund=gc_price_to_fund,
            granger_fund_to_price=gc_fund_to_price,
            is_bidirectional=is_bidirectional,
        )

        self._last_state[symbol] = state
        return state

    def _compute_returns(self, series: List[float]) -> List[float]:
        """Compute percentage returns."""
        returns = []
        for i in range(1, len(series)):
            if series[i-1] != 0:
                ret = (series[i] - series[i-1]) / series[i-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        return returns

    def _granger_causality_test(
        self,
        x: List[float],
        y: List[float],
        max_lag: int,
    ) -> float:
        """
        Simplified Granger causality test.

        Tests if x Granger-causes y.
        Returns p-value.
        """
        n = min(len(x), len(y))
        if n < max_lag + 10:
            return 1.0  # Not enough data

        # Align series
        x = x[-n:]
        y = y[-n:]

        # Restricted model: y ~ y_lags
        y_data = y[max_lag:]
        y_lags = [y[max_lag-i-1:-i-1 if i < max_lag-1 else None] for i in range(max_lag)]

        # Compute RSS for restricted model
        rss_r = self._compute_regression_rss(y_data, y_lags)

        # Unrestricted model: y ~ y_lags + x_lags
        x_lags = [x[max_lag-i-1:-i-1 if i < max_lag-1 else None] for i in range(max_lag)]
        all_lags = y_lags + x_lags

        rss_u = self._compute_regression_rss(y_data, all_lags)

        # F-test
        q = max_lag  # Number of restrictions
        df = len(y_data) - 2 * max_lag - 1

        if df <= 0 or rss_u <= 0:
            return 1.0

        f_stat = ((rss_r - rss_u) / q) / (rss_u / df)

        # Convert to p-value (simplified using chi-square approximation)
        p_value = self._f_to_pvalue(f_stat, q, df)

        return p_value

    def _compute_regression_rss(
        self,
        y: List[float],
        x_lists: List[List[float]],
    ) -> float:
        """Compute residual sum of squares for regression."""
        n = len(y)
        if n == 0 or not x_lists:
            return float('inf')

        # Simple OLS with multiple regressors
        # Using normal equations: beta = (X'X)^(-1) X'y

        k = len(x_lists)

        # Build X matrix and compute X'X and X'y
        xtx = [[0.0] * (k + 1) for _ in range(k + 1)]  # +1 for intercept
        xty = [0.0] * (k + 1)

        for i in range(n):
            row = [1.0] + [x_lists[j][i] if i < len(x_lists[j]) else 0 for j in range(k)]

            for a in range(k + 1):
                for b in range(k + 1):
                    xtx[a][b] += row[a] * row[b]
                xty[a] += row[a] * y[i]

        # Solve using simple Gaussian elimination (for small k)
        try:
            beta = self._solve_linear_system(xtx, xty)
        except Exception:
            return float('inf')

        # Compute predictions and RSS
        rss = 0.0
        for i in range(n):
            pred = beta[0]
            for j in range(k):
                if i < len(x_lists[j]):
                    pred += beta[j + 1] * x_lists[j][i]
            rss += (y[i] - pred) ** 2

        return rss

    def _solve_linear_system(
        self,
        A: List[List[float]],
        b: List[float],
    ) -> List[float]:
        """Solve Ax = b using Gaussian elimination."""
        n = len(b)
        # Augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        # Forward elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            if abs(aug[i][i]) < 1e-10:
                raise ValueError("Singular matrix")

            for k in range(i + 1, n):
                c = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= c * aug[i][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]

        return x

    def _f_to_pvalue(self, f_stat: float, df1: int, df2: int) -> float:
        """
        Convert F-statistic to p-value.

        Uses incomplete beta function approximation.
        """
        if f_stat <= 0:
            return 1.0

        x = df2 / (df2 + df1 * f_stat)

        # Beta distribution CDF approximation
        # P-value = 1 - F_cdf = I_x(df2/2, df1/2)
        a = df2 / 2
        b = df1 / 2

        # Use regularized incomplete beta function approximation
        p_value = self._incomplete_beta(x, a, b)

        return max(0.0, min(1.0, p_value))

    def _incomplete_beta(self, x: float, a: float, b: float) -> float:
        """Approximate regularized incomplete beta function."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # Use continued fraction approximation
        # This is a simplified version
        bt = math.exp(
            a * math.log(x) + b * math.log(1 - x)
            - math.log(a)
            - self._log_beta(a, b)
        )

        if x < (a + 1) / (a + b + 2):
            return bt * self._beta_cf(x, a, b) / a
        else:
            return 1.0 - bt * self._beta_cf(1 - x, b, a) / b

    def _log_beta(self, a: float, b: float) -> float:
        """Log of beta function."""
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def _beta_cf(self, x: float, a: float, b: float) -> float:
        """Continued fraction for incomplete beta."""
        max_iter = 100
        eps = 1e-10

        qab = a + b
        qap = a + 1
        qam = a - 1

        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < eps:
            d = eps
        d = 1.0 / d
        h = d

        for m in range(1, max_iter + 1):
            m2 = 2 * m

            # Even step
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < eps:
                d = eps
            c = 1.0 + aa / c
            if abs(c) < eps:
                c = eps
            d = 1.0 / d
            h *= d * c

            # Odd step
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < eps:
                d = eps
            c = 1.0 + aa / c
            if abs(c) < eps:
                c = eps
            d = 1.0 / d
            delta = d * c
            h *= delta

            if abs(delta - 1.0) < eps:
                break

        return h

    def _compute_coefficient(
        self,
        prices: List[float],
        fundamentals: List[float],
    ) -> float:
        """Compute reflexivity coefficient."""
        if len(prices) < 10 or len(fundamentals) < 10:
            return 0.0

        # Correlation between price changes and fundamental changes
        price_changes = self._compute_returns(prices)
        fund_changes = self._compute_returns(fundamentals)

        n = min(len(price_changes), len(fund_changes))
        if n < 5:
            return 0.0

        price_changes = price_changes[-n:]
        fund_changes = fund_changes[-n:]

        # Correlation
        mean_p = sum(price_changes) / n
        mean_f = sum(fund_changes) / n

        numerator = sum(
            (price_changes[i] - mean_p) * (fund_changes[i] - mean_f)
            for i in range(n)
        )
        denom_p = math.sqrt(sum((p - mean_p) ** 2 for p in price_changes))
        denom_f = math.sqrt(sum((f - mean_f) ** 2 for f in fund_changes))

        if denom_p * denom_f == 0:
            return 0.0

        return numerator / (denom_p * denom_f)

    def _detect_stage(
        self,
        is_bidirectional: bool,
        coefficient: float,
        price_causes_fund: bool,
    ) -> ReflexivityStage:
        """Detect reflexivity stage."""
        if not is_bidirectional and not price_causes_fund:
            return ReflexivityStage.EFFICIENT

        if price_causes_fund and not is_bidirectional:
            return ReflexivityStage.NASCENT

        if is_bidirectional:
            if abs(coefficient) > 0.7:
                return ReflexivityStage.PEAK
            elif abs(coefficient) > 0.4:
                return ReflexivityStage.ACCELERATING
            elif coefficient < -0.2:
                return ReflexivityStage.UNWINDING
            else:
                return ReflexivityStage.NASCENT

        return ReflexivityStage.EFFICIENT

    def _create_default_state(self, symbol: str) -> ReflexivityState:
        """Create default (no reflexivity) state."""
        return ReflexivityState(
            timestamp=datetime.now(),
            symbol=symbol,
            stage=ReflexivityStage.EFFICIENT,
            reflexivity_coefficient=0.0,
            granger_price_to_fund=1.0,
            granger_fund_to_price=1.0,
            is_bidirectional=False,
        )

    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """Generate trading signal from reflexivity state."""
        state = self.compute_state(symbol)

        if state.stage == ReflexivityStage.EFFICIENT:
            return None

        # Trading logic based on reflexivity stage
        if state.stage == ReflexivityStage.NASCENT:
            # Early stage - follow the direction
            direction = state.reflexivity_coefficient
            confidence = 0.4
        elif state.stage == ReflexivityStage.ACCELERATING:
            # Strong reflexivity - follow with conviction
            direction = state.reflexivity_coefficient
            confidence = 0.7
        elif state.stage == ReflexivityStage.PEAK:
            # Peak reflexivity - prepare for reversal
            direction = -state.reflexivity_coefficient * 0.5
            confidence = 0.6
        elif state.stage == ReflexivityStage.UNWINDING:
            # Unwinding - fade the move
            direction = -state.reflexivity_coefficient
            confidence = 0.65
        else:
            return None

        signal_type = SignalType.LONG if direction > 0 else SignalType.SHORT
        urgency = 0.5 if state.stage in (ReflexivityStage.NASCENT, ReflexivityStage.UNWINDING) else 0.7

        return Signal(
            source=SignalSource.REFLEXIVITY,
            signal_type=signal_type,
            symbol=symbol,
            direction=max(-1, min(1, direction)),
            confidence=confidence,
            urgency=urgency,
            metadata={
                "stage": state.stage.name,
                "coefficient": state.reflexivity_coefficient,
                "is_bidirectional": state.is_bidirectional,
            },
        )

    def inject_backtest_data(
        self,
        symbol: str,
        prices: List[Tuple[datetime, float]],
        fundamentals: List[Tuple[datetime, float]],
    ) -> None:
        """Inject historical data for backtesting."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.lookback)
            self._fundamental_history[symbol] = deque(maxlen=self.lookback)

        for ts, price in prices:
            self._price_history[symbol].append((ts, price))

        for ts, fund in fundamentals:
            self._fundamental_history[symbol].append((ts, fund))
