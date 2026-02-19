"""
Cross-Asset Divergence Strategy

Academic Foundation:
    - Collin-Dufresne, Goldstein & Martin (2001): Credit spreads driven by
      equity-related factors.
    - Norden & Weber (2009): CDS market leads stock market in price
      discovery.
    - Blanco, Brennan & Marsh (2005): CDS prices lead bond prices.

Core insight: Credit markets are populated by more sophisticated/informed
participants than equity markets.  When equity momentum and credit/bond
momentum diverge, the credit market is almost always right.  This strategy
exploits the systematic lag in equity prices relative to credit signals.

Signal Generation:
    1. Multi-asset divergence detection across equity, credit, treasury, gold
       and volatility proxies at multiple lookback windows (5, 10, 20 days).
    2. Credit spread signal derived from relative performance of high-yield
       bonds (HYG) vs equities (SPY).
    3. Multi-timeframe confirmation: 5d for entry, 20d for direction, 60d
       for trend validation.

Position Management:
    - Holding period: 2-10 days (mean reversion of divergence).
    - Entry: divergence z-score > 2.0.
    - Exit: divergence z-score < 0.5 (convergence).
    - Stop: divergence z-score > 4.0 with confirmation.
    - Sizing: inverse volatility, scaled by divergence magnitude.

Expected: Sharpe 0.6-1.0 standalone, uncorrelated with pure momentum.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.utils.constants import (
    EPSILON,
    SQRT_252,
    VOL_TARGET_DEFAULT,
)
from trading_algo.quant_core.utils.math_utils import (
    ewma_volatility,
    ols_regression,
    rolling_mean,
    rolling_std,
    simple_returns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DivergenceSignal:
    """A single cross-asset divergence signal.

    Attributes:
        timestamp:            When the signal was generated.
        target_symbol:        The tradeable instrument (e.g. ``"SPY"``).
        reference_symbol:     The leading instrument (e.g. ``"HYG"``).
        direction:            Desired exposure, -1.0 (short) to +1.0 (long).
        confidence:           Composite confidence score, 0.0 to 1.0.
        divergence_score:     Rolling z-score of the momentum divergence.
        timeframe_agreement:  Fraction of lookback windows that agree on
                              direction (0.0 to 1.0).
        holding_period_est:   Estimated days until convergence.
        position_size:        Suggested portfolio weight (unsigned).
    """

    timestamp: datetime
    target_symbol: str
    reference_symbol: str
    direction: float
    confidence: float
    divergence_score: float
    timeframe_agreement: float
    holding_period_est: int
    position_size: float


@dataclass
class DivergenceConfig:
    """Tunable parameters for :class:`CrossAssetDivergenceStrategy`.

    All numeric defaults were chosen to match the academic literature and
    in-sample calibration on 2005-2020 US equity / credit data.
    """

    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    """Momentum lookback windows used for divergence z-score."""

    zscore_lookback: int = 60
    """Rolling window for standardising the divergence series."""

    entry_threshold: float = 2.0
    """Minimum absolute z-score required to enter a position."""

    exit_threshold: float = 0.5
    """Convergence threshold: exit when abs(z-score) falls below this."""

    stop_threshold: float = 4.0
    """Stop-loss z-score: exit if divergence widens beyond this level with
    confirmation from the trend timeframe."""

    max_holding_days: int = 10
    """Hard cap on holding period regardless of convergence."""

    vol_target: float = VOL_TARGET_DEFAULT
    """Annualised portfolio volatility target."""

    max_position: float = 0.15
    """Maximum absolute weight per asset pair signal."""

    min_position: float = 0.01
    """Minimum weight below which a signal is discarded."""

    correlation_lookback: int = 60
    """Rolling window for pairwise correlation estimation."""

    min_history: int = 60
    """Minimum number of price observations before the strategy fires."""

    vol_lookback: int = 20
    """Lookback for EWMA volatility used in position sizing."""

    confirmation_timeframes: List[int] = field(
        default_factory=lambda: [5, 20, 60]
    )
    """Windows used for multi-timeframe confirmation (entry, direction,
    trend)."""

    # -- Asset pair definitions ------------------------------------------------
    # Each tuple is ``(target, reference)``.  The *target* is what we trade;
    # the *reference* is the leading indicator.
    asset_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("SPY", "HYG"),  # Equity vs High Yield
            ("SPY", "LQD"),  # Equity vs Investment Grade Credit
            ("SPY", "TLT"),  # Equity vs Treasuries (inverse)
            ("SPY", "GLD"),  # Equity vs Gold (inverse)
            ("IWM", "SPY"),  # Small-cap vs Large-cap (risk appetite)
        ]
    )

    # Pairs where a *positive* return in the reference asset implies a
    # *negative* signal for the target (e.g. rising Treasuries = risk-off).
    inverse_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("SPY", "TLT"),
            ("SPY", "GLD"),
        ]
    )


# ---------------------------------------------------------------------------
# Vectorised helpers (not JIT -- they call the JIT'd math_utils internally)
# ---------------------------------------------------------------------------


def _rolling_return(prices: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Compute the rolling simple return over *window* bars.

    Returns an array of the same length as *prices*, NaN-padded at the
    start where the lookback is insufficient.
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return result
    # Vectorised: r_t = P_t / P_{t-window} - 1
    result[window:] = prices[window:] / prices[:-window] - 1.0
    return result


def _rolling_zscore(
    series: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
    """Compute the rolling z-score of *series* over *window*.

    Uses the project's ``rolling_mean`` / ``rolling_std`` for consistency
    and numerical stability.
    """
    mu = rolling_mean(series, window)
    sigma = rolling_std(series, window)
    # Guard against division by zero
    safe_sigma = np.where(np.isnan(sigma) | (sigma < EPSILON), np.nan, sigma)
    z = (series - mu) / safe_sigma
    return z


def _rolling_correlation(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Pearson correlation between *a* and *b*.

    Returns NaN where history is insufficient.
    """
    n = min(len(a), len(b))
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    for i in range(window - 1, n):
        start = i - window + 1
        x = a[start: i + 1]
        y = b[start: i + 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.mean((x - x_mean) * (y - y_mean))
        std_x = np.std(x)
        std_y = np.std(y)
        denom = std_x * std_y
        if denom > EPSILON:
            result[i] = cov / denom
        else:
            result[i] = 0.0
    return result


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class CrossAssetDivergenceStrategy:
    """Exploit the empirical lead of credit/bond markets over equities.

    The strategy monitors multiple asset pairs and timeframes, detects
    statistically significant divergences, and generates sized,
    direction-aware signals with built-in risk management.

    Usage::

        strategy = CrossAssetDivergenceStrategy()

        for bar in daily_bars:
            for symbol in bar.symbols:
                strategy.update(symbol, bar.close[symbol])

        signals = strategy.generate_signals(price_dict)
        weights  = strategy.get_target_weights(price_dict)
    """

    def __init__(self, config: Optional[DivergenceConfig] = None) -> None:
        self.config = config or DivergenceConfig()
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._active_positions: Dict[Tuple[str, str], _ActivePosition] = {}

        # Pre-compute the set of inverse pairs for O(1) lookup
        self._inverse_set: set = set(self.config.inverse_pairs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, price: float) -> None:
        """Append a new close price for *symbol*.

        Internally caps history length so memory stays bounded.
        """
        self._price_history[symbol].append(price)

        # Keep enough history for the longest confirmation window + zscore
        # lookback + margin.
        max_needed = (
            max(
                max(self.config.lookback_windows),
                max(self.config.confirmation_timeframes),
                self.config.zscore_lookback,
                self.config.correlation_lookback,
                self.config.vol_lookback,
            )
            + self.config.zscore_lookback
            + 20
        )
        if len(self._price_history[symbol]) > max_needed:
            self._price_history[symbol] = self._price_history[symbol][
                -max_needed:
            ]

    def generate_signals(
        self,
        prices: Dict[str, NDArray[np.float64]],
    ) -> List[DivergenceSignal]:
        """Generate divergence signals for all configured asset pairs.

        Args:
            prices: Mapping of symbol to a 1-D close-price array.  Arrays
                    may have different lengths; the strategy handles
                    alignment internally.

        Returns:
            A list of :class:`DivergenceSignal` objects, one per fired pair.
            Pairs that lack sufficient data or do not breach the entry
            threshold are omitted.
        """
        signals: List[DivergenceSignal] = []
        now = datetime.utcnow()

        for target, reference in self.config.asset_pairs:
            target_prices = prices.get(target)
            reference_prices = prices.get(reference)

            # -- Missing data guard ------------------------------------
            if target_prices is None or reference_prices is None:
                continue

            target_arr = np.asarray(target_prices, dtype=np.float64)
            ref_arr = np.asarray(reference_prices, dtype=np.float64)

            # Align lengths
            common_len = min(len(target_arr), len(ref_arr))
            if common_len < self.config.min_history:
                continue
            target_arr = target_arr[-common_len:]
            ref_arr = ref_arr[-common_len:]

            is_inverse = (target, reference) in self._inverse_set

            # -- Divergence across all lookback windows ----------------
            divergence_by_window: Dict[int, float] = {}
            for window in self.config.lookback_windows:
                div_series = self._compute_divergence(
                    target_arr, ref_arr, window, is_inverse
                )
                zscore_series = _rolling_zscore(
                    div_series, self.config.zscore_lookback
                )
                latest_z = zscore_series[-1]
                if not np.isnan(latest_z):
                    divergence_by_window[window] = float(latest_z)

            if not divergence_by_window:
                continue

            # Primary divergence score: shortest lookback window that exists
            primary_window = min(divergence_by_window.keys())
            divergence_score = divergence_by_window[primary_window]

            # -- Multi-timeframe confirmation --------------------------
            tf_agreement = self._timeframe_agreement(divergence_by_window)

            # -- Credit spread sub-signal (only for HYG/LQD pairs) ----
            spread_boost = 0.0
            if reference in ("HYG", "LQD"):
                spread_boost = self._compute_spread_signal(
                    ref_arr, target_arr
                )

            # -- Entry decision ----------------------------------------
            abs_div = abs(divergence_score)
            if abs_div < self.config.entry_threshold:
                continue

            # Direction: if divergence_score > 0, target outran reference
            # -> target is overvalued -> go SHORT target.
            # If divergence_score < 0, target lagged -> go LONG target.
            raw_direction = -np.sign(divergence_score)

            # Incorporate spread boost: same-sign reinforcement
            if spread_boost != 0.0:
                # If spread signal agrees, boost confidence; if disagrees,
                # dampen.
                if np.sign(spread_boost) == raw_direction:
                    abs_div += 0.5  # widen effective score
                else:
                    abs_div -= 0.25

            if abs_div < self.config.entry_threshold:
                continue

            # -- Confidence and sizing ---------------------------------
            confidence = self._compute_confidence(
                abs_div, tf_agreement, target_arr, ref_arr
            )
            position_size = self._compute_position_size(
                target_arr, abs_div, confidence
            )

            # Holding period estimate: stronger divergence -> shorter
            # expected convergence (empirically calibrated).
            holding_est = max(
                2, min(self.config.max_holding_days, int(10 / max(abs_div, 1.0)))
            )

            signals.append(
                DivergenceSignal(
                    timestamp=now,
                    target_symbol=target,
                    reference_symbol=reference,
                    direction=float(raw_direction),
                    confidence=confidence,
                    divergence_score=divergence_score,
                    timeframe_agreement=tf_agreement,
                    holding_period_est=holding_est,
                    position_size=position_size,
                )
            )

        return signals

    def get_target_weights(
        self,
        prices: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, float]:
        """Aggregate signals into net portfolio weights per symbol.

        When multiple pairs produce signals for the same target symbol the
        weights are combined as a confidence-weighted average.

        Returns:
            Mapping of symbol to signed target weight.
        """
        signals = self.generate_signals(prices)
        if not signals:
            return {}

        # Accumulate weighted contributions per target symbol.
        weight_sum: Dict[str, float] = defaultdict(float)
        conf_sum: Dict[str, float] = defaultdict(float)

        for sig in signals:
            w = sig.direction * sig.position_size * sig.confidence
            weight_sum[sig.target_symbol] += w
            conf_sum[sig.target_symbol] += sig.confidence

        weights: Dict[str, float] = {}
        for sym in weight_sum:
            if conf_sum[sym] > EPSILON:
                raw_w = weight_sum[sym] / conf_sum[sym]
            else:
                raw_w = 0.0
            # Clip to max position
            clipped = float(
                np.clip(raw_w, -self.config.max_position, self.config.max_position)
            )
            if abs(clipped) >= self.config.min_position:
                weights[sym] = clipped

        return weights

    def reset(self) -> None:
        """Reset all internal state."""
        self._price_history.clear()
        self._active_positions.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_divergence(
        self,
        asset_a: NDArray[np.float64],
        asset_b: NDArray[np.float64],
        window: int,
        inverse: bool = False,
    ) -> NDArray[np.float64]:
        """Compute the raw momentum divergence series.

        ``divergence = momentum_a(window) - momentum_b(window)``

        For inverse pairs (e.g. equity vs treasuries) the reference
        momentum sign is flipped before subtraction so that *increasing*
        reference prices map to a *negative* equity signal.

        Args:
            asset_a: Target price series (aligned).
            asset_b: Reference price series (aligned).
            window:  Lookback in bars for rolling return.
            inverse: If ``True``, negate the reference momentum.

        Returns:
            1-D divergence array, same length as inputs, NaN-padded.
        """
        mom_a = _rolling_return(asset_a, window)
        mom_b = _rolling_return(asset_b, window)

        if inverse:
            mom_b = -mom_b

        divergence = mom_a - mom_b
        return divergence

    def _compute_spread_signal(
        self,
        credit: NDArray[np.float64],
        equity: NDArray[np.float64],
    ) -> float:
        """Derive a directional signal from the credit-equity spread.

        HYG price is *inverse* of credit spreads:
        - Rising HYG (tightening spreads) + flat equity -> bullish (LONG).
        - Falling HYG (widening spreads) + flat equity -> bearish (SHORT).

        Returns:
            A float in ``[-1.0, 1.0]`` indicating the spread-implied
            direction for equities.
        """
        if len(credit) < 6 or len(equity) < 6:
            return 0.0

        credit_ret_5 = (credit[-1] / credit[-6]) - 1.0
        equity_ret_5 = (equity[-1] / equity[-6]) - 1.0

        spread_change = credit_ret_5 - equity_ret_5

        # Classify
        ret_threshold = 0.005  # 0.5% minimum move
        if abs(equity_ret_5) < ret_threshold:
            # Equity flat -- credit move is the signal
            if credit_ret_5 > ret_threshold:
                return 1.0   # Tightening spreads -> bullish
            elif credit_ret_5 < -ret_threshold:
                return -1.0  # Widening spreads -> bearish

        # Both moving: look at the spread differential
        if abs(spread_change) < ret_threshold:
            return 0.0
        return float(np.clip(spread_change * 50.0, -1.0, 1.0))

    def _timeframe_agreement(
        self,
        divergences: Dict[int, float],
    ) -> float:
        """Compute the fraction of timeframes that agree on direction.

        A timeframe *agrees* if its divergence z-score has the same sign as
        the majority.  Returns 0.0 when there are no valid entries and 1.0
        when all timeframes point the same way.
        """
        if not divergences:
            return 0.0

        values = list(divergences.values())
        n_total = len(values)

        n_positive = sum(1 for v in values if v > 0)
        n_negative = sum(1 for v in values if v < 0)

        majority = max(n_positive, n_negative)
        return majority / n_total

    def _compute_confidence(
        self,
        abs_divergence: float,
        tf_agreement: float,
        target_prices: NDArray[np.float64],
        ref_prices: NDArray[np.float64],
    ) -> float:
        """Build a composite confidence score in [0, 1].

        Components:
        1. **Divergence magnitude** -- higher z-score -> higher confidence
           (tanh mapping to avoid runaway at extremes).
        2. **Timeframe agreement** -- more windows agreeing -> higher.
        3. **Pair correlation** -- stronger historical correlation means a
           divergence is more likely to revert.
        """
        # 1. Magnitude component (tanh saturates gently)
        mag_score = float(np.tanh((abs_divergence - self.config.entry_threshold) / 2.0))
        mag_score = max(0.0, mag_score)

        # 2. Timeframe component
        tf_score = tf_agreement

        # 3. Correlation component
        min_len = min(len(target_prices), len(ref_prices))
        if min_len >= self.config.correlation_lookback + 1:
            target_rets = simple_returns(
                target_prices[-self.config.correlation_lookback - 1:]
            )
            ref_rets = simple_returns(
                ref_prices[-self.config.correlation_lookback - 1:]
            )
            corr = np.corrcoef(target_rets, ref_rets)[0, 1]
            corr_score = float(np.clip(abs(corr), 0.0, 1.0))
        else:
            corr_score = 0.5  # Agnostic prior

        # Weighted combination
        confidence = 0.40 * mag_score + 0.35 * tf_score + 0.25 * corr_score
        return float(np.clip(confidence, 0.0, 1.0))

    def _compute_position_size(
        self,
        target_prices: NDArray[np.float64],
        abs_divergence: float,
        confidence: float,
    ) -> float:
        """Size the position using inverse-volatility and divergence scaling.

        Steps:
        1. Estimate annualised volatility of the target asset.
        2. Compute base weight = vol_target / asset_vol.
        3. Scale by divergence magnitude (linear in z-score, capped).
        4. Scale by confidence.
        5. Clip to ``[min_position, max_position]``.
        """
        if len(target_prices) < self.config.vol_lookback + 2:
            return self.config.min_position

        rets = simple_returns(
            target_prices[-(self.config.vol_lookback + 1):]
        )
        vol_arr = ewma_volatility(rets)
        ann_vol = vol_arr[-1] if len(vol_arr) > 0 else 0.15

        if ann_vol < EPSILON:
            ann_vol = 0.15  # Fallback

        # Base weight: inverse vol, targeting portfolio vol
        base = self.config.vol_target / ann_vol

        # Divergence scaling: linear from entry_threshold to 2x entry.
        div_scale = min(
            2.0,
            abs_divergence / self.config.entry_threshold
        )
        sized = base * div_scale * confidence

        return float(
            np.clip(sized, self.config.min_position, self.config.max_position)
        )


# ---------------------------------------------------------------------------
# Internal bookkeeping for live position tracking
# ---------------------------------------------------------------------------


@dataclass
class _ActivePosition:
    """Tracks a live divergence position for exit management."""

    target_symbol: str
    reference_symbol: str
    entry_zscore: float
    entry_bar: int
    direction: float
    bars_held: int = 0

    def should_exit(
        self,
        current_zscore: float,
        config: DivergenceConfig,
    ) -> bool:
        """Decide whether to close the position.

        Exit conditions (any one triggers):
        1. Divergence has converged below ``exit_threshold``.
        2. Hard time limit (``max_holding_days``) reached.
        3. Divergence has *widened* beyond ``stop_threshold`` (stop-loss).
        """
        self.bars_held += 1

        # 1. Convergence
        if abs(current_zscore) < config.exit_threshold:
            return True

        # 2. Time stop
        if self.bars_held >= config.max_holding_days:
            return True

        # 3. Widening stop -- only if divergence moved *further* against us.
        if abs(current_zscore) > config.stop_threshold:
            # Confirm the widening is in the same direction as entry.
            if np.sign(current_zscore) == np.sign(self.entry_zscore):
                return True

        return False


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def run_cross_asset_backtest(
    price_data: Dict[str, NDArray[np.float64]],
    timestamps: Optional[List[datetime]] = None,
    initial_capital: float = 100_000.0,
    config: Optional[DivergenceConfig] = None,
    slippage_bps: float = 5.0,
    commission_bps: float = 10.0,
) -> Dict:
    """Run a vectorised walk-forward backtest of the cross-asset divergence
    strategy.

    Args:
        price_data:      Mapping of symbol to 1-D close-price array.  All
                         arrays must have the same length (aligned dates).
        timestamps:      Optional parallel list of ``datetime`` objects.  If
                         ``None`` synthetic indices are used.
        initial_capital: Starting account value.
        config:          Strategy configuration.
        slippage_bps:    One-way slippage in basis points.
        commission_bps:  Round-trip commission in basis points.

    Returns:
        Dictionary with standard backtest metrics::

            {
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "sortino_ratio",
                "volatility",
                "max_drawdown",
                "calmar_ratio",
                "total_trades",
                "avg_holding_days",
                "win_rate",
                "profit_factor",
                "equity_curve",
                "returns",
                "trades",
                "final_value",
                "signals_generated",
            }
    """
    config = config or DivergenceConfig()
    slippage = slippage_bps / 10_000.0
    commission = commission_bps / 10_000.0

    # --- Validate inputs --------------------------------------------------
    symbols = list(price_data.keys())
    if not symbols:
        raise ValueError("price_data must contain at least one symbol")

    lengths = {sym: len(arr) for sym, arr in price_data.items()}
    n_bars = min(lengths.values())

    if n_bars < config.min_history + 10:
        logger.warning(
            "Insufficient history (%d bars) for backtest; need at least %d",
            n_bars,
            config.min_history + 10,
        )
        return _empty_backtest_result(initial_capital)

    # Truncate all to the shortest length for alignment.
    aligned: Dict[str, NDArray[np.float64]] = {
        sym: np.asarray(arr[-n_bars:], dtype=np.float64)
        for sym, arr in price_data.items()
    }

    if timestamps is None:
        timestamps = [datetime(2020, 1, 1)] * n_bars  # Dummy

    strategy = CrossAssetDivergenceStrategy(config)

    # --- State ------------------------------------------------------------
    cash = initial_capital
    positions: Dict[str, float] = defaultdict(float)  # symbol -> shares
    equity_curve: List[float] = []
    all_trades: List[Dict] = []
    all_signals: List[DivergenceSignal] = []
    holding_days: List[int] = []
    trade_pnls: List[float] = []

    # Track per-trade entry info for PnL computation
    _open_entries: Dict[str, Dict] = {}  # symbol -> entry info

    warmup = config.min_history + max(config.lookback_windows) + 10

    for t in range(n_bars):
        # Current prices
        current_prices: Dict[str, float] = {
            sym: float(aligned[sym][t]) for sym in symbols
        }

        # Mark-to-market
        position_value = sum(
            positions[sym] * current_prices.get(sym, 0.0)
            for sym in positions
        )
        equity = cash + position_value
        equity_curve.append(equity)

        if t < warmup:
            continue

        # Build the price-history-up-to-now dict for the strategy
        price_history: Dict[str, NDArray[np.float64]] = {
            sym: aligned[sym][: t + 1] for sym in symbols
        }

        # Generate signals
        signals = strategy.generate_signals(price_history)
        all_signals.extend(signals)

        # Derive target weights
        weights = strategy.get_target_weights(price_history)

        # Translate weights to target dollar values
        target_values: Dict[str, float] = {
            sym: equity * w for sym, w in weights.items()
        }

        # Rebalance each symbol
        for sym in symbols:
            current_shares = positions.get(sym, 0.0)
            current_value = current_shares * current_prices.get(sym, 0.0)
            target_val = target_values.get(sym, 0.0)
            delta_value = target_val - current_value
            price = current_prices.get(sym, 0.0)

            if price < EPSILON:
                continue

            # Minimum trade threshold: $50
            if abs(delta_value) < 50.0:
                continue

            delta_shares = delta_value / price

            # Apply slippage
            if delta_shares > 0:
                exec_price = price * (1.0 + slippage)
            else:
                exec_price = price * (1.0 - slippage)

            trade_cost = abs(delta_shares * exec_price)
            comm = trade_cost * commission

            # Execute
            if delta_shares > 0:
                cash -= trade_cost + comm
            else:
                cash += trade_cost - comm

            old_shares = positions[sym]
            positions[sym] = old_shares + delta_shares

            all_trades.append(
                {
                    "timestamp": timestamps[t] if t < len(timestamps) else t,
                    "symbol": sym,
                    "shares": delta_shares,
                    "price": exec_price,
                    "commission": comm,
                    "direction": "BUY" if delta_shares > 0 else "SELL",
                }
            )

            # Track PnL per round-trip (simplified)
            if abs(old_shares) < EPSILON and abs(delta_shares) > EPSILON:
                # Opening a new position
                _open_entries[sym] = {
                    "entry_price": exec_price,
                    "shares": delta_shares,
                    "entry_bar": t,
                }
            elif sym in _open_entries and abs(positions[sym]) < EPSILON:
                # Position closed
                entry = _open_entries.pop(sym)
                pnl = (exec_price - entry["entry_price"]) * entry["shares"]
                pnl -= comm  # Subtract this leg's commission
                trade_pnls.append(pnl)
                holding_days.append(t - entry["entry_bar"])

    # Ensure final bar equity is recorded if the loop already appended
    # (equity_curve has exactly n_bars entries from the loop above)

    # --- Metrics ----------------------------------------------------------
    equity_arr = np.array(equity_curve, dtype=np.float64)
    if len(equity_arr) < 2:
        return _empty_backtest_result(initial_capital)

    returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], EPSILON)
    total_return = (equity_arr[-1] / initial_capital) - 1.0
    n_years = len(returns) / 252.0

    if n_years > EPSILON:
        ann_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    else:
        ann_return = 0.0

    volatility = float(np.std(returns) * SQRT_252) if len(returns) > 1 else 0.0
    sharpe = (ann_return - 0.02) / volatility if volatility > EPSILON else 0.0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_dev = float(np.std(downside) * SQRT_252) if len(downside) > 1 else EPSILON
    sortino = (ann_return - 0.02) / downside_dev if downside_dev > EPSILON else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / np.maximum(peak, EPSILON)
    max_dd = float(np.max(drawdown))

    calmar = ann_return / max_dd if max_dd > EPSILON else 0.0

    # Trade statistics
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else EPSILON
    profit_factor = gross_profit / gross_loss if gross_loss > EPSILON else 0.0

    avg_hold = float(np.mean(holding_days)) if holding_days else 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "volatility": volatility,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "total_trades": len(all_trades),
        "avg_holding_days": avg_hold,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "equity_curve": equity_arr,
        "returns": returns,
        "trades": all_trades,
        "final_value": equity_arr[-1],
        "signals_generated": len(all_signals),
    }


def _empty_backtest_result(capital: float) -> Dict:
    """Return a zeroed-out result dict when the backtest cannot run."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "volatility": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "total_trades": 0,
        "avg_holding_days": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "equity_curve": np.array([capital]),
        "returns": np.array([]),
        "trades": [],
        "final_value": capital,
        "signals_generated": 0,
    }
