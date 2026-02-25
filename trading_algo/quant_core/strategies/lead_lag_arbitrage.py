"""
Lead-Lag Arbitrage Strategy

Academic Foundation:
    - Lo & MacKinlay (1990): Cross-autocorrelation in stock returns.
    - Hou (2007): Industry information diffusion and the lead-lag effect.
    - Chordia & Swaminathan (2000): Trading volume and cross-autocorrelations.

Core insight: Information propagates across related assets with measurable
delays.  High-liquidity or sector-bellwether stocks (leaders) incorporate
news faster than their peers or ETF baskets (followers).  By measuring
cross-correlation at empirically discovered lags, we can anticipate
follower moves and capture the convergence.

Discovered Pairs (5-minute frequency):
    - NVDA -> AAPL: lag=1 bar (5min), corr=-0.0453 (NEGATIVE = mean-reversion)
    - IWM  -> SPY:  lag=10 bars (50min), corr=+0.0393 (positive = continuation)
    - SMCI -> QQQ:  lag=1 bar (5min), corr=+0.0294 (positive = continuation)
    - SPY  -> QQQ:  lag=10 bars (50min), corr=+0.0364 (positive = continuation)

Signal Generation:
    1. Track rolling cross-correlation at the known optimal lag for each pair.
    2. When the leader makes a significant move (> threshold * ATR):
       - Positive lead-lag correlation: enter FOLLOWER in same direction.
       - Negative lead-lag correlation: enter FOLLOWER in opposite direction.
    3. Exit when follower converges, 15-bar timeout, or leader reverses.
    4. Position size scaled by correlation strength and confidence.

Expected: High-frequency alpha with short holding periods (1-15 bars at 5min).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default pairs discovered from cross-correlation analysis.
# Format: (leader, follower, lag_in_bars, expected_sign)
#   expected_sign = +1  => positive correlation => follower moves same direction
#   expected_sign = -1  => negative correlation => follower moves opposite direction
DEFAULT_PAIRS: List[Tuple[str, str, int, int]] = [
    ("NVDA", "AAPL", 1, -1),    # Negative: mean-reversion
    ("IWM", "SPY", 10, +1),     # Positive: continuation
    ("SMCI", "QQQ", 1, +1),     # Positive: continuation
    ("SPY", "QQQ", 10, +1),     # Positive: continuation
]


@dataclass
class LeadLagConfig:
    """Tunable parameters for :class:`LeadLagArbitrage`.

    All numeric defaults calibrated on 2023-2024 5-minute intraday data
    for the US equity universe.
    """

    pairs: List[Tuple[str, str, int, int]] = field(
        default_factory=lambda: list(DEFAULT_PAIRS)
    )
    """List of (leader, follower, lag_bars, expected_sign) tuples."""

    lookback: int = 390
    """Rolling cross-correlation window in bars.  390 5-minute bars is
    approximately one trading week (5 days * 78 bars/day)."""

    signal_threshold: float = 0.5
    """ATR multiplier: the leader must move at least this many ATRs
    (cumulative over the lag window) to trigger a signal."""

    atr_window: int = 78
    """Window in bars for Average True Range.  78 5-minute bars is one
    trading day."""

    max_hold: int = 15
    """Maximum bars to hold a position before forced exit."""

    min_correlation: float = 0.015
    """Minimum absolute rolling correlation required to trade a pair."""

    warmup: int = 100
    """Minimum bars per symbol before the strategy produces signals."""

    per_pair_weight: float = 0.08
    """Target portfolio weight allocated to each active pair signal."""

    convergence_fraction: float = 0.5
    """Fraction of the expected follower move at which we consider
    convergence complete and exit."""

    leader_reversal_threshold: float = 0.3
    """ATR multiplier: if the leader reverses by this much, exit."""


# ---------------------------------------------------------------------------
# Active position tracker
# ---------------------------------------------------------------------------


@dataclass
class _ActiveTrade:
    """Internal bookkeeping for an open lead-lag position."""

    leader: str
    follower: str
    lag: int
    direction: int                    # +1 long follower, -1 short follower
    entry_bar: int                    # Bar index at entry
    expected_move: float              # Expected absolute follower move (price)
    leader_move_at_entry: float       # Leader's cumulative return at entry
    follower_price_at_entry: float    # Follower close at entry
    bars_held: int = 0


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class LeadLagArbitrage:
    """Exploit cross-asset information flow at 5-minute frequency.

    The strategy monitors known leader-follower pairs with empirically
    discovered lag structures.  When a leader makes a statistically
    significant move, it enters the follower in the direction predicted
    by the sign of the lagged cross-correlation.

    Usage::

        strategy = LeadLagArbitrage()

        for bar in intraday_bars:
            for symbol in bar.symbols:
                strategy.update(symbol, bar.close, bar.high, bar.low, bar.volume)
            signals = strategy.generate_signals(bar.symbols, bar.timestamp)

    """

    def __init__(self, config: Optional[LeadLagConfig] = None) -> None:
        self.config = config or LeadLagConfig()

        # Per-symbol price/volume history
        self._closes: Dict[str, List[float]] = defaultdict(list)
        self._highs: Dict[str, List[float]] = defaultdict(list)
        self._lows: Dict[str, List[float]] = defaultdict(list)
        self._volumes: Dict[str, List[float]] = defaultdict(list)

        self._bars_seen: Dict[str, int] = defaultdict(int)
        self._bar_index: int = 0

        # Active positions keyed by (leader, follower) tuple
        self._active_trades: Dict[Tuple[str, str], _ActiveTrade] = {}

        # Maximum history length needed
        self._max_history = (
            self.config.lookback
            + max(p[2] for p in self.config.pairs)  # max lag
            + 50  # safety margin
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        close: float,
        high: float,
        low: float,
        volume: float,
    ) -> None:
        """Feed a single bar of OHLCV data for one symbol.

        Call once per symbol per bar.  The strategy internally tracks
        histories for all symbols and trims to bounded length.

        Args:
            symbol: Ticker/instrument identifier.
            close:  Close price for this bar.
            high:   High price for this bar.
            low:    Low price for this bar.
            volume: Trade volume for this bar.
        """
        self._closes[symbol].append(close)
        self._highs[symbol].append(high)
        self._lows[symbol].append(low)
        self._volumes[symbol].append(volume)
        self._bars_seen[symbol] += 1

        # Trim to bounded length
        if len(self._closes[symbol]) > self._max_history:
            self._closes[symbol] = self._closes[symbol][-self._max_history:]
            self._highs[symbol] = self._highs[symbol][-self._max_history:]
            self._lows[symbol] = self._lows[symbol][-self._max_history:]
            self._volumes[symbol] = self._volumes[symbol][-self._max_history:]

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[Dict[str, Any]]:
        """Generate lead-lag arbitrage signals for the current bar.

        Should be called once per bar, after all ``update()`` calls for
        that bar are complete.

        Args:
            symbols:   List of symbols in the current universe.
            timestamp: Timestamp of the current bar.

        Returns:
            List of signal dicts, each with keys:
                - ``symbol``: follower to trade
                - ``direction``: +1 (long) or -1 (short)
                - ``confidence``: signal quality score (0 to 1)
                - ``weight``: suggested portfolio weight
                - ``metadata``: dict with leader, lag, correlation info
        """
        self._bar_index += 1
        signals: List[Dict[str, Any]] = []

        for leader, follower, lag, expected_sign in self.config.pairs:
            # Skip pairs where we lack sufficient data
            if (self._bars_seen.get(leader, 0) < self.config.warmup
                    or self._bars_seen.get(follower, 0) < self.config.warmup):
                continue

            # Check for exit on active trades first
            pair_key = (leader, follower)
            if pair_key in self._active_trades:
                trade = self._active_trades[pair_key]
                should_exit, exit_reason = self._check_exit(trade)
                if should_exit:
                    # Emit exit signal
                    signals.append({
                        "symbol": follower,
                        "direction": 0,
                        "confidence": 0.0,
                        "weight": 0.0,
                        "metadata": {
                            "leader": leader,
                            "lag": lag,
                            "corr": 0.0,
                            "action": "exit",
                            "exit_reason": exit_reason,
                            "bars_held": trade.bars_held,
                        },
                    })
                    del self._active_trades[pair_key]
                else:
                    trade.bars_held += 1
                continue  # Don't generate new entry while position is active

            # Compute rolling cross-correlation at the specified lag
            leader_returns = self._get_returns(leader)
            follower_returns = self._get_returns(follower)

            if leader_returns is None or follower_returns is None:
                continue

            rolling_corr = self._compute_rolling_correlation(
                leader_returns, follower_returns, lag
            )

            if np.isnan(rolling_corr) or abs(rolling_corr) < self.config.min_correlation:
                continue

            # Compute ATR for the leader to gauge move significance
            leader_atr = self._compute_atr(
                np.array(self._highs[leader], dtype=np.float64),
                np.array(self._lows[leader], dtype=np.float64),
                np.array(self._closes[leader], dtype=np.float64),
                self.config.atr_window,
            )

            if leader_atr is None or leader_atr <= 0.0:
                continue

            # Check if leader made a significant move over the lag window
            leader_moved, move_direction, move_magnitude = self._detect_leader_move(
                leader_returns, leader_atr, self.config.signal_threshold, lag, leader
            )

            if not leader_moved:
                continue

            # Determine signal direction based on correlation sign
            # If rolling_corr > 0 and expected_sign > 0: continuation
            #   => follower should move in same direction as leader
            # If rolling_corr < 0 or expected_sign < 0: mean-reversion
            #   => follower should move in opposite direction
            corr_sign = np.sign(rolling_corr)
            effective_sign = expected_sign * corr_sign

            if effective_sign > 0:
                # Positive effective correlation: follower follows leader
                signal_direction = move_direction
            else:
                # Negative effective correlation: follower opposes leader
                signal_direction = -move_direction

            # Confidence: scaled by correlation strength and move magnitude
            confidence = float(np.clip(
                abs(rolling_corr) * (move_magnitude / leader_atr) * 2.0,
                0.0,
                1.0,
            ))

            # Estimate expected follower move for exit logic
            follower_atr = self._compute_atr(
                np.array(self._highs[follower], dtype=np.float64),
                np.array(self._lows[follower], dtype=np.float64),
                np.array(self._closes[follower], dtype=np.float64),
                self.config.atr_window,
            )
            expected_move = (follower_atr or leader_atr) * abs(rolling_corr) * 10.0

            # Record the active trade
            follower_close = self._closes[follower][-1]
            self._active_trades[pair_key] = _ActiveTrade(
                leader=leader,
                follower=follower,
                lag=lag,
                direction=signal_direction,
                entry_bar=self._bar_index,
                expected_move=expected_move,
                leader_move_at_entry=move_magnitude * move_direction,
                follower_price_at_entry=follower_close,
                bars_held=0,
            )

            signals.append({
                "symbol": follower,
                "direction": signal_direction,
                "confidence": confidence,
                "weight": self.config.per_pair_weight,
                "metadata": {
                    "leader": leader,
                    "lag": lag,
                    "corr": float(rolling_corr),
                    "action": "entry",
                    "leader_move": float(move_magnitude * move_direction),
                    "leader_atr": float(leader_atr),
                    "expected_sign": expected_sign,
                },
            })

        return signals

    def reset(self) -> None:
        """Reset all internal state for a new session or backtest run."""
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()
        self._volumes.clear()
        self._bars_seen.clear()
        self._bar_index = 0
        self._active_trades.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_returns(self, symbol: str) -> Optional[NDArray[np.float64]]:
        """Compute simple bar-over-bar returns for a symbol.

        Returns None if insufficient history is available.
        """
        closes = self._closes.get(symbol)
        if closes is None or len(closes) < 2:
            return None

        arr = np.array(closes, dtype=np.float64)
        returns = np.diff(arr) / np.maximum(arr[:-1], 1e-10)
        return returns

    def _compute_rolling_correlation(
        self,
        leader_returns: NDArray[np.float64],
        follower_returns: NDArray[np.float64],
        lag: int,
    ) -> float:
        """Compute the rolling cross-correlation at a specific lag.

        Cross-correlation at lag *k* measures the correlation between
        leader_returns[t-k] and follower_returns[t].  We use a rolling
        window of ``self.config.lookback`` bars.

        Args:
            leader_returns:   1-D array of leader simple returns.
            follower_returns: 1-D array of follower simple returns.
            lag:              Number of bars by which the follower lags
                              the leader.

        Returns:
            The Pearson correlation coefficient at the specified lag,
            computed over the most recent ``lookback`` bars.  Returns
            NaN if insufficient data.
        """
        lookback = self.config.lookback
        n_leader = len(leader_returns)
        n_follower = len(follower_returns)

        # We need at least lookback + lag returns in both series
        if n_leader < lookback + lag or n_follower < lookback:
            return np.nan

        # Align: leader_returns[t - lag] vs follower_returns[t]
        # Take the most recent `lookback` aligned observations
        leader_segment = leader_returns[-(lookback + lag): -lag] if lag > 0 else leader_returns[-lookback:]
        follower_segment = follower_returns[-lookback:]

        # Ensure equal length
        min_len = min(len(leader_segment), len(follower_segment))
        if min_len < 20:  # Need minimum observations for meaningful correlation
            return np.nan

        leader_segment = leader_segment[-min_len:]
        follower_segment = follower_segment[-min_len:]

        # Compute Pearson correlation
        leader_mean = np.mean(leader_segment)
        follower_mean = np.mean(follower_segment)

        leader_centered = leader_segment - leader_mean
        follower_centered = follower_segment - follower_mean

        cov = np.mean(leader_centered * follower_centered)
        std_leader = np.std(leader_segment)
        std_follower = np.std(follower_segment)

        denom = std_leader * std_follower
        if denom < 1e-10:
            return 0.0

        return float(cov / denom)

    def _compute_atr(
        self,
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        window: int,
    ) -> Optional[float]:
        """Compute the Average True Range over the most recent ``window`` bars.

        True Range for bar *t* is defined as::

            TR(t) = max(
                high(t) - low(t),
                |high(t) - close(t-1)|,
                |low(t) - close(t-1)|,
            )

        ATR is the simple moving average of TR over the window.

        Args:
            highs:  1-D array of high prices.
            lows:   1-D array of low prices.
            closes: 1-D array of close prices.
            window: Lookback window for the moving average.

        Returns:
            The current ATR value, or None if insufficient data.
        """
        n = len(closes)
        if n < window + 1:
            return None

        # Compute True Range for the most recent `window` bars
        h = highs[-(window + 1):]
        l = lows[-(window + 1):]
        c = closes[-(window + 1):]

        # True Range components (starting from index 1 to have previous close)
        hl = h[1:] - l[1:]
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(l[1:] - c[:-1])

        tr = np.maximum(hl, np.maximum(hc, lc))
        atr = float(np.mean(tr))

        return atr if atr > 0.0 else None

    def _detect_leader_move(
        self,
        leader_returns: NDArray[np.float64],
        atr: float,
        threshold: float,
        lag: int,
        leader_symbol: str = "",
    ) -> Tuple[bool, int, float]:
        """Detect whether the leader has made a significant recent move.

        Examines the cumulative return over the last ``lag`` bars (minimum 1)
        and compares to ``threshold * atr`` relative to the last close.

        Args:
            leader_returns: 1-D array of leader simple returns.
            atr:            Current leader ATR (price-level).
            threshold:      ATR multiplier threshold for significance.
            lag:            Lag window; the cumulative move is measured over
                            ``max(lag, 1)`` recent bars.
            leader_symbol:  Symbol name for looking up the latest close price.

        Returns:
            Tuple of (moved: bool, direction: int, magnitude: float).
            ``direction`` is +1 for up, -1 for down.
            ``magnitude`` is the unsigned absolute move in price terms.
        """
        window = max(lag, 1)

        if len(leader_returns) < window:
            return False, 0, 0.0

        # Cumulative return over the lag window
        recent_returns = leader_returns[-window:]
        cumulative_return = float(np.sum(recent_returns))

        # Convert return to price-level move using last close
        # (the returns are fractional, ATR is in price terms, so we need
        # to compare in consistent units)
        leader_closes = self._closes.get(leader_symbol)

        # Normalize: compare cumulative return * price vs ATR * threshold
        if leader_closes and len(leader_closes) > 0:
            recent_price = leader_closes[-1]
        else:
            recent_price = 1.0

        move_in_price = abs(cumulative_return) * recent_price
        threshold_price = threshold * atr

        moved = move_in_price > threshold_price
        direction = 1 if cumulative_return > 0 else -1
        magnitude = move_in_price

        return moved, direction, magnitude

    def _check_exit(self, trade: _ActiveTrade) -> Tuple[bool, str]:
        """Check whether an active trade should be exited.

        Exit conditions (any one triggers):
            1. **Convergence**: follower has moved enough in the expected
               direction (convergence_fraction of expected_move).
            2. **Timeout**: bars held >= max_hold.
            3. **Leader reversal**: leader has reversed significantly
               against the original move direction.

        Args:
            trade: The active trade to evaluate.

        Returns:
            Tuple of (should_exit: bool, reason: str).
        """
        # 1. Timeout
        if trade.bars_held >= self.config.max_hold:
            return True, "timeout"

        follower_closes = self._closes.get(trade.follower)
        if not follower_closes or len(follower_closes) < 2:
            return False, ""

        current_price = follower_closes[-1]

        # 2. Convergence: follower moved in the expected direction
        price_change = (current_price - trade.follower_price_at_entry) * trade.direction
        if trade.expected_move > 0 and price_change >= (
            trade.expected_move * self.config.convergence_fraction
        ):
            return True, "convergence"

        # 3. Leader reversal
        leader_returns = self._get_returns(trade.leader)
        if leader_returns is not None and len(leader_returns) >= 1:
            leader_atr = self._compute_atr(
                np.array(self._highs[trade.leader], dtype=np.float64),
                np.array(self._lows[trade.leader], dtype=np.float64),
                np.array(self._closes[trade.leader], dtype=np.float64),
                min(self.config.atr_window, len(self._closes[trade.leader]) - 1),
            )
            if leader_atr is not None and leader_atr > 0:
                # Check recent leader move against original direction
                recent_window = min(trade.lag, len(leader_returns))
                if recent_window > 0:
                    recent_cum = float(np.sum(leader_returns[-recent_window:]))
                    recent_price = self._closes[trade.leader][-1]
                    reversal_move = abs(recent_cum) * recent_price
                    original_direction = np.sign(trade.leader_move_at_entry)

                    if (np.sign(recent_cum) != original_direction
                            and reversal_move > self.config.leader_reversal_threshold * leader_atr):
                        return True, "leader_reversal"

        return False, ""

    @property
    def active_trade_count(self) -> int:
        """Number of currently active lead-lag positions."""
        return len(self._active_trades)

    def get_active_trades(self) -> Dict[Tuple[str, str], _ActiveTrade]:
        """Return a copy of active trades for inspection."""
        return dict(self._active_trades)
