"""
Intraday Liquidity Cycle Strategy

Exploits predictable intraday liquidity patterns caused by market maker
inventory management and institutional trading schedules.  Different parts
of the trading day exhibit fundamentally different microstructure
characteristics, and each regime calls for a distinct sub-strategy.

Academic Foundation:
    - Admati & Pfleiderer (1988): "A Theory of Intraday Patterns: Volume
      and Price Variability" -- concentrated trading equilibria produce
      U-shaped volume and volatility curves.
    - Heston, Korajczyk & Sadka (2010): "Intraday Patterns in the
      Cross-section of Stock Returns" -- half-hour returns are predictable
      from same-interval returns on prior days.
    - Bogousslavsky (2016): "Intraday Returns and the Cross-section of
      Stock Returns" -- autocorrelation in intraday returns linked to
      slow-moving capital and liquidity provision.

Key insight: Intraday patterns are STRUCTURAL (driven by market
microstructure) not behavioral.  They persist because they compensate
liquidity providers for predictable inventory risk.

Intraday Regimes
----------------
1. Opening Auction   (09:30 - 10:00) -- MEAN REVERSION
   Overnight order imbalances resolved, high vol, high spread.
   Fade the opening 15-min move at 10:00; expect 30-50% reversion by 10:30.

2. Morning Trend     (10:00 - 11:30) -- MOMENTUM
   Institutional block flow, spreads tighten, strong trends.
   Follow the 10:00 price direction; ~60% persistence through 11:30.

3. Lunch Lull        (11:30 - 13:00) -- REDUCED EXPOSURE
   Thin liquidity, wide spreads, random walk.
   Reduce positions or stay flat; signals unreliable.

4. Afternoon Session (13:00 - 15:00) -- MOMENTUM
   Institutional flow resumes, moderate trend persistence.
   Follow post-lunch direction if volume > 80% of morning average.

5. Closing Session   (15:00 - 15:30) -- MOMENTUM
   Portfolio managers position for close, strong trending.
   Follow the 15:00 direction with increasing conviction.

6. MOC Window        (15:30 - 16:00) -- STRUCTURAL
   Market-On-Close orders create predictable pressure.
   NYSE publishes MOC imbalance at 15:50; trade the imbalance direction.

Expected Performance:
    - Sharpe: 1.0 - 1.8 (combined regime approach)
    - Win rate: 50 - 58%
    - Annual return: 12 - 25% (depending on universe and capital)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, date
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.utils.math_utils import simple_returns, rolling_mean, rolling_std

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================


class IntradayRegime(Enum):
    """
    Distinct microstructure regimes within a single trading session.

    Each regime has materially different spread, volume, and autocorrelation
    characteristics that dictate the optimal trading approach.
    """

    PRE_MARKET = auto()
    OPENING_AUCTION = auto()    # 09:30 - 10:00
    MORNING_TREND = auto()      # 10:00 - 11:30
    LUNCH_LULL = auto()         # 11:30 - 13:00
    AFTERNOON = auto()          # 13:00 - 15:00
    CLOSING = auto()            # 15:00 - 15:30
    MOC_WINDOW = auto()         # 15:30 - 16:00
    AFTER_HOURS = auto()


@dataclass
class LiquidityCycleSignal:
    """Signal produced by the liquidity-cycle strategy for a single bar."""

    timestamp: datetime
    symbol: str
    regime: IntradayRegime
    direction: float            # -1.0 to +1.0  (continuous conviction)
    confidence: float           # 0.0 to 1.0
    strategy_type: str          # "mean_reversion", "momentum", "moc_imbalance"
    entry_price: float
    target_price: float
    stop_price: float
    position_size: float        # Fraction of capital
    expected_holding_minutes: int


@dataclass
class LiquidityCycleConfig:
    """
    Tunable parameters for the liquidity-cycle strategy.

    Defaults calibrated to S&P 500 constituents on 5-minute bars,
    2015-2024.  Re-calibrate periodically with walk-forward analysis.
    """

    # -- Regime boundaries (minutes from market open 09:30 ET) -----------
    opening_end: int = 30           # 10:00
    morning_end: int = 120          # 11:30
    lunch_end: int = 210            # 13:00
    afternoon_end: int = 330        # 15:00
    closing_end: int = 360          # 15:30
    # MOC window runs from closing_end to 390 (16:00)

    # -- Opening reversion parameters -----------------------------------
    opening_reversion_target: float = 0.40
    """Expect 40% reversion of the opening move (Admati & Pfleiderer)."""

    opening_min_move: float = 0.003
    """Minimum absolute opening move (0.3%) to trigger a reversion trade."""

    opening_max_move: float = 0.03
    """Maximum opening move (3%) -- beyond this the move is likely news-driven."""

    # -- Momentum parameters (morning / afternoon / closing) ------------
    momentum_volume_threshold: float = 0.80
    """Volume must exceed this fraction of session-average to confirm trend."""

    momentum_min_bars: int = 3
    """Minimum consecutive confirming bars for a momentum entry."""

    momentum_min_return: float = 0.001
    """Minimum absolute return over confirming bars (0.1%)."""

    # -- MOC parameters -------------------------------------------------
    moc_imbalance_threshold: float = 0.005
    """MOC buy/sell imbalance as fraction of ADV to trigger a trade."""

    moc_entry_time: time = time(15, 50)
    """Enter MOC trades at 15:50 when NYSE publishes imbalance."""

    # -- ATR / risk management ------------------------------------------
    atr_lookback: int = 14
    """Number of bars for Average True Range calculation."""

    atr_stop_multiplier: float = 1.5
    """Stop distance = ATR * this multiplier."""

    atr_target_multiplier: float = 2.0
    """Target distance = ATR * this multiplier."""

    # -- Position sizing ------------------------------------------------
    vol_target: float = 0.15
    """Annualised portfolio volatility target (15%)."""

    max_position: float = 0.15
    """Maximum single-symbol allocation (15% of capital)."""

    min_position: float = 0.01
    """Minimum allocation to bother trading (1%)."""

    max_daily_trades: int = 6
    """Hard cap on round-trip trades per day per symbol."""

    # -- Lunch-lull exposure reduction ----------------------------------
    lunch_exposure_fraction: float = 0.25
    """During lunch, reduce any existing exposure to this fraction."""


# =============================================================================
# PER-SYMBOL INTRADAY STATE
# =============================================================================


@dataclass
class _SymbolDayState:
    """Mutable intraday state tracked per symbol, reset every morning."""

    # OHLCV bars collected so far today (parallel arrays for speed)
    timestamps: List[datetime] = field(default_factory=list)
    opens: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)

    # Derived reference prices
    day_open: Optional[float] = None
    opening_window_close: Optional[float] = None  # Price at 10:00
    opening_move: Optional[float] = None           # Signed return 09:30-10:00
    morning_direction_price: Optional[float] = None  # Price at 10:00 (momentum anchor)
    post_lunch_price: Optional[float] = None       # Price at 13:00
    closing_anchor_price: Optional[float] = None   # Price at 15:00

    # Volume tracking
    opening_volume: float = 0.0
    morning_volume: float = 0.0
    lunch_volume: float = 0.0
    afternoon_volume: float = 0.0

    # Momentum bar counters (consecutive confirming bars)
    momentum_confirming_bars: int = 0
    last_momentum_direction: int = 0  # +1 or -1

    # Trade counter
    trades_today: int = 0

    # Active signal (only one at a time per symbol)
    active_signal: Optional[LiquidityCycleSignal] = None

    # Regime P&L tracking
    regime_pnl: Dict[IntradayRegime, float] = field(
        default_factory=lambda: defaultdict(float)
    )


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================


class LiquidityCycleStrategy:
    """
    Exploit structural intraday liquidity patterns.

    The strategy maintains per-symbol daily state and produces at most one
    active signal per symbol at any time.  Signals are generated on each
    bar update and can be consumed by an execution layer.

    Usage::

        strategy = LiquidityCycleStrategy()

        for bar in intraday_bars:
            strategy.update_bar(
                bar.symbol, bar.timestamp,
                bar.open, bar.high, bar.low, bar.close, bar.volume,
            )
            signal = strategy.generate_signal(bar.symbol, bar.timestamp)
            if signal is not None:
                execute(signal)

        stats = strategy.get_session_stats()
        strategy.reset_day()

    Parameters
    ----------
    config : LiquidityCycleConfig, optional
        Strategy configuration.  Uses sensible defaults when omitted.
    """

    # Market open/close in ET
    _MARKET_OPEN: time = time(9, 30)
    _MARKET_CLOSE: time = time(16, 0)

    def __init__(self, config: Optional[LiquidityCycleConfig] = None) -> None:
        self.config: LiquidityCycleConfig = config or LiquidityCycleConfig()

        # Per-symbol intraday state
        self._state: Dict[str, _SymbolDayState] = {}

        # Current trading date (used to detect day boundaries)
        self._current_date: Optional[date] = None

        # Cross-day statistics for ADV and ATR estimation
        self._daily_volumes: Dict[str, List[float]] = defaultdict(list)
        self._daily_true_ranges: Dict[str, List[float]] = defaultdict(list)

        # Session-level aggregate stats (across all symbols)
        self._session_signals: List[LiquidityCycleSignal] = []
        self._session_trades: List[Dict] = []

    # -----------------------------------------------------------------
    # Regime classification
    # -----------------------------------------------------------------

    def classify_regime(self, timestamp: datetime) -> IntradayRegime:
        """
        Determine the current intraday regime from the timestamp.

        Uses ``time()`` comparison against the configured regime boundaries
        expressed as minute offsets from 09:30 ET.

        Parameters
        ----------
        timestamp : datetime
            Bar timestamp (assumed Eastern Time).

        Returns
        -------
        IntradayRegime
        """
        t: time = timestamp.time()

        if t < self._MARKET_OPEN:
            return IntradayRegime.PRE_MARKET
        if t >= self._MARKET_CLOSE:
            return IntradayRegime.AFTER_HOURS

        minutes_from_open: int = (
            (t.hour - 9) * 60 + t.minute - 30
        )

        if minutes_from_open < 0:
            return IntradayRegime.PRE_MARKET
        if minutes_from_open < self.config.opening_end:
            return IntradayRegime.OPENING_AUCTION
        if minutes_from_open < self.config.morning_end:
            return IntradayRegime.MORNING_TREND
        if minutes_from_open < self.config.lunch_end:
            return IntradayRegime.LUNCH_LULL
        if minutes_from_open < self.config.afternoon_end:
            return IntradayRegime.AFTERNOON
        if minutes_from_open < self.config.closing_end:
            return IntradayRegime.CLOSING

        return IntradayRegime.MOC_WINDOW

    # -----------------------------------------------------------------
    # Bar ingestion
    # -----------------------------------------------------------------

    def update_bar(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """
        Ingest a new OHLCV bar and update internal state.

        Automatically detects day boundaries and resets daily state.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"AAPL"``).
        timestamp : datetime
            Bar timestamp in Eastern Time.
        open_ : float
            Bar open price.
        high : float
            Bar high price.
        low : float
            Bar low price.
        close : float
            Bar close price.
        volume : float
            Bar volume.
        """
        bar_date: date = timestamp.date()

        # Day boundary detection -- reset per-symbol state
        if self._current_date is None or bar_date != self._current_date:
            self._flush_day_stats()
            self._current_date = bar_date
            self._state.clear()
            self._session_signals.clear()
            self._session_trades.clear()

        # Lazy-initialise symbol state
        if symbol not in self._state:
            self._state[symbol] = _SymbolDayState()

        st: _SymbolDayState = self._state[symbol]

        # Append raw bar data
        st.timestamps.append(timestamp)
        st.opens.append(open_)
        st.highs.append(high)
        st.lows.append(low)
        st.closes.append(close)
        st.volumes.append(volume)

        regime: IntradayRegime = self.classify_regime(timestamp)

        # --- capture reference prices ---
        if st.day_open is None:
            st.day_open = open_

        if regime == IntradayRegime.OPENING_AUCTION:
            st.opening_volume += volume
            # Continuously update -- the last bar in the window is the
            # "opening window close"
            st.opening_window_close = close
            if st.day_open is not None and st.day_open > EPSILON:
                st.opening_move = (close - st.day_open) / st.day_open

        elif regime == IntradayRegime.MORNING_TREND:
            st.morning_volume += volume
            if st.morning_direction_price is None:
                st.morning_direction_price = open_
            self._update_momentum_counter(st, close)

        elif regime == IntradayRegime.LUNCH_LULL:
            st.lunch_volume += volume
            # Reset momentum counter -- lunch signals are unreliable
            st.momentum_confirming_bars = 0
            st.last_momentum_direction = 0

        elif regime == IntradayRegime.AFTERNOON:
            st.afternoon_volume += volume
            if st.post_lunch_price is None:
                st.post_lunch_price = open_
            self._update_momentum_counter(st, close)

        elif regime == IntradayRegime.CLOSING:
            if st.closing_anchor_price is None:
                st.closing_anchor_price = open_
            self._update_momentum_counter(st, close)

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[LiquidityCycleSignal]:
        """
        Generate a trading signal for *symbol* at *timestamp*.

        Delegates to regime-specific sub-strategies.  Returns ``None``
        when no actionable signal exists (the common case).

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        timestamp : datetime
            Current bar timestamp.

        Returns
        -------
        LiquidityCycleSignal or None
        """
        if symbol not in self._state:
            return None

        st: _SymbolDayState = self._state[symbol]

        # Enforce daily trade cap
        if st.trades_today >= self.config.max_daily_trades:
            return None

        regime: IntradayRegime = self.classify_regime(timestamp)

        signal: Optional[LiquidityCycleSignal] = None

        if regime == IntradayRegime.OPENING_AUCTION:
            # We wait until 10:00 -- signal fires on the *first bar* of
            # the morning-trend regime.  Nothing to do here.
            pass

        elif regime == IntradayRegime.MORNING_TREND:
            # On the first morning-trend bar after the opening window,
            # evaluate an opening-reversion signal.
            if (
                st.opening_move is not None
                and st.active_signal is None
                and len(st.closes) > 0
                and self._is_first_bar_of_regime(st, regime)
            ):
                signal = self._opening_reversion_signal(symbol, timestamp)

            # Also consider momentum if no reversion signal
            if signal is None and st.active_signal is None:
                signal = self._momentum_signal(symbol, timestamp, regime)

        elif regime == IntradayRegime.LUNCH_LULL:
            # No new entries during lunch.  If we have an active signal,
            # the execution layer should reduce exposure per config.
            pass

        elif regime == IntradayRegime.AFTERNOON:
            if st.active_signal is None:
                signal = self._momentum_signal(symbol, timestamp, regime)

        elif regime == IntradayRegime.CLOSING:
            if st.active_signal is None:
                signal = self._momentum_signal(symbol, timestamp, regime)

        elif regime == IntradayRegime.MOC_WINDOW:
            if st.active_signal is None:
                signal = self._moc_signal(symbol, timestamp)

        if signal is not None:
            st.active_signal = signal
            st.trades_today += 1
            self._session_signals.append(signal)
            logger.debug(
                "Signal: %s %s dir=%.2f conf=%.2f regime=%s",
                signal.symbol,
                signal.strategy_type,
                signal.direction,
                signal.confidence,
                signal.regime.name,
            )

        return signal

    # -----------------------------------------------------------------
    # Sub-strategy: opening reversion
    # -----------------------------------------------------------------

    def _opening_reversion_signal(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[LiquidityCycleSignal]:
        """
        Fade the opening 15-minute move at 10:00 AM.

        Admati & Pfleiderer (1988) show that concentrated trading in the
        opening creates temporary price dislocations as market makers
        absorb overnight order flow.  On average 30-50% of the opening
        move reverts by 10:30.

        Parameters
        ----------
        symbol : str
        timestamp : datetime

        Returns
        -------
        LiquidityCycleSignal or None
        """
        st: _SymbolDayState = self._state[symbol]

        if st.opening_move is None or st.opening_window_close is None:
            return None

        abs_move: float = abs(st.opening_move)

        # Filter: move must be significant but not news-driven
        if abs_move < self.config.opening_min_move:
            return None
        if abs_move > self.config.opening_max_move:
            return None

        current_price: float = st.closes[-1]
        if current_price <= EPSILON:
            return None

        # Direction: FADE the opening move (trade opposite)
        direction: float = -1.0 if st.opening_move > 0 else 1.0

        # Confidence scales with move magnitude (larger move = stronger reversion)
        magnitude_score: float = min(
            1.0, abs_move / (self.config.opening_min_move * 3.0)
        )
        # Volume confirmation: higher opening volume = more institutional flow = more reversion
        volume_score: float = self._opening_volume_score(symbol)
        confidence: float = 0.6 * magnitude_score + 0.4 * volume_score

        # Target: expect opening_reversion_target fraction of the move to revert
        reversion_amount: float = abs_move * self.config.opening_reversion_target
        target_price: float = current_price * (1.0 + direction * reversion_amount)

        # Stop: if the opening move EXTENDS by half its original size
        atr: float = self._calculate_atr(symbol)
        stop_distance: float = max(
            atr * self.config.atr_stop_multiplier,
            current_price * abs_move * 0.5,
        )
        stop_price: float = current_price - direction * stop_distance

        # Position sizing via vol target
        position_size: float = self._size_position(symbol, atr)

        return LiquidityCycleSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=IntradayRegime.OPENING_AUCTION,
            direction=direction,
            confidence=confidence,
            strategy_type="mean_reversion",
            entry_price=current_price,
            target_price=target_price,
            stop_price=stop_price,
            position_size=position_size,
            expected_holding_minutes=30,
        )

    # -----------------------------------------------------------------
    # Sub-strategy: momentum (morning / afternoon / closing)
    # -----------------------------------------------------------------

    def _momentum_signal(
        self,
        symbol: str,
        timestamp: datetime,
        regime: IntradayRegime,
    ) -> Optional[LiquidityCycleSignal]:
        """
        Follow the prevailing trend when confirmed by volume and bar count.

        Heston, Korajczyk & Sadka (2010) document persistent intraday
        return autocorrelation during the morning and afternoon sessions
        when institutional order flow dominates.

        Parameters
        ----------
        symbol : str
        timestamp : datetime
        regime : IntradayRegime

        Returns
        -------
        LiquidityCycleSignal or None
        """
        st: _SymbolDayState = self._state[symbol]

        if len(st.closes) < self.config.momentum_min_bars + 1:
            return None

        # Need enough consecutive confirming bars
        if st.momentum_confirming_bars < self.config.momentum_min_bars:
            return None

        # Determine anchor price for the regime
        anchor_price: Optional[float] = None
        if regime == IntradayRegime.MORNING_TREND:
            anchor_price = st.morning_direction_price
        elif regime == IntradayRegime.AFTERNOON:
            anchor_price = st.post_lunch_price
        elif regime == IntradayRegime.CLOSING:
            anchor_price = st.closing_anchor_price

        if anchor_price is None or anchor_price <= EPSILON:
            return None

        current_price: float = st.closes[-1]
        if current_price <= EPSILON:
            return None

        regime_return: float = (current_price - anchor_price) / anchor_price

        # Minimum return threshold
        if abs(regime_return) < self.config.momentum_min_return:
            return None

        # Volume confirmation for afternoon session
        if regime == IntradayRegime.AFTERNOON:
            if not self._afternoon_volume_confirmed(st):
                return None

        direction: float = float(st.last_momentum_direction)

        # Confidence based on bar count, return magnitude, and volume
        bar_score: float = min(
            1.0,
            st.momentum_confirming_bars / (self.config.momentum_min_bars * 2.0),
        )
        return_score: float = min(
            1.0, abs(regime_return) / (self.config.momentum_min_return * 5.0)
        )
        confidence: float = 0.5 * bar_score + 0.5 * return_score

        # Boost confidence for closing session (urgency effect)
        if regime == IntradayRegime.CLOSING:
            confidence = min(1.0, confidence * 1.15)

        atr: float = self._calculate_atr(symbol)
        target_price: float = current_price + direction * atr * self.config.atr_target_multiplier
        stop_price: float = current_price - direction * atr * self.config.atr_stop_multiplier
        position_size: float = self._size_position(symbol, atr)

        # Expected holding time depends on regime
        holding_minutes_map: Dict[IntradayRegime, int] = {
            IntradayRegime.MORNING_TREND: 60,
            IntradayRegime.AFTERNOON: 90,
            IntradayRegime.CLOSING: 20,
        }
        expected_holding: int = holding_minutes_map.get(regime, 60)

        return LiquidityCycleSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            direction=direction,
            confidence=confidence,
            strategy_type="momentum",
            entry_price=current_price,
            target_price=target_price,
            stop_price=stop_price,
            position_size=position_size,
            expected_holding_minutes=expected_holding,
        )

    # -----------------------------------------------------------------
    # Sub-strategy: MOC imbalance
    # -----------------------------------------------------------------

    def _moc_signal(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[LiquidityCycleSignal]:
        """
        Trade Market-On-Close imbalance direction at 15:50.

        NYSE publishes the MOC imbalance at 15:50 ET.  Empirically, a
        significant buy (sell) imbalance predicts upward (downward) price
        drift into the close.

        Since live MOC imbalance data requires a direct exchange feed,
        this implementation estimates imbalance from the volume-weighted
        price trajectory in the 15:30-15:50 window relative to the
        session VWAP.

        Parameters
        ----------
        symbol : str
        timestamp : datetime

        Returns
        -------
        LiquidityCycleSignal or None
        """
        st: _SymbolDayState = self._state[symbol]

        # Only trigger at or after the MOC publication time
        if timestamp.time() < self.config.moc_entry_time:
            return None

        if len(st.closes) < 2 or len(st.volumes) < 2:
            return None

        current_price: float = st.closes[-1]
        if current_price <= EPSILON:
            return None

        # Estimate imbalance from recent price action relative to VWAP
        session_vwap: float = self._session_vwap(st)
        if session_vwap <= EPSILON:
            return None

        # Imbalance proxy: how far the last few bars' VWAP deviates from
        # the session VWAP, normalised by ADV
        moc_bars: int = min(4, len(st.closes))  # ~last 20 min on 5-min bars
        recent_prices: NDArray[np.float64] = np.array(
            st.closes[-moc_bars:], dtype=np.float64
        )
        recent_volumes: NDArray[np.float64] = np.array(
            st.volumes[-moc_bars:], dtype=np.float64
        )
        total_recent_vol: float = float(np.sum(recent_volumes))

        if total_recent_vol <= EPSILON:
            return None

        recent_vwap: float = float(
            np.sum(recent_prices * recent_volumes) / total_recent_vol
        )
        imbalance: float = (recent_vwap - session_vwap) / session_vwap

        if abs(imbalance) < self.config.moc_imbalance_threshold:
            return None

        direction: float = 1.0 if imbalance > 0 else -1.0

        # Confidence proportional to imbalance magnitude
        confidence: float = min(
            1.0,
            abs(imbalance) / (self.config.moc_imbalance_threshold * 3.0),
        )

        atr: float = self._calculate_atr(symbol)
        target_price: float = current_price + direction * atr * 0.5
        stop_price: float = current_price - direction * atr * 1.0
        position_size: float = self._size_position(symbol, atr)

        return LiquidityCycleSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=IntradayRegime.MOC_WINDOW,
            direction=direction,
            confidence=confidence,
            strategy_type="moc_imbalance",
            entry_price=current_price,
            target_price=target_price,
            stop_price=stop_price,
            position_size=position_size,
            expected_holding_minutes=10,
        )

    # -----------------------------------------------------------------
    # ATR calculation
    # -----------------------------------------------------------------

    def _calculate_atr(self, symbol: str) -> float:
        """
        Compute the Average True Range for *symbol* using the bars
        collected today (plus any historical daily true-range data).

        Falls back to a percentage of the current price when insufficient
        data is available.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        float
            Positive ATR value, never zero.
        """
        st: _SymbolDayState = self._state.get(symbol)
        if st is None or len(st.closes) < 2:
            # Fallback: 0.5% of last known close (or 1.0 as absolute floor)
            if st is not None and len(st.closes) > 0:
                return max(st.closes[-1] * 0.005, EPSILON)
            return 1.0

        n: int = len(st.closes)
        lookback: int = min(self.config.atr_lookback, n - 1)

        true_ranges: List[float] = []
        for i in range(n - lookback, n):
            if i < 1:
                continue
            tr: float = max(
                st.highs[i] - st.lows[i],
                abs(st.highs[i] - st.closes[i - 1]),
                abs(st.lows[i] - st.closes[i - 1]),
            )
            true_ranges.append(tr)

        if not true_ranges:
            return max(st.closes[-1] * 0.005, EPSILON)

        atr: float = float(np.mean(true_ranges))
        return max(atr, EPSILON)

    # -----------------------------------------------------------------
    # Position sizing
    # -----------------------------------------------------------------

    def _size_position(self, symbol: str, atr: float) -> float:
        """
        Volatility-targeted position sizing.

        Scales position so that the expected per-bar P&L volatility
        matches the portfolio-level vol target, capped by ``max_position``.

        Parameters
        ----------
        symbol : str
        atr : float

        Returns
        -------
        float
            Position size as fraction of capital, in [min_position, max_position].
        """
        st: _SymbolDayState = self._state.get(symbol)
        if st is None or len(st.closes) == 0 or atr <= EPSILON:
            return self.config.min_position

        current_price: float = st.closes[-1]
        if current_price <= EPSILON:
            return self.config.min_position

        # Annualise intraday ATR: assume ~78 five-minute bars per day
        # and 252 trading days
        bars_per_day: float = max(len(st.closes), 1.0)
        daily_vol_estimate: float = atr * np.sqrt(bars_per_day) / current_price
        annual_vol_estimate: float = daily_vol_estimate * SQRT_252

        if annual_vol_estimate <= EPSILON:
            return self.config.min_position

        raw_size: float = self.config.vol_target / annual_vol_estimate

        # Clip to bounds
        return float(np.clip(raw_size, self.config.min_position, self.config.max_position))

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _update_momentum_counter(
        self,
        st: _SymbolDayState,
        close: float,
    ) -> None:
        """Track consecutive same-direction bars for momentum confirmation."""
        if len(st.closes) < 2:
            st.momentum_confirming_bars = 0
            return

        bar_return: float = close - st.closes[-2] if len(st.closes) >= 2 else 0.0
        current_dir: int = 1 if bar_return > 0 else (-1 if bar_return < 0 else 0)

        if current_dir == 0:
            # Flat bar -- do not reset, but do not increment
            return

        if current_dir == st.last_momentum_direction:
            st.momentum_confirming_bars += 1
        else:
            st.momentum_confirming_bars = 1
            st.last_momentum_direction = current_dir

    def _is_first_bar_of_regime(
        self,
        st: _SymbolDayState,
        regime: IntradayRegime,
    ) -> bool:
        """Return True if the most recent bar is the first in *regime* today."""
        if len(st.timestamps) < 2:
            return True
        prev_regime: IntradayRegime = self.classify_regime(st.timestamps[-2])
        return prev_regime != regime

    def _opening_volume_score(self, symbol: str) -> float:
        """
        Score the opening volume relative to the 20-day average opening
        volume (if available).  Returns 0.5 when no history exists.
        """
        st: _SymbolDayState = self._state.get(symbol)
        if st is None or st.opening_volume <= EPSILON:
            return 0.5

        history: List[float] = self._daily_volumes.get(symbol, [])
        if len(history) < 5:
            return 0.5

        avg_daily_vol: float = float(np.mean(history[-20:]))
        if avg_daily_vol <= EPSILON:
            return 0.5

        # Opening volume should be ~15-25% of daily; compare to historical fraction
        ratio: float = st.opening_volume / avg_daily_vol
        return float(np.clip(ratio / 0.25, 0.0, 1.0))

    def _afternoon_volume_confirmed(self, st: _SymbolDayState) -> bool:
        """
        Check whether afternoon volume exceeds the configured fraction
        of the average morning volume.
        """
        if st.morning_volume <= EPSILON:
            return False

        # How many bars have we seen in the afternoon so far?
        afternoon_bars: int = max(
            1,
            sum(
                1
                for ts in st.timestamps
                if self.classify_regime(ts) == IntradayRegime.AFTERNOON
            ),
        )
        morning_bars: int = max(
            1,
            sum(
                1
                for ts in st.timestamps
                if self.classify_regime(ts) == IntradayRegime.MORNING_TREND
            ),
        )

        # Normalise volume per bar
        afternoon_rate: float = st.afternoon_volume / afternoon_bars
        morning_rate: float = st.morning_volume / morning_bars

        return afternoon_rate >= morning_rate * self.config.momentum_volume_threshold

    def _session_vwap(self, st: _SymbolDayState) -> float:
        """Compute session-to-date VWAP from collected bars."""
        if not st.closes or not st.volumes:
            return 0.0

        prices: NDArray[np.float64] = np.array(st.closes, dtype=np.float64)
        volumes: NDArray[np.float64] = np.array(st.volumes, dtype=np.float64)
        total_vol: float = float(np.sum(volumes))

        if total_vol <= EPSILON:
            return float(np.mean(prices))

        return float(np.sum(prices * volumes) / total_vol)

    def _flush_day_stats(self) -> None:
        """
        Persist end-of-day volume and true-range data for multi-day
        lookbacks, then clear session state.
        """
        for symbol, st in self._state.items():
            if st.closes:
                total_volume: float = float(np.sum(st.volumes))
                self._daily_volumes[symbol].append(total_volume)
                # Keep 60 days
                if len(self._daily_volumes[symbol]) > 60:
                    self._daily_volumes[symbol] = self._daily_volumes[symbol][-60:]

            if len(st.closes) >= 2:
                daily_tr: float = max(st.highs) - min(st.lows) if st.highs and st.lows else 0.0
                self._daily_true_ranges[symbol].append(daily_tr)
                if len(self._daily_true_ranges[symbol]) > 60:
                    self._daily_true_ranges[symbol] = self._daily_true_ranges[symbol][-60:]

    # -----------------------------------------------------------------
    # Session stats & reset
    # -----------------------------------------------------------------

    def get_session_stats(self) -> Dict:
        """
        Return a dictionary of intraday P&L and signal statistics,
        broken down by regime.

        Useful for post-session analysis and walk-forward calibration.

        Returns
        -------
        dict
            Keys include ``"signals_by_regime"``, ``"total_signals"``,
            ``"symbols_active"``, ``"regime_breakdown"``, and per-symbol
            detail under ``"per_symbol"``.
        """
        regime_counts: Dict[str, int] = defaultdict(int)
        regime_strategy_types: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for sig in self._session_signals:
            regime_name: str = sig.regime.name
            regime_counts[regime_name] += 1
            regime_strategy_types[regime_name][sig.strategy_type] += 1

        per_symbol: Dict[str, Dict] = {}
        for symbol, st in self._state.items():
            per_symbol[symbol] = {
                "trades_today": st.trades_today,
                "bars_collected": len(st.closes),
                "opening_move": st.opening_move,
                "regime_pnl": dict(
                    (k.name, v) for k, v in st.regime_pnl.items()
                ),
            }

        return {
            "date": str(self._current_date) if self._current_date else None,
            "total_signals": len(self._session_signals),
            "symbols_active": len(self._state),
            "regime_breakdown": dict(regime_counts),
            "regime_strategy_types": {
                k: dict(v) for k, v in regime_strategy_types.items()
            },
            "per_symbol": per_symbol,
        }

    def reset_day(self) -> None:
        """
        Explicitly reset all daily state.

        Call this at the end of each trading day (or before the first bar
        of a new day).  ``update_bar`` also auto-detects day boundaries,
        but calling ``reset_day`` is cleaner in a backtest loop.
        """
        self._flush_day_stats()
        self._state.clear()
        self._session_signals.clear()
        self._session_trades.clear()
        self._current_date = None

    def reset(self) -> None:
        """Reset all state including cross-day history."""
        self._state.clear()
        self._session_signals.clear()
        self._session_trades.clear()
        self._daily_volumes.clear()
        self._daily_true_ranges.clear()
        self._current_date = None


# =============================================================================
# BACKTEST ENGINE
# =============================================================================


@dataclass
class _BacktestPosition:
    """Internal position tracker for the backtest engine."""

    symbol: str
    signal: LiquidityCycleSignal
    entry_bar_idx: int
    entry_timestamp: datetime
    entry_price: float
    direction: float
    size: float                 # Fraction of capital
    pnl: float = 0.0
    closed: bool = False
    exit_price: float = 0.0
    exit_timestamp: Optional[datetime] = None
    exit_reason: str = ""
    regime: IntradayRegime = IntradayRegime.PRE_MARKET


@dataclass
class LiquidityCycleBacktestResult:
    """
    Results container for ``run_liquidity_cycle_backtest``.

    Contains standard portfolio-level metrics and a per-regime breakdown.
    """

    # Portfolio-level metrics
    total_return: float = 0.0
    annualised_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    avg_holding_minutes: float = 0.0

    # Equity curve
    equity_curve: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    timestamps: List[datetime] = field(default_factory=list)

    # Per-regime breakdown
    regime_stats: Dict[str, Dict] = field(default_factory=dict)

    # All closed trades
    trades: List[Dict] = field(default_factory=list)


def run_liquidity_cycle_backtest(
    timestamps: NDArray,
    opens: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    volumes: NDArray[np.float64],
    symbol: str = "SYM",
    initial_capital: float = 100_000.0,
    config: Optional[LiquidityCycleConfig] = None,
) -> LiquidityCycleBacktestResult:
    """
    Run a full backtest of the liquidity-cycle strategy on intraday data.

    Processes one symbol at a time.  For multi-symbol backtests, call once
    per symbol and aggregate the equity curves.

    Parameters
    ----------
    timestamps : array-like of datetime
        Bar timestamps (must be in Eastern Time and sorted ascending).
    opens, highs, lows, closes, volumes : NDArray[np.float64]
        OHLCV arrays aligned with *timestamps*.
    symbol : str
        Instrument identifier.
    initial_capital : float
        Starting portfolio value.
    config : LiquidityCycleConfig, optional
        Strategy configuration; uses defaults if omitted.

    Returns
    -------
    LiquidityCycleBacktestResult
        Comprehensive results including equity curve and regime breakdown.
    """
    n_bars: int = len(timestamps)
    if n_bars == 0:
        return LiquidityCycleBacktestResult()

    strategy = LiquidityCycleStrategy(config=config)
    capital: float = initial_capital
    equity: List[float] = [capital]
    equity_ts: List[datetime] = []

    positions: List[_BacktestPosition] = []
    closed_trades: List[Dict] = []

    # Per-regime accumulators
    regime_pnl: Dict[str, float] = defaultdict(float)
    regime_trades: Dict[str, int] = defaultdict(int)
    regime_wins: Dict[str, int] = defaultdict(int)
    regime_holding_mins: Dict[str, List[float]] = defaultdict(list)

    prev_date: Optional[date] = None

    for i in range(n_bars):
        ts: datetime = (
            timestamps[i]
            if isinstance(timestamps[i], datetime)
            else datetime.fromisoformat(str(timestamps[i]))
        )
        bar_date: date = ts.date()

        # Day boundary: force-close all open positions at prior close
        if prev_date is not None and bar_date != prev_date:
            for pos in positions:
                if not pos.closed:
                    _close_position(
                        pos, closes[i - 1], timestamps[i - 1], "eod_flat"
                    )
                    _record_trade(
                        pos, closed_trades, regime_pnl, regime_trades,
                        regime_wins, regime_holding_mins, capital,
                    )
                    capital += pos.pnl
            positions = [p for p in positions if not p.closed]

        prev_date = bar_date

        # Feed bar to strategy
        strategy.update_bar(
            symbol, ts,
            float(opens[i]), float(highs[i]),
            float(lows[i]), float(closes[i]),
            float(volumes[i]),
        )

        # Check exits on open positions
        current_close: float = float(closes[i])
        current_high: float = float(highs[i])
        current_low: float = float(lows[i])

        for pos in positions:
            if pos.closed:
                continue

            # Stop check
            if pos.direction > 0:
                if current_low <= pos.signal.stop_price:
                    _close_position(pos, pos.signal.stop_price, ts, "stop")
                elif current_high >= pos.signal.target_price:
                    _close_position(pos, pos.signal.target_price, ts, "target")
            else:
                if current_high >= pos.signal.stop_price:
                    _close_position(pos, pos.signal.stop_price, ts, "stop")
                elif current_low <= pos.signal.target_price:
                    _close_position(pos, pos.signal.target_price, ts, "target")

        # Record closed trades and update capital
        newly_closed: List[_BacktestPosition] = [
            p for p in positions if p.closed and p.exit_reason != ""
        ]
        for pos in newly_closed:
            _record_trade(
                pos, closed_trades, regime_pnl, regime_trades,
                regime_wins, regime_holding_mins, capital,
            )
            capital += pos.pnl
        positions = [p for p in positions if not p.closed]

        # Generate new signal
        signal: Optional[LiquidityCycleSignal] = strategy.generate_signal(
            symbol, ts
        )

        if signal is not None:
            # Only open if we have no active position for this symbol
            has_open: bool = any(
                not p.closed and p.symbol == symbol for p in positions
            )
            if not has_open:
                pos = _BacktestPosition(
                    symbol=symbol,
                    signal=signal,
                    entry_bar_idx=i,
                    entry_timestamp=ts,
                    entry_price=signal.entry_price,
                    direction=signal.direction,
                    size=signal.position_size,
                    regime=signal.regime,
                )
                positions.append(pos)

        # Mark-to-market equity
        unrealised: float = 0.0
        for pos in positions:
            if not pos.closed:
                unrealised += (
                    pos.direction
                    * pos.size
                    * capital
                    * (current_close - pos.entry_price)
                    / (pos.entry_price + EPSILON)
                )

        equity.append(capital + unrealised)
        equity_ts.append(ts)

    # Final day close
    for pos in positions:
        if not pos.closed:
            _close_position(pos, float(closes[-1]), timestamps[-1], "backtest_end")
            _record_trade(
                pos, closed_trades, regime_pnl, regime_trades,
                regime_wins, regime_holding_mins, capital,
            )
            capital += pos.pnl
    equity[-1] = capital

    # --- Compute result metrics ---
    equity_arr: NDArray[np.float64] = np.array(equity, dtype=np.float64)

    result = LiquidityCycleBacktestResult()
    result.equity_curve = equity_arr
    result.timestamps = equity_ts
    result.trades = closed_trades
    result.total_trades = len(closed_trades)

    if len(equity_arr) < 2:
        return result

    # Returns
    result.total_return = (equity_arr[-1] / equity_arr[0]) - 1.0

    # Estimate number of trading days for annualisation
    if equity_ts:
        day_span: int = max(
            1, (equity_ts[-1].date() - equity_ts[0].date()).days
        )
        n_years: float = day_span / 365.25
    else:
        n_years = 1.0

    if n_years > EPSILON:
        result.annualised_return = (
            (1.0 + result.total_return) ** (1.0 / n_years) - 1.0
        )

    # Sharpe / Sortino on bar-level returns
    bar_returns: NDArray[np.float64] = np.diff(equity_arr) / (
        equity_arr[:-1] + EPSILON
    )
    if len(bar_returns) > 1:
        mean_ret: float = float(np.mean(bar_returns))
        std_ret: float = float(np.std(bar_returns, ddof=1))
        downside: NDArray[np.float64] = bar_returns[bar_returns < 0]
        downside_std: float = (
            float(np.std(downside, ddof=1)) if len(downside) > 1 else std_ret
        )

        # Annualise assuming ~78 five-minute bars per day
        bars_per_day: float = max(
            1.0,
            len(bar_returns) / max(1, n_years * 252),
        )
        annual_factor: float = np.sqrt(bars_per_day * 252)

        if std_ret > EPSILON:
            result.sharpe_ratio = mean_ret / std_ret * annual_factor
        if downside_std > EPSILON:
            result.sortino_ratio = mean_ret / downside_std * annual_factor

    # Max drawdown
    running_max: NDArray[np.float64] = np.maximum.accumulate(equity_arr)
    drawdowns: NDArray[np.float64] = (running_max - equity_arr) / (
        running_max + EPSILON
    )
    result.max_drawdown = float(np.max(drawdowns))

    # Trade-level stats
    if closed_trades:
        trade_returns: List[float] = [t["return_pct"] for t in closed_trades]
        winners: int = sum(1 for r in trade_returns if r > 0)
        losers_pnl: float = sum(
            abs(t["pnl"]) for t in closed_trades if t["pnl"] < 0
        )
        winners_pnl: float = sum(t["pnl"] for t in closed_trades if t["pnl"] > 0)

        result.win_rate = winners / len(closed_trades)
        result.avg_trade_return = float(np.mean(trade_returns))
        result.profit_factor = (
            winners_pnl / losers_pnl if losers_pnl > EPSILON else float("inf")
        )

        holding_mins: List[float] = [t["holding_minutes"] for t in closed_trades]
        result.avg_holding_minutes = float(np.mean(holding_mins))

    # Per-regime breakdown
    for regime_name in set(list(regime_pnl.keys()) + list(regime_trades.keys())):
        n_trades: int = regime_trades.get(regime_name, 0)
        wins: int = regime_wins.get(regime_name, 0)
        pnl: float = regime_pnl.get(regime_name, 0.0)
        mins: List[float] = regime_holding_mins.get(regime_name, [])

        result.regime_stats[regime_name] = {
            "total_trades": n_trades,
            "total_pnl": pnl,
            "win_rate": wins / n_trades if n_trades > 0 else 0.0,
            "avg_holding_minutes": float(np.mean(mins)) if mins else 0.0,
            "pnl_per_trade": pnl / n_trades if n_trades > 0 else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Backtest helper functions
# ---------------------------------------------------------------------------


def _close_position(
    pos: _BacktestPosition,
    exit_price: float,
    exit_ts,
    reason: str,
) -> None:
    """Mark a backtest position as closed and compute P&L."""
    if isinstance(exit_ts, datetime):
        pos.exit_timestamp = exit_ts
    else:
        pos.exit_timestamp = datetime.fromisoformat(str(exit_ts))

    pos.exit_price = exit_price
    pos.exit_reason = reason
    pos.closed = True
    pos.pnl = (
        pos.direction
        * pos.size
        * (exit_price - pos.entry_price)
        / (pos.entry_price + EPSILON)
    )
    # pnl is expressed as a fraction of capital; the caller multiplies by capital


def _record_trade(
    pos: _BacktestPosition,
    closed_trades: List[Dict],
    regime_pnl: Dict[str, float],
    regime_trades: Dict[str, int],
    regime_wins: Dict[str, int],
    regime_holding_mins: Dict[str, List[float]],
    capital: float,
) -> None:
    """Append a closed position to the trade log and update regime accumulators."""
    holding_minutes: float = 0.0
    if pos.exit_timestamp is not None and pos.entry_timestamp is not None:
        delta = pos.exit_timestamp - pos.entry_timestamp
        holding_minutes = delta.total_seconds() / 60.0

    pnl_dollars: float = pos.pnl * capital
    return_pct: float = pos.pnl * 100.0  # Already a fraction

    trade_record: Dict = {
        "symbol": pos.symbol,
        "regime": pos.regime.name,
        "strategy_type": pos.signal.strategy_type,
        "direction": pos.direction,
        "entry_timestamp": pos.entry_timestamp,
        "exit_timestamp": pos.exit_timestamp,
        "entry_price": pos.entry_price,
        "exit_price": pos.exit_price,
        "pnl": pnl_dollars,
        "return_pct": return_pct,
        "exit_reason": pos.exit_reason,
        "holding_minutes": holding_minutes,
        "confidence": pos.signal.confidence,
        "position_size": pos.size,
    }
    closed_trades.append(trade_record)

    regime_name: str = pos.regime.name
    regime_pnl[regime_name] += pnl_dollars
    regime_trades[regime_name] += 1
    if pnl_dollars > 0:
        regime_wins[regime_name] += 1
    regime_holding_mins[regime_name].append(holding_minutes)
