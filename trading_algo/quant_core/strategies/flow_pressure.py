"""
Flow Pressure Strategy -- Exploiting Structural Institutional Rebalancing Flows

This strategy captures alpha from mandatory, non-discretionary trading patterns
created by institutional mechanics.  Unlike behavioral anomalies, these flow
patterns are structural: index funds MUST trade at specific times, pension
contributions arrive on schedule, and options market makers MUST delta-hedge.
The patterns persist because the participants cannot choose to stop.

Academic foundation:
    - Petajisto (2011): "The Index Premium and Its Hidden Cost for Index Funds"
    - Chen, Noronha & Singal (2004): "The Price Response to S&P 500 Index
      Additions and Deletions"
    - Greenwood (2005): "Short- and Long-term Demand Curves for Stocks"
    - Bessembinder et al. (2016): "Passive Investors and Concentrated Ownership"
    - Lakonishok & Smidt (1988): Turn-of-month effect documentation

Flow types exploited:
    1. Turn-of-month (pension / 401k inflows in last 2 + first 2 trading days)
    2. Quarter-end window dressing (fund managers buy winners, sell losers)
    3. Monthly options-expiry pinning (delta-hedging pins price near max-pain)
    4. Tax-loss selling (December selling pressure, January reversal)
    5. Index reconstitution (Russell/S&P addition/deletion pressure)

Target performance:
    - Standalone Sharpe ~1.0 on the turn-of-month sub-strategy
    - Combined flow composite Sharpe ~0.8-1.2 with low correlation to momentum
    - Expected annual contribution: 3-7% when blended into a multi-strategy book
"""

from __future__ import annotations

import calendar
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.utils.math_utils import (
    rolling_mean,
    rolling_std,
    simple_returns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR: int = 252
_TRADING_DAYS_PER_MONTH: int = 21
_TRADING_DAYS_PER_QUARTER: int = 63

# Months that end a quarter
_QUARTER_END_MONTHS: Tuple[int, ...] = (3, 6, 9, 12)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FlowType(Enum):
    """Canonical flow-pressure category."""
    REBALANCE = "rebalance"
    WINDOW_DRESS = "window_dress"
    EXPIRY_PIN = "expiry_pin"
    TURN_OF_MONTH = "turn_of_month"
    TAX_LOSS = "tax_loss"


class FlowRegime(Enum):
    """Aggregate flow-pressure regime for portfolio-level decisions."""
    STRONG_BUY = auto()
    BUY = auto()
    NEUTRAL = auto()
    SELL = auto()
    STRONG_SELL = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FlowSignal:
    """
    A single flow-pressure signal for one symbol.

    Attributes:
        timestamp: Signal generation time.
        symbol: Instrument identifier.
        flow_type: Which structural flow produces this signal.
        direction: Expected direction, continuous in [-1, +1].
        confidence: Probability-weighted conviction in [0, 1].
        expected_magnitude: Anticipated abnormal return over *expected_duration*.
        expected_duration: Trading days until the flow pressure dissipates.
        position_size: Suggested weight (pre-risk-overlay), can be negative.
    """
    timestamp: datetime
    symbol: str
    flow_type: str
    direction: float
    confidence: float
    expected_magnitude: float
    expected_duration: int
    position_size: float


@dataclass
class FlowPressureConfig:
    """
    Full parameterisation of the flow pressure strategy.

    Defaults are calibrated to US equity markets (NYSE/NASDAQ calendar).
    """

    # -- Turn-of-month --------------------------------------------------------
    tom_lookback: int = 20
    """Historical window (trading days) for estimating TOM effect size."""

    tom_entry_days_before_eom: int = 2
    """Enter positions this many trading days before month end."""

    tom_exit_days_after_bom: int = 2
    """Exit positions this many trading days after month start."""

    tom_expected_magnitude: float = 0.004
    """Baseline expected abnormal return per TOM window (~40 bps)."""

    # -- Quarter-end window dressing -------------------------------------------
    qe_momentum_lookback: int = 60
    """Lookback (trading days) for winner/loser classification."""

    qe_entry_days_before_eoq: int = 5
    """Enter window-dressing trade this many trading days before quarter end."""

    qe_reversal_days_after_boq: int = 3
    """Number of trading days into new quarter to capture reversal."""

    qe_n_stocks: int = 10
    """Number of top/bottom stocks used for the long/short legs."""

    qe_expected_magnitude: float = 0.015
    """Expected abnormal return per quarter-end window (~150 bps)."""

    # -- Options expiry pinning ------------------------------------------------
    expiry_lookback: int = 5
    """Days before expiry to evaluate pinning pressure."""

    expiry_pin_threshold: float = 0.02
    """Price within this fraction of a strike is considered in the pinning zone."""

    expiry_expected_magnitude: float = 0.005
    """Expected magnitude of the pinning fade trade (~50 bps)."""

    expiry_strike_spacing: float = 5.0
    """Assumed strike spacing in dollars for ETF/index options."""

    # -- Tax-loss selling ------------------------------------------------------
    tax_loss_momentum_lookback: int = 252
    """YTD performance lookback (trading days) for loser identification."""

    tax_loss_entry_month: int = 12
    """Month to enter the tax-loss selling trade (December)."""

    tax_loss_entry_day: int = 5
    """Enter short leg around this calendar day of December."""

    tax_loss_cover_day: int = 24
    """Cover short leg around this calendar day of December."""

    tax_loss_january_exit_day: int = 15
    """Exit January rebound trade around this calendar day."""

    tax_loss_n_stocks: int = 10
    """Number of worst YTD performers to trade."""

    tax_loss_expected_magnitude: float = 0.03
    """Expected magnitude per leg (~300 bps December short, ~300 bps Jan long)."""

    # -- Index reconstitution --------------------------------------------------
    recon_pre_days: int = 15
    """Trading days before effective date to enter reconstitution trades."""

    recon_expected_add_magnitude: float = 0.06
    """Expected abnormal return for index additions (~600 bps)."""

    recon_expected_del_magnitude: float = 0.04
    """Expected abnormal return for index deletions (~400 bps, negative)."""

    # -- Risk / sizing ---------------------------------------------------------
    vol_target: float = 0.15
    """Annualised portfolio volatility target."""

    max_position: float = 0.10
    """Maximum absolute weight in any single name."""

    max_gross_exposure: float = 1.0
    """Maximum gross exposure (sum of |weights|)."""

    slippage_bps: float = 5.0
    """Assumed one-way slippage in basis points."""

    commission_bps: float = 10.0
    """Assumed round-trip commission in basis points."""

    min_signal_confidence: float = 0.15
    """Signals below this confidence are discarded."""


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """
    Return the third Friday of the given month/year.

    Monthly equity options expire on the third Friday (or the prior Thursday
    if that Friday is a market holiday, but we ignore holidays here).
    """
    # Find the first day of the month
    first_day = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=first_friday_offset)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday


def _last_business_day(year: int, month: int) -> date:
    """Return the last weekday (Mon-Fri) of the given month."""
    last_day_num = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day_num)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


def _first_business_day(year: int, month: int) -> date:
    """Return the first weekday (Mon-Fri) of the given month."""
    d = date(year, month, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _business_days_between(start: date, end: date) -> int:
    """
    Count business days (Mon-Fri) between *start* and *end* inclusive.

    Returns a negative count when *end* < *start*.
    """
    if end < start:
        return -_business_days_between(end, start)

    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


def _add_business_days(start: date, n: int) -> date:
    """
    Advance *start* by *n* business days (skip weekends).

    Negative *n* moves backwards.
    """
    direction = 1 if n >= 0 else -1
    remaining = abs(n)
    current = start
    while remaining > 0:
        current += timedelta(days=direction)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _subtract_business_days(start: date, n: int) -> date:
    """Move *start* backwards by *n* business days."""
    return _add_business_days(start, -n)


def _is_business_day(d: date) -> bool:
    """True if *d* is Monday-Friday (ignores market holidays)."""
    return d.weekday() < 5


def _next_month(year: int, month: int) -> Tuple[int, int]:
    """Return (year, month) for the month after the given one."""
    if month == 12:
        return year + 1, 1
    return year, month + 1


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FlowPressureStrategy:
    """
    Multi-sub-strategy alpha generator exploiting structural flow pressure.

    Usage::

        strategy = FlowPressureStrategy()

        for bar_date, bar_prices in data_stream:
            strategy.update(bar_date, bar_prices)
            signals = strategy.generate_signals(bar_date, bar_prices)
            weights  = strategy.get_target_weights(bar_date, bar_prices)
    """

    def __init__(self, config: Optional[FlowPressureConfig] = None) -> None:
        self.config = config or FlowPressureConfig()

        # Per-symbol price history (close prices keyed by symbol)
        self._price_history: Dict[str, List[float]] = defaultdict(list)

        # Date tracking
        self._dates: List[date] = []

        # Cache for expensive calendar calculations (cleared monthly)
        self._calendar_cache: Dict[str, object] = {}
        self._cache_month: Optional[int] = None

        # Active reconstitution events: list of (symbol, direction, effective_date)
        self._recon_events: List[Tuple[str, float, date]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, dt: datetime, prices: Dict[str, float]) -> None:
        """
        Ingest a new bar of data.

        Args:
            dt: Bar timestamp (or date-like).
            prices: Dict mapping symbol -> closing price for this bar.
        """
        bar_date = dt.date() if isinstance(dt, datetime) else dt

        # Record the date (de-duplicate if called multiple times per day)
        if not self._dates or self._dates[-1] != bar_date:
            self._dates.append(bar_date)

        # Store prices
        max_history = max(
            self.config.tom_lookback,
            self.config.qe_momentum_lookback,
            self.config.tax_loss_momentum_lookback,
            self.config.expiry_lookback,
        ) + 20  # headroom

        for symbol, price in prices.items():
            self._price_history[symbol].append(price)
            if len(self._price_history[symbol]) > max_history:
                self._price_history[symbol] = self._price_history[symbol][-max_history:]

        # Invalidate calendar cache on month change
        if self._cache_month is not None and bar_date.month != self._cache_month:
            self._calendar_cache.clear()
        self._cache_month = bar_date.month

    def add_reconstitution_event(
        self,
        symbol: str,
        direction: float,
        effective_date: date,
    ) -> None:
        """
        Register an index reconstitution event (addition or deletion).

        Args:
            symbol: Instrument to be added or removed.
            direction: +1.0 for addition (buy pressure), -1.0 for deletion.
            effective_date: Date on which the change becomes effective.
        """
        self._recon_events.append((symbol, float(np.clip(direction, -1.0, 1.0)), effective_date))
        logger.info(
            "Registered reconstitution event: %s dir=%.1f effective=%s",
            symbol, direction, effective_date,
        )

    def generate_signals(
        self,
        dt: datetime,
        prices: Dict[str, NDArray],
    ) -> List[FlowSignal]:
        """
        Generate flow-pressure signals across all sub-strategies.

        Args:
            dt: Current bar timestamp.
            prices: Dict of symbol -> price array (full history up to now).

        Returns:
            List of FlowSignal instances, one per symbol per active flow type.
        """
        bar_date = dt.date() if isinstance(dt, datetime) else dt
        all_signals: List[FlowSignal] = []

        all_signals.extend(self._check_turn_of_month(bar_date, prices))
        all_signals.extend(self._check_quarter_end(bar_date, prices))
        all_signals.extend(self._check_options_expiry(bar_date, prices))
        all_signals.extend(self._check_tax_loss_selling(bar_date, prices))
        all_signals.extend(self._check_reconstitution(bar_date, prices))

        # Filter by minimum confidence
        all_signals = [
            s for s in all_signals
            if s.confidence >= self.config.min_signal_confidence
        ]

        return all_signals

    def get_target_weights(
        self,
        dt: datetime,
        prices: Dict[str, NDArray],
    ) -> Dict[str, float]:
        """
        Aggregate all flow signals into a single set of portfolio weights.

        When multiple flow types produce signals for the same symbol, the
        weights are summed (they represent independent sources of pressure).
        The final weights are then volatility-scaled and capped.

        Args:
            dt: Current bar timestamp.
            prices: Dict of symbol -> price array.

        Returns:
            Dict of symbol -> target weight (signed; negative = short).
        """
        signals = self.generate_signals(dt, prices)

        if not signals:
            return {}

        # Aggregate by symbol: sum of direction * position_size
        raw_weights: Dict[str, float] = defaultdict(float)
        for sig in signals:
            raw_weights[sig.symbol] += sig.direction * sig.position_size

        # Volatility-scale each position
        scaled_weights: Dict[str, float] = {}
        for symbol, raw_w in raw_weights.items():
            if abs(raw_w) < EPSILON:
                continue

            vol_scalar = self._estimate_vol_scalar(symbol, prices)
            scaled_w = raw_w * vol_scalar

            # Cap individual position
            scaled_w = float(np.clip(scaled_w, -self.config.max_position, self.config.max_position))
            scaled_weights[symbol] = scaled_w

        # Enforce gross exposure limit
        gross = sum(abs(w) for w in scaled_weights.values())
        if gross > self.config.max_gross_exposure and gross > EPSILON:
            scale = self.config.max_gross_exposure / gross
            scaled_weights = {s: w * scale for s, w in scaled_weights.items()}

        return scaled_weights

    def reset(self) -> None:
        """Clear all internal state."""
        self._price_history.clear()
        self._dates.clear()
        self._calendar_cache.clear()
        self._cache_month = None
        self._recon_events.clear()

    # ------------------------------------------------------------------
    # Sub-strategy: Turn-of-Month
    # ------------------------------------------------------------------

    def _check_turn_of_month(
        self,
        bar_date: date,
        prices: Dict[str, NDArray],
    ) -> List[FlowSignal]:
        """
        Detect turn-of-month buying pressure.

        Pension and 401(k) contributions systematically flow into equities
        in the last 2 and first 2 trading days of each month.
        Lakonishok & Smidt (1988) document a standalone Sharpe of ~1.0.

        Returns long signals for broad symbols during the TOM window.
        """
        signals: List[FlowSignal] = []

        # Determine if we are inside the TOM window
        eom = _last_business_day(bar_date.year, bar_date.month)
        entry_date = _subtract_business_days(eom, self.config.tom_entry_days_before_eom)

        ny, nm = _next_month(bar_date.year, bar_date.month)
        bom_first = _first_business_day(ny, nm)
        exit_date = _add_business_days(bom_first, self.config.tom_exit_days_after_bom - 1)

        # Also check if we're in the first days of the current month
        # (i.e. the TOM window started last month)
        cur_bom_first = _first_business_day(bar_date.year, bar_date.month)
        cur_exit_date = _add_business_days(cur_bom_first, self.config.tom_exit_days_after_bom - 1)

        in_eom_window = entry_date <= bar_date <= eom
        in_bom_window = cur_bom_first <= bar_date <= cur_exit_date

        if not (in_eom_window or in_bom_window):
            return signals

        # Calculate remaining days in the TOM window
        if in_eom_window:
            remaining = _business_days_between(bar_date, exit_date)
        else:
            remaining = _business_days_between(bar_date, cur_exit_date)
        remaining = max(1, remaining)

        # Estimate confidence from historical TOM consistency
        confidence = self._estimate_tom_confidence(prices)

        for symbol, price_arr in prices.items():
            if len(price_arr) < self.config.tom_lookback:
                continue

            signals.append(FlowSignal(
                timestamp=datetime.combine(bar_date, datetime.min.time()),
                symbol=symbol,
                flow_type=FlowType.TURN_OF_MONTH.value,
                direction=1.0,
                confidence=confidence,
                expected_magnitude=self.config.tom_expected_magnitude,
                expected_duration=remaining,
                position_size=self._size_from_magnitude(
                    self.config.tom_expected_magnitude,
                    confidence,
                    remaining,
                ),
            ))

        return signals

    def _estimate_tom_confidence(self, prices: Dict[str, NDArray]) -> float:
        """
        Estimate confidence in the TOM effect from recent history.

        Looks at whether the last several TOM windows were positive.
        Returns a confidence in [0, 1].
        """
        if not prices:
            return 0.5  # prior

        # Use the first available symbol as a market proxy
        proxy_key = next(iter(prices))
        price_arr = prices[proxy_key]

        if len(price_arr) < _TRADING_DAYS_PER_MONTH * 3:
            return 0.5

        # Check trailing 6 months of TOM windows
        rets = simple_returns(price_arr.astype(np.float64))
        n_positive = 0
        n_total = 0

        # Approximate: last 4 trading days of each 21-day block
        block_size = _TRADING_DAYS_PER_MONTH
        n_blocks = min(6, len(rets) // block_size)

        for i in range(n_blocks):
            end_idx = len(rets) - i * block_size
            start_idx = end_idx - 4  # TOM window ~4 days
            if start_idx < 0:
                break
            tom_return = float(np.sum(rets[start_idx:end_idx]))
            n_total += 1
            if tom_return > 0:
                n_positive += 1

        if n_total == 0:
            return 0.5

        # Bayesian: prior 0.5, update with observations
        raw_rate = n_positive / n_total
        # Shrink toward 0.55 (historical base rate for TOM)
        confidence = 0.3 * 0.55 + 0.7 * raw_rate
        return float(np.clip(confidence, 0.1, 0.95))

    # ------------------------------------------------------------------
    # Sub-strategy: Quarter-End Window Dressing
    # ------------------------------------------------------------------

    def _check_quarter_end(
        self,
        bar_date: date,
        prices: Dict[str, NDArray],
    ) -> List[FlowSignal]:
        """
        Detect quarter-end window-dressing pressure.

        Fund managers buy recent winners and sell recent losers in the last
        ~5 trading days of each quarter to make their holdings "look good"
        in quarterly reports.  The effect reverses in the first ~3 days of
        the new quarter.

        Returns:
            Long signals on recent winners (and short on losers) during the
            dressing window; reversed signals during the reversal window.
        """
        signals: List[FlowSignal] = []

        # Determine if we are near a quarter end
        in_dressing, in_reversal, remaining = self._quarter_end_window(bar_date)

        if not (in_dressing or in_reversal):
            return signals

        # Rank symbols by trailing momentum
        momentum_scores = self._rank_by_momentum(
            prices, self.config.qe_momentum_lookback,
        )
        if len(momentum_scores) < self.config.qe_n_stocks * 2:
            return signals

        sorted_symbols = sorted(momentum_scores.keys(), key=lambda s: momentum_scores[s])

        losers = sorted_symbols[: self.config.qe_n_stocks]
        winners = sorted_symbols[-self.config.qe_n_stocks :]

        # During dressing window: managers BUY winners, SELL losers
        # During reversal window: pressure fades, so reverse
        ts = datetime.combine(bar_date, datetime.min.time())

        if in_dressing:
            for sym in winners:
                signals.append(FlowSignal(
                    timestamp=ts,
                    symbol=sym,
                    flow_type=FlowType.WINDOW_DRESS.value,
                    direction=1.0,
                    confidence=0.55,
                    expected_magnitude=self.config.qe_expected_magnitude,
                    expected_duration=remaining,
                    position_size=self._size_from_magnitude(
                        self.config.qe_expected_magnitude, 0.55, remaining,
                    ),
                ))
            for sym in losers:
                signals.append(FlowSignal(
                    timestamp=ts,
                    symbol=sym,
                    flow_type=FlowType.WINDOW_DRESS.value,
                    direction=-1.0,
                    confidence=0.50,
                    expected_magnitude=self.config.qe_expected_magnitude * 0.7,
                    expected_duration=remaining,
                    position_size=self._size_from_magnitude(
                        self.config.qe_expected_magnitude * 0.7, 0.50, remaining,
                    ),
                ))
        else:
            # Reversal window: fade the dressing effect
            for sym in winners:
                signals.append(FlowSignal(
                    timestamp=ts,
                    symbol=sym,
                    flow_type=FlowType.WINDOW_DRESS.value,
                    direction=-1.0,
                    confidence=0.45,
                    expected_magnitude=self.config.qe_expected_magnitude * 0.5,
                    expected_duration=remaining,
                    position_size=self._size_from_magnitude(
                        self.config.qe_expected_magnitude * 0.5, 0.45, remaining,
                    ),
                ))
            for sym in losers:
                signals.append(FlowSignal(
                    timestamp=ts,
                    symbol=sym,
                    flow_type=FlowType.WINDOW_DRESS.value,
                    direction=1.0,
                    confidence=0.45,
                    expected_magnitude=self.config.qe_expected_magnitude * 0.5,
                    expected_duration=remaining,
                    position_size=self._size_from_magnitude(
                        self.config.qe_expected_magnitude * 0.5, 0.45, remaining,
                    ),
                ))

        return signals

    def _quarter_end_window(
        self,
        bar_date: date,
    ) -> Tuple[bool, bool, int]:
        """
        Check whether *bar_date* falls in a quarter-end dressing or reversal
        window.

        Returns:
            (in_dressing_window, in_reversal_window, remaining_days)
        """
        month = bar_date.month

        # Check dressing window: last N days before quarter end
        if month in _QUARTER_END_MONTHS:
            eoq = _last_business_day(bar_date.year, month)
            entry = _subtract_business_days(eoq, self.config.qe_entry_days_before_eoq)
            if entry <= bar_date <= eoq:
                remaining = _business_days_between(bar_date, eoq)
                return True, False, max(1, remaining)

        # Check reversal window: first N days of the quarter
        quarter_start_months = (1, 4, 7, 10)
        if month in quarter_start_months:
            boq = _first_business_day(bar_date.year, month)
            exit_date = _add_business_days(
                boq, self.config.qe_reversal_days_after_boq - 1,
            )
            if boq <= bar_date <= exit_date:
                remaining = _business_days_between(bar_date, exit_date)
                return False, True, max(1, remaining)

        return False, False, 0

    # ------------------------------------------------------------------
    # Sub-strategy: Options Expiry Pinning
    # ------------------------------------------------------------------

    def _check_options_expiry(
        self,
        bar_date: date,
        prices: Dict[str, NDArray],
    ) -> List[FlowSignal]:
        """
        Detect options-expiry pinning pressure.

        Market makers who are short options delta-hedge continuously.  As
        expiry approaches, gamma increases and hedging activity pins the
        underlying price near the strike with maximum open interest
        ("max-pain").

        Because we do not have live open-interest data in a backtest, we
        approximate the max-pain strike as the nearest round strike to the
        current price (using ``expiry_strike_spacing``).

        The trade: fade moves away from the nearest strike in the last
        ``expiry_lookback`` days before the third Friday.
        """
        signals: List[FlowSignal] = []

        expiry_date = self._next_options_expiry(bar_date)
        days_to_expiry = _business_days_between(bar_date, expiry_date)

        if days_to_expiry < 0 or days_to_expiry > self.config.expiry_lookback:
            return signals

        # Proximity to expiry increases the pinning effect (gamma increases)
        proximity_weight = 1.0 - (days_to_expiry / max(1, self.config.expiry_lookback))

        ts = datetime.combine(bar_date, datetime.min.time())

        for symbol, price_arr in prices.items():
            if len(price_arr) < 10:
                continue

            current_price = float(price_arr[-1])
            if current_price < EPSILON:
                continue

            # Nearest strike (round to strike spacing)
            spacing = self.config.expiry_strike_spacing
            nearest_strike = round(current_price / spacing) * spacing

            if nearest_strike < EPSILON:
                continue

            distance_pct = abs(current_price - nearest_strike) / nearest_strike

            # Only generate signal if price is within the pinning zone
            if distance_pct > self.config.expiry_pin_threshold:
                continue

            # Direction: fade the move away from the strike
            if current_price > nearest_strike:
                direction = -1.0  # expect price to come back down toward strike
            elif current_price < nearest_strike:
                direction = 1.0   # expect price to come back up toward strike
            else:
                continue  # already pinned, no trade

            # Strength of signal scales with proximity and distance
            strength = proximity_weight * (distance_pct / self.config.expiry_pin_threshold)
            confidence = float(np.clip(0.3 + 0.4 * proximity_weight, 0.1, 0.75))

            signals.append(FlowSignal(
                timestamp=ts,
                symbol=symbol,
                flow_type=FlowType.EXPIRY_PIN.value,
                direction=direction,
                confidence=confidence,
                expected_magnitude=self.config.expiry_expected_magnitude * strength,
                expected_duration=max(1, days_to_expiry),
                position_size=self._size_from_magnitude(
                    self.config.expiry_expected_magnitude * strength,
                    confidence,
                    max(1, days_to_expiry),
                ),
            ))

        return signals

    # ------------------------------------------------------------------
    # Sub-strategy: Tax-Loss Selling
    # ------------------------------------------------------------------

    def _check_tax_loss_selling(
        self,
        bar_date: date,
        prices: Dict[str, NDArray],
    ) -> List[FlowSignal]:
        """
        Detect tax-loss selling and January effect opportunities.

        In December, investors sell YTD losers to harvest tax losses,
        depressing their prices further.  In January, the selling pressure
        disappears and many of these stocks rebound ("January effect").

        Trade legs:
        1. Short YTD losers in early December (ride the selling pressure).
        2. Cover shorts and go long those same names in late December/January.
        """
        signals: List[FlowSignal] = []

        month = bar_date.month
        day = bar_date.day

        if month not in (12, 1):
            return signals

        # Identify YTD losers
        losers = self._identify_ytd_losers(prices)
        if not losers:
            return signals

        ts = datetime.combine(bar_date, datetime.min.time())

        if month == 12:
            if day <= self.config.tax_loss_cover_day:
                # December selling pressure phase: short losers
                remaining = max(1, self.config.tax_loss_cover_day - day)
                for symbol, ytd_return in losers:
                    # Stronger signal for bigger losers
                    magnitude_scale = min(2.0, abs(ytd_return) / 0.20)
                    confidence = float(np.clip(0.45 + 0.2 * magnitude_scale, 0.2, 0.80))

                    signals.append(FlowSignal(
                        timestamp=ts,
                        symbol=symbol,
                        flow_type=FlowType.TAX_LOSS.value,
                        direction=-1.0,
                        confidence=confidence,
                        expected_magnitude=self.config.tax_loss_expected_magnitude * magnitude_scale,
                        expected_duration=remaining,
                        position_size=self._size_from_magnitude(
                            self.config.tax_loss_expected_magnitude * magnitude_scale,
                            confidence,
                            remaining,
                        ),
                    ))
            else:
                # Late December: begin accumulating longs for January rebound
                remaining = max(1, 31 - day + self.config.tax_loss_january_exit_day)
                for symbol, ytd_return in losers:
                    magnitude_scale = min(2.0, abs(ytd_return) / 0.20)
                    confidence = float(np.clip(0.40 + 0.15 * magnitude_scale, 0.2, 0.70))

                    signals.append(FlowSignal(
                        timestamp=ts,
                        symbol=symbol,
                        flow_type=FlowType.TAX_LOSS.value,
                        direction=1.0,
                        confidence=confidence,
                        expected_magnitude=self.config.tax_loss_expected_magnitude * 0.8 * magnitude_scale,
                        expected_duration=remaining,
                        position_size=self._size_from_magnitude(
                            self.config.tax_loss_expected_magnitude * 0.8 * magnitude_scale,
                            confidence,
                            remaining,
                        ),
                    ))

        elif month == 1 and day <= self.config.tax_loss_january_exit_day:
            # January rebound: long the same losers
            remaining = max(1, self.config.tax_loss_january_exit_day - day)
            for symbol, ytd_return in losers:
                magnitude_scale = min(2.0, abs(ytd_return) / 0.20)
                confidence = float(np.clip(0.40 + 0.15 * magnitude_scale, 0.2, 0.70))

                signals.append(FlowSignal(
                    timestamp=ts,
                    symbol=symbol,
                    flow_type=FlowType.TAX_LOSS.value,
                    direction=1.0,
                    confidence=confidence,
                    expected_magnitude=self.config.tax_loss_expected_magnitude * 0.8 * magnitude_scale,
                    expected_duration=remaining,
                    position_size=self._size_from_magnitude(
                        self.config.tax_loss_expected_magnitude * 0.8 * magnitude_scale,
                        confidence,
                        remaining,
                    ),
                ))

        return signals

    def _identify_ytd_losers(
        self,
        prices: Dict[str, NDArray],
    ) -> List[Tuple[str, float]]:
        """
        Identify the worst YTD performers from available price data.

        Returns:
            List of (symbol, ytd_return) sorted worst-first, truncated to
            ``config.tax_loss_n_stocks``.
        """
        ytd_returns: List[Tuple[str, float]] = []
        lookback = self.config.tax_loss_momentum_lookback

        for symbol, price_arr in prices.items():
            if len(price_arr) < lookback:
                continue
            start_price = float(price_arr[-lookback])
            end_price = float(price_arr[-1])
            if start_price < EPSILON:
                continue
            ret = (end_price - start_price) / start_price
            ytd_returns.append((symbol, ret))

        if not ytd_returns:
            return []

        # Sort ascending (worst performers first), take bottom N
        ytd_returns.sort(key=lambda x: x[1])
        # Only include actual losers
        losers = [(s, r) for s, r in ytd_returns if r < 0]
        return losers[: self.config.tax_loss_n_stocks]

    # ------------------------------------------------------------------
    # Sub-strategy: Index Reconstitution
    # ------------------------------------------------------------------

    def _check_reconstitution(
        self,
        bar_date: date,
        prices: Dict[str, NDArray],
    ) -> List[FlowSignal]:
        """
        Generate signals for pending index reconstitution events.

        Events must be registered via ``add_reconstitution_event()`` before
        this method will produce signals.  In a live system the events come
        from an external data feed (e.g. Russell reconstitution announcements).
        """
        signals: List[FlowSignal] = []

        # Clean up expired events
        self._recon_events = [
            (sym, d, eff) for sym, d, eff in self._recon_events
            if eff >= bar_date
        ]

        ts = datetime.combine(bar_date, datetime.min.time())

        for symbol, direction, effective_date in self._recon_events:
            days_to_effective = _business_days_between(bar_date, effective_date)

            if days_to_effective < 0 or days_to_effective > self.config.recon_pre_days:
                continue

            # Only trade if we have price data
            if symbol not in prices or len(prices[symbol]) < 20:
                continue

            if direction > 0:
                expected_mag = self.config.recon_expected_add_magnitude
            else:
                expected_mag = self.config.recon_expected_del_magnitude

            # Confidence increases as effective date approaches
            time_weight = 1.0 - (days_to_effective / max(1, self.config.recon_pre_days))
            confidence = float(np.clip(0.40 + 0.30 * time_weight, 0.30, 0.75))

            signals.append(FlowSignal(
                timestamp=ts,
                symbol=symbol,
                flow_type=FlowType.REBALANCE.value,
                direction=direction,
                confidence=confidence,
                expected_magnitude=expected_mag,
                expected_duration=max(1, days_to_effective),
                position_size=self._size_from_magnitude(
                    expected_mag, confidence, max(1, days_to_effective),
                ),
            ))

        return signals

    # ------------------------------------------------------------------
    # Calendar helpers (public, thin wrappers for testability)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trading_day(dt: datetime) -> bool:
        """
        Check whether *dt* falls on a business day (Mon-Fri).

        Does **not** account for market holidays; a holiday calendar can
        be plugged in via subclass override.
        """
        d = dt.date() if isinstance(dt, datetime) else dt
        return _is_business_day(d)

    @staticmethod
    def _days_to_month_end(dt: datetime) -> int:
        """
        Return the number of business days remaining until the last
        business day of the current month (inclusive of that last day,
        exclusive of today).
        """
        d = dt.date() if isinstance(dt, datetime) else dt
        eom = _last_business_day(d.year, d.month)
        if d >= eom:
            return 0
        return _business_days_between(d, eom) - 1  # exclude today

    @staticmethod
    def _days_to_quarter_end(dt: datetime) -> int:
        """
        Return the number of business days remaining until the last
        business day of the current quarter.
        """
        d = dt.date() if isinstance(dt, datetime) else dt
        # Find the quarter-end month
        qe_month = 3 * ((d.month - 1) // 3 + 1)  # 3, 6, 9, or 12
        eoq = _last_business_day(d.year, qe_month)
        if d >= eoq:
            return 0
        return _business_days_between(d, eoq) - 1

    @staticmethod
    def _next_options_expiry(dt) -> date:
        """
        Return the next monthly options-expiry date (third Friday) on or
        after *dt*.
        """
        d = dt.date() if isinstance(dt, datetime) else dt
        tf = _third_friday(d.year, d.month)
        if tf >= d:
            return tf
        # Already past this month's expiry; use next month
        ny, nm = _next_month(d.year, d.month)
        return _third_friday(ny, nm)

    # ------------------------------------------------------------------
    # Internal sizing / volatility helpers
    # ------------------------------------------------------------------

    def _size_from_magnitude(
        self,
        expected_magnitude: float,
        confidence: float,
        duration_days: int,
    ) -> float:
        """
        Convert an expected magnitude and confidence into a position size.

        Uses a simplified Kelly-like formula:
            raw_size = (expected_magnitude * confidence) / target_daily_vol

        The result is clamped to ``[0, max_position]``.
        """
        if expected_magnitude < EPSILON or confidence < EPSILON:
            return 0.0

        # Annualise the expected return then compute an edge/vol ratio
        daily_target_vol = self.config.vol_target / SQRT_252
        edge = expected_magnitude * confidence

        # Scale by inverse duration (shorter = more aggressive)
        if duration_days > 0:
            daily_edge = edge / duration_days
        else:
            daily_edge = edge

        if daily_target_vol < EPSILON:
            return 0.0

        raw_size = daily_edge / daily_target_vol
        # Apply a dampener to avoid over-betting
        raw_size *= 0.5

        return float(np.clip(raw_size, 0.0, self.config.max_position))

    def _estimate_vol_scalar(
        self,
        symbol: str,
        prices: Dict[str, NDArray],
    ) -> float:
        """
        Compute a volatility scalar for *symbol* so that each position
        contributes roughly equally to portfolio risk.

        Returns a multiplier (>1 for low-vol names, <1 for high-vol names).
        """
        if symbol not in prices:
            return 1.0

        price_arr = prices[symbol]
        if len(price_arr) < 22:
            return 1.0

        rets = simple_returns(price_arr[-60:].astype(np.float64) if len(price_arr) >= 60 else price_arr.astype(np.float64))
        if len(rets) < 5:
            return 1.0

        ann_vol = float(np.std(rets) * SQRT_252)
        if ann_vol < EPSILON:
            return 1.0

        # Target each position to contribute vol_target / sqrt(N) of vol
        scalar = self.config.vol_target / ann_vol
        # Bound the scalar to prevent extreme leverage
        return float(np.clip(scalar, 0.25, 3.0))

    def _rank_by_momentum(
        self,
        prices: Dict[str, NDArray],
        lookback: int,
    ) -> Dict[str, float]:
        """
        Rank all symbols by simple momentum (price return over *lookback* bars).

        Returns:
            Dict of symbol -> momentum return.
        """
        scores: Dict[str, float] = {}
        for symbol, price_arr in prices.items():
            if len(price_arr) < lookback + 1:
                continue
            start = float(price_arr[-(lookback + 1)])
            end = float(price_arr[-1])
            if start < EPSILON:
                continue
            scores[symbol] = (end - start) / start
        return scores


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_flow_pressure_backtest(
    historical_data: Dict[str, NDArray[np.float64]],
    timestamps: List[datetime],
    initial_capital: float = 100_000.0,
    config: Optional[FlowPressureConfig] = None,
    recon_events: Optional[List[Tuple[str, float, date]]] = None,
) -> Dict:
    """
    Run a full backtest of the flow-pressure strategy.

    Args:
        historical_data:
            Dict of symbol -> OHLCV array with shape ``(T, 5)``
            where columns are ``[open, high, low, close, volume]``.
        timestamps:
            Ordered list of bar timestamps, length ``T``.
        initial_capital:
            Starting equity.
        config:
            Strategy configuration (uses defaults if ``None``).
        recon_events:
            Optional list of ``(symbol, direction, effective_date)`` tuples
            representing index reconstitution events to inject.

    Returns:
        Dict containing:
            - ``equity_curve``: NDArray of portfolio equity over time.
            - ``returns``: NDArray of per-bar simple returns.
            - ``trades``: List of trade dicts.
            - ``signals_log``: List of all signals generated.
            - ``total_return``, ``annualized_return``, ``sharpe_ratio``,
              ``volatility``, ``max_drawdown``, ``calmar_ratio``,
              ``total_trades``, ``final_value``.
    """
    config = config or FlowPressureConfig()
    strategy = FlowPressureStrategy(config)

    # Register reconstitution events if provided
    if recon_events:
        for sym, direction, eff_date in recon_events:
            strategy.add_reconstitution_event(sym, direction, eff_date)

    symbols = list(historical_data.keys())
    n_bars = len(timestamps)

    # Extract close prices (column index 3)
    close_prices: Dict[str, NDArray] = {}
    for sym in symbols:
        data = historical_data[sym]
        if data.ndim == 2 and data.shape[1] >= 4:
            close_prices[sym] = data[:, 3]
        else:
            close_prices[sym] = data  # assume 1-D close prices

    # Warmup period: need enough history for the longest lookback
    warmup = max(
        config.tom_lookback,
        config.qe_momentum_lookback,
        config.tax_loss_momentum_lookback,
    ) + 10

    if n_bars <= warmup:
        logger.warning(
            "Not enough bars (%d) for warmup (%d). Returning empty results.",
            n_bars, warmup,
        )
        return _empty_result(initial_capital)

    # State
    cash = initial_capital
    positions: Dict[str, float] = {}  # symbol -> shares held
    equity_curve: List[float] = []
    trades: List[Dict] = []
    signals_log: List[FlowSignal] = []

    # Fill warmup equity
    for _ in range(warmup):
        equity_curve.append(initial_capital)

    # Slippage / commission helpers (one-way, in fraction)
    slippage_frac = config.slippage_bps / 10_000.0
    commission_frac = config.commission_bps / 10_000.0  # round-trip split in half

    for t in range(warmup, n_bars):
        dt = timestamps[t]

        # Build price snapshot up to bar t
        price_snapshot: Dict[str, NDArray] = {
            sym: close_prices[sym][: t + 1] for sym in symbols
        }

        # Current prices
        current_prices = {sym: float(close_prices[sym][t]) for sym in symbols}

        # Feed strategy
        strategy.update(dt, current_prices)

        # Current equity
        position_value = sum(
            positions.get(sym, 0.0) * current_prices.get(sym, 0.0)
            for sym in symbols
        )
        equity = cash + position_value

        # Target weights
        target_weights = strategy.get_target_weights(dt, price_snapshot)

        # Log signals (for analysis)
        bar_signals = strategy.generate_signals(dt, price_snapshot)
        signals_log.extend(bar_signals)

        # Compute target positions in dollar terms
        target_values = {sym: equity * w for sym, w in target_weights.items()}

        # Rebalance
        for sym in symbols:
            current_shares = positions.get(sym, 0.0)
            current_value = current_shares * current_prices.get(sym, 0.0)
            target_value = target_values.get(sym, 0.0)
            delta_value = target_value - current_value

            price = current_prices.get(sym, 0.0)
            if price < EPSILON or abs(delta_value) < 50.0:
                continue

            delta_shares = delta_value / price

            # Apply slippage
            if delta_shares > 0:
                exec_price = price * (1.0 + slippage_frac)
            else:
                exec_price = price * (1.0 - slippage_frac)

            trade_cost = abs(delta_shares) * exec_price
            commission = trade_cost * commission_frac * 0.5  # half round-trip

            if delta_shares > 0:
                cash -= trade_cost + commission
            else:
                cash += trade_cost - commission

            positions[sym] = current_shares + delta_shares

            trades.append({
                "timestamp": dt,
                "symbol": sym,
                "shares": delta_shares,
                "price": exec_price,
                "commission": commission,
                "delta_value": delta_value,
            })

        # Mark-to-market
        position_value = sum(
            positions.get(sym, 0.0) * current_prices.get(sym, 0.0)
            for sym in symbols
        )
        equity = cash + position_value
        equity_curve.append(equity)

    # ------------------------------------------------------------------
    # Compute performance metrics
    # ------------------------------------------------------------------
    equity_arr = np.array(equity_curve, dtype=np.float64)

    if len(equity_arr) < 2:
        return _empty_result(initial_capital)

    returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], EPSILON)

    total_return = (equity_arr[-1] / initial_capital) - 1.0
    n_years = len(returns) / _TRADING_DAYS_PER_YEAR
    if n_years > EPSILON:
        ann_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    else:
        ann_return = 0.0

    volatility = float(np.std(returns) * SQRT_252) if len(returns) > 1 else 0.0
    sharpe = (ann_return - 0.02) / volatility if volatility > EPSILON else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / np.maximum(peak, EPSILON)
    max_dd = float(np.max(drawdown))

    calmar = ann_return / max_dd if max_dd > EPSILON else 0.0

    # Win rate
    trade_pnl = []
    for i in range(1, len(trades)):
        # Approximate per-trade P&L from consecutive trades on same symbol
        t_cur = trades[i]
        t_prev = trades[i - 1]
        if t_cur["symbol"] == t_prev["symbol"] and np.sign(t_cur["shares"]) != np.sign(t_prev["shares"]):
            pnl = (t_cur["price"] - t_prev["price"]) * t_prev["shares"]
            trade_pnl.append(pnl)
    win_rate = float(np.mean([1 if p > 0 else 0 for p in trade_pnl])) if trade_pnl else 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "volatility": volatility,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "equity_curve": equity_arr,
        "returns": returns,
        "trades": trades,
        "signals_log": signals_log,
        "final_value": float(equity_arr[-1]),
    }


def _empty_result(initial_capital: float) -> Dict:
    """Return a metrics dict for degenerate cases (no bars, no trades)."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "volatility": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "equity_curve": np.array([initial_capital]),
        "returns": np.array([]),
        "trades": [],
        "signals_log": [],
        "final_value": initial_capital,
    }
