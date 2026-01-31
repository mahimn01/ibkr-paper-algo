"""
Options Flow and Short Interest Analyzer.

Analyzes options market data and short interest to detect:
- Unusual options activity (smart money signals)
- Gamma squeeze potential
- Short squeeze setups
- Implied volatility signals

Key Concepts:
- Options flow reveals institutional positioning
- Unusual volume often precedes price moves
- High short interest + bullish flow = squeeze potential
- IV rank indicates expected move magnitude
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from trading_algo.stock_selector.models import OptionsScore, ShortInterestScore


@dataclass
class OptionTrade:
    """Single options trade/order."""
    timestamp: datetime
    symbol: str
    expiration: datetime
    strike: float
    option_type: str  # 'call' or 'put'
    side: str  # 'buy' or 'sell'
    quantity: int
    premium: float  # Total premium
    implied_vol: float
    is_sweep: bool  # Aggressive order hitting multiple exchanges
    is_block: bool  # Large institutional trade


@dataclass
class ShortInterestData:
    """Short interest metrics."""
    symbol: str
    short_interest: float  # Total shares short
    float_shares: float    # Total float
    avg_volume: float      # Average daily volume
    days_to_cover: float   # Short interest / avg volume
    short_percent_float: float  # SI as % of float
    previous_si: float     # Previous report SI
    cost_to_borrow: float  # Borrow rate %
    utilization: float     # % of lendable shares borrowed
    as_of_date: datetime


class OptionsFlowAnalyzer:
    """
    Analyze options flow for smart money signals.

    Smart money indicators:
    - Large block trades (>$100K premium)
    - Sweep orders (aggressive, hitting multiple exchanges)
    - Unusual volume (>2x average)
    - Put/call ratio extremes
    - IV skew changes
    """

    def __init__(
        self,
        block_threshold: float = 100_000,  # $100K minimum for block trade
        sweep_threshold: float = 50_000,   # $50K minimum for sweep
        unusual_volume_ratio: float = 2.0,  # 2x normal = unusual
    ):
        self.block_threshold = block_threshold
        self.sweep_threshold = sweep_threshold
        self.unusual_volume_ratio = unusual_volume_ratio

    def analyze(
        self,
        trades: List[OptionTrade],
        historical_volume: Optional[Dict[str, float]] = None,
        current_price: float = 0,
    ) -> OptionsScore:
        """
        Analyze options flow.

        Args:
            trades: Recent options trades
            historical_volume: Average daily call/put volume
            current_price: Current stock price

        Returns:
            OptionsScore with all metrics
        """
        if not trades:
            return self._empty_score()

        # Separate calls and puts
        calls = [t for t in trades if t.option_type == 'call']
        puts = [t for t in trades if t.option_type == 'put']

        # Total volume
        call_volume = sum(t.quantity for t in calls)
        put_volume = sum(t.quantity for t in puts)
        total_volume = call_volume + put_volume

        # Put/call ratio
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0

        # Check for unusual activity
        unusual_activity = False
        if historical_volume:
            avg_volume = historical_volume.get('total', total_volume / 2)
            if total_volume > avg_volume * self.unusual_volume_ratio:
                unusual_activity = True

        # Flow analysis
        call_premium = sum(t.premium for t in calls if t.side == 'buy')
        call_premium -= sum(t.premium for t in calls if t.side == 'sell')
        put_premium = sum(t.premium for t in puts if t.side == 'buy')
        put_premium -= sum(t.premium for t in puts if t.side == 'sell')

        net_premium = call_premium - put_premium

        # Score flows
        max_premium = max(abs(call_premium), abs(put_premium), 1)
        call_flow_score = (call_premium / max_premium + 1) / 2 * 100 if max_premium > 0 else 50
        put_flow_score = (put_premium / max_premium + 1) / 2 * 100 if max_premium > 0 else 50

        # IV analysis
        avg_iv = sum(t.implied_vol for t in trades) / len(trades) if trades else 0.3
        call_iv = sum(t.implied_vol for t in calls) / len(calls) if calls else avg_iv
        put_iv = sum(t.implied_vol for t in puts) / len(puts) if puts else avg_iv
        iv_skew = put_iv - call_iv  # Positive = puts more expensive (bearish)

        # IV rank (0-100) - would need historical IV data
        # Approximating based on absolute level
        iv_rank = min(100, avg_iv * 200)  # 50% IV = 100 rank (rough estimate)

        # Detect large/sweep orders
        blocks = [t for t in trades if t.premium >= self.block_threshold]
        sweeps = [t for t in trades if t.is_sweep and t.premium >= self.sweep_threshold]

        # Smart money direction
        bullish_flow = sum(t.premium for t in blocks + sweeps
                         if t.option_type == 'call' and t.side == 'buy')
        bearish_flow = sum(t.premium for t in blocks + sweeps
                         if t.option_type == 'put' and t.side == 'buy')

        total_smart = bullish_flow + bearish_flow
        if total_smart > 0:
            smart_money_direction = (bullish_flow - bearish_flow) / total_smart
        else:
            smart_money_direction = 0.0

        # Calculate derived scores
        options_signal_score = self._score_options_signal(
            net_premium, unusual_activity, len(blocks), len(sweeps), smart_money_direction
        )

        # Gamma squeeze potential
        squeeze_potential = self._calculate_gamma_squeeze_potential(
            calls, current_price, call_flow_score
        )

        return OptionsScore(
            options_volume=total_volume,
            put_call_ratio=put_call_ratio,
            unusual_activity=unusual_activity,
            call_flow_score=call_flow_score,
            put_flow_score=put_flow_score,
            net_premium=net_premium,
            iv_rank=iv_rank,
            iv_skew=iv_skew,
            large_trades_detected=len(blocks),
            sweep_orders_detected=len(sweeps),
            smart_money_direction=smart_money_direction,
            options_signal_score=options_signal_score,
            squeeze_potential=squeeze_potential,
        )

    def _empty_score(self) -> OptionsScore:
        """Return empty/neutral score when no data."""
        return OptionsScore(
            options_volume=0,
            put_call_ratio=1.0,
            unusual_activity=False,
            call_flow_score=50,
            put_flow_score=50,
            net_premium=0,
            iv_rank=50,
            iv_skew=0,
            large_trades_detected=0,
            sweep_orders_detected=0,
            smart_money_direction=0,
            options_signal_score=50,
            squeeze_potential=0,
        )

    def _score_options_signal(
        self,
        net_premium: float,
        unusual: bool,
        blocks: int,
        sweeps: int,
        smart_direction: float,
    ) -> float:
        """Score overall options signal strength."""
        score = 50  # Neutral base

        # Unusual activity is significant
        if unusual:
            score += 15

        # Large block trades
        score += min(15, blocks * 5)

        # Sweep orders (very aggressive)
        score += min(15, sweeps * 7)

        # Smart money direction magnitude
        score += abs(smart_direction) * 10

        return min(100, score)

    def _calculate_gamma_squeeze_potential(
        self,
        calls: List[OptionTrade],
        current_price: float,
        call_flow_score: float,
    ) -> float:
        """
        Calculate gamma squeeze potential.

        Gamma squeeze occurs when:
        - High call volume near current price
        - Market makers need to buy stock to hedge
        - Creates feedback loop
        """
        if not calls or current_price <= 0:
            return 0.0

        # Find calls near the money (within 5% of current price)
        near_money_calls = [
            c for c in calls
            if abs(c.strike - current_price) / current_price < 0.05
        ]

        if not near_money_calls:
            return 0.0

        # Volume concentration near money
        near_money_volume = sum(c.quantity for c in near_money_calls)
        total_call_volume = sum(c.quantity for c in calls)

        concentration = near_money_volume / total_call_volume if total_call_volume > 0 else 0

        # Combine with flow score
        squeeze_potential = concentration * 0.5 + (call_flow_score / 100) * 0.5

        return min(1.0, squeeze_potential)


class ShortInterestAnalyzer:
    """
    Analyze short interest for squeeze potential.

    Squeeze indicators:
    - High short interest (>20% of float)
    - High days to cover (>5 days)
    - Rising cost to borrow
    - High utilization (>90%)
    - Decreasing shares available
    """

    def __init__(
        self,
        high_si_threshold: float = 0.20,  # 20% of float
        high_dtc_threshold: float = 5.0,  # 5 days to cover
        high_utilization_threshold: float = 0.90,
    ):
        self.high_si = high_si_threshold
        self.high_dtc = high_dtc_threshold
        self.high_util = high_utilization_threshold

    def analyze(
        self,
        data: Optional[ShortInterestData],
    ) -> ShortInterestScore:
        """
        Analyze short interest metrics.

        Args:
            data: Short interest data

        Returns:
            ShortInterestScore with all metrics
        """
        if data is None:
            return self._empty_score()

        # Calculate SI change
        if data.previous_si > 0:
            si_change = (data.short_interest - data.previous_si) / data.previous_si
        else:
            si_change = 0.0

        # Calculate scores
        squeeze_setup = self._score_squeeze_setup(
            data.short_percent_float,
            data.days_to_cover,
            data.utilization,
            data.cost_to_borrow,
        )

        short_pressure = self._score_short_pressure(
            data.short_percent_float,
            si_change,
            data.cost_to_borrow,
        )

        return ShortInterestScore(
            short_interest_ratio=data.short_percent_float,
            days_to_cover=data.days_to_cover,
            short_interest_change=si_change,
            cost_to_borrow=data.cost_to_borrow,
            shares_available=int(data.float_shares - data.short_interest),
            utilization=data.utilization,
            squeeze_setup_score=squeeze_setup,
            short_pressure_score=short_pressure,
        )

    def _empty_score(self) -> ShortInterestScore:
        """Return neutral score when no data."""
        return ShortInterestScore(
            short_interest_ratio=0.0,
            days_to_cover=0.0,
            short_interest_change=0.0,
            cost_to_borrow=0.0,
            shares_available=0,
            utilization=0.0,
            squeeze_setup_score=0.0,
            short_pressure_score=0.0,
        )

    def _score_squeeze_setup(
        self,
        si_pct: float,
        dtc: float,
        utilization: float,
        ctb: float,
    ) -> float:
        """Score short squeeze setup potential (0-100)."""
        score = 0

        # High short interest
        if si_pct > 0.30:  # >30%
            score += 35
        elif si_pct > self.high_si:  # >20%
            score += 25
        elif si_pct > 0.10:  # >10%
            score += 15

        # Days to cover (harder for shorts to exit)
        if dtc > 10:
            score += 25
        elif dtc > self.high_dtc:  # >5
            score += 15
        elif dtc > 3:
            score += 10

        # Utilization (shares hard to borrow)
        if utilization > 0.95:
            score += 25
        elif utilization > self.high_util:  # >90%
            score += 15
        elif utilization > 0.70:
            score += 10

        # Cost to borrow (expensive to maintain short)
        if ctb > 50:  # >50% annual rate
            score += 15
        elif ctb > 20:
            score += 10
        elif ctb > 5:
            score += 5

        return min(100, score)

    def _score_short_pressure(
        self,
        si_pct: float,
        si_change: float,
        ctb: float,
    ) -> float:
        """Score pressure on shorts (0-100)."""
        score = 30  # Base

        # SI increasing = more shorts = more potential pressure
        if si_change > 0.20:  # SI up 20%+
            score += 20
        elif si_change > 0.10:
            score += 15
        elif si_change > 0:
            score += 10
        elif si_change < -0.10:  # Shorts covering
            score -= 15

        # Absolute SI level
        if si_pct > 0.25:
            score += 25
        elif si_pct > 0.15:
            score += 15

        # CTB pressure
        if ctb > 30:
            score += 15
        elif ctb > 10:
            score += 10

        return max(0, min(100, score))
