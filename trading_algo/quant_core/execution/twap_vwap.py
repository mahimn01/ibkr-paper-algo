"""
TWAP and VWAP Execution Algorithms

Standard benchmark execution algorithms:
    - TWAP: Time-Weighted Average Price
    - VWAP: Volume-Weighted Average Price

These are simpler alternatives to Almgren-Chriss when:
    - Market impact is less of a concern
    - Benchmark tracking is the primary goal
    - Simpler implementation is preferred

References:
    - Berkowitz et al. (1988): "The Total Cost of Transactions on the NYSE"
    - Almgren (2009): "Execution Costs"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import EPSILON


@dataclass
class TWAPSchedule:
    """TWAP execution schedule."""
    time_points: NDArray[np.float64]
    trade_sizes: NDArray[np.float64]
    total_shares: float
    horizon_minutes: float
    interval_minutes: float


@dataclass
class VWAPSchedule:
    """VWAP execution schedule."""
    time_points: NDArray[np.float64]
    trade_sizes: NDArray[np.float64]
    volume_profile: NDArray[np.float64]
    total_shares: float
    horizon_minutes: float


class TWAPExecutor:
    """
    Time-Weighted Average Price Executor.

    Splits order evenly across time intervals.

    Usage:
        twap = TWAPExecutor()
        schedule = twap.generate_schedule(
            total_shares=10000,
            horizon_minutes=60,
            interval_minutes=5,
        )

        for t, size in zip(schedule.time_points, schedule.trade_sizes):
            execute_at_time(t, size)
    """

    def __init__(
        self,
        min_trade_size: float = 100,
        randomize: bool = True,
        randomize_pct: float = 0.1,
    ):
        """
        Initialize TWAP executor.

        Args:
            min_trade_size: Minimum trade size
            randomize: Add randomness to sizes to reduce predictability
            randomize_pct: Randomization as fraction of trade size
        """
        self.min_trade_size = min_trade_size
        self.randomize = randomize
        self.randomize_pct = randomize_pct

    def generate_schedule(
        self,
        total_shares: float,
        horizon_minutes: float,
        interval_minutes: float = 5.0,
    ) -> TWAPSchedule:
        """
        Generate TWAP schedule.

        Args:
            total_shares: Total shares to trade (negative for sell)
            horizon_minutes: Execution horizon
            interval_minutes: Time between trades

        Returns:
            TWAPSchedule with trade times and sizes
        """
        if abs(total_shares) < EPSILON:
            return TWAPSchedule(
                time_points=np.array([0.0]),
                trade_sizes=np.array([0.0]),
                total_shares=0.0,
                horizon_minutes=horizon_minutes,
                interval_minutes=interval_minutes,
            )

        # Calculate number of intervals
        n_intervals = max(1, int(horizon_minutes / interval_minutes))

        # Time points
        time_points = np.linspace(0, horizon_minutes, n_intervals + 1)[:-1]

        # Equal trade sizes
        base_size = total_shares / n_intervals
        trade_sizes = np.full(n_intervals, base_size)

        # Add randomization if enabled
        if self.randomize and n_intervals > 1:
            # Random adjustments that sum to zero
            adjustments = np.random.randn(n_intervals)
            adjustments = adjustments - adjustments.mean()  # Mean zero
            adjustments *= self.randomize_pct * abs(base_size)
            trade_sizes += adjustments

        # Ensure minimum trade size
        trade_sizes = np.sign(trade_sizes) * np.maximum(
            np.abs(trade_sizes), self.min_trade_size
        )

        # Adjust to match total
        trade_sizes[-1] += total_shares - trade_sizes.sum()

        return TWAPSchedule(
            time_points=time_points,
            trade_sizes=trade_sizes,
            total_shares=total_shares,
            horizon_minutes=horizon_minutes,
            interval_minutes=interval_minutes,
        )

    def calculate_expected_vwap_deviation(
        self,
        schedule: TWAPSchedule,
        volume_profile: NDArray[np.float64],
    ) -> float:
        """
        Calculate expected deviation from VWAP.

        TWAP will deviate from VWAP if volume is not uniform.
        """
        if len(volume_profile) != len(schedule.trade_sizes):
            return 0.0

        # Normalize profiles
        vol_weights = volume_profile / volume_profile.sum()
        twap_weights = np.abs(schedule.trade_sizes) / np.abs(schedule.trade_sizes).sum()

        # VWAP deviation is the tracking error
        deviation = np.sqrt(np.sum((twap_weights - vol_weights) ** 2))
        return float(deviation)


class VWAPExecutor:
    """
    Volume-Weighted Average Price Executor.

    Trades proportionally to expected volume throughout the day.

    Usage:
        vwap = VWAPExecutor()

        # With historical volume profile
        schedule = vwap.generate_schedule(
            total_shares=10000,
            horizon_minutes=60,
            volume_profile=historical_profile,
        )

        # Or use default U-shaped profile
        schedule = vwap.generate_schedule_with_default_profile(
            total_shares=10000,
            horizon_minutes=60,
        )
    """

    # Default U-shaped intraday volume profile
    # (Higher volume at open and close)
    DEFAULT_PROFILE = np.array([
        1.5, 1.3, 1.1, 1.0, 0.9, 0.8,  # Morning
        0.7, 0.7, 0.7, 0.8, 0.9, 1.0,  # Midday
        1.1, 1.2, 1.4, 1.6, 1.8, 2.0,  # Afternoon close
    ])

    def __init__(
        self,
        min_trade_size: float = 100,
        min_participation: float = 0.01,
        max_participation: float = 0.25,
    ):
        """
        Initialize VWAP executor.

        Args:
            min_trade_size: Minimum trade size
            min_participation: Minimum fraction of interval volume
            max_participation: Maximum fraction of interval volume
        """
        self.min_trade_size = min_trade_size
        self.min_participation = min_participation
        self.max_participation = max_participation

    def generate_schedule(
        self,
        total_shares: float,
        horizon_minutes: float,
        volume_profile: NDArray[np.float64],
        start_minute: float = 0,
    ) -> VWAPSchedule:
        """
        Generate VWAP schedule from volume profile.

        Args:
            total_shares: Total shares to trade
            horizon_minutes: Execution horizon
            volume_profile: Expected volume at each interval
            start_minute: Starting minute of trading day

        Returns:
            VWAPSchedule with trade sizes proportional to volume
        """
        if abs(total_shares) < EPSILON or len(volume_profile) == 0:
            return VWAPSchedule(
                time_points=np.array([0.0]),
                trade_sizes=np.array([0.0]),
                volume_profile=np.array([1.0]),
                total_shares=0.0,
                horizon_minutes=horizon_minutes,
            )

        n_intervals = len(volume_profile)

        # Normalize volume profile
        vol_weights = volume_profile / volume_profile.sum()

        # Calculate trade sizes proportional to volume
        trade_sizes = total_shares * vol_weights

        # Ensure minimum trade size
        trade_sizes = np.sign(trade_sizes) * np.maximum(
            np.abs(trade_sizes), self.min_trade_size
        )

        # Adjust to match total
        trade_sizes[-1] += total_shares - trade_sizes.sum()

        # Time points
        interval_minutes = horizon_minutes / n_intervals
        time_points = np.arange(n_intervals) * interval_minutes + start_minute

        return VWAPSchedule(
            time_points=time_points,
            trade_sizes=trade_sizes,
            volume_profile=volume_profile,
            total_shares=total_shares,
            horizon_minutes=horizon_minutes,
        )

    def generate_schedule_with_default_profile(
        self,
        total_shares: float,
        horizon_minutes: float,
        num_intervals: Optional[int] = None,
    ) -> VWAPSchedule:
        """
        Generate VWAP schedule using default U-shaped profile.

        Args:
            total_shares: Total shares to trade
            horizon_minutes: Execution horizon
            num_intervals: Number of trading intervals

        Returns:
            VWAPSchedule with default volume profile
        """
        if num_intervals is None:
            num_intervals = max(1, int(horizon_minutes / 5))  # 5-min intervals

        # Interpolate default profile to match intervals
        x_default = np.linspace(0, 1, len(self.DEFAULT_PROFILE))
        x_new = np.linspace(0, 1, num_intervals)
        volume_profile = np.interp(x_new, x_default, self.DEFAULT_PROFILE)

        return self.generate_schedule(
            total_shares=total_shares,
            horizon_minutes=horizon_minutes,
            volume_profile=volume_profile,
        )

    def update_schedule_with_real_volume(
        self,
        current_schedule: VWAPSchedule,
        realized_volume: NDArray[np.float64],
        remaining_shares: float,
        current_interval: int,
    ) -> VWAPSchedule:
        """
        Update schedule based on realized volume.

        If actual volume differs from expected, adjust remaining trades.

        Args:
            current_schedule: Original schedule
            realized_volume: Actual volume so far
            remaining_shares: Shares still to execute
            current_interval: Current time interval index

        Returns:
            Updated VWAPSchedule
        """
        if current_interval >= len(current_schedule.volume_profile):
            return current_schedule

        # Calculate remaining volume profile
        remaining_profile = current_schedule.volume_profile[current_interval:]

        if len(remaining_profile) == 0:
            return current_schedule

        # Update with realized volume information
        if len(realized_volume) > 0:
            # Scale remaining profile based on realized/expected ratio
            expected_so_far = current_schedule.volume_profile[:current_interval].sum()
            realized_so_far = realized_volume.sum()

            if expected_so_far > 0:
                scale_factor = realized_so_far / expected_so_far
                # Dampen adjustment to avoid over-reaction
                scale_factor = 0.5 + 0.5 * scale_factor
                remaining_profile = remaining_profile * scale_factor

        # Generate new schedule for remaining execution
        return self.generate_schedule(
            total_shares=remaining_shares,
            horizon_minutes=current_schedule.horizon_minutes * len(remaining_profile) / len(current_schedule.volume_profile),
            volume_profile=remaining_profile,
        )

    def calculate_vwap(
        self,
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
    ) -> float:
        """
        Calculate Volume-Weighted Average Price.

        Args:
            prices: Price at each interval
            volumes: Volume at each interval

        Returns:
            VWAP
        """
        total_volume = volumes.sum()
        if total_volume < EPSILON:
            return float(prices[-1]) if len(prices) > 0 else 0.0

        return float(np.sum(prices * volumes) / total_volume)

    def calculate_execution_quality(
        self,
        execution_prices: NDArray[np.float64],
        execution_volumes: NDArray[np.float64],
        market_vwap: float,
    ) -> Tuple[float, float]:
        """
        Calculate execution quality vs VWAP benchmark.

        Args:
            execution_prices: Prices at which we executed
            execution_volumes: Volumes we executed
            market_vwap: Market VWAP during execution

        Returns:
            Tuple of (execution_vwap, slippage_bps)
        """
        exec_vwap = self.calculate_vwap(execution_prices, execution_volumes)
        slippage_bps = (exec_vwap - market_vwap) / market_vwap * 10000

        return exec_vwap, slippage_bps


class POVExecutor:
    """
    Percentage of Volume (POV) Executor.

    Trades at a fixed percentage of market volume.
    Adaptive to real-time volume.
    """

    def __init__(
        self,
        target_pov: float = 0.10,
        min_pov: float = 0.05,
        max_pov: float = 0.25,
    ):
        """
        Initialize POV executor.

        Args:
            target_pov: Target participation rate
            min_pov: Minimum participation rate
            max_pov: Maximum participation rate
        """
        self.target_pov = target_pov
        self.min_pov = min_pov
        self.max_pov = max_pov

    def calculate_trade_size(
        self,
        interval_volume: float,
        remaining_shares: float,
        remaining_expected_volume: float,
    ) -> float:
        """
        Calculate trade size for current interval.

        Args:
            interval_volume: Volume in current interval
            remaining_shares: Shares still to execute
            remaining_expected_volume: Expected remaining market volume

        Returns:
            Trade size for this interval
        """
        # Base trade on target POV
        base_trade = interval_volume * self.target_pov

        # Adjust if we're behind/ahead of schedule
        if remaining_expected_volume > 0:
            expected_pov = remaining_shares / remaining_expected_volume
            if expected_pov > self.max_pov:
                # Need to trade more aggressively
                base_trade = interval_volume * self.max_pov
            elif expected_pov < self.min_pov:
                # Can trade more passively
                base_trade = interval_volume * self.min_pov

        # Don't trade more than remaining
        return min(base_trade, remaining_shares)

    def estimate_completion_time(
        self,
        total_shares: float,
        expected_volume_profile: NDArray[np.float64],
        interval_minutes: float,
    ) -> float:
        """
        Estimate time to complete order at target POV.

        Args:
            total_shares: Total shares to execute
            expected_volume_profile: Expected volume per interval
            interval_minutes: Length of each interval

        Returns:
            Estimated completion time in minutes
        """
        remaining = abs(total_shares)
        time_elapsed = 0.0

        for vol in expected_volume_profile:
            trade_size = vol * self.target_pov
            remaining -= trade_size
            time_elapsed += interval_minutes

            if remaining <= 0:
                # Interpolate final interval
                if trade_size > 0:
                    time_elapsed -= (abs(remaining) / trade_size) * interval_minutes
                break

        return time_elapsed
