"""
Almgren-Chriss Optimal Execution Framework

Implements optimal trade execution based on:
    Almgren, R., & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
    Journal of Risk, 3(2), 5-39.

The model minimizes execution cost while controlling for timing risk.

Key Concepts:
    - Permanent Impact: Price change that persists after trading
    - Temporary Impact: Price pressure during execution only
    - Implementation Shortfall: Difference between decision price and execution price
    - Efficient Frontier: Optimal cost-risk tradeoff curve

Mathematical Framework:
    The optimal trajectory minimizes: E[Cost] + λ * Var[Cost]

    where:
        - E[Cost] = ½γX² + εX + η∑(n_k²/τ)  (expected cost)
        - Var[Cost] = σ²∑(x_k²τ)             (execution risk)
        - λ = risk aversion parameter
        - γ = permanent impact coefficient
        - η = temporary impact coefficient
        - ε = fixed cost per share
        - σ = volatility
        - X = total shares to trade
        - T = execution horizon

Optimal Solution:
    x*(t) = X * sinh(κ(T-t)) / sinh(κT)

    where κ = sqrt(λσ²/η)

References:
    - https://www.math.nyu.edu/~almgren/papers/optliq.pdf
    - Almgren (2003): "Optimal Execution with Nonlinear Impact Functions"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import (
    EPSILON,
    SQRT_252,
    TRADING_DAYS_PER_YEAR,
)


class ExecutionUrgency(Enum):
    """Execution urgency levels mapping to risk aversion."""
    PASSIVE = auto()      # Low urgency, minimize impact
    NEUTRAL = auto()      # Balanced approach
    AGGRESSIVE = auto()   # High urgency, accept more impact
    IMMEDIATE = auto()    # Execute as fast as possible


@dataclass
class MarketImpactModel:
    """
    Market impact parameters for Almgren-Chriss model.

    Attributes:
        permanent_impact (gamma): Price impact per share that persists
        temporary_impact (eta): Temporary price impact coefficient
        fixed_cost (epsilon): Fixed cost per share (bid-ask spread)
        volatility (sigma): Daily volatility of the asset

    Impact Functions:
        - Permanent: g(v) = γv (linear in trading rate)
        - Temporary: h(v) = ε + ηv (affine in trading rate)
    """
    permanent_impact: float  # γ (gamma)
    temporary_impact: float  # η (eta)
    fixed_cost: float        # ε (epsilon) - half spread
    volatility: float        # σ (sigma) - daily vol

    @classmethod
    def estimate_from_data(
        cls,
        avg_daily_volume: float,
        avg_spread: float,
        volatility: float,
        market_cap: Optional[float] = None,
    ) -> "MarketImpactModel":
        """
        Estimate impact parameters from market data.

        Uses empirical relationships from market microstructure literature.

        Args:
            avg_daily_volume: Average daily trading volume
            avg_spread: Average bid-ask spread (as decimal)
            volatility: Daily volatility
            market_cap: Market capitalization (optional)

        Returns:
            MarketImpactModel with estimated parameters
        """
        # Fixed cost is half the spread
        epsilon = avg_spread / 2

        # Temporary impact scales with volatility and inverse of liquidity
        # η ≈ σ / (ADV^0.5) based on square-root law
        eta = volatility / np.sqrt(avg_daily_volume) if avg_daily_volume > 0 else 0.001

        # Permanent impact is typically smaller than temporary
        # γ ≈ 0.1 * η for liquid stocks
        gamma = 0.1 * eta

        # Adjust for market cap if available (smaller cap = higher impact)
        if market_cap is not None and market_cap > 0:
            cap_adjustment = np.sqrt(1e10 / market_cap)  # Normalize to $10B
            cap_adjustment = np.clip(cap_adjustment, 0.5, 3.0)
            eta *= cap_adjustment
            gamma *= cap_adjustment

        return cls(
            permanent_impact=gamma,
            temporary_impact=eta,
            fixed_cost=epsilon,
            volatility=volatility,
        )

    @classmethod
    def from_kyle_lambda(
        cls,
        kyle_lambda: float,
        volatility: float,
        avg_spread: float = 0.001,
    ) -> "MarketImpactModel":
        """
        Create model from Kyle's lambda (price impact per unit trade).

        Kyle (1985) showed λ = σ / √(informed trading volume)
        """
        return cls(
            permanent_impact=kyle_lambda * 0.1,
            temporary_impact=kyle_lambda,
            fixed_cost=avg_spread / 2,
            volatility=volatility,
        )


@dataclass
class ExecutionPlan:
    """
    Optimal execution plan from Almgren-Chriss.

    Attributes:
        time_steps: Array of time points
        holdings: Optimal holdings at each time point
        trade_schedule: Shares to trade in each period
        expected_cost: Expected implementation shortfall
        cost_variance: Variance of implementation shortfall
        risk_adjusted_cost: E[Cost] + λ * Var[Cost]
    """
    time_steps: NDArray[np.float64]     # t_0, t_1, ..., t_N
    holdings: NDArray[np.float64]       # x_0, x_1, ..., x_N (remaining shares)
    trade_schedule: NDArray[np.float64] # n_1, n_2, ..., n_N (shares per period)
    expected_cost: float
    cost_variance: float
    risk_adjusted_cost: float
    kappa: float                        # Urgency parameter

    @property
    def num_periods(self) -> int:
        return len(self.trade_schedule)

    @property
    def completion_time(self) -> float:
        return self.time_steps[-1] if len(self.time_steps) > 0 else 0.0

    def get_trade_at_time(self, t: float) -> float:
        """Get trade size for a given time."""
        if len(self.time_steps) == 0:
            return 0.0

        idx = np.searchsorted(self.time_steps[:-1], t)
        if idx >= len(self.trade_schedule):
            return 0.0
        return float(self.trade_schedule[idx])


@dataclass
class ExecutionMetrics:
    """Metrics for evaluating execution quality."""
    implementation_shortfall: float     # Actual cost vs decision price
    expected_shortfall: float           # Expected cost from model
    slippage: float                     # Price deviation from VWAP
    market_impact: float                # Estimated permanent impact
    timing_cost: float                  # Cost due to price movement during execution
    participation_rate: float           # Our volume / total volume
    realized_volatility: float          # Vol during execution


class AlmgrenChrissExecutor:
    """
    Optimal execution using Almgren-Chriss framework.

    Computes optimal trading trajectory that minimizes the combination
    of expected execution cost and execution risk.

    Usage:
        executor = AlmgrenChrissExecutor(impact_model)

        # Generate optimal plan
        plan = executor.generate_plan(
            total_shares=10000,
            horizon_minutes=60,
            risk_aversion=1e-6,
        )

        # Execute trades according to plan
        for t, trade_size in zip(plan.time_steps[:-1], plan.trade_schedule):
            execute_trade(trade_size)
    """

    def __init__(
        self,
        impact_model: MarketImpactModel,
        min_trade_size: float = 100,
        max_participation_rate: float = 0.10,
    ):
        """
        Initialize executor.

        Args:
            impact_model: Market impact parameters
            min_trade_size: Minimum trade size (shares)
            max_participation_rate: Max fraction of volume to trade
        """
        self.impact = impact_model
        self.min_trade_size = min_trade_size
        self.max_participation_rate = max_participation_rate

    def generate_plan(
        self,
        total_shares: float,
        horizon_minutes: float,
        risk_aversion: float = 1e-6,
        num_periods: Optional[int] = None,
        urgency: Optional[ExecutionUrgency] = None,
    ) -> ExecutionPlan:
        """
        Generate optimal execution plan.

        The optimal trajectory minimizes:
            E[Cost] + λ * Var[Cost]

        Args:
            total_shares: Total shares to trade (negative for sell)
            horizon_minutes: Execution horizon in minutes
            risk_aversion: λ parameter (higher = more aggressive)
            num_periods: Number of trading periods (default: horizon_minutes)
            urgency: Override risk_aversion with urgency level

        Returns:
            ExecutionPlan with optimal trajectory
        """
        X = abs(total_shares)
        if X < EPSILON:
            return self._empty_plan()

        # Convert horizon to trading days
        T = horizon_minutes / (6.5 * 60)  # Fraction of trading day

        # Override risk aversion with urgency if provided
        if urgency is not None:
            risk_aversion = self._urgency_to_lambda(urgency)

        # Number of periods
        N = num_periods if num_periods else max(1, int(horizon_minutes))
        tau = T / N  # Time step size

        # Calculate kappa (urgency parameter)
        # κ = sqrt(λσ²/η)
        sigma = self.impact.volatility
        eta = self.impact.temporary_impact

        if eta < EPSILON:
            eta = EPSILON

        kappa_squared = risk_aversion * sigma**2 / eta
        kappa = np.sqrt(kappa_squared) if kappa_squared > 0 else EPSILON

        # Generate time steps
        time_steps = np.linspace(0, T, N + 1)

        # Calculate optimal holdings using sinh formula
        # x*(t) = X * sinh(κ(T-t)) / sinh(κT)
        sinh_kT = np.sinh(kappa * T)
        if abs(sinh_kT) < EPSILON:
            # Nearly zero kappa means linear liquidation
            holdings = X * (1 - time_steps / T)
        else:
            holdings = X * np.sinh(kappa * (T - time_steps)) / sinh_kT

        # Ensure final holdings are zero
        holdings[-1] = 0.0

        # Calculate trade schedule (differences in holdings)
        trade_schedule = -np.diff(holdings)  # Positive for sells

        # Adjust sign for buys vs sells
        if total_shares < 0:
            trade_schedule = -trade_schedule

        # Calculate expected cost
        # E[Cost] = ½γX² + εX + ηΣ(n_k²/τ)
        gamma = self.impact.permanent_impact
        epsilon = self.impact.fixed_cost

        permanent_cost = 0.5 * gamma * X**2
        fixed_cost = epsilon * X
        temporary_cost = eta * np.sum(trade_schedule**2) / tau
        expected_cost = permanent_cost + fixed_cost + temporary_cost

        # Calculate cost variance
        # Var[Cost] = σ²Σ(x_k²τ)
        cost_variance = sigma**2 * np.sum(holdings[:-1]**2) * tau

        # Risk-adjusted cost
        risk_adjusted_cost = expected_cost + risk_aversion * cost_variance

        return ExecutionPlan(
            time_steps=time_steps,
            holdings=holdings,
            trade_schedule=trade_schedule,
            expected_cost=expected_cost,
            cost_variance=cost_variance,
            risk_adjusted_cost=risk_adjusted_cost,
            kappa=kappa,
        )

    def generate_efficient_frontier(
        self,
        total_shares: float,
        horizon_minutes: float,
        num_points: int = 20,
    ) -> List[Tuple[float, float]]:
        """
        Generate efficient frontier of cost vs risk tradeoff.

        Returns list of (expected_cost, cost_std) pairs for different
        risk aversion levels.
        """
        frontier = []

        # Range of risk aversion from very low to very high
        lambdas = np.logspace(-8, -3, num_points)

        for lam in lambdas:
            plan = self.generate_plan(total_shares, horizon_minutes, lam)
            frontier.append((
                plan.expected_cost,
                np.sqrt(plan.cost_variance),
            ))

        return frontier

    def optimal_horizon(
        self,
        total_shares: float,
        risk_aversion: float = 1e-6,
        max_horizon_minutes: float = 390,  # Full trading day
    ) -> float:
        """
        Calculate optimal execution horizon.

        Finds horizon that minimizes risk-adjusted cost.
        """
        best_horizon = 60.0
        best_cost = float('inf')

        # Search over horizons
        horizons = np.linspace(5, max_horizon_minutes, 50)

        for h in horizons:
            plan = self.generate_plan(total_shares, h, risk_aversion)
            if plan.risk_adjusted_cost < best_cost:
                best_cost = plan.risk_adjusted_cost
                best_horizon = h

        return best_horizon

    def adaptive_execution(
        self,
        remaining_shares: float,
        remaining_time_minutes: float,
        current_price: float,
        arrival_price: float,
        realized_vol: Optional[float] = None,
    ) -> Tuple[float, ExecutionPlan]:
        """
        Adaptive execution that updates based on realized conditions.

        If price has moved favorably, can slow down.
        If price has moved adversely, may need to speed up.

        Args:
            remaining_shares: Shares still to execute
            remaining_time_minutes: Time left in horizon
            current_price: Current market price
            arrival_price: Price at start of execution
            realized_vol: Realized volatility (if available)

        Returns:
            Tuple of (next_trade_size, updated_plan)
        """
        if remaining_time_minutes <= 0 or remaining_shares <= 0:
            return remaining_shares, self._empty_plan()

        # Update volatility estimate if provided
        if realized_vol is not None:
            impact = MarketImpactModel(
                permanent_impact=self.impact.permanent_impact,
                temporary_impact=self.impact.temporary_impact,
                fixed_cost=self.impact.fixed_cost,
                volatility=realized_vol,
            )
            executor = AlmgrenChrissExecutor(impact)
        else:
            executor = self

        # Calculate price move
        price_move = (current_price - arrival_price) / arrival_price

        # Adjust risk aversion based on price move
        # If price moved against us, increase urgency
        base_lambda = 1e-6
        if price_move < -0.005:  # Price dropped (bad for buyer)
            adjusted_lambda = base_lambda * 2  # More aggressive
        elif price_move > 0.005:  # Price rose (good for buyer)
            adjusted_lambda = base_lambda * 0.5  # More passive
        else:
            adjusted_lambda = base_lambda

        # Generate new plan for remaining execution
        plan = executor.generate_plan(
            remaining_shares,
            remaining_time_minutes,
            risk_aversion=adjusted_lambda,
        )

        # Return first trade size
        next_trade = plan.trade_schedule[0] if len(plan.trade_schedule) > 0 else remaining_shares

        return next_trade, plan

    def calculate_execution_metrics(
        self,
        executed_prices: NDArray[np.float64],
        executed_volumes: NDArray[np.float64],
        decision_price: float,
        market_vwap: float,
        plan: ExecutionPlan,
    ) -> ExecutionMetrics:
        """
        Calculate post-execution metrics.

        Args:
            executed_prices: Prices at which trades executed
            executed_volumes: Volumes executed at each price
            decision_price: Price when trade decision was made
            market_vwap: Market VWAP during execution
            plan: Original execution plan

        Returns:
            ExecutionMetrics with performance analysis
        """
        total_volume = np.sum(executed_volumes)
        if total_volume < EPSILON:
            return ExecutionMetrics(
                implementation_shortfall=0.0,
                expected_shortfall=plan.expected_cost,
                slippage=0.0,
                market_impact=0.0,
                timing_cost=0.0,
                participation_rate=0.0,
                realized_volatility=0.0,
            )

        # Execution VWAP
        exec_vwap = np.sum(executed_prices * executed_volumes) / total_volume

        # Implementation shortfall
        impl_shortfall = (exec_vwap - decision_price) / decision_price

        # Slippage vs market VWAP
        slippage = (exec_vwap - market_vwap) / market_vwap

        # Estimate market impact (permanent)
        final_price = executed_prices[-1]
        market_impact = (final_price - decision_price) / decision_price

        # Timing cost (cost due to delay)
        timing_cost = (market_vwap - decision_price) / decision_price

        # Realized volatility during execution
        if len(executed_prices) > 1:
            price_returns = np.diff(executed_prices) / executed_prices[:-1]
            realized_vol = np.std(price_returns) * SQRT_252
        else:
            realized_vol = self.impact.volatility

        return ExecutionMetrics(
            implementation_shortfall=impl_shortfall,
            expected_shortfall=plan.expected_cost,
            slippage=slippage,
            market_impact=market_impact,
            timing_cost=timing_cost,
            participation_rate=0.0,  # Would need market volume data
            realized_volatility=realized_vol,
        )

    def _urgency_to_lambda(self, urgency: ExecutionUrgency) -> float:
        """Convert urgency level to risk aversion parameter."""
        mapping = {
            ExecutionUrgency.PASSIVE: 1e-8,
            ExecutionUrgency.NEUTRAL: 1e-6,
            ExecutionUrgency.AGGRESSIVE: 1e-4,
            ExecutionUrgency.IMMEDIATE: 1e-2,
        }
        return mapping.get(urgency, 1e-6)

    def _empty_plan(self) -> ExecutionPlan:
        """Return empty execution plan."""
        return ExecutionPlan(
            time_steps=np.array([0.0]),
            holdings=np.array([0.0]),
            trade_schedule=np.array([]),
            expected_cost=0.0,
            cost_variance=0.0,
            risk_adjusted_cost=0.0,
            kappa=0.0,
        )


def estimate_market_impact(
    trade_size: float,
    avg_daily_volume: float,
    volatility: float,
    is_buy: bool = True,
) -> Tuple[float, float]:
    """
    Quick estimate of market impact using square-root law.

    Impact ≈ σ * √(Q/V) * sign

    Args:
        trade_size: Order size in shares
        avg_daily_volume: Average daily volume
        volatility: Daily volatility
        is_buy: True for buy, False for sell

    Returns:
        Tuple of (temporary_impact, permanent_impact) as percentages
    """
    if avg_daily_volume < EPSILON:
        return 0.0, 0.0

    participation = trade_size / avg_daily_volume
    sign = 1 if is_buy else -1

    # Square-root law for temporary impact
    temp_impact = sign * volatility * np.sqrt(participation)

    # Permanent impact is typically 30-50% of temporary
    perm_impact = 0.4 * temp_impact

    return float(temp_impact), float(perm_impact)
