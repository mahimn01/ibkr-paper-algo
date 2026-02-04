"""
Execution Manager

Handles optimal order execution using:
    - Almgren-Chriss optimal execution
    - TWAP/VWAP as fallbacks
    - Market impact estimation
    - Execution quality monitoring

Responsibilities:
    - Convert target positions to orders
    - Optimize execution timing
    - Monitor fills and slippage
    - Track execution costs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import logging

from trading_algo.quant_core.utils.constants import EPSILON
from trading_algo.quant_core.execution.almgren_chriss import (
    AlmgrenChrissExecutor,
    MarketImpactModel,
    ExecutionPlan,
    ExecutionUrgency,
)
from trading_algo.quant_core.execution.twap_vwap import TWAPExecutor, VWAPExecutor
from trading_algo.quant_core.engine.portfolio_manager import TargetPosition
from trading_algo.quant_core.engine.trading_context import (
    TradingContext, Order, OrderSide, OrderType, OrderStatus
)


logger = logging.getLogger(__name__)


class ExecutionMethod(Enum):
    """Execution algorithm."""
    MARKET = auto()           # Immediate market order
    LIMIT = auto()            # Limit order
    ALMGREN_CHRISS = auto()   # Optimal execution
    TWAP = auto()             # Time-weighted
    VWAP = auto()             # Volume-weighted
    POV = auto()              # Percentage of volume


@dataclass
class OrderRequest:
    """Request to execute an order."""
    symbol: str
    shares: float             # Positive = buy, negative = sell
    urgency: ExecutionUrgency
    method: ExecutionMethod
    limit_price: Optional[float] = None
    max_slippage_bps: float = 50  # Max slippage in basis points
    horizon_minutes: float = 60   # Execution horizon

    @property
    def side(self) -> OrderSide:
        return OrderSide.BUY if self.shares > 0 else OrderSide.SELL

    @property
    def quantity(self) -> float:
        return abs(self.shares)


@dataclass
class ExecutionResult:
    """Result of order execution."""
    order_request: OrderRequest
    orders_submitted: List[Order]
    total_filled: float
    avg_fill_price: float
    slippage_bps: float       # Actual slippage
    execution_cost: float     # Total cost (commission + impact)
    completion_time: float    # Time to complete in minutes
    success: bool


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    # Default method
    default_method: ExecutionMethod = ExecutionMethod.ALMGREN_CHRISS

    # Almgren-Chriss parameters
    default_urgency: ExecutionUrgency = ExecutionUrgency.NEUTRAL
    default_horizon_minutes: float = 30
    risk_aversion: float = 1e-6

    # Market impact estimates (defaults)
    default_permanent_impact: float = 0.0001  # 1bp
    default_temporary_impact: float = 0.0005  # 5bp
    default_spread: float = 0.001             # 10bp

    # Execution limits
    max_order_size: float = 10000             # Max shares per order
    min_order_value: float = 100              # Min trade value
    max_slippage_bps: float = 100             # Max acceptable slippage

    # Timing
    check_interval_seconds: float = 5         # Order check interval


class ExecutionManager:
    """
    Manages order execution with optimal algorithms.

    Converts target positions into executable orders and
    monitors execution quality.

    Usage:
        manager = ExecutionManager(config, context)

        # Execute target portfolio
        for symbol, pos in target_portfolio.positions.items():
            if abs(pos.delta_shares) > 10:  # Min trade size
                result = manager.execute(OrderRequest(
                    symbol=symbol,
                    shares=pos.delta_shares,
                    urgency=ExecutionUrgency.NEUTRAL,
                    method=ExecutionMethod.ALMGREN_CHRISS,
                ))
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        context: Optional[TradingContext] = None,
    ):
        """
        Initialize execution manager.

        Args:
            config: Execution configuration
            context: Trading context (live or backtest)
        """
        self.config = config or ExecutionConfig()
        self.context = context

        # Initialize executors
        self.ac_executors: Dict[str, AlmgrenChrissExecutor] = {}
        self.twap = TWAPExecutor()
        self.vwap = VWAPExecutor()

        # Execution tracking
        self._pending_orders: Dict[str, List[Order]] = {}
        self._execution_history: List[ExecutionResult] = []

    def set_context(self, context: TradingContext) -> None:
        """Set trading context."""
        self.context = context

    def execute(
        self,
        request: OrderRequest,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute an order request.

        Args:
            request: Order request details
            market_data: Optional market data for impact estimation

        Returns:
            ExecutionResult with fill details
        """
        if self.context is None:
            raise ValueError("Trading context not set")

        if abs(request.shares) < EPSILON:
            return self._empty_result(request)

        # Get current market data
        current_data = self.context.get_market_data(request.symbol)
        if current_data is None:
            logger.error(f"No market data for {request.symbol}")
            return self._empty_result(request)

        arrival_price = current_data.close

        # Select execution method
        if request.method == ExecutionMethod.MARKET:
            result = self._execute_market(request, arrival_price)

        elif request.method == ExecutionMethod.LIMIT:
            result = self._execute_limit(request, arrival_price)

        elif request.method == ExecutionMethod.ALMGREN_CHRISS:
            result = self._execute_almgren_chriss(request, arrival_price, market_data)

        elif request.method == ExecutionMethod.TWAP:
            result = self._execute_twap(request, arrival_price)

        elif request.method == ExecutionMethod.VWAP:
            result = self._execute_vwap(request, arrival_price)

        else:
            # Default to market order
            result = self._execute_market(request, arrival_price)

        self._execution_history.append(result)
        return result

    def _execute_market(
        self,
        request: OrderRequest,
        arrival_price: float,
    ) -> ExecutionResult:
        """Execute as immediate market order."""
        order = self.context.submit_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=OrderType.MARKET,
        )

        # Calculate slippage (for backtest, this is already included)
        fill_price = order.avg_fill_price if order.avg_fill_price > 0 else arrival_price
        slippage_bps = abs(fill_price - arrival_price) / arrival_price * 10000

        return ExecutionResult(
            order_request=request,
            orders_submitted=[order],
            total_filled=order.filled_quantity,
            avg_fill_price=fill_price,
            slippage_bps=slippage_bps,
            execution_cost=slippage_bps * request.quantity * fill_price / 10000,
            completion_time=0.0,
            success=order.status == OrderStatus.FILLED,
        )

    def _execute_limit(
        self,
        request: OrderRequest,
        arrival_price: float,
    ) -> ExecutionResult:
        """Execute as limit order."""
        limit_price = request.limit_price
        if limit_price is None:
            # Set limit at favorable price
            slippage = request.max_slippage_bps / 10000
            if request.side == OrderSide.BUY:
                limit_price = arrival_price * (1 + slippage)
            else:
                limit_price = arrival_price * (1 - slippage)

        order = self.context.submit_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
        )

        fill_price = order.avg_fill_price if order.avg_fill_price > 0 else limit_price
        slippage_bps = abs(fill_price - arrival_price) / arrival_price * 10000

        return ExecutionResult(
            order_request=request,
            orders_submitted=[order],
            total_filled=order.filled_quantity,
            avg_fill_price=fill_price,
            slippage_bps=slippage_bps,
            execution_cost=slippage_bps * request.quantity * fill_price / 10000,
            completion_time=0.0,
            success=order.filled_quantity > 0,
        )

    def _execute_almgren_chriss(
        self,
        request: OrderRequest,
        arrival_price: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute using Almgren-Chriss optimal execution."""
        symbol = request.symbol

        # For small orders (< 100 shares), just execute as market order
        # Almgren-Chriss is overkill for retail-size orders
        if abs(request.shares) < 100:
            return self._execute_market(request, arrival_price)

        # Get or create executor
        if symbol not in self.ac_executors:
            impact_model = self._estimate_impact_model(symbol, market_data)
            self.ac_executors[symbol] = AlmgrenChrissExecutor(impact_model)

        executor = self.ac_executors[symbol]

        # Use fewer periods for moderate-size orders
        num_periods = max(1, min(int(request.shares / 10), int(request.horizon_minutes)))

        # Generate execution plan
        plan = executor.generate_plan(
            total_shares=request.shares,
            horizon_minutes=request.horizon_minutes,
            num_periods=num_periods,
            urgency=request.urgency,
        )

        # Execute according to plan
        orders_submitted = []
        total_filled = 0.0
        total_value = 0.0
        accumulated_shares = 0.0

        for i, trade_size in enumerate(plan.trade_schedule):
            # Accumulate small trades
            accumulated_shares += trade_size

            # Only submit when we have at least 1 share accumulated
            if abs(accumulated_shares) < 1 and i < len(plan.trade_schedule) - 1:
                continue

            if abs(accumulated_shares) < 0.5:
                continue

            # Submit order for accumulated shares
            side = OrderSide.BUY if accumulated_shares > 0 else OrderSide.SELL
            order = self.context.submit_order(
                symbol=symbol,
                side=side,
                quantity=abs(accumulated_shares),
                order_type=OrderType.MARKET,
            )

            orders_submitted.append(order)
            total_filled += order.filled_quantity
            total_value += order.filled_quantity * order.avg_fill_price
            accumulated_shares = 0.0

            # In live trading, would wait for next time slice
            # In backtest, this executes immediately

        avg_fill = total_value / total_filled if total_filled > 0 else arrival_price
        slippage_bps = abs(avg_fill - arrival_price) / arrival_price * 10000

        return ExecutionResult(
            order_request=request,
            orders_submitted=orders_submitted,
            total_filled=total_filled,
            avg_fill_price=avg_fill,
            slippage_bps=slippage_bps,
            execution_cost=plan.expected_cost,
            completion_time=plan.completion_time * 6.5 * 60,  # Convert to minutes
            success=total_filled >= request.quantity * 0.95,
        )

    def _execute_twap(
        self,
        request: OrderRequest,
        arrival_price: float,
    ) -> ExecutionResult:
        """Execute using TWAP algorithm."""
        # Generate TWAP schedule
        schedule = self.twap.generate_schedule(
            total_shares=request.shares,
            horizon_minutes=request.horizon_minutes,
            interval_minutes=5,
        )

        orders_submitted = []
        total_filled = 0.0
        total_value = 0.0

        for trade_size in schedule.trade_sizes:
            if abs(trade_size) < 1:
                continue

            side = OrderSide.BUY if trade_size > 0 else OrderSide.SELL
            order = self.context.submit_order(
                symbol=request.symbol,
                side=side,
                quantity=abs(trade_size),
                order_type=OrderType.MARKET,
            )

            orders_submitted.append(order)
            total_filled += order.filled_quantity
            total_value += order.filled_quantity * order.avg_fill_price

        avg_fill = total_value / total_filled if total_filled > 0 else arrival_price
        slippage_bps = abs(avg_fill - arrival_price) / arrival_price * 10000

        return ExecutionResult(
            order_request=request,
            orders_submitted=orders_submitted,
            total_filled=total_filled,
            avg_fill_price=avg_fill,
            slippage_bps=slippage_bps,
            execution_cost=slippage_bps * total_value / 10000,
            completion_time=request.horizon_minutes,
            success=total_filled >= request.quantity * 0.95,
        )

    def _execute_vwap(
        self,
        request: OrderRequest,
        arrival_price: float,
    ) -> ExecutionResult:
        """Execute using VWAP algorithm."""
        # Use default volume profile
        schedule = self.vwap.generate_schedule_with_default_profile(
            total_shares=request.shares,
            horizon_minutes=request.horizon_minutes,
        )

        orders_submitted = []
        total_filled = 0.0
        total_value = 0.0

        for trade_size in schedule.trade_sizes:
            if abs(trade_size) < 1:
                continue

            side = OrderSide.BUY if trade_size > 0 else OrderSide.SELL
            order = self.context.submit_order(
                symbol=request.symbol,
                side=side,
                quantity=abs(trade_size),
                order_type=OrderType.MARKET,
            )

            orders_submitted.append(order)
            total_filled += order.filled_quantity
            total_value += order.filled_quantity * order.avg_fill_price

        avg_fill = total_value / total_filled if total_filled > 0 else arrival_price
        slippage_bps = abs(avg_fill - arrival_price) / arrival_price * 10000

        return ExecutionResult(
            order_request=request,
            orders_submitted=orders_submitted,
            total_filled=total_filled,
            avg_fill_price=avg_fill,
            slippage_bps=slippage_bps,
            execution_cost=slippage_bps * total_value / 10000,
            completion_time=request.horizon_minutes,
            success=total_filled >= request.quantity * 0.95,
        )

    def _estimate_impact_model(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> MarketImpactModel:
        """Estimate market impact model for a symbol."""
        if market_data is None:
            # Use defaults
            return MarketImpactModel(
                permanent_impact=self.config.default_permanent_impact,
                temporary_impact=self.config.default_temporary_impact,
                fixed_cost=self.config.default_spread / 2,
                volatility=0.02,  # 2% daily vol default
            )

        # Estimate from market data
        adv = market_data.get('avg_daily_volume', 1000000)
        spread = market_data.get('spread', self.config.default_spread)
        volatility = market_data.get('volatility', 0.02)

        return MarketImpactModel.estimate_from_data(
            avg_daily_volume=adv,
            avg_spread=spread,
            volatility=volatility,
        )

    def _empty_result(self, request: OrderRequest) -> ExecutionResult:
        """Return empty execution result."""
        return ExecutionResult(
            order_request=request,
            orders_submitted=[],
            total_filled=0.0,
            avg_fill_price=0.0,
            slippage_bps=0.0,
            execution_cost=0.0,
            completion_time=0.0,
            success=True,  # Zero-size order is "successful"
        )

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {}

        total_trades = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        avg_slippage = np.mean([r.slippage_bps for r in self._execution_history])
        total_cost = sum(r.execution_cost for r in self._execution_history)

        return {
            'total_trades': total_trades,
            'success_rate': successful / total_trades if total_trades > 0 else 0,
            'avg_slippage_bps': avg_slippage,
            'total_execution_cost': total_cost,
        }

    def cancel_all_pending(self) -> int:
        """Cancel all pending orders."""
        cancelled = 0

        for symbol, orders in self._pending_orders.items():
            for order in orders:
                if self.context.cancel_order(order.order_id):
                    cancelled += 1

        self._pending_orders.clear()
        return cancelled
