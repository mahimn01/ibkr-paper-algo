"""
Trading Context Abstraction

Provides a unified interface for both live trading and backtesting.
This abstraction allows the same trading logic to run in either mode
without modification.

The context handles:
    - Market data retrieval
    - Order submission
    - Position tracking
    - Account management
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol, Callable
from datetime import datetime, timedelta
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging

from trading_algo.quant_core.utils.constants import EPSILON


logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Order type."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class OrderStatus(Enum):
    """Order status."""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


@dataclass
class MarketData:
    """
    Market data snapshot for a single asset.

    Contains OHLCV data plus optional quote data.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    @property
    def mid(self) -> float:
        """Mid price (average of bid/ask or close if not available)."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.close

    @property
    def spread(self) -> float:
        """Bid-ask spread as decimal."""
        if self.bid is not None and self.ask is not None and self.mid > 0:
            return (self.ask - self.bid) / self.mid
        return 0.001  # Default 10bps


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float           # Positive = long, negative = short
    avg_cost: float           # Average cost basis
    current_price: float      # Current market price
    unrealized_pnl: float     # Unrealized P&L
    realized_pnl: float       # Realized P&L for the day

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < EPSILON


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass
class AccountInfo:
    """Account information."""
    equity: float                    # Total equity
    cash: float                      # Available cash
    buying_power: float              # Buying power
    margin_used: float               # Margin in use
    unrealized_pnl: float            # Total unrealized P&L
    realized_pnl: float              # Total realized P&L for day
    positions: Dict[str, Position] = field(default_factory=dict)


class TradingContext(ABC):
    """
    Abstract base class for trading contexts.

    Provides unified interface for both live and backtest modes.
    """

    @abstractmethod
    def get_current_time(self) -> datetime:
        """Get current time in the context."""
        pass

    @abstractmethod
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol."""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        lookback: int,
        include_current: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        """
        Get historical price data.

        Returns array of shape (lookback, 5) with OHLCV data.
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        pass

    @abstractmethod
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get current account information."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Submit an order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        pass

    @abstractmethod
    def advance(self) -> bool:
        """
        Advance to next bar (for backtesting).

        For live context, this waits for next data.
        Returns False when no more data available.
        """
        pass


class BacktestContext(TradingContext):
    """
    Backtesting context implementation.

    Simulates trading using historical data with realistic
    execution assumptions.
    """

    def __init__(
        self,
        historical_data: Dict[str, NDArray[np.float64]],
        timestamps: NDArray,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,      # 10bps
        slippage_rate: float = 0.0005,       # 5bps
        fill_ratio: float = 1.0,             # Full fills
    ):
        """
        Initialize backtest context.

        Args:
            historical_data: Dict of symbol -> OHLCV arrays (T, 5)
            timestamps: Array of timestamps
            initial_capital: Starting capital
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of price
            fill_ratio: Fraction of order that fills (1.0 = full)
        """
        self.historical_data = historical_data
        self.timestamps = timestamps
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.fill_ratio = fill_ratio

        # State
        self.current_bar = 0
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.trade_log: List[Dict] = []       # Fill events
        self.closed_trades: List[Dict] = []   # Round-trip trades with realized PnL
        self._open_lots: Dict[str, List[Dict[str, Any]]] = {}

        # Symbols in universe
        self.symbols = list(historical_data.keys())
        self.n_bars = len(timestamps)

    def get_current_time(self) -> datetime:
        if self.current_bar < len(self.timestamps):
            ts = self.timestamps[self.current_bar]
            if isinstance(ts, (int, float, np.integer, np.floating)):
                return datetime.fromtimestamp(float(ts))
            return ts
        return datetime.now()

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        if symbol not in self.historical_data:
            return None

        data = self.historical_data[symbol]
        if self.current_bar >= len(data):
            return None

        bar = data[self.current_bar]
        return MarketData(
            symbol=symbol,
            timestamp=self.get_current_time(),
            open=float(bar[0]),
            high=float(bar[1]),
            low=float(bar[2]),
            close=float(bar[3]),
            volume=float(bar[4]) if len(bar) > 4 else 0.0,
        )

    def get_historical_data(
        self,
        symbol: str,
        lookback: int,
        include_current: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        if symbol not in self.historical_data:
            return None

        data = self.historical_data[symbol]
        end_idx = self.current_bar + (1 if include_current else 0)
        start_idx = max(0, end_idx - lookback)

        if start_idx >= end_idx:
            return None

        return data[start_idx:end_idx].copy()

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        return self.positions.copy()

    def get_account_info(self) -> AccountInfo:
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(p.realized_pnl for p in self.positions.values())
        total_market_value = sum(p.market_value for p in self.positions.values())

        equity = self.cash + total_market_value

        return AccountInfo(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash,  # Simplified: no margin
            margin_used=0.0,
            unrealized_pnl=total_unrealized,
            realized_pnl=total_realized,
            positions=self.positions.copy(),
        )

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        self.order_counter += 1
        order_id = f"BT_{self.order_counter}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.SUBMITTED,
            submitted_time=self.get_current_time(),
        )

        # For market orders, execute immediately
        if order_type == OrderType.MARKET:
            self._execute_order(order)
        else:
            self.orders[order_id] = order

        return order

    def _execute_order(self, order: Order) -> None:
        """Execute an order with slippage and commission."""
        market_data = self.get_market_data(order.symbol)
        if market_data is None:
            order.status = OrderStatus.REJECTED
            return

        # Base execution price
        if order.side == OrderSide.BUY:
            base_price = market_data.close * (1 + self.slippage_rate)
        else:
            base_price = market_data.close * (1 - self.slippage_rate)

        # Apply limit price constraint
        if order.limit_price is not None:
            if order.side == OrderSide.BUY and base_price > order.limit_price:
                return  # Don't fill
            if order.side == OrderSide.SELL and base_price < order.limit_price:
                return  # Don't fill

        # Calculate fill
        fill_qty = order.quantity * self.fill_ratio
        fill_value = fill_qty * base_price
        commission = fill_value * self.commission_rate
        slippage = abs(base_price - market_data.close) * fill_qty

        # Update order
        order.filled_quantity = fill_qty
        order.avg_fill_price = base_price
        order.filled_time = self.get_current_time()
        order.status = OrderStatus.FILLED

        # Update position
        self._update_position(order, base_price, commission, slippage)

        # Log trade
        self.trade_log.append({
            'timestamp': self.get_current_time(),
            'symbol': order.symbol,
            'side': order.side.name,
            'quantity': fill_qty,
            'price': base_price,
            'commission': commission,
            'slippage': slippage,
            'fill_value': fill_value,
            'bar_index': self.current_bar,
            'order_id': order.order_id,
        })

    def _update_position(
        self,
        order: Order,
        fill_price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """Update position after fill."""
        symbol = order.symbol
        fill_qty = order.filled_quantity

        if order.side == OrderSide.SELL:
            fill_qty = -fill_qty

        self._match_lots(
            symbol=symbol,
            signed_qty=fill_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=self.get_current_time(),
        )

        # Get or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + fill_qty

            if abs(new_qty) < EPSILON:
                # Position closed
                del self.positions[symbol]
            elif (old_qty > 0 and fill_qty < 0) or (old_qty < 0 and fill_qty > 0):
                # Reducing position
                close_qty = min(abs(old_qty), abs(fill_qty))
                realized = (fill_price - pos.avg_cost) * close_qty * np.sign(old_qty)
                pos.quantity = new_qty
                pos.realized_pnl += realized
            else:
                # Adding to position
                total_cost = pos.avg_cost * abs(old_qty) + fill_price * abs(fill_qty)
                pos.quantity = new_qty
                pos.avg_cost = total_cost / abs(new_qty)
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=fill_qty,
                avg_cost=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

        # Update cash
        trade_value = abs(fill_qty) * fill_price
        if order.side == OrderSide.BUY:
            self.cash -= trade_value + commission
        else:
            self.cash += trade_value - commission

    def _match_lots(
        self,
        symbol: str,
        signed_qty: float,
        fill_price: float,
        commission: float,
        slippage: float,
        timestamp: datetime,
    ) -> None:
        """Match fills against open lots and emit realized round-trip trades."""
        if abs(signed_qty) < EPSILON:
            return

        total_qty = abs(signed_qty)
        commission_per_share = commission / total_qty if total_qty > EPSILON else 0.0
        slippage_per_share = slippage / total_qty if total_qty > EPSILON else 0.0

        lots = self._open_lots.setdefault(symbol, [])
        remaining = signed_qty

        while (
            abs(remaining) > EPSILON and lots and
            np.sign(lots[0]["qty"]) != np.sign(remaining)
        ):
            lot = lots[0]
            lot_sign = 1.0 if lot["qty"] > 0 else -1.0
            match_qty = min(abs(remaining), abs(lot["qty"]))

            gross_pnl = (fill_price - lot["entry_price"]) * match_qty * lot_sign
            entry_cost = (
                lot["commission_per_share"] * match_qty +
                lot["slippage_per_share"] * match_qty
            )
            exit_cost = (
                commission_per_share * match_qty +
                slippage_per_share * match_qty
            )
            net_pnl = gross_pnl - entry_cost - exit_cost

            self.closed_trades.append({
                "symbol": symbol,
                "direction": "LONG" if lot_sign > 0 else "SHORT",
                "quantity": float(match_qty),
                "entry_time": lot["entry_time"],
                "exit_time": timestamp,
                "entry_price": float(lot["entry_price"]),
                "exit_price": float(fill_price),
                "gross_pnl": float(gross_pnl),
                "commission": float(entry_cost + exit_cost),
                "slippage": float(
                    lot["slippage_per_share"] * match_qty +
                    slippage_per_share * match_qty
                ),
                "pnl": float(net_pnl),
                "bars_held": max(0, self.current_bar - lot["entry_bar"]),
            })

            lot["qty"] -= lot_sign * match_qty
            remaining -= np.sign(remaining) * match_qty

            if abs(lot["qty"]) < EPSILON:
                lots.pop(0)

        if abs(remaining) > EPSILON:
            lots.append({
                "qty": float(remaining),
                "entry_price": float(fill_price),
                "entry_time": timestamp,
                "entry_bar": int(self.current_bar),
                "commission_per_share": float(commission_per_share),
                "slippage_per_share": float(slippage_per_share),
            })

        if not lots:
            self._open_lots.pop(symbol, None)

    def _infer_periods_per_year(self) -> float:
        """Infer annualization factor from timestamp spacing."""
        if len(self.timestamps) < 3:
            return 252.0

        sample = self.timestamps[: min(len(self.timestamps), 400)]
        dt_list: List[datetime] = []
        for ts in sample:
            if isinstance(ts, (int, float, np.integer, np.floating)):
                dt_list.append(datetime.fromtimestamp(float(ts)))
            else:
                dt_list.append(ts)

        deltas = [
            (dt_list[i] - dt_list[i - 1]).total_seconds()
            for i in range(1, len(dt_list))
            if (dt_list[i] - dt_list[i - 1]).total_seconds() > 0
        ]
        if not deltas:
            return 252.0

        intraday = [d for d in deltas if d <= 2 * 3600]
        if intraday:
            step = float(np.median(intraday))
            periods_per_day = max(1.0, min(390.0, 23400.0 / max(step, 1.0)))
            return periods_per_day * 252.0

        step_days = float(np.median([d / 86400.0 for d in deltas]))
        if step_days <= 1.5:
            return 252.0 / max(step_days, 1e-9)
        if step_days <= 8.0:
            return 52.0 / max(step_days / 7.0, 1e-9)
        return max(1.0, 365.0 / step_days)

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            del self.orders[order_id]
            return True
        return False

    def get_open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if not o.is_complete]

    def is_market_open(self) -> bool:
        # In backtest, market is always "open" during data
        return self.current_bar < self.n_bars

    def advance(self) -> bool:
        """Advance to next bar."""
        # Process pending orders
        for order in list(self.orders.values()):
            if not order.is_complete:
                self._execute_order(order)

        # Update position prices
        for symbol, pos in self.positions.items():
            market_data = self.get_market_data(symbol)
            if market_data:
                pos.current_price = market_data.close
                pos.unrealized_pnl = (pos.current_price - pos.avg_cost) * pos.quantity

        # Record equity
        account = self.get_account_info()
        self.equity_curve.append(account.equity)

        # Advance bar
        self.current_bar += 1
        return self.current_bar < self.n_bars

    def get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        periods_per_year = self._infer_periods_per_year()
        total_return = (equity[-1] / self.initial_capital - 1) if self.initial_capital > 0 else 0.0

        if len(returns) > 1:
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))
            downside = returns[returns < 0]
            downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        else:
            mean_ret = 0.0
            std_ret = 0.0
            downside_std = 0.0

        sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > EPSILON else 0.0
        sortino = (
            mean_ret / downside_std * np.sqrt(periods_per_year)
            if downside_std > EPSILON else 0.0
        )

        # Max drawdown from cumulative equity
        if len(returns) > 0:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            dd_series = 1 - cumulative / np.maximum(running_max, EPSILON)
            max_dd = float(np.max(dd_series))
        else:
            max_dd = 0.0

        years = (len(returns) / periods_per_year) if periods_per_year > 0 else 0.0
        if years > 0 and (1 + total_return) > 0:
            annualized_return = float((1 + total_return) ** (1 / years) - 1)
        else:
            annualized_return = -1.0 if total_return <= -1 else 0.0

        calmar = annualized_return / max_dd if max_dd > EPSILON else 0.0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': equity[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0.0,
            'periods_per_year': periods_per_year,
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'max_drawdown': float(max_dd),
            'n_trades': len(self.closed_trades),
            'n_fills': len(self.trade_log),
            'equity_curve': equity,
            'returns': returns,
            'trades': self.closed_trades,
            'fills': self.trade_log,
        }


class LiveContext(TradingContext):
    """
    Live trading context for IBKR integration.

    Wraps IBKR broker interface to provide unified context.
    """

    def __init__(
        self,
        ibkr_broker,
        data_provider,
        universe: List[str],
    ):
        """
        Initialize live context.

        Args:
            ibkr_broker: IBKR broker instance
            data_provider: Data provider for historical data
            universe: List of symbols to trade
        """
        self.broker = ibkr_broker
        self.data_provider = data_provider
        self.universe = universe

        # Cache for market data
        self._market_data_cache: Dict[str, MarketData] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=1)

    def get_current_time(self) -> datetime:
        return datetime.now()

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        # Check cache
        now = datetime.now()
        if self._cache_time and (now - self._cache_time) < self._cache_ttl:
            if symbol in self._market_data_cache:
                return self._market_data_cache[symbol]

        try:
            # Fetch from broker
            quote = self.broker.get_quote(symbol)
            if quote is None:
                return None

            data = MarketData(
                symbol=symbol,
                timestamp=now,
                open=quote.get('open', quote.get('last', 0)),
                high=quote.get('high', quote.get('last', 0)),
                low=quote.get('low', quote.get('last', 0)),
                close=quote.get('last', 0),
                volume=quote.get('volume', 0),
                bid=quote.get('bid'),
                ask=quote.get('ask'),
                bid_size=quote.get('bid_size'),
                ask_size=quote.get('ask_size'),
            )

            self._market_data_cache[symbol] = data
            self._cache_time = now
            return data

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        lookback: int,
        include_current: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        try:
            return self.data_provider.get_historical_bars(
                symbol, lookback, include_current
            )
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.get_all_positions()
        return positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        try:
            broker_positions = self.broker.get_positions()
            positions = {}

            for pos in broker_positions:
                symbol = pos.get('symbol')
                if symbol:
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=pos.get('quantity', 0),
                        avg_cost=pos.get('avg_cost', 0),
                        current_price=pos.get('current_price', 0),
                        unrealized_pnl=pos.get('unrealized_pnl', 0),
                        realized_pnl=pos.get('realized_pnl', 0),
                    )

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_account_info(self) -> AccountInfo:
        try:
            account = self.broker.get_account()
            positions = self.get_all_positions()

            return AccountInfo(
                equity=account.get('equity', 0),
                cash=account.get('cash', 0),
                buying_power=account.get('buying_power', 0),
                margin_used=account.get('margin_used', 0),
                unrealized_pnl=account.get('unrealized_pnl', 0),
                realized_pnl=account.get('realized_pnl', 0),
                positions=positions,
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return AccountInfo(
                equity=0, cash=0, buying_power=0, margin_used=0,
                unrealized_pnl=0, realized_pnl=0
            )

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        try:
            # Convert to IBKR order
            ibkr_side = "BUY" if side == OrderSide.BUY else "SELL"
            ibkr_type = {
                OrderType.MARKET: "MKT",
                OrderType.LIMIT: "LMT",
                OrderType.STOP: "STP",
                OrderType.STOP_LIMIT: "STP LMT",
            }.get(order_type, "MKT")

            result = self.broker.place_order(
                symbol=symbol,
                action=ibkr_side,
                quantity=int(quantity),
                order_type=ibkr_type,
                limit_price=limit_price,
                stop_price=stop_price,
            )

            return Order(
                order_id=str(result.get('order_id', '')),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                status=OrderStatus.SUBMITTED,
                submitted_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return Order(
                order_id="",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                status=OrderStatus.REJECTED,
            )

    def cancel_order(self, order_id: str) -> bool:
        try:
            return self.broker.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_open_orders(self) -> List[Order]:
        try:
            broker_orders = self.broker.get_open_orders()
            orders = []

            for o in broker_orders:
                orders.append(Order(
                    order_id=str(o.get('order_id', '')),
                    symbol=o.get('symbol', ''),
                    side=OrderSide.BUY if o.get('side') == 'BUY' else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=o.get('quantity', 0),
                    status=OrderStatus.SUBMITTED,
                ))

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def is_market_open(self) -> bool:
        try:
            return self.broker.is_market_open()
        except Exception:
            # Fallback: check time
            now = datetime.now()
            # US market hours (simplified)
            return (
                now.weekday() < 5 and
                9 <= now.hour < 16
            )

    def advance(self) -> bool:
        """
        For live context, wait for next bar.

        Returns True if we should continue trading.
        """
        import time
        time.sleep(1)  # Short sleep to prevent tight loop
        return self.is_market_open()
