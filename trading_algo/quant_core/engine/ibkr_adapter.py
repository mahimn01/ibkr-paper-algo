"""
IBKR Broker Adapter

Adapts the IBKR broker interface to work with the QuantOrchestrator.
Handles:
    - Connection management
    - Order submission and tracking
    - Position and account retrieval
    - Market data subscription

This adapter wraps the existing IBKR broker implementation to
provide a clean interface for the trading engine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import time

from trading_algo.quant_core.engine.trading_context import (
    LiveContext, MarketData, Position, Order, OrderSide, OrderType
)


logger = logging.getLogger(__name__)


@dataclass
class IBKRConfig:
    """IBKR connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading port (7496 for live)
    client_id: int = 1
    account: str = ""

    # Timeouts
    connect_timeout: int = 30
    order_timeout: int = 60
    data_timeout: int = 10

    # Rate limiting
    max_requests_per_second: float = 45.0
    order_delay_ms: int = 100


class IBKRDataProvider:
    """
    Market data provider using IBKR.

    Fetches historical and real-time market data.
    """

    def __init__(self, ibkr_broker):
        """
        Initialize data provider.

        Args:
            ibkr_broker: IBKR broker instance
        """
        self.broker = ibkr_broker
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=5)

    def get_historical_bars(
        self,
        symbol: str,
        lookback: int,
        include_current: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Asset symbol
            lookback: Number of bars
            include_current: Include current bar

        Returns:
            Array of shape (lookback, 5) with OHLCV
        """
        try:
            # Calculate duration string for IBKR
            if lookback <= 30:
                duration = f"{lookback} D"
            elif lookback <= 252:
                duration = f"{lookback // 20} M"
            else:
                duration = f"{lookback // 252} Y"

            bars = self.broker.get_historical_data(
                symbol=symbol,
                duration=duration,
                bar_size="1 day",
                what_to_show="TRADES",
            )

            if bars is None or len(bars) == 0:
                return None

            # Convert to numpy array
            data = np.array([
                [b['open'], b['high'], b['low'], b['close'], b['volume']]
                for b in bars
            ])

            if not include_current:
                data = data[:-1]

            return data[-lookback:]

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote."""
        # Check cache
        now = datetime.now()
        if symbol in self._cache:
            if (now - self._cache_time.get(symbol, datetime.min)) < self._cache_ttl:
                return self._cache[symbol]

        try:
            quote = self.broker.get_quote(symbol)
            if quote:
                self._cache[symbol] = quote
                self._cache_time[symbol] = now
            return quote

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None


class IBKRBrokerAdapter:
    """
    Adapter for IBKR broker operations.

    Provides a clean interface for the trading engine to interact
    with Interactive Brokers.
    """

    def __init__(
        self,
        config: Optional[IBKRConfig] = None,
        broker_instance = None,
    ):
        """
        Initialize adapter.

        Args:
            config: IBKR configuration
            broker_instance: Existing broker instance (optional)
        """
        self.config = config or IBKRConfig()
        self._broker = broker_instance
        self._connected = False
        self._order_counter = 0
        self._pending_orders: Dict[str, Dict] = {}

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / self.config.max_requests_per_second

    def connect(self) -> bool:
        """
        Connect to IBKR.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        try:
            if self._broker is None:
                # Import and create broker
                from trading_algo.broker.ibkr import IBKRBroker
                self._broker = IBKRBroker(
                    host=self.config.host,
                    port=self.config.port,
                    client_id=self.config.client_id,
                )

            # Connect
            self._broker.connect()
            self._connected = True
            logger.info("Connected to IBKR")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._broker:
            try:
                self._broker.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        self._rate_limit()

        try:
            account = self._broker.get_account_summary()
            return {
                'equity': account.get('NetLiquidation', 0),
                'cash': account.get('AvailableFunds', 0),
                'buying_power': account.get('BuyingPower', 0),
                'margin_used': account.get('MaintMarginReq', 0),
                'unrealized_pnl': account.get('UnrealizedPnL', 0),
                'realized_pnl': account.get('RealizedPnL', 0),
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        """Get all positions."""
        self._rate_limit()

        try:
            positions = self._broker.get_positions()
            return [
                {
                    'symbol': p.contract.symbol if hasattr(p, 'contract') else p.get('symbol'),
                    'quantity': p.position if hasattr(p, 'position') else p.get('quantity', 0),
                    'avg_cost': p.avgCost if hasattr(p, 'avgCost') else p.get('avg_cost', 0),
                    'current_price': p.marketPrice if hasattr(p, 'marketPrice') else p.get('market_price', 0),
                    'unrealized_pnl': p.unrealizedPNL if hasattr(p, 'unrealizedPNL') else p.get('unrealized_pnl', 0),
                    'realized_pnl': 0,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current market quote."""
        self._rate_limit()

        try:
            quote = self._broker.get_market_data(symbol)
            if quote:
                return {
                    'symbol': symbol,
                    'last': quote.get('last', quote.get('close', 0)),
                    'bid': quote.get('bid'),
                    'ask': quote.get('ask'),
                    'bid_size': quote.get('bid_size'),
                    'ask_size': quote.get('ask_size'),
                    'volume': quote.get('volume', 0),
                    'open': quote.get('open'),
                    'high': quote.get('high'),
                    'low': quote.get('low'),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
    ) -> Optional[List[Dict]]:
        """Get historical bars."""
        self._rate_limit()

        try:
            bars = self._broker.get_historical_bars(
                symbol=symbol,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
            )
            return bars
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        action: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """
        Place an order.

        Returns:
            Dict with order_id and status
        """
        self._rate_limit()

        # Small delay between orders
        time.sleep(self.config.order_delay_ms / 1000)

        try:
            order_id = self._broker.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
            )

            self._pending_orders[str(order_id)] = {
                'order_id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'status': 'SUBMITTED',
            }

            return {'order_id': order_id, 'status': 'SUBMITTED'}

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'order_id': None, 'status': 'REJECTED', 'error': str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._rate_limit()

        try:
            self._broker.cancel_order(int(order_id))
            if order_id in self._pending_orders:
                self._pending_orders[order_id]['status'] = 'CANCELLED'
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        self._rate_limit()

        try:
            orders = self._broker.get_open_orders()
            return [
                {
                    'order_id': str(o.orderId) if hasattr(o, 'orderId') else str(o.get('order_id')),
                    'symbol': o.contract.symbol if hasattr(o, 'contract') else o.get('symbol'),
                    'side': o.action if hasattr(o, 'action') else o.get('side'),
                    'quantity': o.totalQuantity if hasattr(o, 'totalQuantity') else o.get('quantity'),
                    'status': o.status if hasattr(o, 'status') else o.get('status'),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            # Simple check based on time
            now = datetime.now()

            # Skip weekends
            if now.weekday() >= 5:
                return False

            # US market hours (9:30 AM - 4:00 PM ET)
            # Simplified check without timezone handling
            hour = now.hour
            minute = now.minute

            market_open = 9 * 60 + 30   # 9:30 AM
            market_close = 16 * 60       # 4:00 PM
            current = hour * 60 + minute

            return market_open <= current < market_close

        except Exception:
            return False


def create_live_context(
    config: Optional[IBKRConfig] = None,
    universe: List[str] = None,
) -> LiveContext:
    """
    Create a live trading context connected to IBKR.

    Args:
        config: IBKR configuration
        universe: List of symbols to trade

    Returns:
        LiveContext ready for trading
    """
    config = config or IBKRConfig()
    universe = universe or []

    # Create adapter
    adapter = IBKRBrokerAdapter(config)

    # Connect
    if not adapter.connect():
        raise ConnectionError("Failed to connect to IBKR")

    # Create data provider
    data_provider = IBKRDataProvider(adapter._broker)

    # Create context
    return LiveContext(
        ibkr_broker=adapter,
        data_provider=data_provider,
        universe=universe,
    )
