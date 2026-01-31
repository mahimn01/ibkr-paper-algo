"""
RAT Integration with IBKR Broker

Bridges the RAT framework with the existing IBKR broker infrastructure.

Usage:
    from trading_algo.broker.ibkr import IBKRBroker
    from trading_algo.rat.integration import RATTrader

    broker = IBKRBroker()
    broker.connect()

    trader = RATTrader(broker, symbols=["AAPL", "MSFT"])
    trader.start()

    # In event loop:
    trader.process_market_data()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from trading_algo.broker.base import (
    Broker,
    MarketDataSnapshot,
    OrderRequest,
    Position,
)
from trading_algo.instruments import InstrumentSpec
from trading_algo.rat.config import RATConfig
from trading_algo.rat.engine import RATEngine, RATState
from trading_algo.rat.signals import SignalType


logger = logging.getLogger(__name__)


@dataclass
class TradeExecution:
    """Record of an executed trade."""

    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    order_id: str
    signal_source: str
    confidence: float


class RATTrader:
    """
    RAT-based trader integrated with IBKR broker.

    Manages the full trading cycle:
    1. Receive market data from broker
    2. Process through RAT engine
    3. Execute trading decisions
    4. Track positions and P&L
    """

    def __init__(
        self,
        broker: Broker,
        symbols: List[str],
        config: Optional[RATConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.broker = broker
        self.symbols = [s.upper() for s in symbols]
        self.config = config or RATConfig.from_env()

        # Initialize RAT engine
        self.engine = RATEngine(
            config=self.config,
            broker=broker,
            llm_client=llm_client,
        )

        # Instrument mapping
        self._instruments: Dict[str, InstrumentSpec] = {}

        # Execution tracking
        self._executions: List[TradeExecution] = []
        self._pending_orders: Dict[str, str] = {}  # order_id -> symbol

        # State
        self._running = False
        self._last_update: Dict[str, datetime] = {}

    def start(self) -> None:
        """Start the RAT trader."""
        logger.info("Starting RAT Trader")

        # Create instruments for each symbol
        for symbol in self.symbols:
            self._instruments[symbol] = InstrumentSpec(
                symbol=symbol,
                sec_type="STK",
                exchange="SMART",
                currency="USD",
            )

        self.engine.start()
        self._running = True

    def stop(self) -> None:
        """Stop the RAT trader."""
        logger.info("Stopping RAT Trader")
        self._running = False
        self.engine.stop()

    def process_market_data(self) -> Dict[str, Optional[RATState]]:
        """
        Process current market data for all symbols.

        Should be called periodically (e.g., every second).
        Returns state for each symbol.
        """
        if not self._running:
            return {}

        states: Dict[str, Optional[RATState]] = {}

        for symbol in self.symbols:
            try:
                state = self._process_symbol(symbol)
                states[symbol] = state
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                states[symbol] = None

        # Check for order fills
        self._check_order_status()

        return states

    def _process_symbol(self, symbol: str) -> Optional[RATState]:
        """Process market data for a single symbol."""
        instrument = self._instruments.get(symbol)
        if not instrument:
            return None

        # Get market data from broker
        try:
            snapshot = self.broker.get_market_data_snapshot(instrument)
        except Exception as e:
            logger.debug(f"Could not get market data for {symbol}: {e}")
            return None

        if snapshot.last is None:
            return None

        # Extract price data
        price = snapshot.last
        bid = snapshot.bid or price
        ask = snapshot.ask or price
        volume = snapshot.volume or 0

        # Process through RAT engine
        state = self.engine.process_tick(
            symbol=symbol,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            timestamp=datetime.fromtimestamp(snapshot.timestamp_epoch_s),
        )

        self._last_update[symbol] = datetime.now()

        # Execute if appropriate
        if state and state.decision and state.decision.should_trade():
            self._execute_decision(symbol, state, snapshot)

        return state

    def _execute_decision(
        self,
        symbol: str,
        state: RATState,
        snapshot: MarketDataSnapshot,
    ) -> None:
        """Execute a trading decision."""
        decision = state.decision
        instrument = self._instruments.get(symbol)

        if not instrument or not decision:
            return

        # Get current position
        current_position = self._get_position(symbol)

        # Determine order parameters
        if decision.action == "buy":
            side = "BUY"
            # Calculate quantity based on account value
            account = self.broker.get_account_snapshot()
            equity = account.values.get("NetLiquidation", 100000)
            position_value = equity * decision.position_size_pct
            price = snapshot.ask or snapshot.last or 0
            if price <= 0:
                return
            quantity = position_value / price

            # Adjust for existing position
            if current_position > 0:
                quantity = max(0, quantity - current_position)

        elif decision.action == "sell":
            side = "SELL"
            # Sell existing position
            if current_position <= 0:
                return
            quantity = current_position

        else:
            return

        if quantity <= 0:
            return

        # Round to appropriate lot size
        quantity = round(quantity, 2)

        # Create and place order
        try:
            order = OrderRequest(
                instrument=instrument,
                side=side,
                quantity=quantity,
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)

            self._pending_orders[result.order_id] = symbol

            logger.info(
                f"Placed {side} order for {quantity} {symbol} "
                f"(confidence: {decision.confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")

    def _get_position(self, symbol: str) -> float:
        """Get current position for a symbol."""
        try:
            positions = self.broker.get_positions()
            for pos in positions:
                if pos.instrument.symbol == symbol:
                    return pos.quantity
        except Exception:
            pass
        return 0.0

    def _check_order_status(self) -> None:
        """Check status of pending orders."""
        filled_orders = []

        for order_id, symbol in list(self._pending_orders.items()):
            try:
                status = self.broker.get_order_status(order_id)

                if status.status == "Filled":
                    filled_orders.append(order_id)

                    execution = TradeExecution(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        side=status.status,
                        quantity=status.filled,
                        price=status.avg_fill_price or 0,
                        order_id=order_id,
                        signal_source="RAT",
                        confidence=0,  # Would track from original decision
                    )
                    self._executions.append(execution)

                    logger.info(
                        f"Order filled: {status.filled} {symbol} @ "
                        f"{status.avg_fill_price}"
                    )

                elif status.status in ("Cancelled", "Error"):
                    filled_orders.append(order_id)
                    logger.warning(f"Order {order_id} {status.status}")

            except Exception as e:
                logger.debug(f"Could not check order {order_id}: {e}")

        # Remove completed orders
        for order_id in filled_orders:
            del self._pending_orders[order_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get trader statistics."""
        engine_stats = self.engine.get_stats()

        return {
            "symbols": self.symbols,
            "running": self._running,
            "pending_orders": len(self._pending_orders),
            "total_executions": len(self._executions),
            "last_update": {
                sym: ts.isoformat() for sym, ts in self._last_update.items()
            },
            **engine_stats,
        }

    def get_latest_states(self) -> Dict[str, Optional[RATState]]:
        """Get latest state for each symbol."""
        return {
            symbol: self.engine.get_last_state(symbol)
            for symbol in self.symbols
        }


class RATBacktestAdapter:
    """
    Adapter to use RAT with existing backtest infrastructure.

    Wraps the BacktestBroker and runs RAT engine against it.
    """

    def __init__(
        self,
        broker,  # BacktestBroker from trading_algo.backtest.broker
        config: Optional[RATConfig] = None,
    ):
        from trading_algo.backtest.broker import BacktestBroker

        if not isinstance(broker, BacktestBroker):
            raise TypeError("broker must be a BacktestBroker")

        self.broker = broker
        self.config = config or RATConfig.from_env()

        self.engine = RATEngine(
            config=self.config,
            broker=None,
            llm_client=None,
        )
        self.engine.reset_for_backtest()

        self._states: List[RATState] = []
        self._decisions: List[Dict] = []

    def run(self) -> Dict[str, Any]:
        """Run RAT against the backtest broker."""
        self.broker.connect()
        symbol = self.broker.instrument.symbol

        while self.broker.step():
            bar = self.broker.current_bar()
            snapshot = self.broker.get_market_data_snapshot(self.broker.instrument)

            # Process through RAT
            state = self.engine.inject_backtest_tick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(bar.timestamp_epoch_s),
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume or 0,
            )

            if state:
                self._states.append(state)

                if state.decision and state.decision.should_trade():
                    self._execute_backtest_decision(state, snapshot)

        self.broker.disconnect()

        # Return results
        account = self.broker.get_account_snapshot()

        return {
            "final_equity": account.values.get("NetLiquidation", 0),
            "total_states": len(self._states),
            "total_decisions": len(self._decisions),
            "engine_stats": self.engine.get_stats(),
        }

    def _execute_backtest_decision(
        self,
        state: RATState,
        snapshot: MarketDataSnapshot,
    ) -> None:
        """Execute a decision in backtest mode."""
        decision = state.decision
        if not decision:
            return

        symbol = state.symbol

        if decision.action == "buy":
            side = "BUY"
            # Simple position sizing
            account = self.broker.get_account_snapshot()
            equity = account.values.get("NetLiquidation", 100000)
            position_value = equity * decision.position_size_pct
            price = snapshot.last or snapshot.close or 0
            if price <= 0:
                return
            quantity = position_value / price

        elif decision.action == "sell":
            side = "SELL"
            positions = self.broker.get_positions()
            quantity = 0
            for pos in positions:
                if pos.quantity > 0:
                    quantity = pos.quantity
                    break
            if quantity <= 0:
                return

        else:
            return

        # Place order
        try:
            order = OrderRequest(
                instrument=self.broker.instrument,
                side=side,
                quantity=quantity,
                order_type="MKT",
            )
            self.broker.place_order(order)

            self._decisions.append({
                "timestamp": state.timestamp,
                "action": decision.action,
                "quantity": quantity,
                "confidence": decision.confidence,
            })

        except Exception as e:
            logger.debug(f"Backtest order failed: {e}")


def create_rat_trader(
    broker: Broker,
    symbols: List[str],
    **kwargs,
) -> RATTrader:
    """
    Factory function to create a RAT trader.

    Args:
        broker: Connected broker instance
        symbols: List of symbols to trade
        **kwargs: Additional configuration

    Returns:
        Configured RATTrader instance
    """
    config = RATConfig.from_env()

    # Override config from kwargs
    if "confidence_threshold" in kwargs:
        config.signal.confidence_threshold = kwargs["confidence_threshold"]
    if "max_position_pct" in kwargs:
        config.signal.max_position_pct = kwargs["max_position_pct"]

    return RATTrader(
        broker=broker,
        symbols=symbols,
        config=config,
        llm_client=kwargs.get("llm_client"),
    )
