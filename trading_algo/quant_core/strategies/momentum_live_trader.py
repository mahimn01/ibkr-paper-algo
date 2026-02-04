"""
Live Trading Integration for Pure Momentum Strategy

Production-ready momentum trader with IBKR integration.
Designed for maximum profit with acceptable risk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.strategies.pure_momentum import (
    PureMomentumStrategy, MomentumConfig, MomentumSignal
)


logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Live trading position."""
    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    signal_strength: float


@dataclass
class LiveTradeStats:
    """Live trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0


class MomentumLiveTrader:
    """
    Live trading implementation of Pure Momentum.

    Features:
    - Real-time signal generation
    - IBKR integration
    - Risk management
    - Performance tracking
    - Position management

    Usage:
        trader = MomentumLiveTrader(config, ibkr_broker)
        trader.start()
    """

    def __init__(
        self,
        config: MomentumConfig,
        ibkr_broker,
        universe: List[str],
        initial_capital: float = 100000.0,
    ):
        """
        Initialize live trader.

        Args:
            config: Momentum strategy config
            ibkr_broker: Connected IBKRBroker instance
            universe: List of symbols to trade
            initial_capital: Starting capital
        """
        self.config = config
        self.broker = ibkr_broker
        self.universe = universe
        self.initial_capital = initial_capital

        # Strategy
        self.strategy = PureMomentumStrategy(config)

        # State
        self._price_history: Dict[str, List[float]] = {s: [] for s in universe}
        self._positions: Dict[str, LivePosition] = {}
        self._stats = LiveTradeStats(current_equity=initial_capital, peak_equity=initial_capital)
        self._last_rebalance: Optional[datetime] = None
        self._running = False

        logger.info(f"Initialized MomentumLiveTrader with {len(universe)} symbols")

    def update_prices(self) -> Dict[str, float]:
        """
        Fetch current prices from IBKR.

        Returns:
            Dict of symbol -> current price
        """
        from trading_algo.instruments import InstrumentSpec

        current_prices = {}

        for symbol in self.universe:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                snapshot = self.broker.get_market_data_snapshot(instrument)
                price = snapshot.last or snapshot.close

                if price:
                    current_prices[symbol] = float(price)
                    self._price_history[symbol].append(float(price))

                    # Keep limited history
                    max_len = max(self.config.trend_ma, self.config.momentum_lookback) + 50
                    if len(self._price_history[symbol]) > max_len:
                        self._price_history[symbol] = self._price_history[symbol][-max_len:]

            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")

        return current_prices

    def get_account_value(self) -> float:
        """Get current account equity."""
        try:
            snapshot = self.broker.get_account_snapshot()
            nav = snapshot.values.get('NetLiquidation', self.initial_capital)
            return float(nav)
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return self._stats.current_equity

    def generate_signals(self) -> Dict[str, MomentumSignal]:
        """
        Generate trading signals based on current data.

        Returns:
            Dict of symbol -> MomentumSignal
        """
        # Convert price history to numpy arrays
        price_arrays = {}
        for symbol in self.universe:
            if len(self._price_history[symbol]) >= self.config.trend_ma:
                price_arrays[symbol] = np.array(self._price_history[symbol])

        if not price_arrays:
            logger.warning("Not enough price history for signals")
            return {}

        # Generate signals
        signals = self.strategy.generate_signals(
            list(price_arrays.keys()),
            price_arrays,
        )

        return signals

    def calculate_target_positions(
        self,
        signals: Dict[str, MomentumSignal],
        current_prices: Dict[str, float],
        equity: float,
    ) -> Dict[str, float]:
        """
        Calculate target dollar positions for each symbol.

        Args:
            signals: Current signals
            current_prices: Current prices
            equity: Current account equity

        Returns:
            Dict of symbol -> target dollar position
        """
        # Get target weights from strategy
        weight_dict = self.strategy.get_target_weights(
            list(signals.keys()),
            {s: np.array(self._price_history[s]) for s in signals.keys()}
        )

        # Convert weights to dollar positions
        target_positions = {}
        for symbol, weight in weight_dict.items():
            if symbol in current_prices:
                target_value = equity * weight
                target_positions[symbol] = target_value

        return target_positions

    def execute_rebalance(
        self,
        target_positions: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> None:
        """
        Execute trades to reach target positions.

        Args:
            target_positions: Dict of symbol -> target dollar value
            current_prices: Current prices
        """
        from trading_algo.instruments import InstrumentSpec
        from trading_algo.broker.base import OrderRequest

        # Get current positions
        current_positions = {}
        try:
            positions = self.broker.get_positions()
            for pos in positions:
                if pos.instrument.symbol in self.universe:
                    current_value = pos.quantity * current_prices.get(pos.instrument.symbol, 0)
                    current_positions[pos.instrument.symbol] = current_value
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return

        # Calculate trades needed
        trades_executed = 0

        for symbol in self.universe:
            current_value = current_positions.get(symbol, 0)
            target_value = target_positions.get(symbol, 0)

            delta_value = target_value - current_value

            # Min trade size $500
            if abs(delta_value) < 500:
                continue

            price = current_prices.get(symbol)
            if not price:
                continue

            delta_shares = int(delta_value / price)

            if delta_shares == 0:
                continue

            # Create order
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                order = OrderRequest(
                    instrument=instrument,
                    side="BUY" if delta_shares > 0 else "SELL",
                    quantity=abs(delta_shares),
                    order_type="MKT",
                    tif="DAY",
                )

                result = self.broker.place_order(order)

                if result.status in ["Submitted", "Filled", "PreSubmitted"]:
                    trades_executed += 1
                    logger.info(f"Executed: {order.side} {abs(delta_shares)} {symbol} @ ${price:.2f}")

                    # Update position tracking
                    if delta_shares > 0 or symbol in self._positions:
                        self._positions[symbol] = LivePosition(
                            symbol=symbol,
                            shares=delta_shares if delta_shares > 0 else 0,
                            entry_price=price,
                            entry_time=datetime.now(),
                            current_price=price,
                            unrealized_pnl=0.0,
                            signal_strength=0.0,
                        )
                    elif delta_shares < 0 and symbol in self._positions:
                        del self._positions[symbol]

                    self._stats.total_trades += 1

            except Exception as e:
                logger.error(f"Failed to execute trade for {symbol}: {e}")

        if trades_executed > 0:
            logger.info(f"Rebalance complete: {trades_executed} trades executed")

    def update_stats(self, current_prices: Dict[str, float]) -> None:
        """Update trading statistics."""
        equity = self.get_account_value()
        self._stats.current_equity = equity

        # Update peak and drawdown
        if equity > self._stats.peak_equity:
            self._stats.peak_equity = equity

        self._stats.current_drawdown = (self._stats.peak_equity - equity) / self._stats.peak_equity

        # Calculate exposure
        gross = 0.0
        net = 0.0
        unrealized = 0.0

        for symbol, pos in self._positions.items():
            if symbol in current_prices:
                current_value = pos.shares * current_prices[symbol]
                gross += abs(current_value)
                net += current_value
                unrealized += (current_prices[symbol] - pos.entry_price) * pos.shares

        self._stats.gross_exposure = gross / equity if equity > 0 else 0
        self._stats.net_exposure = net / equity if equity > 0 else 0
        self._stats.unrealized_pnl = unrealized

    def should_rebalance(self) -> bool:
        """Check if we should rebalance."""
        if self._last_rebalance is None:
            return True

        # Rebalance daily
        elapsed = datetime.now() - self._last_rebalance
        return elapsed >= timedelta(days=1)

    def run_once(self) -> None:
        """Run one iteration of the trading loop."""
        logger.info("Running trading iteration...")

        # 1. Update prices
        current_prices = self.update_prices()

        if not current_prices:
            logger.warning("No prices available")
            return

        # 2. Update stats
        self.update_stats(current_prices)

        # 3. Log current state
        logger.info(f"Equity: ${self._stats.current_equity:,.2f}, "
                   f"DD: {self._stats.current_drawdown:.2%}, "
                   f"Exposure: {self._stats.gross_exposure:.1%}")

        # 4. Check if rebalance needed
        if not self.should_rebalance():
            logger.info("Not time to rebalance yet")
            return

        # 5. Generate signals
        signals = self.generate_signals()

        if not signals:
            logger.warning("No signals generated")
            return

        logger.info(f"Generated {len(signals)} signals")

        # 6. Calculate target positions
        equity = self.get_account_value()
        target_positions = self.calculate_target_positions(signals, current_prices, equity)

        logger.info(f"Target positions: {len(target_positions)} symbols")

        # 7. Execute rebalance
        self.execute_rebalance(target_positions, current_prices)

        self._last_rebalance = datetime.now()

        logger.info("Trading iteration complete")

    def start(self, iterations: Optional[int] = None) -> None:
        """
        Start live trading.

        Args:
            iterations: Number of iterations (None = run forever)
        """
        self._running = True
        count = 0

        logger.info("Starting live trading...")

        try:
            while self._running:
                if iterations and count >= iterations:
                    break

                self.run_once()
                count += 1

                # Wait before next iteration (e.g., end of day)
                if iterations is None:
                    import time
                    time.sleep(3600)  # Check hourly

        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop live trading."""
        self._running = False
        logger.info("Live trading stopped")

        # Print final stats
        logger.info("="*60)
        logger.info("FINAL STATS")
        logger.info("="*60)
        logger.info(f"Total Trades: {self._stats.total_trades}")
        logger.info(f"Final Equity: ${self._stats.current_equity:,.2f}")
        logger.info(f"Total Return: {(self._stats.current_equity / self.initial_capital - 1):.2%}")
        logger.info(f"Max Drawdown: {self._stats.current_drawdown:.2%}")
        logger.info("="*60)

    def get_stats(self) -> LiveTradeStats:
        """Get current trading statistics."""
        return self._stats
