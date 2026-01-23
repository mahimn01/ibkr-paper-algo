#!/usr/bin/env python3
"""
Live Paper Trading with Chameleon Strategy.

Runs the Chameleon Strategy in real-time on IBKR Paper account.
Uses 5-minute bars for intraday trading decisions.

Usage:
    # Run for rest of trading day on AAPL and MSFT
    python run_chameleon_live.py AAPL MSFT

    # Run for specific duration (seconds)
    python run_chameleon_live.py AAPL MSFT --duration 7200

    # Dry run (no real orders)
    python run_chameleon_live.py AAPL MSFT --dry-run

Prerequisites:
    - TWS or IB Gateway running (paper trading, port 7497)
    - API access enabled in TWS settings
"""

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.broker.base import OrderRequest
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.rat.chameleon_strategy import (
    ChameleonStrategy,
    MarketMode,
    create_chameleon_strategy,
)


@dataclass
class LivePosition:
    """Track a live position."""
    symbol: str
    direction: int  # 1=long, -1=short
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class ChameleonLiveTrader:
    """
    Live trader using the Chameleon Strategy.

    Fetches 5-minute bars periodically and makes trading decisions.
    """

    def __init__(
        self,
        broker: IBKRBroker,
        symbols: List[str],
        max_position_pct: float = 0.25,
        dry_run: bool = False,
    ):
        self.broker = broker
        self.symbols = [s.upper() for s in symbols]
        self.max_position_pct = max_position_pct
        self.dry_run = dry_run

        # Create strategy instances (one per symbol for isolation)
        self.strategies: Dict[str, ChameleonStrategy] = {
            symbol: create_chameleon_strategy()
            for symbol in self.symbols
        }

        # Position tracking
        self.positions: Dict[str, LivePosition] = {}
        self.pending_orders: Dict[str, str] = {}  # symbol -> order_id

        # Stats
        self.signals_generated = 0
        self.orders_placed = 0
        self.total_pnl = 0.0

        # Account info
        self.account_value = 100_000  # Will be updated from broker

        # Running flag
        self.running = False

    def start(self):
        """Start the trader."""
        self.running = True

        # Get initial account value
        try:
            account = self.broker.get_account_snapshot()
            self.account_value = account.values.get('NetLiquidation', 100_000)
            print(f"Account Value: ${self.account_value:,.2f}")
        except Exception as e:
            print(f"Warning: Could not get account value: {e}")

        # Sync existing positions
        self._sync_positions()

        # Warm up strategies with historical data
        self._warmup_strategies()

    def stop(self):
        """Stop the trader."""
        self.running = False

    def _sync_positions(self):
        """Sync positions from broker."""
        try:
            broker_positions = self.broker.get_positions()
            for pos in broker_positions:
                symbol = pos.instrument.symbol
                if symbol in self.symbols and pos.quantity != 0:
                    direction = 1 if pos.quantity > 0 else -1
                    self.positions[symbol] = LivePosition(
                        symbol=symbol,
                        direction=direction,
                        quantity=abs(pos.quantity),
                        entry_price=pos.avg_cost or 0,
                        entry_time=datetime.now(),
                        current_price=pos.avg_cost or 0,
                    )
                    print(f"  Synced position: {symbol} {pos.quantity} shares @ ${pos.avg_cost:.2f}")
        except Exception as e:
            print(f"Warning: Could not sync positions: {e}")

    def _warmup_strategies(self):
        """Warm up strategies with historical data."""
        print("\nWarming up strategies with historical data...")

        for symbol in self.symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                # Get 10 days of 5-minute bars for warmup
                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="10 D",
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                print(f"  {symbol}: {len(bars)} bars for warmup")

                strategy = self.strategies[symbol]
                for bar in bars:
                    ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
                    strategy.update(
                        symbol=symbol,
                        timestamp=ts,
                        open_price=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume or 0,
                    )

            except Exception as e:
                print(f"  Warning: Could not warm up {symbol}: {e}")

    def update(self) -> Dict[str, dict]:
        """
        Fetch latest data and update strategies.

        Returns dict of symbol -> decision
        """
        decisions = {}

        for symbol in self.symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                # Get latest 5-minute bars (use 1 hour to ensure we get recent data)
                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="3600 S",  # Last hour of 5-min bars
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                if not bars:
                    continue

                bar = bars[-1]
                ts = datetime.fromtimestamp(bar.timestamp_epoch_s)

                # Update position prices
                if symbol in self.positions:
                    self.positions[symbol].current_price = bar.close
                    pos = self.positions[symbol]
                    if pos.direction > 0:
                        pos.unrealized_pnl = (bar.close - pos.entry_price) * pos.quantity
                    else:
                        pos.unrealized_pnl = (pos.entry_price - bar.close) * pos.quantity

                # Get strategy decision
                strategy = self.strategies[symbol]
                decision = strategy.update(
                    symbol=symbol,
                    timestamp=ts,
                    open_price=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume or 0,
                )

                if decision:
                    decisions[symbol] = decision
                    self.signals_generated += 1

                    # Log the decision
                    action = decision.get('action', 'hold')
                    regime = decision.get('regime')
                    regime_name = regime.name if regime else 'UNKNOWN'
                    print(f"  [{symbol}] ${bar.close:.2f} | Regime: {regime_name} | Action: {action.upper()}")

            except Exception as e:
                print(f"  Error updating {symbol}: {e}")

        return decisions

    def execute_decisions(self, decisions: Dict[str, dict]):
        """Execute trading decisions."""
        for symbol, decision in decisions.items():
            action = decision.get('action', 'hold')

            if action == 'hold':
                continue

            regime = decision.get('regime')
            regime_name = regime.name if regime else 'UNKNOWN'

            if action == 'buy':
                self._handle_buy(symbol, decision, regime_name)
            elif action == 'short':
                self._handle_short(symbol, decision, regime_name)
            elif action == 'sell':
                self._handle_sell(symbol, decision, regime_name)
            elif action == 'cover':
                self._handle_cover(symbol, decision, regime_name)

    def _handle_buy(self, symbol: str, decision: dict, regime: str):
        """Handle buy signal."""
        if symbol in self.positions:
            return  # Already have position

        size_pct = min(decision.get('size', 0.1), self.max_position_pct)

        try:
            # Get current price
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            snap = self.broker.get_market_data_snapshot(instrument)
            price = snap.last or snap.close or 0

            if price <= 0:
                return

            # Calculate quantity
            position_value = self.account_value * size_pct
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> BUY SIGNAL: {symbol}")
            print(f"    Regime: {regime}")
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity}")
            print(f"    Value: ${quantity * price:,.2f}")

            if self.dry_run:
                print("    [DRY RUN - no order placed]")
                return

            # Place order
            order = OrderRequest(
                instrument=instrument,
                side="BUY",
                quantity=quantity,
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)
            self.orders_placed += 1
            print(f"    Order ID: {result.order_id}, Status: {result.status}")

            # Track position
            self.positions[symbol] = LivePosition(
                symbol=symbol,
                direction=1,
                quantity=quantity,
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price,
            )

        except Exception as e:
            print(f"    Error placing buy order: {e}")

    def _handle_short(self, symbol: str, decision: dict, regime: str):
        """Handle short signal."""
        if symbol in self.positions:
            return

        size_pct = min(decision.get('size', 0.1), self.max_position_pct)

        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            snap = self.broker.get_market_data_snapshot(instrument)
            price = snap.last or snap.close or 0

            if price <= 0:
                return

            position_value = self.account_value * size_pct
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> SHORT SIGNAL: {symbol}")
            print(f"    Regime: {regime}")
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity}")

            if self.dry_run:
                print("    [DRY RUN - no order placed]")
                return

            order = OrderRequest(
                instrument=instrument,
                side="SELL",  # Short sell
                quantity=quantity,
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)
            self.orders_placed += 1
            print(f"    Order ID: {result.order_id}, Status: {result.status}")

            self.positions[symbol] = LivePosition(
                symbol=symbol,
                direction=-1,
                quantity=quantity,
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price,
            )

        except Exception as e:
            print(f"    Error placing short order: {e}")

    def _handle_sell(self, symbol: str, decision: dict, regime: str):
        """Handle sell signal (close long)."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if pos.direction != 1:
            return

        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            snap = self.broker.get_market_data_snapshot(instrument)
            price = snap.last or snap.close or pos.current_price

            pnl = (price - pos.entry_price) * pos.quantity

            print(f"\n>>> SELL SIGNAL: {symbol}")
            print(f"    Regime: {regime}")
            print(f"    Entry: ${pos.entry_price:.2f}")
            print(f"    Exit: ${price:.2f}")
            print(f"    P&L: ${pnl:+,.2f}")

            if self.dry_run:
                print("    [DRY RUN - no order placed]")
                return

            order = OrderRequest(
                instrument=instrument,
                side="SELL",
                quantity=int(pos.quantity),
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)
            self.orders_placed += 1
            print(f"    Order ID: {result.order_id}, Status: {result.status}")

            self.total_pnl += pnl
            del self.positions[symbol]

        except Exception as e:
            print(f"    Error placing sell order: {e}")

    def _handle_cover(self, symbol: str, decision: dict, regime: str):
        """Handle cover signal (close short)."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if pos.direction != -1:
            return

        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            snap = self.broker.get_market_data_snapshot(instrument)
            price = snap.last or snap.close or pos.current_price

            pnl = (pos.entry_price - price) * pos.quantity

            print(f"\n>>> COVER SIGNAL: {symbol}")
            print(f"    Regime: {regime}")
            print(f"    Entry: ${pos.entry_price:.2f}")
            print(f"    Exit: ${price:.2f}")
            print(f"    P&L: ${pnl:+,.2f}")

            if self.dry_run:
                print("    [DRY RUN - no order placed]")
                return

            order = OrderRequest(
                instrument=instrument,
                side="BUY",  # Buy to cover
                quantity=int(pos.quantity),
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)
            self.orders_placed += 1
            print(f"    Order ID: {result.order_id}, Status: {result.status}")

            self.total_pnl += pnl
            del self.positions[symbol]

        except Exception as e:
            print(f"    Error placing cover order: {e}")

    def print_status(self, decisions: Dict[str, dict] = None):
        """Print current status."""
        now = datetime.now()
        print(f"\n[{now.strftime('%H:%M:%S')}] Status Update")
        print(f"  Signals: {self.signals_generated}, Orders: {self.orders_placed}")

        # Show current regime for each symbol
        if decisions:
            for symbol, decision in decisions.items():
                regime = decision.get('regime')
                regime_name = regime.name if regime else 'UNKNOWN'
                action = decision.get('action', 'hold')
                print(f"  {symbol}: Regime={regime_name}, Action={action}")
        print(f"  Realized P&L: ${self.total_pnl:+,.2f}")

        if self.positions:
            print(f"  Open Positions:")
            total_unrealized = 0
            for symbol, pos in self.positions.items():
                direction = "LONG" if pos.direction > 0 else "SHORT"
                print(f"    {symbol}: {direction} {pos.quantity} @ ${pos.entry_price:.2f}")
                print(f"           Current: ${pos.current_price:.2f}, P&L: ${pos.unrealized_pnl:+,.2f}")
                total_unrealized += pos.unrealized_pnl
            print(f"  Unrealized P&L: ${total_unrealized:+,.2f}")
        else:
            print(f"  No open positions")


def main():
    parser = argparse.ArgumentParser(
        description="Live paper trading with Chameleon Strategy"
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbols to trade (e.g., AAPL MSFT)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (default: until market close or Ctrl+C)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - generate signals but don't place orders"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TWS/Gateway host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="TWS/Gateway port (7497=paper)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CHAMELEON STRATEGY - LIVE PAPER TRADING")
    print("=" * 70)
    print(f"Symbols:  {', '.join(args.symbols)}")
    print(f"Interval: {args.interval} seconds")
    print(f"Dry Run:  {args.dry_run}")
    print("=" * 70)

    # Connect to IBKR
    config = IBKRConfig(host=args.host, port=args.port, client_id=10)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR at {args.host}:{args.port}...")
    try:
        broker.connect()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    # Create trader
    trader = ChameleonLiveTrader(
        broker=broker,
        symbols=args.symbols,
        dry_run=args.dry_run,
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        trader.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        trader.start()

        print("\n" + "=" * 70)
        print("TRADING STARTED - Press Ctrl+C to stop")
        print("=" * 70)

        start_time = time.time()
        update_count = 0

        while trader.running:
            # Check duration
            if args.duration and (time.time() - start_time) >= args.duration:
                print("\nDuration reached, stopping...")
                break

            # Update and execute
            decisions = trader.update()
            trader.execute_decisions(decisions)

            update_count += 1

            # Print status every 5 updates
            if update_count % 5 == 0:
                trader.print_status(decisions)

            # Wait for next interval
            time.sleep(args.interval)

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        trader.stop()
        broker.disconnect()

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Total Signals:    {trader.signals_generated}")
        print(f"Total Orders:     {trader.orders_placed}")
        print(f"Realized P&L:     ${trader.total_pnl:+,.2f}")

        if trader.positions:
            print(f"\nOpen Positions (not closed):")
            for symbol, pos in trader.positions.items():
                direction = "LONG" if pos.direction > 0 else "SHORT"
                print(f"  {symbol}: {direction} {pos.quantity} @ ${pos.entry_price:.2f}")

        print("=" * 70)


if __name__ == "__main__":
    main()
