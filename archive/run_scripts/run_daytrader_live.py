#!/usr/bin/env python3
"""
Chameleon Day Trader v2 - Live Trading.

Features:
- ATR-based adaptive stops (adjusts to each stock's volatility)
- Trailing stops (locks in profits as trade moves favorably)
- Momentum exhaustion filter (avoids chasing RSI extremes)
- VWAP-relative mean-reversion entries in choppy markets
- Cooldown after stop-loss exits (prevents revenge trading)
- Time-of-day awareness (avoids open/close chaos)

Usage:
    # Dry run first!
    python run_daytrader_live.py AAPL MSFT --dry-run

    # Live trading
    python run_daytrader_live.py AAPL MSFT

    # Conservative mode
    python run_daytrader_live.py AAPL MSFT --conservative
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.broker.base import OrderRequest
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.rat.chameleon_daytrader import (
    ChameleonDayTrader,
    DayTradeMode,
    DayTradeSignal,
    MarketConfig,
    MARKET_PRESETS,
    create_daytrader,
    list_markets,
)


class DayTraderLive:
    """Live day trading execution."""

    def __init__(
        self,
        broker: IBKRBroker,
        symbols: List[str],
        aggressive: bool = True,
        dry_run: bool = False,
        max_position_dollars: float = 10000,
        market: str = 'NYSE',
    ):
        self.broker = broker
        self.symbols = [s.upper() for s in symbols]
        self.dry_run = dry_run
        self.max_position_dollars = max_position_dollars
        self.market = market.upper()
        self.market_config = MARKET_PRESETS.get(self.market, MARKET_PRESETS['NYSE'])

        # Create day traders (one per symbol)
        self.traders: Dict[str, ChameleonDayTrader] = {
            symbol: create_daytrader(
                aggressive=aggressive,
                max_position_dollars=max_position_dollars,
                market=self.market,
            )
            for symbol in self.symbols
        }

        # Stats
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

        # Account
        self.account_value = 100_000

        self.running = False

    def start(self):
        """Start the day trader."""
        self.running = True

        # Get account value
        try:
            account = self.broker.get_account_snapshot()
            self.account_value = account.values.get('NetLiquidation', 100_000)
            print(f"Account Value: ${self.account_value:,.2f}")
        except Exception as e:
            print(f"Warning: Could not get account value: {e}")

        # Warm up with historical data
        self._warmup()

    def stop(self):
        """Stop trading."""
        self.running = False

    def _warmup(self):
        """Warm up strategies with recent data."""
        print("\nWarming up with recent 5-min bars...")

        for symbol in self.symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange=self.market_config.exchange,
                    currency=self.market_config.currency,
                )

                # Get full day of 5-min bars for warmup (need 30+ bars)
                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="1 D",  # Full trading day
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                print(f"  {symbol}: {len(bars)} bars loaded")

                trader = self.traders[symbol]
                for bar in bars:
                    ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
                    trader.update(
                        symbol=symbol,
                        timestamp=ts,
                        open_price=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume or 0,
                    )

                # Clear any positions created during warmup
                trader.clear_positions()

            except Exception as e:
                print(f"  Warning: Could not warm up {symbol}: {e}")

    def update(self) -> Dict[str, DayTradeSignal]:
        """Fetch latest data and get signals."""
        signals = {}

        for symbol in self.symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange=self.market_config.exchange,
                    currency=self.market_config.currency,
                )

                # Get latest bars
                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="1800 S",  # Last 30 minutes
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                if not bars:
                    continue

                bar = bars[-1]
                ts = datetime.fromtimestamp(bar.timestamp_epoch_s)

                trader = self.traders[symbol]
                signal = trader.update(
                    symbol=symbol,
                    timestamp=ts,
                    open_price=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume or 0,
                )

                if signal:
                    signals[symbol] = signal
                    self.signals_generated += 1

                    # Print signal info
                    mode_name = signal.mode.name
                    print(f"  [{symbol}] ${signal.entry_price:.2f} | {mode_name} | {signal.action.upper()}")
                    if signal.reason and signal.action != 'hold':
                        print(f"           Reason: {signal.reason}")

            except Exception as e:
                print(f"  Error updating {symbol}: {e}")

        return signals

    def execute(self, signals: Dict[str, DayTradeSignal]):
        """Execute trading signals."""
        for symbol, signal in signals.items():
            if signal.action == 'hold':
                continue

            if signal.action == 'buy':
                self._execute_buy(signal)
            elif signal.action == 'short':
                self._execute_short(signal)
            elif signal.action == 'sell':
                self._execute_sell(signal)
            elif signal.action == 'cover':
                self._execute_cover(signal)

    def _execute_buy(self, signal: DayTradeSignal):
        """Execute buy order."""
        try:
            price = signal.entry_price
            # signal.size is a percentage, apply dollar cap
            position_value = self.account_value * signal.size
            position_value = min(position_value, self.max_position_dollars)
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> BUY {signal.symbol}")
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity} shares")
            print(f"    Value: ${quantity * price:,.2f}")
            print(f"    Stop: ${signal.stop_loss:.2f}")
            print(f"    Target: ${signal.take_profit:.2f}")
            print(f"    Confidence: {signal.confidence:.0%}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange=self.market_config.exchange, currency=self.market_config.currency)
            order = OrderRequest(
                instrument=instrument,
                side="BUY",
                quantity=quantity,
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)
            self.trades_executed += 1
            print(f"    Order ID: {result.order_id}")

        except Exception as e:
            print(f"    Error: {e}")

    def _execute_short(self, signal: DayTradeSignal):
        """Execute short order."""
        try:
            price = signal.entry_price
            # signal.size is a percentage, apply dollar cap
            position_value = self.account_value * signal.size
            position_value = min(position_value, self.max_position_dollars)
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> SHORT {signal.symbol}")
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity} shares")
            print(f"    Stop: ${signal.stop_loss:.2f}")
            print(f"    Target: ${signal.take_profit:.2f}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange=self.market_config.exchange, currency=self.market_config.currency)
            order = OrderRequest(
                instrument=instrument,
                side="SELL",
                quantity=quantity,
                order_type="MKT",
                tif="DAY",
            )

            result = self.broker.place_order(order)
            self.trades_executed += 1
            print(f"    Order ID: {result.order_id}")

        except Exception as e:
            print(f"    Error: {e}")

    def _execute_sell(self, signal: DayTradeSignal):
        """Execute sell order (close long)."""
        try:
            print(f"\n>>> SELL {signal.symbol}")
            print(f"    {signal.reason}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            # Get current position
            positions = self.broker.get_positions()
            for pos in positions:
                if pos.instrument.symbol == signal.symbol and pos.quantity > 0:
                    instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange=self.market_config.exchange, currency=self.market_config.currency)
                    order = OrderRequest(
                        instrument=instrument,
                        side="SELL",
                        quantity=int(pos.quantity),
                        order_type="MKT",
                        tif="DAY",
                    )
                    result = self.broker.place_order(order)
                    self.trades_executed += 1
                    print(f"    Closed {int(pos.quantity)} shares, Order ID: {result.order_id}")
                    break

        except Exception as e:
            print(f"    Error: {e}")

    def _execute_cover(self, signal: DayTradeSignal):
        """Execute cover order (close short)."""
        try:
            print(f"\n>>> COVER {signal.symbol}")
            print(f"    {signal.reason}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            positions = self.broker.get_positions()
            for pos in positions:
                if pos.instrument.symbol == signal.symbol and pos.quantity < 0:
                    instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange=self.market_config.exchange, currency=self.market_config.currency)
                    order = OrderRequest(
                        instrument=instrument,
                        side="BUY",
                        quantity=int(abs(pos.quantity)),
                        order_type="MKT",
                        tif="DAY",
                    )
                    result = self.broker.place_order(order)
                    self.trades_executed += 1
                    print(f"    Covered {int(abs(pos.quantity))} shares, Order ID: {result.order_id}")
                    break

        except Exception as e:
            print(f"    Error: {e}")

    def print_status(self):
        """Print trading status."""
        now = datetime.now()
        print(f"\n[{now.strftime('%H:%M:%S')}] Day Trading Status")
        print(f"  Signals: {self.signals_generated}")
        print(f"  Trades: {self.trades_executed}")


def main():
    parser = argparse.ArgumentParser(description="Aggressive Day Trading")
    parser.add_argument("symbols", nargs="+", help="Symbols to trade")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--interval", type=int, default=30, help="Update interval (default: 30s)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--aggressive", action="store_true", default=True, help="Aggressive mode (default)")
    parser.add_argument("--conservative", action="store_true", help="Conservative mode")
    parser.add_argument("--max-position", type=float, default=10000, help="Max position size cap in dollars (default: $10000)")
    parser.add_argument("--market", type=str, default="NYSE", help="Market preset: NYSE, HKEX, TSE, LSE, ASX")
    parser.add_argument("--list-markets", action="store_true", help="List available market presets and exit")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port")

    args = parser.parse_args()

    if args.list_markets:
        list_markets()
        sys.exit(0)

    aggressive = not args.conservative
    market_key = args.market.upper()
    mc = MARKET_PRESETS.get(market_key, MARKET_PRESETS['NYSE'])

    print("=" * 70)
    print("CHAMELEON DAY TRADER v2 - LIVE TRADING")
    print("=" * 70)
    print(f"Market:     {mc.name} ({market_key}) | {mc.exchange} | {mc.currency}")
    print(f"Hours:      {mc.market_open.strftime('%H:%M')}-{mc.market_close.strftime('%H:%M')} ({mc.timezone})")
    if mc.lunch_start:
        print(f"Lunch:      {mc.lunch_start.strftime('%H:%M')}-{mc.lunch_end.strftime('%H:%M')} (no new entries)")
    print(f"Symbols:    {', '.join(args.symbols)}")
    print(f"Interval:   {args.interval} seconds")
    print(f"Mode:       {'AGGRESSIVE' if aggressive else 'CONSERVATIVE'}")
    print(f"Sizing:     {'2-5%' if aggressive else '1-3%'} of account, max ${args.max_position:,.0f}")
    print(f"Stops:      ATR-based ({'1.5x' if aggressive else '2x'} ATR), trailing after {'1.5x' if aggressive else '2x'} ATR")
    print(f"Dry Run:    {args.dry_run}")
    print()
    print("WARNING: Day trading is risky! Use at your own risk.")
    print("=" * 70)

    # Connect
    config = IBKRConfig(host="127.0.0.1", port=args.port, client_id=20)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

    trader = DayTraderLive(
        broker=broker,
        symbols=args.symbols,
        aggressive=aggressive,
        dry_run=args.dry_run,
        max_position_dollars=args.max_position,
        market=market_key,
    )

    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        trader.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        trader.start()

        print("\n" + "=" * 70)
        print("DAY TRADING STARTED - Press Ctrl+C to stop")
        print("=" * 70)

        start_time = time.time()
        update_count = 0

        while trader.running:
            if args.duration and (time.time() - start_time) >= args.duration:
                print("\nDuration reached.")
                break

            # Auto-stop after market close (5 min grace period)
            now = datetime.now().time()
            close_dt = datetime.combine(datetime.today(), mc.market_close) + timedelta(minutes=5)
            if now >= close_dt.time():
                print(f"\nMarket closed at {mc.market_close.strftime('%H:%M')}. Auto-stopping.")
                break

            signals = trader.update()
            trader.execute(signals)

            update_count += 1
            if update_count % 4 == 0:
                trader.print_status()

            time.sleep(args.interval)

    finally:
        trader.stop()
        broker.disconnect()

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Total Signals: {trader.signals_generated}")
        print(f"Total Trades:  {trader.trades_executed}")
        print("=" * 70)


if __name__ == "__main__":
    main()
