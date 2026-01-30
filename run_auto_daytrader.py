#!/usr/bin/env python3
"""
Auto Day Trader - Scan, Select, Trade.

Automatically scans the market for the best day trading candidates,
selects the top volatile stocks, and trades them using the Chameleon
Day Trader v3 algorithm.

Pipeline:
    1. Connect to IBKR
    2. Scan ~80 stocks for day trading suitability (full scan or today's movers)
    3. Rank by composite score (volatility, volume, momentum, technicals)
    4. Select top N candidates
    5. Initialize Chameleon Day Trader v3 for each
    6. Trade all day with adaptive risk management
    7. Mid-day rescan (optional) to swap underperforming symbols

Usage:
    # Full auto mode (scan + trade, dry run first!)
    python run_auto_daytrader.py --dry-run

    # Today's movers mode (faster, finds today's biggest gaps/moves)
    python run_auto_daytrader.py --movers --dry-run

    # With mid-day rescan every 60 minutes
    python run_auto_daytrader.py --movers --rescan 60 --dry-run

    # Specify number of stocks to trade
    python run_auto_daytrader.py --top 5 --dry-run

    # Conservative mode
    python run_auto_daytrader.py --top 3 --conservative --dry-run

    # Live trading
    python run_auto_daytrader.py --top 5

    # Scan only (no trading)
    python run_auto_daytrader.py --scan-only

    # Trade specific symbols (skip scan)
    python run_auto_daytrader.py --symbols TSLA AMD COIN
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
from trading_algo.stock_selector.ibkr_scanner import (
    IBKRStockScanner,
    ScanResult,
    print_scan_results,
)
from trading_algo.rat.chameleon_daytrader import (
    ChameleonDayTrader,
    DayTradeMode,
    DayTradeSignal,
    MarketConfig,
    MARKET_PRESETS,
    create_daytrader,
    list_markets,
)


class AutoDayTrader:
    """
    Automated day trading system.

    Combines stock scanning with the Chameleon Day Trader v3.
    Scans for the best candidates, then trades them all day.
    Supports mid-day rescans to swap underperforming symbols.
    """

    def __init__(
        self,
        broker: IBKRBroker,
        aggressive: bool = True,
        dry_run: bool = False,
        max_position_dollars: float = 10000,
        market: str = 'NYSE',
        top_n: int = 5,
        use_movers: bool = False,
        rescan_minutes: int = 0,  # 0 = no rescan
    ):
        self.broker = broker
        self.aggressive = aggressive
        self.dry_run = dry_run
        self.max_position_dollars = max_position_dollars
        self.market = market.upper()
        self.market_config = MARKET_PRESETS.get(self.market, MARKET_PRESETS['NYSE'])
        self.top_n = top_n
        self.use_movers = use_movers
        self.rescan_minutes = rescan_minutes

        # Will be populated after scan
        self.symbols: List[str] = []
        self.traders: Dict[str, ChameleonDayTrader] = {}
        self.scan_result: Optional[ScanResult] = None
        self._scanner: Optional[IBKRStockScanner] = None
        self._last_rescan_time: float = 0.0

        # Stats
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0

        # Account
        self.account_value = 100_000

        self.running = False

    def scan_and_select(self) -> List[str]:
        """Scan the market and select top day trading candidates."""
        print("\n" + "=" * 70)
        if self.use_movers:
            print("SCANNING TODAY'S MOVERS (fast mode)")
        else:
            print("FULL MARKET SCAN FOR DAY TRADING CANDIDATES")
        print("=" * 70)

        self._scanner = IBKRStockScanner(
            broker=self.broker,
            exchange=self.market_config.exchange,
            currency=self.market_config.currency,
            min_atr_pct=0.015,  # 1.5% minimum daily ATR
            min_avg_volume=500_000,
            data_days=60,
        )

        if self.use_movers:
            self.scan_result = self._scanner.scan_todays_movers(
                top_n=self.top_n,
                verbose=True,
            )
        else:
            self.scan_result = self._scanner.scan(
                top_n=self.top_n,
                verbose=True,
            )

        # Print results
        print_scan_results(self.scan_result)

        # Extract symbols
        symbols = [c.symbol for c in self.scan_result.candidates]

        if not symbols:
            print("\nWARNING: No suitable day trading candidates found!")
            return []

        print(f"\nSELECTED FOR TRADING: {', '.join(symbols)}")
        self._last_rescan_time = time.time()
        return symbols

    def check_rescan(self) -> bool:
        """Check if it's time for a mid-day rescan. Returns True if symbols changed."""
        if self.rescan_minutes <= 0 or self._scanner is None:
            return False

        elapsed = (time.time() - self._last_rescan_time) / 60
        if elapsed < self.rescan_minutes:
            return False

        print("\n" + "=" * 70)
        print("MID-DAY RESCAN")
        print("=" * 70)

        # Rescan using today's movers (always fast mode for mid-day)
        new_result = self._scanner.scan_todays_movers(
            top_n=self.top_n,
            verbose=True,
        )

        new_symbols = [c.symbol for c in new_result.candidates]
        self._last_rescan_time = time.time()

        if not new_symbols:
            print("  No better candidates found. Keeping current symbols.")
            return False

        # Check if the new picks are different
        old_set = set(self.symbols)
        new_set = set(new_symbols)
        added = new_set - old_set
        removed = old_set - new_set

        if not added:
            print("  Same candidates. No changes needed.")
            return False

        print(f"\n  NEW:     {', '.join(added)}")
        print(f"  REMOVED: {', '.join(removed)}")
        print(f"  KEPT:    {', '.join(old_set & new_set)}")

        # Add new symbols
        for symbol in added:
            self.symbols.append(symbol)
            self.traders[symbol] = create_daytrader(
                aggressive=self.aggressive,
                max_position_dollars=self.max_position_dollars,
                market=self.market,
            )
            print(f"  Added {symbol} to trading roster")

        # Remove symbols that are no longer in the top (only if no open position)
        for symbol in removed:
            trader = self.traders.get(symbol)
            if trader and not trader.positions:
                self.symbols.remove(symbol)
                del self.traders[symbol]
                print(f"  Removed {symbol} (no open position)")
            else:
                print(f"  Keeping {symbol} (has open position)")

        # Warm up new symbols
        for symbol in added:
            self._warmup_symbol(symbol)

        self.scan_result = new_result
        print(f"\n  Now trading: {', '.join(self.symbols)}")
        return True

    def initialize(self, symbols: List[str]):
        """Initialize traders for the given symbols."""
        self.symbols = [s.upper() for s in symbols]

        # Create day traders (one per symbol)
        self.traders = {
            symbol: create_daytrader(
                aggressive=self.aggressive,
                max_position_dollars=self.max_position_dollars,
                market=self.market,
            )
            for symbol in self.symbols
        }

    def start(self):
        """Start the auto day trader."""
        self.running = True

        # Get account value
        try:
            account = self.broker.get_account_snapshot()
            self.account_value = account.values.get('NetLiquidation', 100_000)
            print(f"\nAccount Value: ${self.account_value:,.2f}")
        except Exception as e:
            print(f"Warning: Could not get account value: {e}")

        # Warm up with historical data
        self._warmup()

    def stop(self):
        """Stop trading."""
        self.running = False

    def _warmup(self):
        """Warm up strategies with recent intraday data."""
        print("\nWarming up with recent 5-min bars...")
        for symbol in self.symbols:
            self._warmup_symbol(symbol)

    def _warmup_symbol(self, symbol: str):
        """Warm up a single symbol's strategy."""
        try:
            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange=self.market_config.exchange,
                currency=self.market_config.currency,
            )

            bars = self.broker.get_historical_bars(
                instrument,
                duration="1 D",
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

            # Clear positions created during warmup
            trader.clear_positions()

        except Exception as e:
            print(f"  Warning: Could not warm up {symbol}: {e}")

    def update(self) -> Dict[str, DayTradeSignal]:
        """Fetch latest data and get signals for all symbols."""
        signals = {}

        for symbol in self.symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange=self.market_config.exchange,
                    currency=self.market_config.currency,
                )

                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="1800 S",
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
            position_value = self.account_value * signal.size
            position_value = min(position_value, self.max_position_dollars)
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> BUY {signal.symbol}")
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity} shares (${quantity * price:,.2f})")
            print(f"    Stop: ${signal.stop_loss:.2f} | Target: ${signal.take_profit:.2f}")
            print(f"    Confidence: {signal.confidence:.0%}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            instrument = InstrumentSpec(
                kind="STK", symbol=signal.symbol,
                exchange=self.market_config.exchange,
                currency=self.market_config.currency,
            )
            order = OrderRequest(
                instrument=instrument,
                side="BUY", quantity=quantity,
                order_type="MKT", tif="DAY",
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
            position_value = self.account_value * signal.size
            position_value = min(position_value, self.max_position_dollars)
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> SHORT {signal.symbol}")
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity} shares")
            print(f"    Stop: ${signal.stop_loss:.2f} | Target: ${signal.take_profit:.2f}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            instrument = InstrumentSpec(
                kind="STK", symbol=signal.symbol,
                exchange=self.market_config.exchange,
                currency=self.market_config.currency,
            )
            order = OrderRequest(
                instrument=instrument,
                side="SELL", quantity=quantity,
                order_type="MKT", tif="DAY",
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

            positions = self.broker.get_positions()
            for pos in positions:
                if pos.instrument.symbol == signal.symbol and pos.quantity > 0:
                    instrument = InstrumentSpec(
                        kind="STK", symbol=signal.symbol,
                        exchange=self.market_config.exchange,
                        currency=self.market_config.currency,
                    )
                    order = OrderRequest(
                        instrument=instrument,
                        side="SELL", quantity=int(pos.quantity),
                        order_type="MKT", tif="DAY",
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
                    instrument = InstrumentSpec(
                        kind="STK", symbol=signal.symbol,
                        exchange=self.market_config.exchange,
                        currency=self.market_config.currency,
                    )
                    order = OrderRequest(
                        instrument=instrument,
                        side="BUY", quantity=int(abs(pos.quantity)),
                        order_type="MKT", tif="DAY",
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
        active_positions = sum(
            1 for t in self.traders.values() if t.positions
        )
        print(f"\n[{now.strftime('%H:%M:%S')}] Auto Day Trader Status")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Signals: {self.signals_generated}")
        print(f"  Trades: {self.trades_executed}")
        print(f"  Active Positions: {active_positions}")


def main():
    parser = argparse.ArgumentParser(description="Auto Day Trading - Scan + Select + Trade")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to trade (skip scan)")
    parser.add_argument("--top", type=int, default=5, help="Number of stocks to select (default: 5)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan, don't trade")
    parser.add_argument("--movers", action="store_true", help="Use today's movers scan (faster, finds gaps/surges)")
    parser.add_argument("--rescan", type=int, default=0, help="Mid-day rescan interval in minutes (0=disabled)")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--interval", type=int, default=30, help="Update interval (default: 30s)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--aggressive", action="store_true", default=True, help="Aggressive mode (default)")
    parser.add_argument("--conservative", action="store_true", help="Conservative mode")
    parser.add_argument("--max-position", type=float, default=10000, help="Max position $ (default: 10000)")
    parser.add_argument("--market", type=str, default="NYSE", help="Market: NYSE, HKEX, TSE, LSE, ASX")
    parser.add_argument("--list-markets", action="store_true", help="List markets and exit")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port")

    args = parser.parse_args()

    if args.list_markets:
        list_markets()
        sys.exit(0)

    aggressive = not args.conservative
    market_key = args.market.upper()
    mc = MARKET_PRESETS.get(market_key, MARKET_PRESETS['NYSE'])

    print("=" * 70)
    print("CHAMELEON AUTO DAY TRADER v3")
    print("=" * 70)
    print(f"Market:     {mc.name} ({market_key}) | {mc.exchange} | {mc.currency}")
    print(f"Hours:      {mc.market_open.strftime('%H:%M')}-{mc.market_close.strftime('%H:%M')} ({mc.timezone})")
    if mc.lunch_start:
        print(f"Lunch:      {mc.lunch_start.strftime('%H:%M')}-{mc.lunch_end.strftime('%H:%M')}")
    print(f"Mode:       {'AGGRESSIVE' if aggressive else 'CONSERVATIVE'}")
    print(f"Max Pos:    ${args.max_position:,.0f}")
    print(f"Dry Run:    {args.dry_run}")
    if args.symbols:
        print(f"Symbols:    {', '.join(args.symbols)} (manual)")
    else:
        scan_mode = "Today's movers (fast)" if args.movers else "Full multi-factor scan"
        print(f"Selection:  Auto-scan top {args.top} ({scan_mode})")
    if args.rescan > 0:
        print(f"Rescan:     Every {args.rescan} minutes (swap underperformers)")
    print()
    print("v3 FEATURES:")
    print("  - Auto stock scanning (selects best volatile candidates)")
    print("  - Today's movers detection (finds gaps and surges)")
    print("  - Mid-day rescan (swaps underperforming symbols)")
    print("  - Trend clarity filter (avoids choppy stocks)")
    print("  - Chop detection (rejects whipsaw-prone conditions)")
    print("  - Min ATR gate (skips tight-range stocks)")
    print("  - Volatility-adaptive sizing (risk-parity)")
    print()
    print("WARNING: Day trading is risky! Use at your own risk.")
    print("=" * 70)

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=args.port, client_id=20)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

    auto_trader = AutoDayTrader(
        broker=broker,
        aggressive=aggressive,
        dry_run=args.dry_run,
        max_position_dollars=args.max_position,
        market=market_key,
        top_n=args.top,
        use_movers=args.movers,
        rescan_minutes=args.rescan,
    )

    try:
        # Phase 1: Stock Selection
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
            print(f"\nUsing manually specified symbols: {', '.join(symbols)}")
        else:
            symbols = auto_trader.scan_and_select()

        if not symbols:
            print("\nNo stocks to trade. Exiting.")
            sys.exit(0)

        if args.scan_only:
            print("\nScan complete. Use --symbols to trade specific picks.")
            sys.exit(0)

        # Phase 2: Initialize and Trade
        auto_trader.initialize(symbols)
        auto_trader.start()

        print("\n" + "=" * 70)
        print(f"TRADING STARTED - {', '.join(symbols)} - Press Ctrl+C to stop")
        print("=" * 70)

        def signal_handler(sig, frame):
            print("\n\nShutting down...")
            auto_trader.stop()

        signal.signal(signal.SIGINT, signal_handler)

        start_time = time.time()
        update_count = 0

        while auto_trader.running:
            if args.duration and (time.time() - start_time) >= args.duration:
                print("\nDuration reached.")
                break

            # Auto-stop after market close
            now = datetime.now().time()
            close_dt = datetime.combine(datetime.today(), mc.market_close) + timedelta(minutes=5)
            if now >= close_dt.time():
                print(f"\nMarket closed at {mc.market_close.strftime('%H:%M')}. Auto-stopping.")
                break

            signals = auto_trader.update()
            auto_trader.execute(signals)

            update_count += 1
            if update_count % 4 == 0:
                auto_trader.print_status()

            # Mid-day rescan check
            if auto_trader.rescan_minutes > 0 and update_count % 10 == 0:
                auto_trader.check_rescan()

            time.sleep(args.interval)

    finally:
        auto_trader.stop()
        broker.disconnect()

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Symbols Traded: {', '.join(auto_trader.symbols)}")
        print(f"Total Signals:  {auto_trader.signals_generated}")
        print(f"Total Trades:   {auto_trader.trades_executed}")
        if auto_trader.scan_result:
            print(f"Market Regime:  {auto_trader.scan_result.market_regime.name}")
        print("=" * 70)


if __name__ == "__main__":
    main()
