#!/usr/bin/env python3
"""
Orchestrator Auto Trader - Automatic Stock Selection + Multi-Edge Trading

This is the main entry point for live trading. It:
1. Scans for today's biggest movers (gaps, intraday surges)
2. Selects the top N most volatile/active stocks
3. Runs the Orchestrator multi-edge strategy on them
4. Trades automatically when 4+ edges agree

The Orchestrator uses 6 independent edge sources:
1. Market Regime - What type of day? (Trend/Range/Reversal)
2. Relative Strength - Is stock leading or lagging sector?
3. Statistical Extremes - Is price at a measurable extreme?
4. Volume Profile - Where is fair value?
5. Cross-Asset Confirmation - Do related stocks confirm?
6. Time-of-Day - What time pattern is active?

Only trades when 4+ edges agree. Any edge can veto.

Usage:
    # Dry run (always do this first!)
    python run_orchestrator_auto.py --dry-run

    # Live trading with auto-selected stocks
    python run_orchestrator_auto.py

    # Specify number of stocks to trade
    python run_orchestrator_auto.py --top 5

    # Trade specific symbols instead of scanning
    python run_orchestrator_auto.py --symbols INTC AMD NVDA

    # Mid-day rescan for new movers
    python run_orchestrator_auto.py --rescan 60
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Set

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.broker.base import OrderRequest
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.strategies.orchestrator import (
    Orchestrator,
    OrchestratorSignal,
    MarketRegime,
    EdgeVote,
    create_orchestrator,
)
from trading_algo.stock_selector.ibkr_scanner import (
    IBKRStockScanner,
    DAY_TRADE_UNIVERSE,
)


# Core reference assets (always loaded for market context)
CORE_REFERENCE_ASSETS = [
    "SPY", "QQQ", "IWM",  # Market indices
]

# Sector ETFs (loaded based on what stocks are selected)
SECTOR_ETFS = {
    "tech": ["XLK", "SMH"],
    "financials": ["XLF"],
    "energy": ["XLE"],
    "consumer": ["XLY"],
    "healthcare": ["XLV"],
    "materials": ["XME", "GDX"],
    "precious_metals": ["GLD", "SLV"],
}


def get_sector_etfs_for_symbols(symbols: List[str]) -> List[str]:
    """Determine which sector ETFs to load based on symbols being traded."""
    etfs = set()

    # Mapping of symbols to sectors
    symbol_sectors = {
        # Tech / Semiconductors
        "INTC": "tech", "AMD": "tech", "NVDA": "tech", "MU": "tech",
        "AVGO": "tech", "QCOM": "tech", "TSM": "tech", "SMCI": "tech",
        "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
        "SOXL": "tech", "TQQQ": "tech",
        # Financials
        "JPM": "financials", "BAC": "financials", "GS": "financials",
        "MS": "financials", "C": "financials", "WFC": "financials",
        # Energy
        "XOM": "energy", "CVX": "energy", "OXY": "energy", "SLB": "energy",
        # Consumer
        "TSLA": "consumer", "AMZN": "consumer", "SBUX": "consumer",
        "NKE": "consumer", "DIS": "consumer", "MCD": "consumer",
        # Healthcare
        "UNH": "healthcare", "JNJ": "healthcare", "PFE": "healthcare",
        # Materials / Mining
        "GOLD": "materials", "NEM": "materials", "FCX": "materials",
        # Precious metals ETFs
        "GLD": "precious_metals", "SLV": "precious_metals",
        "SIVR": "precious_metals", "GDX": "precious_metals",
    }

    for symbol in symbols:
        sector = symbol_sectors.get(symbol.upper())
        if sector and sector in SECTOR_ETFS:
            for etf in SECTOR_ETFS[sector]:
                etfs.add(etf)

    # Always include core tech ETFs as they're most commonly traded
    etfs.add("SMH")
    etfs.add("XLK")

    return list(etfs)


class OrchestratorAutoTrader:
    """Automatic stock selection + Orchestrator trading."""

    def __init__(
        self,
        broker: IBKRBroker,
        symbols: Optional[List[str]] = None,
        top_n: int = 5,
        dry_run: bool = False,
        max_position_dollars: float = 10000,
        rescan_interval: Optional[int] = None,
    ):
        self.broker = broker
        self.initial_symbols = [s.upper() for s in symbols] if symbols else None
        self.top_n = top_n
        self.dry_run = dry_run
        self.max_position_dollars = max_position_dollars
        self.rescan_interval = rescan_interval

        # Will be set after scanning
        self.symbols: List[str] = []
        self.reference_assets: List[str] = []
        self.all_symbols: List[str] = []

        # Create the orchestrator
        self.orchestrator = create_orchestrator()

        # Scanner for finding movers
        self.scanner = IBKRStockScanner(broker)

        # Stats
        self.signals_generated = 0
        self.trades_executed = 0

        # Account
        self.account_value = 100_000

        self.running = False
        self._last_bar_time: Dict[str, datetime] = {}
        self._last_scan_time: Optional[datetime] = None

    def scan_for_movers(self) -> List[str]:
        """Scan for today's biggest movers."""
        print("\n" + "=" * 70)
        print("SCANNING FOR TODAY'S MOVERS")
        print("=" * 70)

        try:
            result = self.scanner.scan_todays_movers(
                top_n=self.top_n * 2,  # Scan for more, then pick top N
                min_gap_pct=0.01,  # At least 1% gap/move
                verbose=True,
            )

            if not result.candidates:
                print("No movers found! Using default universe subset.")
                return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"][:self.top_n]

            symbols = [c.symbol for c in result.candidates[:self.top_n]]

            print(f"\nSelected for trading: {', '.join(symbols)}")
            return symbols

        except Exception as e:
            print(f"Error during scan: {e}")
            print("Using default universe subset.")
            return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"][:self.top_n]

    def start(self):
        """Start the auto trader."""
        self.running = True

        # Get account value
        try:
            account = self.broker.get_account_snapshot()
            self.account_value = account.values.get('NetLiquidation', 100_000)
            print(f"Account Value: ${self.account_value:,.2f}")
        except Exception as e:
            print(f"Warning: Could not get account value: {e}")

        # Determine symbols to trade
        if self.initial_symbols:
            self.symbols = self.initial_symbols
            print(f"\nUsing specified symbols: {', '.join(self.symbols)}")
        else:
            self.symbols = self.scan_for_movers()
            self._last_scan_time = datetime.now()

        # Determine reference assets
        sector_etfs = get_sector_etfs_for_symbols(self.symbols)
        self.reference_assets = list(set(CORE_REFERENCE_ASSETS + sector_etfs))
        self.all_symbols = list(set(self.symbols + self.reference_assets))

        print(f"Reference assets: {', '.join(self.reference_assets)}")

        # Warm up with historical data
        self._warmup()

    def stop(self):
        """Stop trading."""
        self.running = False

    def _warmup(self):
        """Warm up the orchestrator with recent data."""
        print("\nWarming up with historical data...")

        for i, symbol in enumerate(self.all_symbols):
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="2 D",
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                is_ref = symbol in self.reference_assets and symbol not in self.symbols
                marker = "(ref)" if is_ref else ""
                print(f"  [{i+1}/{len(self.all_symbols)}] {symbol}: {len(bars)} bars {marker}")

                for bar in bars:
                    ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
                    self.orchestrator.update_asset(
                        symbol=symbol,
                        timestamp=ts,
                        open_price=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume or 0,
                    )
                    self._last_bar_time[symbol] = ts

            except Exception as e:
                print(f"  Warning: Could not warm up {symbol}: {e}")

            time.sleep(0.5)

        self.orchestrator.clear_positions()
        print("Warmup complete!")

    def maybe_rescan(self):
        """Check if we should rescan for new movers."""
        if self.rescan_interval is None or self.initial_symbols:
            return  # No rescan if disabled or using fixed symbols

        if self._last_scan_time is None:
            return

        elapsed = (datetime.now() - self._last_scan_time).total_seconds() / 60
        if elapsed >= self.rescan_interval:
            print(f"\n[Rescan] {elapsed:.0f} minutes since last scan, checking for new movers...")

            # Only rescan if we don't have open positions
            if self.orchestrator.positions:
                print(f"[Rescan] Skipping - have {len(self.orchestrator.positions)} open positions")
                return

            new_symbols = self.scan_for_movers()

            # Check if symbols changed
            if set(new_symbols) != set(self.symbols):
                print(f"[Rescan] New symbols detected! Switching from {self.symbols} to {new_symbols}")
                self.symbols = new_symbols
                sector_etfs = get_sector_etfs_for_symbols(self.symbols)
                self.reference_assets = list(set(CORE_REFERENCE_ASSETS + sector_etfs))
                self.all_symbols = list(set(self.symbols + self.reference_assets))
                self._warmup()
            else:
                print("[Rescan] Same symbols, continuing...")

            self._last_scan_time = datetime.now()

    def update(self) -> Dict[str, OrchestratorSignal]:
        """Fetch latest data and get signals."""
        signals = {}

        # Update all assets
        for symbol in self.all_symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
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

                for bar in bars:
                    ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
                    if symbol in self._last_bar_time and ts <= self._last_bar_time[symbol]:
                        continue

                    self.orchestrator.update_asset(
                        symbol=symbol,
                        timestamp=ts,
                        open_price=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume or 0,
                    )
                    self._last_bar_time[symbol] = ts

            except Exception as e:
                if "pacing" not in str(e).lower():
                    print(f"  Error updating {symbol}: {e}")

        # Generate signals for trading symbols only
        for symbol in self.symbols:
            try:
                if symbol not in self._last_bar_time:
                    continue

                ts = self._last_bar_time[symbol]
                signal = self.orchestrator.generate_signal(symbol, ts)

                if signal:
                    signals[symbol] = signal
                    self.signals_generated += 1

                    if signal.action != "hold":
                        self._print_signal(signal)
                    else:
                        state = self.orchestrator.asset_states.get(symbol)
                        price = state.prices[-1] if state and state.prices else 0
                        regime = signal.market_regime.name if signal.market_regime else "?"
                        reason = signal.reason[:50] if signal.reason else ""
                        print(f"  [{symbol}] ${price:.2f} | {regime} | HOLD: {reason}")

            except Exception as e:
                print(f"  Error generating signal for {symbol}: {e}")

        return signals

    def _print_signal(self, signal: OrchestratorSignal):
        """Print a trading signal with details."""
        print(f"\n{'='*60}")
        print(f">>> {signal.action.upper()} {signal.symbol} @ ${signal.entry_price:.2f}")
        print(f"{'='*60}")
        print(f"  Regime:     {signal.market_regime.name}")
        print(f"  Consensus:  {signal.consensus_score:.2f}")
        print(f"  Confidence: {signal.confidence:.0%}")
        if signal.stop_loss:
            print(f"  Stop:       ${signal.stop_loss:.2f}")
        if signal.take_profit:
            print(f"  Target:     ${signal.take_profit:.2f}")
        print(f"  Votes: {', '.join([f'{k}:{v.name}' for k, v in signal.edge_votes.items()])}")
        print(f"{'='*60}")

    def execute(self, signals: Dict[str, OrchestratorSignal]):
        """Execute trading signals."""
        for symbol, signal in signals.items():
            if signal.action == "hold":
                continue

            if signal.action == "buy":
                self._execute_buy(signal)
            elif signal.action == "short":
                self._execute_short(signal)
            elif signal.action == "sell":
                self._execute_sell(signal)
            elif signal.action == "cover":
                self._execute_cover(signal)

    def _execute_buy(self, signal: OrchestratorSignal):
        """Execute buy order."""
        try:
            price = signal.entry_price
            position_value = self.account_value * signal.size
            position_value = min(position_value, self.max_position_dollars)
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> BUY {signal.symbol}")
            print(f"    Quantity: {quantity} shares @ ${price:.2f}")
            print(f"    Value: ${quantity * price:,.2f}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange="SMART", currency="USD")
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

    def _execute_short(self, signal: OrchestratorSignal):
        """Execute short order."""
        try:
            price = signal.entry_price
            position_value = self.account_value * signal.size
            position_value = min(position_value, self.max_position_dollars)
            quantity = int(position_value / price)

            if quantity <= 0:
                return

            print(f"\n>>> SHORT {signal.symbol}")
            print(f"    Quantity: {quantity} shares @ ${price:.2f}")

            if self.dry_run:
                print("    [DRY RUN]")
                return

            instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange="SMART", currency="USD")
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

    def _execute_sell(self, signal: OrchestratorSignal):
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
                    instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange="SMART", currency="USD")
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

    def _execute_cover(self, signal: OrchestratorSignal):
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
                    instrument = InstrumentSpec(kind="STK", symbol=signal.symbol, exchange="SMART", currency="USD")
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
        """Print current status."""
        now = datetime.now()
        regime, conf, reason = self.orchestrator.regime_engine.detect_regime(now)

        print(f"\n[{now.strftime('%H:%M:%S')}] Orchestrator Auto Trader")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Signals: {self.signals_generated}")
        print(f"  Trades:  {self.trades_executed}")
        print(f"  Active:  {len(self.orchestrator.positions)}")
        print(f"  Regime:  {regime.name} ({conf:.0%})")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator Auto Trader - Automatic Stock Selection + Multi-Edge Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_orchestrator_auto.py --dry-run              # Dry run with auto-selected stocks
  python run_orchestrator_auto.py --top 5                # Trade top 5 movers
  python run_orchestrator_auto.py --symbols GLD SIVR     # Trade specific symbols
  python run_orchestrator_auto.py --rescan 60            # Rescan for movers every 60 min
        """
    )
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to trade (skips scanning)")
    parser.add_argument("--top", type=int, default=5, help="Number of top movers to trade (default: 5)")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no real orders)")
    parser.add_argument("--max-position", type=float, default=10000, help="Max position size in dollars (default: 10000)")
    parser.add_argument("--rescan", type=int, help="Rescan for new movers every N minutes")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port (default: 7497)")

    args = parser.parse_args()

    print("=" * 70)
    print("ORCHESTRATOR AUTO TRADER")
    print("Multi-Edge Ensemble Trading with Automatic Stock Selection")
    print("=" * 70)
    print()
    print("Strategy: 6 independent edge sources must agree (4+ required)")
    print("  1. Market Regime    - What type of day?")
    print("  2. Relative Strength - Leading or lagging sector?")
    print("  3. Statistics       - At measurable extreme?")
    print("  4. Volume Profile   - Where is fair value?")
    print("  5. Cross-Asset      - Do related stocks confirm?")
    print("  6. Time-of-Day      - What time pattern?")
    print()
    if args.symbols:
        print(f"Mode:       Trading specified symbols: {', '.join(args.symbols)}")
    else:
        print(f"Mode:       Auto-scan for top {args.top} movers")
    print(f"Max Pos:    ${args.max_position:,.0f}")
    print(f"Interval:   {args.interval} seconds")
    print(f"Dry Run:    {args.dry_run}")
    if args.rescan:
        print(f"Rescan:     Every {args.rescan} minutes")
    print()
    print("WARNING: Day trading is risky! Use at your own risk.")
    print("=" * 70)

    # Connect
    config = IBKRConfig(host="127.0.0.1", port=args.port, client_id=50)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

    trader = OrchestratorAutoTrader(
        broker=broker,
        symbols=args.symbols,
        top_n=args.top,
        dry_run=args.dry_run,
        max_position_dollars=args.max_position,
        rescan_interval=args.rescan,
    )

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
            if args.duration and (time.time() - start_time) >= args.duration:
                print("\nDuration reached.")
                break

            # Check market hours
            now = datetime.now().time()
            if now >= dt_time(16, 5):
                print("\nMarket closed. Auto-stopping.")
                break

            # Maybe rescan for new movers
            trader.maybe_rescan()

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
