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
import threading
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Set

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.broker.base import OrderRequest
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.orchestrator import (
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

# Dashboard imports (optional, graceful fallback if textual not installed)
try:
    from trading_algo.dashboard import (
        TradingDashboard,
        OrchestratorAdapter,
        get_event_bus,
        get_store,
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Backtest imports
try:
    from trading_algo.backtest_v2 import (
        BacktestEngine,
        BacktestConfig,
        BacktestExporter,
        DataProvider,
        DataRequest,
    )
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False


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
        use_dashboard: bool = False,
    ):
        self.broker = broker
        self.initial_symbols = [s.upper() for s in symbols] if symbols else None
        self.top_n = top_n
        self.dry_run = dry_run
        self.max_position_dollars = max_position_dollars
        self.rescan_interval = rescan_interval
        self.use_dashboard = use_dashboard and DASHBOARD_AVAILABLE

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

        # Dashboard adapter
        self.adapter: Optional[OrchestratorAdapter] = None
        if self.use_dashboard:
            self.adapter = OrchestratorAdapter()

    def log(self, level: str, message: str):
        """Log a message (to console and dashboard if available)."""
        if self.adapter:
            self.adapter.log(level, message)
        else:
            print(f"[{level}] {message}")

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

        # Start dashboard adapter
        if self.adapter:
            self.adapter.start()

        # Get account value
        try:
            account = self.broker.get_account_snapshot()
            self.account_value = account.values.get('NetLiquidation', 100_000)
            self.log("INFO", f"Account Value: ${self.account_value:,.2f}")
        except Exception as e:
            self.log("WARNING", f"Could not get account value: {e}")

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
        if self.adapter:
            self.adapter.stop()

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

                    # Emit to dashboard
                    if self.adapter:
                        self.adapter.on_orchestrator_signal(
                            symbol=symbol,
                            action=signal.action,
                            price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            confidence=signal.confidence,
                            consensus_score=signal.consensus_score,
                            edge_votes={k: v.name for k, v in signal.edge_votes.items()},
                            reason=signal.reason,
                        )

                    if signal.action != "hold":
                        self._print_signal(signal)
                    elif not self.use_dashboard:
                        state = self.orchestrator.asset_states.get(symbol)
                        price = state.prices[-1] if state and state.prices else 0
                        regime = signal.market_regime.name if signal.market_regime else "?"
                        reason = signal.reason[:50] if signal.reason else ""
                        print(f"  [{symbol}] ${price:.2f} | {regime} | HOLD: {reason}")

            except Exception as e:
                if self.adapter:
                    self.adapter.error("SignalError", str(e))
                else:
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
  python run.py --dry-run                    # Dry run with auto-selected stocks
  python run.py --dashboard                  # Run with interactive dashboard
  python run.py --top 5                      # Trade top 5 movers
  python run.py --symbols GLD SIVR           # Trade specific symbols
  python run.py --rescan 60                  # Rescan for movers every 60 min
  python run.py --dashboard --dry-run        # Dashboard in dry-run mode
  python run.py --backtest --symbols SPY     # Run backtest for SPY
  python run.py --backtest --start 2025-01-01 --end 2025-12-31  # Custom date range
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
    parser.add_argument("--dashboard", action="store_true", help="Run with interactive TUI dashboard")

    # Backtest arguments
    parser.add_argument("--backtest", action="store_true", help="Run backtest instead of live trading")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital for backtest (default: 100000)")
    parser.add_argument("--export", action="store_true", help="Export backtest results to folder")

    args = parser.parse_args()

    # Check if backtest mode
    if args.backtest:
        _run_backtest(args)
        return

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

    # Check dashboard availability
    if args.dashboard and not DASHBOARD_AVAILABLE:
        print("\nWARNING: Dashboard requested but 'textual' is not installed.")
        print("Install with: pip install textual")
        print("Falling back to console mode.\n")
        args.dashboard = False

    trader = OrchestratorAutoTrader(
        broker=broker,
        symbols=args.symbols,
        top_n=args.top,
        dry_run=args.dry_run,
        max_position_dollars=args.max_position,
        rescan_interval=args.rescan,
        use_dashboard=args.dashboard,
    )

    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        trader.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        trader.start()

        if args.dashboard:
            # Run with dashboard UI
            _run_with_dashboard(trader, broker, args)
        else:
            # Run in console mode
            _run_console_mode(trader, broker, args)

    finally:
        trader.stop()
        broker.disconnect()

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Total Signals: {trader.signals_generated}")
        print(f"Total Trades:  {trader.trades_executed}")
        print("=" * 70)


def _run_console_mode(trader: OrchestratorAutoTrader, broker: IBKRBroker, args):
    """Run in console mode (no dashboard)."""
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


def _run_with_dashboard(trader: OrchestratorAutoTrader, broker: IBKRBroker, args):
    """Run with interactive TUI dashboard."""
    from trading_algo.dashboard import TradingDashboard, get_store

    # Create backtest callback for dashboard
    async def run_backtest_callback(config: dict, progress_callback) -> "BacktestResults":
        """Callback for running backtests from dashboard."""
        if not BACKTEST_AVAILABLE:
            raise RuntimeError("Backtest module not available")

        from datetime import date

        # Create backtest config
        bt_config = BacktestConfig(
            symbols=config["symbols"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            initial_capital=config["initial_capital"],
            bar_size=config["bar_size"],
            algorithm_name="Orchestrator",
        )

        # Get data
        data_provider = DataProvider()
        progress_callback(0.1, "Fetching historical data...")

        requests = [
            DataRequest(
                symbol=s,
                start_date=config["start_date"],
                end_date=config["end_date"],
                bar_size=config["bar_size"],
            )
            for s in config["symbols"]
        ]

        data = data_provider.get_data(
            requests,
            lambda pct, msg: progress_callback(0.1 + pct * 0.4, msg),
        )

        if not data:
            # Use sample data if no real data
            progress_callback(0.2, "No data found, generating sample data...")
            for s in config["symbols"]:
                data[s] = data_provider.generate_sample_data(
                    s,
                    config["start_date"],
                    config["end_date"],
                    config["bar_size"],
                )

        # Create orchestrator for backtest
        progress_callback(0.5, "Running backtest...")
        test_orchestrator = create_orchestrator()

        # Create engine
        engine = BacktestEngine(bt_config, test_orchestrator)

        # Run backtest
        results = engine.run(data, lambda pct, msg: progress_callback(0.5 + pct * 0.5, msg))

        return results

    # Create trading loop thread
    stop_event = threading.Event()

    def trading_loop():
        """Background trading loop."""
        start_time = time.time()

        while trader.running and not stop_event.is_set():
            if args.duration and (time.time() - start_time) >= args.duration:
                trader.log("INFO", "Duration reached. Stopping...")
                trader.stop()
                break

            # Check market hours
            now = datetime.now().time()
            if now >= dt_time(16, 5):
                trader.log("INFO", "Market closed. Auto-stopping.")
                trader.stop()
                break

            try:
                # Maybe rescan for new movers
                trader.maybe_rescan()

                # Update and execute
                signals = trader.update()
                trader.execute(signals)

            except Exception as e:
                if trader.adapter:
                    trader.adapter.error("TradingLoop", str(e))

            time.sleep(args.interval)

    # Start trading in background thread
    trading_thread = threading.Thread(target=trading_loop, daemon=True)
    trading_thread.start()

    # Run dashboard in main thread
    try:
        dashboard = TradingDashboard(
            algorithm_name="Orchestrator Auto Trader",
            store=get_store(),
            backtest_callback=run_backtest_callback if BACKTEST_AVAILABLE else None,
        )
        dashboard.run()
    finally:
        stop_event.set()
        trader.stop()


def _run_backtest(args):
    """Run backtest mode."""
    if not BACKTEST_AVAILABLE:
        print("ERROR: Backtest module not available.")
        print("Make sure trading_algo/backtest_v2 exists.")
        sys.exit(1)

    from datetime import date

    print("=" * 70)
    print("ORCHESTRATOR BACKTEST")
    print("Historical Performance Analysis")
    print("=" * 70)
    print()

    # Parse dates
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = date.today() - timedelta(days=365)

    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today()

    # Get symbols
    symbols = args.symbols if args.symbols else ["SPY"]

    print(f"Symbols:     {', '.join(symbols)}")
    print(f"Period:      {start_date} to {end_date}")
    print(f"Capital:     ${args.capital:,.2f}")
    print()

    # Create config
    config = BacktestConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        bar_size="5 mins",
        algorithm_name="Orchestrator",
    )

    # Get data
    print("Loading historical data...")
    data_provider = DataProvider()

    requests = [
        DataRequest(
            symbol=s,
            start_date=start_date,
            end_date=end_date,
            bar_size="5 mins",
        )
        for s in symbols
    ]

    def progress(pct, msg):
        bars = int(pct * 40)
        print(f"\r  [{('=' * bars).ljust(40)}] {pct*100:.0f}% {msg}", end="", flush=True)

    data = data_provider.get_data(requests, progress)
    print()

    if not data:
        print("\nNo historical data found. Generating sample data for testing...")
        for s in symbols:
            data[s] = data_provider.generate_sample_data(s, start_date, end_date, "5 mins")
            print(f"  Generated {len(data[s])} bars for {s}")

    # Create orchestrator and engine
    print("\nRunning backtest...")
    orchestrator = create_orchestrator()
    engine = BacktestEngine(config, orchestrator)

    # Run backtest
    results = engine.run(data, progress)
    print()

    # Print results
    m = results.metrics
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    pnl_color = "\033[92m" if m.net_profit >= 0 else "\033[91m"
    reset = "\033[0m"

    print(f"\n{'Performance Summary':^70}")
    print("-" * 70)
    print(f"  Total P&L:         {pnl_color}${m.net_profit:>15,.2f}{reset}")
    print(f"  Total Return:      {pnl_color}{m.total_return_pct:>15.2f}%{reset}")
    print(f"  Annualized Return: {pnl_color}{m.annualized_return:>15.2f}%{reset}")
    print(f"  Sharpe Ratio:      {m.sharpe_ratio:>15.2f}")
    print(f"  Sortino Ratio:     {m.sortino_ratio:>15.2f}")
    print(f"  Max Drawdown:      \033[91m{m.max_drawdown_pct:>15.2f}%{reset}")

    print(f"\n{'Trade Statistics':^70}")
    print("-" * 70)
    print(f"  Total Trades:      {m.total_trades:>15}")
    print(f"  Winning Trades:    {m.winning_trades:>15}")
    print(f"  Losing Trades:     {m.losing_trades:>15}")
    print(f"  Win Rate:          {m.win_rate:>15.1f}%")
    print(f"  Profit Factor:     {m.profit_factor:>15.2f}")
    print(f"  Expectancy:        ${m.expectancy:>14.2f}")
    print(f"  Avg Win:           ${m.avg_win:>14.2f}")
    print(f"  Avg Loss:          ${m.avg_loss:>14.2f}")

    print(f"\n{'Daily Performance':^70}")
    print("-" * 70)
    print(f"  Trading Days:      {len(results.daily_results):>15}")
    print(f"  Avg Daily P&L:     ${m.avg_daily_pnl:>14.2f}")
    print(f"  Best Day:          ${m.best_day:>14.2f}")
    print(f"  Worst Day:         ${m.worst_day:>14.2f}")
    print(f"  Win Streak:        {m.max_consecutive_wins:>15}")
    print(f"  Loss Streak:       {m.max_consecutive_losses:>15}")

    print("=" * 70)

    # Export results if requested
    if args.export:
        print("\nExporting results...")
        exporter = BacktestExporter()
        export_path = exporter.export(results)
        print(f"Results exported to: {export_path}")
        print(f"  - summary.json")
        print(f"  - trades.csv")
        print(f"  - daily_pnl.csv")
        print(f"  - equity_curve.csv")
        print(f"  - metrics.json")
        print(f"  - report.html")
        if (export_path / "charts").exists():
            print(f"  - charts/ (PNG images)")

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
