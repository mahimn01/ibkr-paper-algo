#!/usr/bin/env python3
"""
The Orchestrator - Multi-Edge Ensemble Day Trading System

A fundamentally different approach that combines 6 independent edge sources
into a unified decision framework. Only trades when multiple edges agree.

Key Differences from Old Approach:
1. CONTEXT FIRST: Understands market regime before any trade
2. MULTI-ASSET: Never trades a stock in isolation
3. STATISTICAL: Only acts at measurable extremes
4. ENSEMBLE: Requires 4+ edges to agree (with veto power)
5. ANTI-FRAGILE: Detects and avoids trap setups

The 6 Edge Sources:
1. Market Regime - What type of day is it? (Trend/Range/Reversal)
2. Relative Strength - Is this stock leading or lagging its sector?
3. Statistics - Is price/volume at a statistical extreme?
4. Volume Profile - Where is "fair value" and are we extended?
5. Cross-Asset - Do related stocks confirm the move?
6. Time-of-Day - Is this the right time for this type of trade?

Usage:
    # Dry run first - always!
    python run_orchestrator.py INTC AMD NVDA --dry-run

    # Live trading
    python run_orchestrator.py INTC AMD NVDA

    # With specific reference assets
    python run_orchestrator.py INTC AMD NVDA --refs SPY QQQ SMH
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


# Reference assets we always need
DEFAULT_REFERENCE_ASSETS = [
    "SPY", "QQQ", "IWM",  # Market indices
    "SMH",  # Semiconductors
    "XLK", "XLF", "XLE", "XLY", "XLV",  # Sector ETFs
]


class OrchestratorLive:
    """Live trading execution for the Orchestrator strategy."""

    def __init__(
        self,
        broker: IBKRBroker,
        symbols: List[str],
        reference_assets: List[str],
        dry_run: bool = False,
        max_position_dollars: float = 10000,
    ):
        self.broker = broker
        self.symbols = [s.upper() for s in symbols]
        self.reference_assets = [s.upper() for s in reference_assets]
        self.all_symbols = list(set(self.symbols + self.reference_assets))
        self.dry_run = dry_run
        self.max_position_dollars = max_position_dollars

        # Create the orchestrator
        self.orchestrator = create_orchestrator()

        # Stats
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0

        # Account
        self.account_value = 100_000

        self.running = False

        # Track which bars we've seen (avoid double-processing)
        self._last_bar_time: Dict[str, datetime] = {}

    def start(self):
        """Start the orchestrator."""
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
        """Warm up the orchestrator with recent data."""
        print("\nWarming up with historical data...")
        print(f"  Loading {len(self.all_symbols)} assets...")

        for i, symbol in enumerate(self.all_symbols):
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                # Get 2 days of 5-min bars for warmup
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

            # IBKR pacing
            time.sleep(0.5)

        # Clear any positions from warmup
        self.orchestrator.clear_positions()
        print("  Warmup complete!")

    def update(self) -> Dict[str, OrchestratorSignal]:
        """Fetch latest data and get signals."""
        signals = {}

        # Update all assets (references first, then trading symbols)
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
                    duration="1800 S",  # Last 30 minutes
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                if not bars:
                    continue

                # Process new bars
                for bar in bars:
                    ts = datetime.fromtimestamp(bar.timestamp_epoch_s)

                    # Skip if we've already seen this bar
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

        # Generate signals for trading symbols only (not references)
        for symbol in self.symbols:
            try:
                if symbol not in self._last_bar_time:
                    continue

                ts = self._last_bar_time[symbol]
                signal = self.orchestrator.generate_signal(symbol, ts)

                if signal:
                    signals[symbol] = signal
                    self.signals_generated += 1

                    # Print signal info
                    if signal.action != "hold":
                        self._print_signal(signal)
                    else:
                        # Print hold reason (shortened)
                        state = self.orchestrator.asset_states.get(symbol)
                        price = state.prices[-1] if state and state.prices else 0
                        regime = signal.market_regime.name if signal.market_regime else "?"
                        reason = signal.reason[:60] if signal.reason else ""
                        print(f"  [{symbol}] ${price:.2f} | {regime} | HOLD: {reason}")

            except Exception as e:
                print(f"  Error generating signal for {symbol}: {e}")

        return signals

    def _print_signal(self, signal: OrchestratorSignal):
        """Print a trading signal with details."""
        print(f"\n{'='*60}")
        print(f"  SIGNAL: {signal.action.upper()} {signal.symbol}")
        print(f"{'='*60}")
        print(f"  Trade Type:  {signal.trade_type.name}")
        print(f"  Regime:      {signal.market_regime.name}")
        print(f"  Price:       ${signal.entry_price:.2f}")
        print(f"  Stop:        ${signal.stop_loss:.2f}" if signal.stop_loss else "")
        print(f"  Target:      ${signal.take_profit:.2f}" if signal.take_profit else "")
        print(f"  Size:        {signal.size*100:.1f}% of account")
        print(f"  Confidence:  {signal.confidence:.0%}")
        print(f"  Consensus:   {signal.consensus_score:.2f}")
        print()
        print("  Edge Votes:")
        for edge, vote in signal.edge_votes.items():
            reason = signal.edge_reasons.get(edge, "")[:50]
            print(f"    {edge:20s} {vote.name:15s} {reason}")
        print()
        print(f"  Reason: {signal.reason[:80]}")
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
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity} shares")
            print(f"    Value: ${quantity * price:,.2f}")
            print(f"    Stop: ${signal.stop_loss:.2f}")
            print(f"    Target: ${signal.take_profit:.2f}")

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
            print(f"    Price: ${price:.2f}")
            print(f"    Quantity: {quantity} shares")
            print(f"    Stop: ${signal.stop_loss:.2f}")
            print(f"    Target: ${signal.take_profit:.2f}")

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
        print(f"\n[{now.strftime('%H:%M:%S')}] Orchestrator Status")
        print(f"  Signals: {self.signals_generated}")
        print(f"  Trades:  {self.trades_executed}")
        print(f"  Active:  {len(self.orchestrator.positions)}")

        # Show current regime
        regime, conf, reason = self.orchestrator.regime_engine.detect_regime(now)
        print(f"  Regime:  {regime.name} ({conf:.0%})")


def main():
    parser = argparse.ArgumentParser(description="Orchestrator Day Trading System")
    parser.add_argument("symbols", nargs="+", help="Symbols to trade")
    parser.add_argument("--refs", nargs="*", default=[], help="Additional reference assets")
    parser.add_argument("--duration", type=int, help="Duration in seconds")
    parser.add_argument("--interval", type=int, default=30, help="Update interval (default: 30s)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--max-position", type=float, default=10000, help="Max position size in dollars")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port")

    args = parser.parse_args()

    # Combine default and user-specified reference assets
    reference_assets = list(set(DEFAULT_REFERENCE_ASSETS + args.refs))

    print("=" * 70)
    print("THE ORCHESTRATOR - MULTI-EDGE ENSEMBLE DAY TRADING")
    print("=" * 70)
    print()
    print("  A fundamentally different approach:")
    print("  - 6 independent edge sources")
    print("  - Context-first (market regime)")
    print("  - Multi-asset confirmation")
    print("  - Statistical extremes only")
    print("  - Ensemble voting (4+ edges must agree)")
    print()
    print(f"  Trading:     {', '.join(args.symbols)}")
    print(f"  References:  {', '.join(reference_assets[:5])}...")
    print(f"  Interval:    {args.interval} seconds")
    print(f"  Max Pos:     ${args.max_position:,.0f}")
    print(f"  Dry Run:     {args.dry_run}")
    print()
    print("WARNING: This is experimental. Day trading is risky!")
    print("=" * 70)

    # Connect
    config = IBKRConfig(host="127.0.0.1", port=args.port, client_id=30)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

    trader = OrchestratorLive(
        broker=broker,
        symbols=args.symbols,
        reference_assets=reference_assets,
        dry_run=args.dry_run,
        max_position_dollars=args.max_position,
    )

    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        trader.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        trader.start()

        print("\n" + "=" * 70)
        print("ORCHESTRATOR STARTED - Press Ctrl+C to stop")
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
