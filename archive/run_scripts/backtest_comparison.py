#!/usr/bin/env python3
"""
Backtest Comparison: Chameleon v3 vs Orchestrator

Runs both strategies on the exact same historical data and compares results.
This allows direct comparison of the old momentum-based approach vs the new
multi-edge ensemble approach.

Usage:
    python backtest_comparison.py
"""

import sys
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec

# Import both strategies
from trading_algo.rat.chameleon_daytrader import (
    ChameleonDayTrader,
    DayTradeSignal,
    create_daytrader,
)
from trading_algo.strategies.orchestrator import (
    Orchestrator,
    OrchestratorSignal,
    create_orchestrator,
)


@dataclass
class Trade:
    """Record of a completed trade."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    trades: List[Trade] = field(default_factory=list)
    signals_generated: int = 0

    # Calculated after backtest
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0

    def calculate_metrics(self):
        """Calculate performance metrics from trades."""
        closed_trades = [t for t in self.trades if t.exit_time is not None]
        self.total_trades = len(closed_trades)

        if self.total_trades == 0:
            return

        wins = [t for t in closed_trades if t.pnl_pct > 0]
        losses = [t for t in closed_trades if t.pnl_pct <= 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        self.total_pnl_pct = sum(t.pnl_pct for t in closed_trades)
        self.avg_win_pct = statistics.mean([t.pnl_pct for t in wins]) if wins else 0
        self.avg_loss_pct = statistics.mean([t.pnl_pct for t in losses]) if losses else 0

        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate drawdown
        equity_curve = []
        cumulative = 0
        for t in closed_trades:
            cumulative += t.pnl_pct
            equity_curve.append(cumulative)

        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown_pct = max_dd


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BacktestEngine:
    """Runs backtests on historical data."""

    def __init__(self, broker: IBKRBroker):
        self.broker = broker
        self.bars_cache: Dict[str, List[Bar]] = {}

    def load_data(self, symbols: List[str], date: str = "today", days: int = 1):
        """Load historical data for symbols."""
        print(f"\nLoading historical data for {len(symbols)} symbols...")

        # We need reference assets for the Orchestrator
        reference_assets = ["SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE", "XLY"]
        all_symbols = list(set(symbols + reference_assets))

        for symbol in all_symbols:
            try:
                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                )

                # Get intraday 5-min bars
                duration = f"{days} D"
                raw_bars = self.broker.get_historical_bars(
                    instrument,
                    duration=duration,
                    bar_size="5 mins",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                bars = []
                for b in raw_bars:
                    bars.append(Bar(
                        timestamp=datetime.fromtimestamp(b.timestamp_epoch_s),
                        open=b.open,
                        high=b.high,
                        low=b.low,
                        close=b.close,
                        volume=b.volume or 0,
                    ))

                self.bars_cache[symbol] = bars
                is_ref = symbol in reference_assets and symbol not in symbols
                marker = "(ref)" if is_ref else ""
                print(f"  {symbol}: {len(bars)} bars {marker}")

            except Exception as e:
                print(f"  {symbol}: Error - {e}")

            # IBKR pacing
            import time
            time.sleep(0.5)

    def run_chameleon(self, symbols: List[str], max_position_dollars: float = 10000) -> BacktestResult:
        """Run backtest with Chameleon strategy."""
        print("\n" + "=" * 60)
        print("BACKTESTING: CHAMELEON v3 (Old Strategy)")
        print("=" * 60)

        result = BacktestResult(strategy_name="Chameleon v3")

        # Create traders for each symbol
        traders: Dict[str, ChameleonDayTrader] = {
            symbol: create_daytrader(aggressive=True, max_position_dollars=max_position_dollars)
            for symbol in symbols
        }

        # Active positions
        positions: Dict[str, Trade] = {}

        # Get all timestamps from the first symbol
        if not symbols or symbols[0] not in self.bars_cache:
            print("No data available")
            return result

        # Collect all unique timestamps
        all_timestamps = set()
        for symbol in symbols:
            if symbol in self.bars_cache:
                for bar in self.bars_cache[symbol]:
                    all_timestamps.add(bar.timestamp)

        sorted_timestamps = sorted(all_timestamps)

        # Warmup period (first 30 bars)
        warmup_count = 30

        for i, ts in enumerate(sorted_timestamps):
            is_warmup = i < warmup_count

            for symbol in symbols:
                if symbol not in self.bars_cache:
                    continue

                # Find the bar for this timestamp
                bar = None
                for b in self.bars_cache[symbol]:
                    if b.timestamp == ts:
                        bar = b
                        break

                if bar is None:
                    continue

                trader = traders[symbol]
                signal = trader.update(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open_price=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )

                if signal is None or is_warmup:
                    continue

                result.signals_generated += 1

                # Process signal
                if signal.action in ('buy', 'short') and symbol not in positions:
                    # Open position
                    positions[symbol] = Trade(
                        symbol=symbol,
                        side='long' if signal.action == 'buy' else 'short',
                        entry_time=bar.timestamp,
                        entry_price=bar.close,
                    )
                    print(f"  {bar.timestamp.strftime('%H:%M')} | {signal.action.upper():5s} {symbol} @ ${bar.close:.2f}")

                elif signal.action in ('sell', 'cover') and symbol in positions:
                    # Close position
                    pos = positions[symbol]
                    pos.exit_time = bar.timestamp
                    pos.exit_price = bar.close
                    pos.exit_reason = signal.reason

                    if pos.side == 'long':
                        pos.pnl = pos.exit_price - pos.entry_price
                        pos.pnl_pct = (pos.exit_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pos.pnl = pos.entry_price - pos.exit_price
                        pos.pnl_pct = (pos.entry_price - pos.exit_price) / pos.entry_price * 100

                    result.trades.append(pos)
                    del positions[symbol]

                    print(f"  {bar.timestamp.strftime('%H:%M')} | {signal.action.upper():5s} {symbol} @ ${bar.close:.2f} | P&L: {pos.pnl_pct:+.2f}%")

        # Close any remaining positions at last price
        for symbol, pos in positions.items():
            if symbol in self.bars_cache and self.bars_cache[symbol]:
                last_bar = self.bars_cache[symbol][-1]
                pos.exit_time = last_bar.timestamp
                pos.exit_price = last_bar.close
                pos.exit_reason = "End of backtest"

                if pos.side == 'long':
                    pos.pnl = pos.exit_price - pos.entry_price
                    pos.pnl_pct = (pos.exit_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pos.pnl = pos.entry_price - pos.exit_price
                    pos.pnl_pct = (pos.entry_price - pos.exit_price) / pos.entry_price * 100

                result.trades.append(pos)
                print(f"  {last_bar.timestamp.strftime('%H:%M')} | CLOSE {symbol} @ ${last_bar.close:.2f} | P&L: {pos.pnl_pct:+.2f}%")

        result.calculate_metrics()
        return result

    def run_orchestrator(self, symbols: List[str], max_position_dollars: float = 10000) -> BacktestResult:
        """Run backtest with Orchestrator strategy."""
        print("\n" + "=" * 60)
        print("BACKTESTING: ORCHESTRATOR (New Strategy)")
        print("=" * 60)

        result = BacktestResult(strategy_name="Orchestrator")

        # Create orchestrator
        orchestrator = create_orchestrator()

        # Reference assets
        reference_assets = ["SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE", "XLY"]
        all_symbols = list(set(symbols + reference_assets))

        # Active positions
        positions: Dict[str, Trade] = {}

        # Collect all unique timestamps
        all_timestamps = set()
        for symbol in all_symbols:
            if symbol in self.bars_cache:
                for bar in self.bars_cache[symbol]:
                    all_timestamps.add(bar.timestamp)

        sorted_timestamps = sorted(all_timestamps)

        # Warmup period (first 30 bars)
        warmup_count = 30

        for i, ts in enumerate(sorted_timestamps):
            is_warmup = i < warmup_count

            # Update all assets (references and trading symbols)
            for symbol in all_symbols:
                if symbol not in self.bars_cache:
                    continue

                # Find the bar for this timestamp
                bar = None
                for b in self.bars_cache[symbol]:
                    if b.timestamp == ts:
                        bar = b
                        break

                if bar is None:
                    continue

                orchestrator.update_asset(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open_price=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )

            if is_warmup:
                continue

            # Generate signals for trading symbols only
            for symbol in symbols:
                if symbol not in self.bars_cache:
                    continue

                # Find the bar for this timestamp
                bar = None
                for b in self.bars_cache[symbol]:
                    if b.timestamp == ts:
                        bar = b
                        break

                if bar is None:
                    continue

                signal = orchestrator.generate_signal(symbol, bar.timestamp)

                if signal is None:
                    continue

                result.signals_generated += 1

                # Process signal
                if signal.action in ('buy', 'short') and symbol not in positions:
                    # Open position
                    positions[symbol] = Trade(
                        symbol=symbol,
                        side='long' if signal.action == 'buy' else 'short',
                        entry_time=bar.timestamp,
                        entry_price=bar.close,
                    )

                    # Show edge breakdown
                    votes_str = ", ".join([f"{k}:{v.name}" for k, v in signal.edge_votes.items()])
                    print(f"  {bar.timestamp.strftime('%H:%M')} | {signal.action.upper():5s} {symbol} @ ${bar.close:.2f}")
                    print(f"           Regime: {signal.market_regime.name}, Consensus: {signal.consensus_score:.2f}")
                    print(f"           Votes: {votes_str}")

                elif signal.action in ('sell', 'cover') and symbol in positions:
                    # Close position
                    pos = positions[symbol]
                    pos.exit_time = bar.timestamp
                    pos.exit_price = bar.close
                    pos.exit_reason = signal.reason

                    if pos.side == 'long':
                        pos.pnl = pos.exit_price - pos.entry_price
                        pos.pnl_pct = (pos.exit_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pos.pnl = pos.entry_price - pos.exit_price
                        pos.pnl_pct = (pos.entry_price - pos.exit_price) / pos.entry_price * 100

                    result.trades.append(pos)
                    del positions[symbol]

                    print(f"  {bar.timestamp.strftime('%H:%M')} | {signal.action.upper():5s} {symbol} @ ${bar.close:.2f} | P&L: {pos.pnl_pct:+.2f}%")

        # Close any remaining positions at last price
        for symbol, pos in positions.items():
            if symbol in self.bars_cache and self.bars_cache[symbol]:
                last_bar = self.bars_cache[symbol][-1]
                pos.exit_time = last_bar.timestamp
                pos.exit_price = last_bar.close
                pos.exit_reason = "End of backtest"

                if pos.side == 'long':
                    pos.pnl = pos.exit_price - pos.entry_price
                    pos.pnl_pct = (pos.exit_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pos.pnl = pos.entry_price - pos.exit_price
                    pos.pnl_pct = (pos.entry_price - pos.exit_price) / pos.entry_price * 100

                result.trades.append(pos)
                print(f"  {last_bar.timestamp.strftime('%H:%M')} | CLOSE {symbol} @ ${last_bar.close:.2f} | P&L: {pos.pnl_pct:+.2f}%")

        result.calculate_metrics()
        return result


def print_comparison(chameleon: BacktestResult, orchestrator: BacktestResult):
    """Print side-by-side comparison of results."""
    print("\n")
    print("=" * 80)
    print("BACKTEST COMPARISON: CHAMELEON v3 vs ORCHESTRATOR")
    print("=" * 80)
    print()

    # Header
    print(f"{'Metric':<30} {'Chameleon v3':>20} {'Orchestrator':>20}")
    print("-" * 70)

    # Metrics
    print(f"{'Signals Generated':<30} {chameleon.signals_generated:>20} {orchestrator.signals_generated:>20}")
    print(f"{'Total Trades':<30} {chameleon.total_trades:>20} {orchestrator.total_trades:>20}")
    print(f"{'Winning Trades':<30} {chameleon.winning_trades:>20} {orchestrator.winning_trades:>20}")
    print(f"{'Losing Trades':<30} {chameleon.losing_trades:>20} {orchestrator.losing_trades:>20}")
    print(f"{'Win Rate':<30} {chameleon.win_rate:>19.1%} {orchestrator.win_rate:>19.1%}")
    print("-" * 70)
    print(f"{'Total P&L':<30} {chameleon.total_pnl_pct:>19.2f}% {orchestrator.total_pnl_pct:>19.2f}%")
    print(f"{'Avg Win':<30} {chameleon.avg_win_pct:>19.2f}% {orchestrator.avg_win_pct:>19.2f}%")
    print(f"{'Avg Loss':<30} {chameleon.avg_loss_pct:>19.2f}% {orchestrator.avg_loss_pct:>19.2f}%")
    print(f"{'Profit Factor':<30} {chameleon.profit_factor:>20.2f} {orchestrator.profit_factor:>20.2f}")
    print(f"{'Max Drawdown':<30} {chameleon.max_drawdown_pct:>19.2f}% {orchestrator.max_drawdown_pct:>19.2f}%")
    print("=" * 80)

    # Summary
    print()
    if orchestrator.total_pnl_pct > chameleon.total_pnl_pct:
        diff = orchestrator.total_pnl_pct - chameleon.total_pnl_pct
        print(f"RESULT: Orchestrator outperformed Chameleon by {diff:.2f}%")
    elif chameleon.total_pnl_pct > orchestrator.total_pnl_pct:
        diff = chameleon.total_pnl_pct - orchestrator.total_pnl_pct
        print(f"RESULT: Chameleon outperformed Orchestrator by {diff:.2f}%")
    else:
        print("RESULT: Both strategies performed equally")

    print()

    # Trade count analysis
    trade_reduction = ((chameleon.total_trades - orchestrator.total_trades) /
                       chameleon.total_trades * 100) if chameleon.total_trades > 0 else 0
    print(f"Trade Reduction: {trade_reduction:.0f}% fewer trades with Orchestrator")

    print()
    print("=" * 80)

    # Show individual trades
    print("\nCHAMELEON TRADES:")
    print("-" * 60)
    for t in chameleon.trades:
        print(f"  {t.entry_time.strftime('%H:%M')} {t.side:5s} {t.symbol:5s} "
              f"${t.entry_price:.2f} -> ${t.exit_price:.2f} | {t.pnl_pct:+.2f}%")

    print("\nORCHESTRATOR TRADES:")
    print("-" * 60)
    for t in orchestrator.trades:
        print(f"  {t.entry_time.strftime('%H:%M')} {t.side:5s} {t.symbol:5s} "
              f"${t.entry_price:.2f} -> ${t.exit_price:.2f} | {t.pnl_pct:+.2f}%")


def main():
    # The exact symbols from today's session
    symbols = ["MU", "SBUX", "INTC", "SOXL", "PLUG"]

    print("=" * 80)
    print("BACKTEST COMPARISON")
    print("=" * 80)
    print()
    print("Comparing two strategies on the SAME data:")
    print()
    print("  1. CHAMELEON v3 (Old): Momentum + RSI + chop filter")
    print("     - Single indicator approach")
    print("     - No market context")
    print("     - Trades in isolation")
    print()
    print("  2. ORCHESTRATOR (New): Multi-edge ensemble")
    print("     - 6 independent edge sources")
    print("     - Market regime awareness")
    print("     - Cross-asset confirmation")
    print("     - Statistical extremes only")
    print()
    print(f"Symbols: {', '.join(symbols)}")
    print("=" * 80)

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=40)
    broker = IBKRBroker(config=config, require_paper=True)

    print("\nConnecting to IBKR...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    try:
        # Create backtest engine
        engine = BacktestEngine(broker)

        # Load data (today's data)
        engine.load_data(symbols, days=1)

        # Run both strategies
        chameleon_result = engine.run_chameleon(symbols)
        orchestrator_result = engine.run_orchestrator(symbols)

        # Print comparison
        print_comparison(chameleon_result, orchestrator_result)

    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
