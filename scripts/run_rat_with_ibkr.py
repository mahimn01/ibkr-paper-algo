#!/usr/bin/env python3
"""
Complete RAT trading workflow with IBKR data.

This script demonstrates:
1. Pulling historical data from IBKR
2. Running backtest with RAT framework
3. Optionally running live paper trading

Prerequisites:
    pip install ib_insync

Usage:
    # Backtest mode (pull data, run backtest)
    python scripts/run_rat_with_ibkr.py backtest AAPL MSFT --days 60

    # Live paper trading mode
    python scripts/run_rat_with_ibkr.py live AAPL MSFT --duration 3600
"""

import argparse
import time
from datetime import datetime
from typing import Optional


def run_backtest(
    symbols: list[str],
    days: int = 60,
    initial_capital: float = 100_000,
    host: str = "127.0.0.1",
    port: int = 7497,
) -> dict:
    """Run RAT backtest with IBKR historical data."""
    from trading_algo.broker.ibkr import IBKRBroker
    from trading_algo.config import IBKRConfig
    from trading_algo.instruments import InstrumentSpec
    from trading_algo.rat.config import RATConfig
    from trading_algo.rat.engine import RATEngine
    from trading_algo.rat.backtest.backtester import BacktestConfig, SimulatedBroker
    from trading_algo.rat.backtest.analytics import PerformanceAnalytics

    print("=" * 70)
    print("RAT BACKTEST WITH IBKR DATA")
    print("=" * 70)

    # Connect to IBKR for data
    config = IBKRConfig(host=host, port=port, client_id=1)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR at {host}:{port}...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return {}

    # Pull historical data
    all_bars = []
    try:
        for symbol in symbols:
            print(f"Fetching {days} days of data for {symbol}...")
            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange="SMART",
                currency="USD",
            )

            bars = broker.get_historical_bars(
                instrument,
                duration=f"{days} D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=False,
            )
            print(f"  Got {len(bars)} bars")

            # Convert to our Bar format
            for bar in bars:
                all_bars.append({
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(bar.timestamp_epoch_s),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume or 0,
                })
    finally:
        broker.disconnect()

    if not all_bars:
        print("No data retrieved!")
        return {}

    # Sort by timestamp
    all_bars.sort(key=lambda b: b["timestamp"])
    print(f"\nTotal bars: {len(all_bars)}")

    # Initialize RAT
    rat_config = RATConfig.from_env()
    bt_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_per_share=0.005,
        slippage_pct=0.001,
        max_position_pct=0.25,
        warmup_bars=20,
    )

    sim_broker = SimulatedBroker(bt_config)
    analytics = PerformanceAnalytics(initial_capital=initial_capital)
    engine = RATEngine(config=rat_config, broker=None, llm_client=None)
    engine.reset_for_backtest()

    # Lower thresholds for daily data
    engine._filter._confidence_threshold = 0.2
    engine._combiner.min_signals_required = 1

    # Run backtest
    print("\nRunning backtest...")
    current_prices = {}
    signals_generated = 0
    trades_executed = 0

    for i, bar in enumerate(all_bars):
        current_prices[bar["symbol"]] = bar["close"]
        sim_broker.update_prices(current_prices)

        if i < bt_config.warmup_bars:
            continue

        # Process through RAT
        state = engine.inject_backtest_tick(
            symbol=bar["symbol"],
            timestamp=bar["timestamp"],
            open_price=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
        )

        # Execute any signals
        if state and state.decision and state.decision.should_trade(min_confidence=0.15):
            signals_generated += 1
            decision = state.decision

            if decision.action == "buy" and bar["symbol"] not in sim_broker.positions:
                qty = (sim_broker.equity * decision.position_size_pct) / bar["close"]
                if qty > 0:
                    trade = sim_broker.place_order(
                        symbol=bar["symbol"],
                        side="BUY",
                        quantity=qty,
                        price=bar["close"],
                        timestamp=bar["timestamp"],
                    )
                    if trade:
                        trades_executed += 1
                        analytics.record_trade(trade)

            elif decision.action == "sell" and bar["symbol"] in sim_broker.positions:
                pos = sim_broker.positions[bar["symbol"]]
                trade = sim_broker.place_order(
                    symbol=bar["symbol"],
                    side="SELL",
                    quantity=pos.quantity,
                    price=bar["close"],
                    timestamp=bar["timestamp"],
                )
                if trade:
                    trades_executed += 1
                    analytics.record_trade(trade)

        analytics.record_equity(bar["timestamp"], sim_broker.equity)

    # Close remaining positions
    if all_bars:
        final_bar = all_bars[-1]
        for symbol in list(sim_broker.positions.keys()):
            if symbol in current_prices:
                pos = sim_broker.positions[symbol]
                trade = sim_broker.place_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=pos.quantity,
                    price=current_prices[symbol],
                    timestamp=final_bar["timestamp"],
                )
                if trade:
                    analytics.record_trade(trade)

    # Calculate metrics
    metrics = analytics.calculate_metrics()

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial Capital:  ${initial_capital:>15,.2f}")
    print(f"Final Equity:     ${sim_broker.equity:>15,.2f}")
    print(f"Total Return:     {metrics.total_return_pct:>15.2%}")
    print(f"Sharpe Ratio:     {metrics.sharpe_ratio:>15.2f}")
    print(f"Max Drawdown:     {metrics.max_drawdown:>15.2%}")
    print(f"Win Rate:         {metrics.win_rate:>15.2%}")
    print(f"Total Trades:     {metrics.total_trades:>15}")
    print(f"Signals:          {signals_generated:>15}")
    print("=" * 70)

    return {
        "metrics": metrics,
        "final_equity": sim_broker.equity,
        "trades": trades_executed,
    }


def run_live(
    symbols: list[str],
    duration_seconds: int = 3600,
    host: str = "127.0.0.1",
    port: int = 7497,
) -> None:
    """Run live paper trading with RAT."""
    from trading_algo.broker.ibkr import IBKRBroker
    from trading_algo.config import IBKRConfig
    from trading_algo.rat.integration import RATTrader

    print("=" * 70)
    print("RAT LIVE PAPER TRADING")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Duration: {duration_seconds} seconds")
    print("=" * 70)

    # Connect to IBKR
    config = IBKRConfig(host=host, port=port, client_id=2)
    broker = IBKRBroker(config=config, require_paper=True)

    print(f"\nConnecting to IBKR at {host}:{port}...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Create RAT trader
    trader = RATTrader(broker, symbols)
    trader.start()

    print("\nStarting live trading loop...")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    update_count = 0

    try:
        while time.time() - start_time < duration_seconds:
            # Process market data
            states = trader.process_market_data()

            update_count += 1
            if update_count % 10 == 0:
                elapsed = int(time.time() - start_time)
                stats = trader.get_stats()
                print(
                    f"[{elapsed:4d}s] Updates: {update_count}, "
                    f"Pending orders: {stats['pending_orders']}, "
                    f"Executions: {stats['total_executions']}"
                )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        trader.stop()
        broker.disconnect()

    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    stats = trader.get_stats()
    print(f"Total updates:    {update_count}")
    print(f"Total executions: {stats['total_executions']}")
    print(f"Final P&L:        ${stats['total_pnl']:,.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="RAT trading with IBKR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest with 60 days of AAPL and MSFT data
  python scripts/run_rat_with_ibkr.py backtest AAPL MSFT --days 60

  # Run live paper trading for 1 hour
  python scripts/run_rat_with_ibkr.py live AAPL MSFT --duration 3600

  # Connect to different port
  python scripts/run_rat_with_ibkr.py backtest AAPL --port 4002
        """
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Backtest subcommand
    bt_parser = subparsers.add_parser("backtest", help="Run backtest with historical data")
    bt_parser.add_argument("symbols", nargs="+", help="Stock symbols")
    bt_parser.add_argument("--days", type=int, default=60, help="Days of history")
    bt_parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    bt_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    bt_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port")

    # Live subcommand
    live_parser = subparsers.add_parser("live", help="Run live paper trading")
    live_parser.add_argument("symbols", nargs="+", help="Stock symbols")
    live_parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds")
    live_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    live_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port")

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(
            symbols=[s.upper() for s in args.symbols],
            days=args.days,
            initial_capital=args.capital,
            host=args.host,
            port=args.port,
        )
    elif args.mode == "live":
        run_live(
            symbols=[s.upper() for s in args.symbols],
            duration_seconds=args.duration,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
