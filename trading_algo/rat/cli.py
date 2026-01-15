"""
RAT CLI: Command-line interface for RAT framework.

Commands:
1. backtest - Run backtests
2. live - Run live trading
3. analyze - Analyze historical data
4. status - Check engine status

Usage:
    python -m trading_algo.rat.cli backtest --symbols AAPL,MSFT --start 2023-01-01 --end 2023-12-31
    python -m trading_algo.rat.cli live --symbols AAPL,MSFT
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from trading_algo.rat.config import RATConfig
from trading_algo.rat.engine import RATEngine
from trading_algo.rat.backtest import (
    RATBacktester,
    BacktestConfig,
    CSVLoader,
    YahooLoader,
    run_walk_forward,
    aggregate_walk_forward_results,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rat.cli")


def parse_date(date_str: str) -> datetime:
    """Parse date string."""
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {date_str}")


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols."""
    return [s.strip().upper() for s in symbols_str.split(",")]


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run backtest command."""
    logger.info("Starting RAT Backtest")

    # Parse arguments
    symbols = parse_symbols(args.symbols)
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    # Load configs
    rat_config = RATConfig.from_env()

    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        commission_per_share=args.commission,
        slippage_pct=args.slippage / 10000,  # Convert bps to decimal
        max_position_pct=args.max_position,
        max_daily_loss_pct=args.daily_loss_limit,
        max_drawdown_pct=args.max_drawdown,
    )

    # Select data loader
    if args.data_source == "yahoo":
        data_loader = YahooLoader(interval=args.interval)
    else:
        data_loader = CSVLoader(data_dir=args.data_dir)

    # Run backtest
    backtester = RATBacktester(
        rat_config=rat_config,
        backtest_config=backtest_config,
        data_loader=data_loader,
    )

    try:
        result = backtester.run(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            benchmark_symbol=args.benchmark,
        )

        # Print report
        print(result.report)

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(result.report)
            logger.info(f"Report saved to {output_path}")

        # Return based on Sharpe ratio
        if result.metrics.sharpe_ratio > 0:
            return 0
        else:
            return 1

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1


def cmd_walk_forward(args: argparse.Namespace) -> int:
    """Run walk-forward optimization."""
    logger.info("Starting Walk-Forward Optimization")

    symbols = parse_symbols(args.symbols)
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    rat_config = RATConfig.from_env()
    backtest_config = BacktestConfig(initial_capital=args.capital)

    if args.data_source == "yahoo":
        data_loader = YahooLoader()
    else:
        data_loader = CSVLoader(data_dir=args.data_dir)

    results = run_walk_forward(
        rat_config=rat_config,
        backtest_config=backtest_config,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_period_days=args.train_days,
        test_period_days=args.test_days,
        data_loader=data_loader,
    )

    # Print summary
    summary = aggregate_walk_forward_results(results)
    print("\n" + "=" * 60)
    print("WALK-FORWARD OPTIMIZATION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

    # Print individual period results
    for i, result in enumerate(results):
        print(f"\nPeriod {i+1}:")
        print(f"  Return: {result.metrics.total_return_pct:.2%}")
        print(f"  Sharpe: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Trades: {result.metrics.total_trades}")

    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run live trading (paper mode)."""
    logger.info("Starting RAT Live Trading")

    # This would integrate with actual IBKR broker
    # For now, just show configuration
    symbols = parse_symbols(args.symbols)
    rat_config = RATConfig.from_env()

    print("\n" + "=" * 60)
    print("RAT LIVE TRADING CONFIGURATION")
    print("=" * 60)
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Mode: {'Paper' if args.paper else 'LIVE'}")
    print(f"  Confidence threshold: {rat_config.signal.confidence_threshold}")
    print(f"  Max position %: {rat_config.signal.max_position_pct:.0%}")
    print("=" * 60)

    if not args.paper:
        print("\nWARNING: Live trading not yet implemented for safety.")
        print("Use --paper flag for paper trading.")
        return 1

    print("\nTo run live trading, integrate with IBKR broker.")
    print("See trading_algo/rat/integration.py for details.")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze historical data with RAT modules."""
    logger.info("Starting RAT Analysis")

    symbols = parse_symbols(args.symbols)
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    # Load data
    if args.data_source == "yahoo":
        loader = YahooLoader()
    else:
        loader = CSVLoader(data_dir=args.data_dir)

    rat_config = RATConfig.from_env()
    engine = RATEngine(config=rat_config)
    engine.reset_for_backtest()

    for symbol in symbols:
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS: {symbol}")
        print("=" * 60)

        try:
            bars = loader.load(symbol, start_date, end_date)
            print(f"Loaded {len(bars)} bars")

            # Process through engine
            for bar in bars:
                engine.inject_backtest_tick(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open_price=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )

            # Get final state
            state = engine.get_last_state(symbol)
            if state:
                print(f"\nFinal State:")
                print(f"  Attention Score: {state.attention_score:.3f}")
                print(f"  Reflexivity Stage: {state.reflexivity_stage.name}")
                print(f"  Topology Regime: {state.topology_regime.name}")
                print(f"  Adversarial: {state.adversarial_archetype or 'None'}")
                print(f"  Alpha Health: {state.alpha_health:.2%}")

            # Get engine stats
            stats = engine.get_stats()
            print(f"\nAlpha Factors:")
            print(f"  Active: {stats['active_factors']}")
            print(f"  Decaying: {stats['decaying_factors']}")

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Check RAT system status."""
    print("\n" + "=" * 60)
    print("RAT SYSTEM STATUS")
    print("=" * 60)

    # Check config
    try:
        config = RATConfig.from_env()
        print("  Configuration: OK")
        print(f"    - Attention window: {config.attention.flow_window}")
        print(f"    - Topology dim: {config.topology.embedding_dim}")
        print(f"    - Signal threshold: {config.signal.confidence_threshold}")
    except Exception as e:
        print(f"  Configuration: ERROR - {e}")

    # Check data directory
    data_dir = Path(args.data_dir if hasattr(args, 'data_dir') else "data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"  Data directory: OK ({len(csv_files)} CSV files)")
    else:
        print(f"  Data directory: NOT FOUND ({data_dir})")

    # Check optional dependencies
    deps = {
        "yfinance": "Yahoo Finance data loading",
        "ripser": "Fast persistent homology",
        "numpy": "Numerical operations",
    }

    print("\n  Optional dependencies:")
    for dep, desc in deps.items():
        try:
            __import__(dep)
            print(f"    - {dep}: OK ({desc})")
        except ImportError:
            print(f"    - {dep}: NOT INSTALLED ({desc})")

    print("=" * 60)
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAT: Reflexive Attention Topology Trading Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    bt_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    bt_parser.add_argument("--commission", type=float, default=0.005, help="Commission per share")
    bt_parser.add_argument("--slippage", type=float, default=5, help="Slippage in basis points")
    bt_parser.add_argument("--max-position", type=float, default=0.25, help="Max position %")
    bt_parser.add_argument("--daily-loss-limit", type=float, default=0.02, help="Daily loss limit %")
    bt_parser.add_argument("--max-drawdown", type=float, default=0.15, help="Max drawdown %")
    bt_parser.add_argument("--data-source", choices=["csv", "yahoo"], default="yahoo", help="Data source")
    bt_parser.add_argument("--data-dir", default="data", help="Data directory for CSV")
    bt_parser.add_argument("--interval", default="1d", help="Data interval")
    bt_parser.add_argument("--benchmark", help="Benchmark symbol")
    bt_parser.add_argument("--output", "-o", help="Output file for report")
    bt_parser.set_defaults(func=cmd_backtest)

    # Walk-forward command
    wf_parser = subparsers.add_parser("walk-forward", help="Run walk-forward optimization")
    wf_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    wf_parser.add_argument("--start", required=True, help="Start date")
    wf_parser.add_argument("--end", required=True, help="End date")
    wf_parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    wf_parser.add_argument("--train-days", type=int, default=252, help="Training period days")
    wf_parser.add_argument("--test-days", type=int, default=63, help="Test period days")
    wf_parser.add_argument("--data-source", choices=["csv", "yahoo"], default="yahoo")
    wf_parser.add_argument("--data-dir", default="data")
    wf_parser.set_defaults(func=cmd_walk_forward)

    # Live command
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    live_parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    live_parser.set_defaults(func=cmd_live)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data with RAT")
    analyze_parser.add_argument("--symbols", "-s", required=True, help="Comma-separated symbols")
    analyze_parser.add_argument("--start", required=True, help="Start date")
    analyze_parser.add_argument("--end", required=True, help="End date")
    analyze_parser.add_argument("--data-source", choices=["csv", "yahoo"], default="yahoo")
    analyze_parser.add_argument("--data-dir", default="data")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")
    status_parser.add_argument("--data-dir", default="data")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
