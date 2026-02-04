#!/usr/bin/env python3
"""
Quantitative Trading Engine Runner

Command-line interface for running backtests and live trading.

Usage:
    # Backtest mode
    python -m trading_algo.quant_core.run backtest --config config.json

    # Paper trading (IBKR paper account)
    python -m trading_algo.quant_core.run paper --config config.json

    # Live trading (use with caution!)
    python -m trading_algo.quant_core.run live --config config.json --confirm
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_algo.quant_core.engine.orchestrator import (
    QuantOrchestrator,
    EngineConfig,
    EngineMode,
    BacktestResult,
)
from trading_algo.quant_core.engine.signal_aggregator import AggregatorConfig
from trading_algo.quant_core.engine.risk_controller import RiskConfig
from trading_algo.quant_core.engine.portfolio_manager import PortfolioConfig, SizingMethod
from trading_algo.quant_core.engine.execution_manager import ExecutionConfig, ExecutionMethod
from trading_algo.quant_core.engine.ibkr_adapter import (
    IBKRConfig,
    IBKRBrokerAdapter,
    IBKRDataProvider,
    create_live_context,
)


# Setup logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_engine_config(config: Dict[str, Any], mode: EngineMode) -> EngineConfig:
    """Create EngineConfig from dictionary."""
    # Signal config
    signal_cfg = config.get('signal', {})
    signal_config = AggregatorConfig(
        ou_weight=signal_cfg.get('ou_weight', 0.25),
        tsmom_weight=signal_cfg.get('tsmom_weight', 0.25),
        vol_mom_weight=signal_cfg.get('vol_mom_weight', 0.25),
        ml_weight=signal_cfg.get('ml_weight', 0.25),
        regime_scaling=signal_cfg.get('regime_scaling', True),
        use_ml_combination=signal_cfg.get('use_ml_combination', True),
    )

    # Risk config
    risk_cfg = config.get('risk', {})
    risk_config = RiskConfig(
        max_position_pct=risk_cfg.get('max_position_pct', 0.10),
        max_drawdown=risk_cfg.get('max_drawdown', 0.15),
        daily_loss_limit=risk_cfg.get('daily_loss_limit', 0.03),
        max_gross_exposure=risk_cfg.get('max_gross_exposure', 1.0),
        vol_target=risk_cfg.get('vol_target', 0.15),
    )

    # Portfolio config
    port_cfg = config.get('portfolio', {})
    sizing_method = SizingMethod[port_cfg.get('sizing_method', 'KELLY').upper()]
    portfolio_config = PortfolioConfig(
        sizing_method=sizing_method,
        kelly_fraction=port_cfg.get('kelly_fraction', 0.25),
        target_volatility=port_cfg.get('target_volatility', 0.15),
        min_weight=port_cfg.get('min_weight', 0.01),
        max_weight=port_cfg.get('max_weight', 0.10),
        allow_shorting=port_cfg.get('allow_shorting', True),
    )

    # Execution config
    exec_cfg = config.get('execution', {})
    exec_method = ExecutionMethod[exec_cfg.get('method', 'ALMGREN_CHRISS').upper()]
    execution_config = ExecutionConfig(
        default_method=exec_method,
        default_horizon_minutes=exec_cfg.get('horizon_minutes', 30),
    )

    return EngineConfig(
        mode=mode,
        universe=config.get('universe', []),
        benchmark_symbol=config.get('benchmark', 'SPY'),
        warmup_bars=config.get('warmup_bars', 60),
        rebalance_frequency=config.get('rebalance_frequency', 'daily'),
        signal_config=signal_config,
        risk_config=risk_config,
        portfolio_config=portfolio_config,
        execution_config=execution_config,
    )


def load_historical_data(
    data_path: str,
    symbols: list,
) -> tuple:
    """
    Load historical data from files.

    Supports CSV files with OHLCV columns.
    """
    import pandas as pd

    historical_data = {}
    timestamps = None

    for symbol in symbols:
        file_path = os.path.join(data_path, f"{symbol}.csv")

        if not os.path.exists(file_path):
            logging.warning(f"No data file for {symbol}")
            continue

        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date')

        # Extract OHLCV
        ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values
        historical_data[symbol] = ohlcv

        if timestamps is None:
            timestamps = df['date'].values

    return historical_data, timestamps


def run_backtest(args):
    """Run backtest mode."""
    logging.info("Starting backtest mode")

    # Load config
    config = load_config(args.config)
    engine_config = create_engine_config(config, EngineMode.BACKTEST)

    # Load data
    data_path = args.data or config.get('data_path', './data')
    historical_data, timestamps = load_historical_data(data_path, engine_config.universe)

    if not historical_data:
        logging.error("No historical data loaded")
        return 1

    # Create and run engine
    engine = QuantOrchestrator(engine_config)

    result = engine.run_backtest(
        historical_data=historical_data,
        timestamps=timestamps,
        initial_capital=config.get('initial_capital', 100000),
        validate=args.validate,
    )

    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return:      {result.total_return:>10.2%}")
    print(f"Annualized Return: {result.annualized_return:>10.2%}")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:     {result.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:      {result.calmar_ratio:>10.2f}")
    print(f"Max Drawdown:      {result.max_drawdown:>10.2%}")
    print(f"Volatility:        {result.volatility:>10.2%}")
    print("-"*60)
    print(f"Total Trades:      {result.total_trades:>10}")
    print(f"Win Rate:          {result.win_rate:>10.2%}")
    print(f"Profit Factor:     {result.profit_factor:>10.2f}")
    print("-"*60)
    print(f"Validation Passed: {result.validation_passed}")
    print(f"PBO:               {result.pbo:>10.2%}")
    print(f"Deflated Sharpe:   {result.deflated_sharpe:>10.2f}")
    print("="*60)

    # Save results
    if args.output:
        engine.save_results(args.output)

        # Save equity curve
        np.save(
            os.path.join(args.output, 'equity_curve.npy'),
            result.equity_curve
        )

    return 0


def run_paper(args):
    """Run paper trading mode."""
    logging.info("Starting paper trading mode")

    # Load config
    config = load_config(args.config)
    engine_config = create_engine_config(config, EngineMode.PAPER)

    # IBKR config (paper trading port)
    ibkr_config = IBKRConfig(
        host=config.get('ibkr', {}).get('host', '127.0.0.1'),
        port=config.get('ibkr', {}).get('paper_port', 7497),
        client_id=config.get('ibkr', {}).get('client_id', 1),
    )

    # Create engine
    engine = QuantOrchestrator(engine_config)

    try:
        # Connect to IBKR
        adapter = IBKRBrokerAdapter(ibkr_config)
        if not adapter.connect():
            logging.error("Failed to connect to IBKR paper trading")
            return 1

        data_provider = IBKRDataProvider(adapter._broker)

        # Run live trading
        engine.run_live(
            ibkr_broker=adapter,
            data_provider=data_provider,
            run_duration_minutes=args.duration,
        )

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        adapter.disconnect()

    # Save results
    if args.output:
        engine.save_results(args.output)

    return 0


def run_live(args):
    """Run live trading mode (use with caution!)."""
    # Safety check
    if not args.confirm:
        print("ERROR: Live trading requires --confirm flag")
        print("WARNING: Live trading uses real money. Use at your own risk!")
        return 1

    logging.warning("="*60)
    logging.warning("STARTING LIVE TRADING - REAL MONEY AT RISK")
    logging.warning("="*60)

    # Load config
    config = load_config(args.config)
    engine_config = create_engine_config(config, EngineMode.LIVE)

    # IBKR config (live trading port)
    ibkr_config = IBKRConfig(
        host=config.get('ibkr', {}).get('host', '127.0.0.1'),
        port=config.get('ibkr', {}).get('live_port', 7496),
        client_id=config.get('ibkr', {}).get('client_id', 1),
    )

    # Create engine
    engine = QuantOrchestrator(engine_config)

    try:
        # Connect to IBKR
        adapter = IBKRBrokerAdapter(ibkr_config)
        if not adapter.connect():
            logging.error("Failed to connect to IBKR live trading")
            return 1

        data_provider = IBKRDataProvider(adapter._broker)

        # Run live trading
        engine.run_live(
            ibkr_broker=adapter,
            data_provider=data_provider,
            run_duration_minutes=args.duration,
        )

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        adapter.disconnect()

    # Save results
    if args.output:
        engine.save_results(args.output)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantitative Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level',
    )
    parser.add_argument(
        '--log-file',
        help='Log file path',
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run backtest')
    bt_parser.add_argument('--config', required=True, help='Config file path')
    bt_parser.add_argument('--data', help='Historical data directory')
    bt_parser.add_argument('--output', help='Output directory')
    bt_parser.add_argument('--validate', action='store_true', help='Run validation')

    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--config', required=True, help='Config file path')
    paper_parser.add_argument('--output', help='Output directory')
    paper_parser.add_argument('--duration', type=int, help='Duration in minutes')

    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--config', required=True, help='Config file path')
    live_parser.add_argument('--output', help='Output directory')
    live_parser.add_argument('--duration', type=int, help='Duration in minutes')
    live_parser.add_argument('--confirm', action='store_true',
                             help='Confirm live trading with real money')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Run appropriate command
    if args.command == 'backtest':
        return run_backtest(args)
    elif args.command == 'paper':
        return run_paper(args)
    elif args.command == 'live':
        return run_live(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
