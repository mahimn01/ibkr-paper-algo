#!/usr/bin/env python3
"""
Run Aggressive Backtest with Real IBKR Historical Data

Fetches historical data directly from TWS and runs backtest.
Requires TWS Desktop to be running with API enabled.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Fix asyncio event loop issue with ib_insync on Python 3.10+
import asyncio
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import numpy as np
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.quant_core.engine.orchestrator import QuantOrchestrator, EngineConfig, EngineMode
from trading_algo.quant_core.engine.signal_aggregator import AggregatorConfig
from trading_algo.quant_core.engine.risk_controller import RiskConfig
from trading_algo.quant_core.engine.portfolio_manager import PortfolioConfig
from trading_algo.quant_core.engine.execution_manager import ExecutionConfig


def fetch_historical_data_from_ibkr(
    broker: IBKRBroker,
    symbols: list[str],
    duration: str = "2 Y",  # 2 years of data
    bar_size: str = "1 day",
) -> tuple[dict[str, np.ndarray], list[datetime]]:
    """
    Fetch historical data from IBKR for multiple symbols.

    Args:
        broker: Connected IBKRBroker instance
        symbols: List of stock symbols
        duration: IBKR duration string (e.g., "1 Y", "6 M", "30 D")
        bar_size: IBKR bar size (e.g., "1 day", "1 hour", "5 mins")

    Returns:
        Tuple of (symbol -> OHLCV array, timestamps)
    """
    all_data = {}
    all_timestamps = {}

    for symbol in symbols:
        logger.info(f"Fetching historical data for {symbol}...")

        try:
            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange="SMART",
                currency="USD",
            )

            bars = broker.get_historical_bars(
                instrument,
                duration=duration,
                bar_size=bar_size,
                what_to_show="TRADES",
                use_rth=True,  # Regular trading hours only
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                continue

            # Convert to OHLCV numpy array
            ohlcv = np.zeros((len(bars), 5))
            timestamps = []

            for i, bar in enumerate(bars):
                ohlcv[i] = [bar.open, bar.high, bar.low, bar.close, bar.volume or 0]
                timestamps.append(datetime.fromtimestamp(bar.timestamp_epoch_s))

            all_data[symbol] = ohlcv
            all_timestamps[symbol] = timestamps

            logger.info(f"  Loaded {len(bars)} bars for {symbol} "
                       f"({timestamps[0].date()} to {timestamps[-1].date()})")

            # Rate limiting - IBKR allows ~6 historical requests per 10 seconds
            time.sleep(2)

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            continue

    if not all_data:
        raise ValueError("No data loaded for any symbol")

    # Find common timestamps and align data
    ref_symbol = max(all_data.keys(), key=lambda s: len(all_data[s]))
    ref_timestamps = all_timestamps[ref_symbol]

    logger.info(f"Using {ref_symbol} as reference with {len(ref_timestamps)} bars")

    # Align all data to reference timestamps
    ts_to_idx = {ts.date(): i for i, ts in enumerate(ref_timestamps)}
    aligned_data = {}

    for symbol, ohlcv in all_data.items():
        if symbol == ref_symbol:
            aligned_data[symbol] = ohlcv
            continue

        aligned = np.full((len(ref_timestamps), 5), np.nan)
        for i, ts in enumerate(all_timestamps[symbol]):
            date = ts.date()
            if date in ts_to_idx:
                aligned[ts_to_idx[date]] = ohlcv[i]

        # Forward fill NaN values
        for col in range(5):
            mask = np.isnan(aligned[:, col])
            if mask.any() and not mask.all():
                idx = np.where(~mask, np.arange(len(aligned)), 0)
                np.maximum.accumulate(idx, out=idx)
                aligned[:, col] = aligned[idx, col]

        aligned_data[symbol] = aligned

    return aligned_data, ref_timestamps


def get_aggressive_config(universe: list[str]) -> EngineConfig:
    """
    Get aggressive configuration for higher returns.

    Target: 25-40% annual returns with 30-40% max drawdown.
    """
    return EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=universe,
        benchmark_symbol=universe[0] if universe else "SPY",
        bar_frequency="1D",

        signal_config=AggregatorConfig(
            # Shorter lookbacks for faster response
            ou_lookback=30,
            momentum_lookback=60,

            # Favor momentum
            ou_weight=0.20,
            tsmom_weight=0.35,
            vol_mom_weight=0.30,
            ml_weight=0.15,

            vol_target=0.20,
            min_signal_threshold=0.05,
        ),

        risk_config=RiskConfig(
            max_drawdown=0.30,
            daily_loss_limit=0.05,
            max_position_pct=0.25,
            max_gross_exposure=1.8,  # 180% leverage
            max_sector_pct=0.40,
            var_limit_95=0.06,
        ),

        portfolio_config=PortfolioConfig(
            max_gross_exposure=1.5,
            max_weight=0.20,
            min_weight=0.03,
            kelly_fraction=0.60,  # 60% Kelly
            max_kelly_leverage=2.0,
            target_volatility=0.20,
            min_trade_value=500,
        ),

        execution_config=ExecutionConfig(),
        warmup_bars=30,
        rebalance_frequency='daily',
    )


def run_aggressive_backtest():
    """Main function to run aggressive backtest with IBKR data."""

    # Universe of stocks to trade
    # Mix of tech, financials, and broad market
    universe = [
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Google
        "AMZN",  # Amazon
        "NVDA",  # Nvidia
        "META",  # Meta
        "XLF",   # Financials ETF
        "XLE",   # Energy ETF
    ]

    logger.info("="*60)
    logger.info("AGGRESSIVE BACKTEST WITH REAL IBKR DATA")
    logger.info("="*60)
    logger.info(f"Universe: {universe}")

    # Connect to IBKR TWS
    logger.info("\nConnecting to IBKR TWS...")

    config = IBKRConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=10,  # Use different client ID to avoid conflicts
    )

    broker = IBKRBroker(config=config, require_paper=True)

    try:
        broker.connect()
        logger.info("Connected to IBKR TWS (Paper Trading)")

        # Fetch historical data
        logger.info("\nFetching 2 years of historical data...")
        historical_data, timestamps = fetch_historical_data_from_ibkr(
            broker=broker,
            symbols=universe,
            duration="2 Y",
            bar_size="1 day",
        )

        logger.info(f"\nData loaded for {len(historical_data)} symbols:")
        for symbol in historical_data:
            logger.info(f"  {symbol}: {len(historical_data[symbol])} bars")

        # Filter universe to available data
        available_universe = list(historical_data.keys())

        # Create aggressive config
        config = get_aggressive_config(available_universe)

        # Run backtest
        logger.info("\n" + "="*60)
        logger.info("RUNNING BACKTEST")
        logger.info("="*60)

        engine = QuantOrchestrator(config)

        result = engine.run_backtest(
            historical_data=historical_data,
            timestamps=np.array(timestamps),
            initial_capital=100000.0,
            validate=True,
        )

        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS - AGGRESSIVE STRATEGY")
        print("="*60)
        print(f"Period: {timestamps[0].date()} to {timestamps[-1].date()}")
        print(f"Days: {len(timestamps)}")
        print()
        print("PERFORMANCE:")
        print(f"  Total Return:      {result.total_return:>10.2%}")
        print(f"  Annualized Return: {result.annualized_return:>10.2%}")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:     {result.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:      {result.calmar_ratio:>10.2f}")
        print()
        print("RISK:")
        print(f"  Max Drawdown:      {result.max_drawdown:>10.2%}")
        print(f"  Volatility:        {result.volatility:>10.2%}")
        print(f"  VaR (95%):         {result.var_95:>10.2%}")
        print(f"  Avg Exposure:      {result.avg_exposure:>10.2%}")
        print()
        print("TRADING:")
        print(f"  Total Trades:      {result.total_trades:>10}")
        print(f"  Win Rate:          {result.win_rate:>10.2%}")
        print(f"  Profit Factor:     {result.profit_factor:>10.2f}")
        print(f"  Avg Trade P&L:     ${result.avg_trade_pnl:>9.2f}")
        print()
        print("VALIDATION:")
        print(f"  Passed:            {result.validation_passed}")
        print(f"  PBO:               {result.pbo:>10.2%}")
        print(f"  Deflated Sharpe:   {result.deflated_sharpe:>10.2f}")
        print("="*60)

        # Calculate and show equity curve stats
        equity = result.equity_curve
        final_equity = equity[-1] if len(equity) > 0 else 100000

        print(f"\nFinal Portfolio Value: ${final_equity:,.2f}")
        print(f"Total P&L: ${final_equity - 100000:,.2f}")

        # Save results
        output_dir = Path(__file__).parent.parent.parent.parent / "backtest_results"
        output_dir.mkdir(exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"aggressive_backtest_{timestamp_str}.json"

        engine.save_results(str(output_dir / f"aggressive_{timestamp_str}"))

        logger.info(f"\nResults saved to: {output_dir}")

        return result

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        logger.info("\nDisconnecting from IBKR...")
        broker.disconnect()


if __name__ == "__main__":
    result = run_aggressive_backtest()
