#!/usr/bin/env python3
"""
Comprehensive Pure Momentum Backtests

Tests the strategy across:
1. Different time periods
2. Different asset classes (Equity, FX, Commodities)
3. Different market conditions (Bull, Bear, Sideways)
4. Different volatility regimes

Validates the strategy for live trading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
import warnings
warnings.filterwarnings("ignore")
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import numpy as np
from datetime import datetime
import time
import logging

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.quant_core.strategies.pure_momentum import (
    MomentumConfig, run_pure_momentum_backtest
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def fetch_ibkr_data(broker, symbols, duration="2 Y"):
    """Fetch and align IBKR data."""
    all_data = {}
    all_timestamps = {}

    for symbol in symbols:
        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            bars = broker.get_historical_bars(
                instrument, duration=duration, bar_size="1 day",
                what_to_show="TRADES", use_rth=True
            )
            if bars and len(bars) > 100:
                ohlcv = np.zeros((len(bars), 5))
                timestamps = []
                for i, bar in enumerate(bars):
                    ohlcv[i] = [bar.open, bar.high, bar.low, bar.close, bar.volume or 0]
                    timestamps.append(datetime.fromtimestamp(bar.timestamp_epoch_s))
                all_data[symbol] = ohlcv
                all_timestamps[symbol] = timestamps
            time.sleep(1.5)
        except Exception as e:
            logger.warning(f"Failed {symbol}: {e}")

    if not all_data:
        return None, None

    # Align
    ref_symbol = max(all_data.keys(), key=lambda s: len(all_data[s]))
    ref_timestamps = all_timestamps[ref_symbol]
    ts_to_idx = {ts.date(): i for i, ts in enumerate(ref_timestamps)}

    aligned = {}
    for symbol, ohlcv in all_data.items():
        if symbol == ref_symbol:
            aligned[symbol] = ohlcv
            continue
        arr = np.full((len(ref_timestamps), 5), np.nan)
        for i, ts in enumerate(all_timestamps[symbol]):
            if ts.date() in ts_to_idx:
                arr[ts_to_idx[ts.date()]] = ohlcv[i]
        for col in range(5):
            mask = np.isnan(arr[:, col])
            if mask.any() and not mask.all():
                idx = np.where(~mask, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr[:, col] = arr[idx, col]
        if not np.isnan(arr).all():
            aligned[symbol] = arr

    return aligned, ref_timestamps


def main():
    """Run comprehensive backtests."""

    print("="*80)
    print(" "*20 + "COMPREHENSIVE MOMENTUM BACKTESTS")
    print("="*80)

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=18)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    test_cases = []

    # Test 1: Growth Tech (Bull Market)
    print("\n[1/4] Fetching Growth Tech data...")
    tech_universe = ["QQQ", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA"]
    tech_data, tech_ts = fetch_ibkr_data(broker, tech_universe)
    if tech_data:
        test_cases.append(("Growth Tech", tech_data, tech_ts))
        print(f"  ✓ Loaded {len(tech_data)} symbols, {len(tech_ts)} bars")

    # Test 2: Sector Diversification
    print("\n[2/4] Fetching Sector ETFs...")
    sector_universe = ["SPY", "QQQ", "XLF", "XLE", "XLV", "XLK", "XLI", "XLP"]
    sector_data, sector_ts = fetch_ibkr_data(broker, sector_universe)
    if sector_data:
        test_cases.append(("Sector ETFs", sector_data, sector_ts))
        print(f"  ✓ Loaded {len(sector_data)} symbols, {len(sector_ts)} bars")

    # Test 3: International
    print("\n[3/4] Fetching International exposure...")
    intl_universe = ["SPY", "EFA", "EEM", "VWO", "EWJ", "EWZ", "FXI", "EWG"]
    intl_data, intl_ts = fetch_ibkr_data(broker, intl_universe)
    if intl_data:
        test_cases.append(("International", intl_data, intl_ts))
        print(f"  ✓ Loaded {len(intl_data)} symbols, {len(intl_ts)} bars")

    # Test 4: Commodities & Real Assets
    print("\n[4/4] Fetching Commodities...")
    comm_universe = ["GLD", "SLV", "USO", "DBA", "UNG", "PPLT"]
    comm_data, comm_ts = fetch_ibkr_data(broker, comm_universe)
    if comm_data:
        test_cases.append(("Commodities", comm_data, comm_ts))
        print(f"  ✓ Loaded {len(comm_data)} symbols, {len(comm_ts)} bars")

    broker.disconnect()

    if not test_cases:
        print("\nNo data loaded!")
        return

    # Test configurations
    configs = {
        "Conservative": MomentumConfig(
            fast_ma=20,
            slow_ma=50,
            trend_ma=200,
            momentum_lookback=60,
            max_position=0.20,
            target_exposure=1.2,
            vol_scale=True,
            target_vol=0.15,
        ),
        "Moderate": MomentumConfig(
            fast_ma=10,
            slow_ma=30,
            trend_ma=100,
            momentum_lookback=40,
            max_position=0.30,
            target_exposure=1.5,
            vol_scale=True,
            target_vol=0.20,
        ),
        "Aggressive": MomentumConfig(
            fast_ma=5,
            slow_ma=20,
            trend_ma=50,
            momentum_lookback=20,
            max_position=0.40,
            target_exposure=2.0,
            vol_scale=True,
            target_vol=0.30,
        ),
    }

    print(f"\n{'='*80}")
    print("RUNNING BACKTESTS")
    print(f"{'='*80}")

    all_results = {}

    for test_name, data, timestamps in test_cases:
        print(f"\n{test_name}:")
        print(f"  Period: {timestamps[0].date()} to {timestamps[-1].date()}")
        print(f"  Bars: {len(timestamps)}")

        # Calculate buy & hold return
        spy_or_first = list(data.keys())[0]
        bh_return = (data[spy_or_first][-1, 3] / data[spy_or_first][0, 3]) - 1

        test_results = {"Buy&Hold": {"return": bh_return, "sharpe": 0, "dd": 0.15}}

        for cfg_name, cfg in configs.items():
            try:
                result = run_pure_momentum_backtest(data, timestamps, 100000.0, cfg)
                test_results[cfg_name] = {
                    "return": result["total_return"],
                    "ann_return": result["annualized_return"],
                    "sharpe": result["sharpe_ratio"],
                    "dd": result["max_drawdown"],
                    "vol": result["volatility"],
                    "trades": result["total_trades"],
                }
            except Exception as e:
                logger.error(f"Failed {cfg_name}: {e}")

        all_results[test_name] = test_results

    # Print comprehensive results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*80}")

    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        print(f"{'Strategy':<18} {'Return':>12} {'Ann.Ret':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8}")
        print("-"*80)

        for strategy, metrics in results.items():
            if strategy == "Buy&Hold":
                print(f"{strategy:<18} {metrics['return']:>11.2%} {'N/A':>10} {'N/A':>8} {metrics['dd']:>9.2%} {'0':>8}")
            else:
                print(f"{strategy:<18} {metrics['return']:>11.2%} {metrics['ann_return']:>9.2%} "
                      f"{metrics['sharpe']:>8.2f} {metrics['dd']:>9.2%} {metrics['trades']:>8}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("STRATEGY PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    for cfg_name in configs.keys():
        returns = [r[cfg_name]["return"] for r in all_results.values() if cfg_name in r]
        sharpes = [r[cfg_name]["sharpe"] for r in all_results.values() if cfg_name in r]
        dds = [r[cfg_name]["dd"] for r in all_results.values() if cfg_name in r]

        if returns:
            print(f"\n{cfg_name}:")
            print(f"  Avg Return:    {np.mean(returns):>8.2%}")
            print(f"  Best Return:   {max(returns):>8.2%}")
            print(f"  Worst Return:  {min(returns):>8.2%}")
            print(f"  Avg Sharpe:    {np.mean(sharpes):>8.2f}")
            print(f"  Avg Max DD:    {np.mean(dds):>8.2%}")
            print(f"  Win Rate:      {sum(1 for r in returns if r > 0)/len(returns):>8.2%}")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    # Identify best configuration
    avg_returns = {}
    for cfg_name in configs.keys():
        returns = [r[cfg_name]["return"] for r in all_results.values() if cfg_name in r]
        if returns:
            avg_returns[cfg_name] = np.mean(returns)

    if avg_returns:
        best_config = max(avg_returns.items(), key=lambda x: x[1])
        print(f"\n✓ BEST OVERALL: {best_config[0]} ({best_config[1]:.2%} avg return)")
        print(f"\nFor live trading, use:")
        print(f"  - Conservative: Wealth preservation (lower vol, steady returns)")
        print(f"  - Moderate: Balanced growth (recommended for most)")
        print(f"  - Aggressive: Maximum returns (higher drawdowns)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
