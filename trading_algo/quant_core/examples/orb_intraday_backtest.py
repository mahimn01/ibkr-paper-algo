#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) - Proper Intraday Backtest

Uses REAL 5-minute intraday bars from IBKR to test ORB strategy.

Strategy:
1. Calculate opening range from first 15-30 minutes (3-6 bars of 5-min data)
2. Buy breakout above range high + threshold
3. Short breakdown below range low - threshold
4. Target: 1.5x range size
5. Stop: Opposite side of range
6. Exit all positions at market close (3:55pm ET)

Expected Performance:
- 50-100+ trades per year on volatile stocks
- Win rate: 45-55%
- Sharpe: 0.8-1.2
- Annual return: 15-25%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import numpy as np
from datetime import datetime, time, timedelta
import time as time_module
import logging
from collections import defaultdict

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.quant_core.strategies.intraday.opening_range_breakout import (
    OpeningRangeBreakout, ORBConfig, ORBPosition
)

logging.basicConfig(level=logging.WARNING)


def backtest_orb_intraday(
    symbol: str,
    intraday_bars: list,
    initial_capital: float = 100000.0,
    range_minutes: int = 30,
    bar_size_minutes: int = 5
):
    """
    Backtest ORB strategy on intraday data.

    Args:
        symbol: Stock symbol
        intraday_bars: List of intraday bars (5-min bars)
        initial_capital: Starting capital
        range_minutes: Opening range period in minutes (default 30)
        bar_size_minutes: Size of each bar in minutes (default 5)

    Returns:
        Dictionary with performance metrics
    """

    # Create ORB strategy
    config = ORBConfig(
        range_minutes=range_minutes,
        breakout_threshold=0.002,  # 0.2% threshold
        target_multiplier=1.5,
        stop_multiplier=1.0,
        position_size=0.30,  # 30% of capital per trade
        min_range_size=0.003,  # Min 0.3% range
        max_range_size=0.05,  # Max 5% range
        volume_confirmation=True
    )

    strategy = OpeningRangeBreakout(config=config, initial_capital=initial_capital)

    # Group bars by trading day
    days = defaultdict(list)
    for bar in intraday_bars:
        bar_dt = datetime.fromtimestamp(bar.timestamp_epoch_s)
        day_key = bar_dt.date()
        days[day_key].append({
            'datetime': bar_dt,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    # Sort days
    sorted_days = sorted(days.keys())

    print(f"\n  Trading days: {len(sorted_days)}")
    print(f"  Total bars: {len(intraday_bars)}")

    # Process each trading day
    num_bars_for_range = range_minutes // bar_size_minutes

    for day in sorted_days:
        day_bars = days[day]

        # Need enough bars for opening range
        if len(day_bars) < num_bars_for_range + 5:
            continue

        # Market open time (9:30 AM ET)
        market_open = time(9, 30)

        # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
        trading_bars = [
            b for b in day_bars
            if time(9, 30) <= b['datetime'].time() <= time(16, 0)
        ]

        if len(trading_bars) < num_bars_for_range + 5:
            continue

        # Build opening range from first N bars
        opening_bars = trading_bars[:num_bars_for_range]

        # Calculate opening range
        highs = [b['high'] for b in opening_bars]
        lows = [b['low'] for b in opening_bars]
        volumes = [b['volume'] for b in opening_bars]

        range_high, range_low, range_size = strategy.calculate_opening_range(
            highs, lows, num_bars_for_range
        )

        if range_high is None or not strategy.is_valid_range(range_size):
            continue

        # Update strategy's opening range
        strategy.opening_ranges[symbol] = {
            'high': range_high,
            'low': range_low,
            'size': range_size,
            'confirmed': True,
            'date': day
        }

        avg_volume = np.mean(volumes)

        # Process bars after opening range
        for i, bar in enumerate(trading_bars[num_bars_for_range:], start=num_bars_for_range):
            current_dt = bar['datetime']
            current_price = bar['close']
            current_volume = bar['volume']

            # Update positions first
            current_prices = {symbol: current_price}
            strategy.update_positions(current_prices, current_dt)

            # Check for new signals (only if no position)
            if not any(p.symbol == symbol for p in strategy.positions):
                signals = strategy.generate_signals(
                    symbol,
                    current_price,
                    current_volume,
                    avg_volume,
                    current_dt
                )

                # Open position if signal
                for signal in signals:
                    strategy.open_position(signal, current_dt)

            # Close all positions at 3:55 PM (5 minutes before close)
            if current_dt.time() >= time(15, 55):
                positions_to_close = [p for p in strategy.positions if p.symbol == symbol]
                for position in positions_to_close:
                    strategy.close_position(position, current_price, current_dt, 'eod')

        # Close any remaining positions at end of day
        positions_to_close = [p for p in strategy.positions if p.symbol == symbol]
        for position in positions_to_close:
            last_price = trading_bars[-1]['close']
            last_dt = trading_bars[-1]['datetime']
            strategy.close_position(position, last_price, last_dt, 'eod_force')

    # Get performance stats
    return strategy.get_performance_stats()


def main():
    """Run proper intraday ORB backtest."""

    print("="*90)
    print(" "*20 + "OPENING RANGE BREAKOUT - INTRADAY BACKTEST")
    print("="*90)
    print("\nUsing REAL 5-minute intraday bars from IBKR")
    print("="*90)

    # Connect to IBKR
    print("\nConnecting to IBKR TWS...")
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=32)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Test symbols - focus on most volatile day trading stocks
    test_symbols = [
        'TSLA',   # Tesla - very volatile
        'NVDA',   # Nvidia - high volume
        'AMD',    # AMD - day trader favorite
        'PLTR',   # Palantir - very volatile
        'COIN',   # Coinbase - crypto proxy
        'SPY',    # S&P 500 ETF - baseline
    ]

    all_data = {}

    print("\nFetching 5-minute intraday data...")
    print("(This will take a while - IBKR rate limits apply)")

    for symbol in test_symbols:
        print(f"\n{symbol}:")
        try:
            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange="SMART",
                currency="USD"
            )

            # Fetch 1 month of 5-minute bars
            # Note: IBKR has limits on intraday data (usually ~30 days max)
            print(f"  Fetching 5-min bars (30 days)...", end=" ")
            bars = broker.get_historical_bars(
                instrument,
                duration="30 D",
                bar_size="5 mins",
                what_to_show="TRADES",
                use_rth=True  # Regular trading hours only
            )

            if bars and len(bars) > 100:
                all_data[symbol] = bars
                print(f"✓ {len(bars)} bars")
            else:
                print("✗ Not enough data")

            # Rate limiting - IBKR requires pauses
            time_module.sleep(3)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            time_module.sleep(3)

    broker.disconnect()

    if not all_data:
        print("\n✗ No data loaded")
        return

    print(f"\n✓ Loaded {len(all_data)} symbols")

    # Run backtests
    print(f"\n{'='*90}")
    print("RUNNING INTRADAY ORB BACKTESTS")
    print(f"{'='*90}")

    all_results = {}

    for symbol, bars in all_data.items():
        print(f"\n{symbol}:")
        print(f"  Backtesting...", end=" ")

        try:
            results = backtest_orb_intraday(
                symbol,
                bars,
                initial_capital=100000.0,
                range_minutes=30,
                bar_size_minutes=5
            )

            all_results[symbol] = results

            if results:
                print(f"✓ {results.get('ann_return', 0):.2f}% | " +
                      f"{results.get('total_trades', 0)} trades | " +
                      f"Sharpe {results.get('sharpe', 0):.2f}")
            else:
                print("✗ No trades executed")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Print comprehensive results
    print(f"\n{'='*90}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*90}")

    if not all_results or all(not r for r in all_results.values()):
        print("\n✗ No successful backtests")
        return

    print(f"\n{'Symbol':<8} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>8}")
    print("-"*90)

    for symbol, results in all_results.items():
        if not results:
            continue

        ret = results.get('ann_return', 0)
        sharpe = results.get('sharpe', 0)
        max_dd = results.get('max_dd', 0)
        trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0)

        print(f"{symbol:<8} {ret:>9.2f}% {sharpe:>8.2f} {max_dd:>9.2f}% {trades:>8} {win_rate:>7.1f}%")

    # Statistics
    print(f"\n{'='*90}")
    print("SUMMARY STATISTICS")
    print(f"{'='*90}")

    valid_results = [r for r in all_results.values() if r]

    if not valid_results:
        print("\n✗ No valid results")
        return

    returns = [r['ann_return'] for r in valid_results]
    sharpes = [r['sharpe'] for r in valid_results]
    trades = [r['total_trades'] for r in valid_results]
    win_rates = [r['win_rate'] for r in valid_results]

    print(f"\nAnnualized Returns:")
    print(f"  Average:    {np.mean(returns):>8.2f}%")
    print(f"  Median:     {np.median(returns):>8.2f}%")
    print(f"  Best:       {max(returns):>8.2f}%")
    print(f"  Worst:      {min(returns):>8.2f}%")

    print(f"\nTrades Per Month (30 days tested):")
    print(f"  Average:    {np.mean(trades):>8.1f}")
    print(f"  Median:     {np.median(trades):>8.0f}")
    print(f"  Most:       {max(trades):>8.0f}")
    print(f"  Least:      {min(trades):>8.0f}")
    print(f"  Annualized: {np.mean(trades) * 12:>8.1f} trades/year")

    print(f"\nWin Rates:")
    print(f"  Average:    {np.mean(win_rates):>8.1f}%")
    print(f"  Median:     {np.median(win_rates):>8.1f}%")

    print(f"\nSharpe Ratios:")
    print(f"  Average:    {np.mean(sharpes):>8.2f}")
    print(f"  Median:     {np.median(sharpes):>8.2f}")

    # Best performers
    print(f"\n{'='*90}")
    print("TOP PERFORMERS")
    print(f"{'='*90}")

    sorted_by_return = sorted(
        [(s, r) for s, r in all_results.items() if r],
        key=lambda x: x[1]['ann_return'],
        reverse=True
    )

    for i, (symbol, results) in enumerate(sorted_by_return[:3]):
        print(f"\n{i+1}. {symbol}:")
        print(f"   Return:        {results['ann_return']:>8.2f}%")
        print(f"   Sharpe:        {results['sharpe']:>8.2f}")
        print(f"   Trades:        {results['total_trades']:>8}")
        print(f"   Win Rate:      {results['win_rate']:>8.1f}%")
        print(f"   Max Drawdown:  {results['max_dd']:>8.2f}%")

    # Most active
    print(f"\n{'='*90}")
    print("MOST ACTIVE (By Trades)")
    print(f"{'='*90}")

    sorted_by_trades = sorted(
        [(s, r) for s, r in all_results.items() if r],
        key=lambda x: x[1]['total_trades'],
        reverse=True
    )

    for i, (symbol, results) in enumerate(sorted_by_trades[:3]):
        print(f"\n{i+1}. {symbol}:")
        print(f"   Trades:        {results['total_trades']:>8}")
        print(f"   Return:        {results['ann_return']:>8.2f}%")
        print(f"   Win Rate:      {results['win_rate']:>8.1f}%")
        print(f"   Sharpe:        {results['sharpe']:>8.2f}")

    # Conclusion
    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")

    print(f"\nReal Intraday ORB Strategy Performance:")
    print(f"  Average Return:   {np.mean(returns):>8.2f}% annualized")
    print(f"  Average Trades:   {np.mean(trades):>8.1f} per month")
    print(f"  Estimated Annual: {np.mean(trades) * 12:.0f} trades/year")
    print(f"  Average Win Rate: {np.mean(win_rates):>8.1f}%")
    print(f"  Average Sharpe:   {np.mean(sharpes):>8.2f}")

    # Compare to mean reversion
    print(f"\n📊 COMPARISON TO MEAN REVERSION:")
    print(f"  Mean Reversion:  1-4 trades/year")
    print(f"  ORB Strategy:    {np.mean(trades) * 12:.0f} trades/year")
    print(f"  ORB = {(np.mean(trades) * 12) / 2:.0f}x MORE ACTIVE")

    if np.mean(returns) > 15:
        print(f"\n✓ ORB strategy suitable for active day trading")
        print(f"  - High trade frequency ({np.mean(trades) * 12:.0f}/year)")
        print(f"  - Positive returns ({np.mean(returns):.1f}%)")
        print(f"  - Acceptable Sharpe ({np.mean(sharpes):.2f})")
    else:
        print(f"\n⚠️  ORB strategy shows moderate performance")
        print(f"  - Trade frequency is good ({np.mean(trades) * 12:.0f}/year)")
        print(f"  - Returns need improvement")
        print(f"  - Consider parameter optimization")

    print("="*90)


if __name__ == "__main__":
    main()
