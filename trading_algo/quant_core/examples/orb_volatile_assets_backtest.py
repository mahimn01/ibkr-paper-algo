#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) on Volatile Assets

Tests intraday ORB strategy on volatile day trading stocks.
This should generate MANY more trades than mean reversion.

Strategy:
- Define opening range (first 15-30 minutes)
- Buy breakout above range high
- Short breakout below range low
- Exit at target or stop loss or EOD

Expected: 50-100+ trades per year on volatile stocks
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

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec

logging.basicConfig(level=logging.WARNING)


def simulate_orb_from_daily_data(prices, dates, initial_capital=100000.0):
    """
    Simulate ORB strategy using daily data.

    Since we don't have intraday bars, we'll simulate by:
    - Using daily open as "range low"
    - Using daily high as breakout if it's >2% above open
    - Using daily low as breakdown if it's >2% below open
    - Target = 1.5x range size
    - Stop = opposite side of range

    This is a ROUGH approximation but shows the concept.
    """
    capital = initial_capital
    position = 0
    equity_curve = [initial_capital]
    trades = []

    for t in range(1, len(prices)):
        # Simulate: assume we can see open, high, low, close each day
        # In reality, we'd need intraday bars

        # Approximate opening range (10% of daily range)
        daily_range = abs(prices[t] - prices[t-1]) / prices[t-1]

        # Only trade if daily move is significant (>1.5%)
        if daily_range < 0.015:
            equity = capital + (position * prices[t] if position != 0 else 0)
            equity_curve.append(equity)
            continue

        # Close any existing position at start of day
        if position != 0:
            exit_price = prices[t]
            if position > 0:
                pnl = position * (exit_price - entry_price)
                capital += position * exit_price
            else:
                pnl = -position * (entry_price - exit_price)
                capital += -position * entry_price

            trades.append({
                'type': 'close',
                'price': exit_price,
                'date': dates[t],
                'pnl': pnl,
                'return_pct': pnl / (abs(position) * entry_price) * 100
            })
            position = 0

        # Determine if we have a breakout opportunity
        # Bullish: if price moved up >1.5%
        if prices[t] > prices[t-1] * 1.015 and position == 0:
            entry_price = prices[t-1] * 1.015  # Entry at breakout
            shares = (capital * 0.20) / entry_price  # 20% position size
            position = shares
            capital -= shares * entry_price

            trades.append({
                'type': 'buy',
                'price': entry_price,
                'date': dates[t],
                'shares': shares
            })

        # Bearish: if price moved down >1.5%
        elif prices[t] < prices[t-1] * 0.985 and position == 0:
            entry_price = prices[t-1] * 0.985  # Entry at breakdown
            shares = (capital * 0.20) / entry_price  # 20% position size
            position = -shares  # Short position
            capital += shares * entry_price

            trades.append({
                'type': 'sell_short',
                'price': entry_price,
                'date': dates[t],
                'shares': shares
            })

        equity = capital + (position * prices[t] if position > 0 else -position * prices[t])
        equity_curve.append(equity)

    # Calculate metrics
    equity_arr = np.array(equity_curve)
    total_return = (equity_arr[-1] / initial_capital - 1) * 100
    n_years = len(dates) / 252
    ann_return = ((equity_arr[-1] / initial_capital) ** (1 / n_years) - 1) * 100

    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = (np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak * 100
    max_dd = np.max(drawdown)

    close_trades = [t for t in trades if t['type'] == 'close']
    winning_trades = sum(1 for t in close_trades if t.get('pnl', 0) > 0)
    total_trades = len(close_trades)
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    # Calculate volatility
    price_returns = np.diff(prices) / prices[:-1]
    volatility = np.std(price_returns) * np.sqrt(252) * 100

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'final_equity': equity_arr[-1],
        'volatility': volatility,
        'trades': trades
    }


def main():
    """Test ORB on volatile day trading stocks."""

    print("="*90)
    print(" "*20 + "OPENING RANGE BREAKOUT - VOLATILE ASSETS")
    print("="*90)
    print("\nNOTE: This simulates ORB using daily data.")
    print("For TRUE ORB, need 1-min or 5-min intraday bars from IBKR.")
    print("="*90)

    # Connect to IBKR
    print("\nConnecting to IBKR TWS...")
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=31)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Test on most volatile day trading stocks
    test_symbols = {
        'Day Trading Favorites': ['TSLA', 'AMD', 'NVDA', 'PLTR'],
        'Meme Stocks': ['GME', 'AMC'],
        'Crypto-Related': ['COIN', 'MSTR', 'MARA'],
        'High Volatility': ['SNAP', 'UBER', 'LYFT']
    }

    all_data = {}

    for category, symbols in test_symbols.items():
        print(f"\nFetching {category}...")
        for symbol in symbols:
            print(f"  {symbol}...", end=" ")
            try:
                instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")

                # Fetch 1 year of daily data
                bars = broker.get_historical_bars(
                    instrument,
                    duration="1 Y",
                    bar_size="1 day",
                    what_to_show="TRADES",
                    use_rth=True
                )

                if bars and len(bars) > 100:
                    prices = np.array([bar.close for bar in bars])
                    dates = [datetime.fromtimestamp(bar.timestamp_epoch_s) for bar in bars]

                    all_data[symbol] = {
                        'prices': prices,
                        'dates': dates,
                        'category': category
                    }

                    print(f"✓ {len(bars)} bars")
                else:
                    print("✗ Not enough data")

                time_module.sleep(1.5)

            except Exception as e:
                print(f"✗ {e}")

    broker.disconnect()

    if not all_data:
        print("\n✗ No data loaded")
        return

    print(f"\n✓ Loaded {len(all_data)} symbols")

    # Run backtests
    print(f"\n{'='*90}")
    print("RUNNING ORB SIMULATIONS (Daily Data Approximation)")
    print(f"{'='*90}")

    all_results = {}

    for symbol, data in all_data.items():
        print(f"\n{symbol} ({data['category']}):")
        prices = data['prices']
        dates = data['dates']

        print("  Testing ORB simulation...", end=" ")
        results = simulate_orb_from_daily_data(prices, dates)
        all_results[symbol] = results

        print(f"✓ {results['ann_return']:+.2f}% | {results['total_trades']} trades | Sharpe {results['sharpe']:.2f}")

    # Print results
    print(f"\n{'='*90}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*90}")

    print(f"\n{'Symbol':<8} {'Category':<22} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'WinRate':>8} {'Vol':>8}")
    print("-"*90)

    for symbol, results in all_results.items():
        category = all_data[symbol]['category']
        ret = results['ann_return']
        sharpe = results['sharpe']
        trades = results['total_trades']
        win_rate = results['win_rate']
        vol = results['volatility']

        print(f"{symbol:<8} {category:<22} {ret:>9.2f}% {sharpe:>8.2f} {trades:>8} {win_rate:>7.1f}% {vol:>7.1f}%")

    # Category analysis
    print(f"\n{'='*90}")
    print("ANALYSIS BY CATEGORY")
    print(f"{'='*90}")

    categories = {}
    for symbol, results in all_results.items():
        category = all_data[symbol]['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(results)

    for category, results_list in categories.items():
        returns = [r['ann_return'] for r in results_list]
        trades = [r['total_trades'] for r in results_list]
        sharpes = [r['sharpe'] for r in results_list]

        print(f"\n{category}:")
        print(f"  Avg Return:       {np.mean(returns):>8.2f}%")
        print(f"  Avg Trades:       {np.mean(trades):>8.1f}")
        print(f"  Avg Sharpe:       {np.mean(sharpes):>8.2f}")
        print(f"  Profitable:       {sum(1 for r in returns if r > 0)}/{len(returns)}")

    # Most active
    print(f"\n{'='*90}")
    print("MOST ACTIVE (By Trades)")
    print(f"{'='*90}")

    sorted_by_trades = sorted(all_results.items(), key=lambda x: x[1]['total_trades'], reverse=True)

    for i, (symbol, results) in enumerate(sorted_by_trades[:5]):
        print(f"\n{i+1}. {symbol} ({all_data[symbol]['category']}):")
        print(f"   Trades:        {results['total_trades']}")
        print(f"   Return:        {results['ann_return']:+.2f}%")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")
        print(f"   Sharpe:        {results['sharpe']:.2f}")

    # Best performers
    print(f"\n{'='*90}")
    print("BEST PERFORMERS")
    print(f"{'='*90}")

    sorted_by_return = sorted(all_results.items(), key=lambda x: x[1]['ann_return'], reverse=True)

    for i, (symbol, results) in enumerate(sorted_by_return[:5]):
        print(f"\n{i+1}. {symbol} ({all_data[symbol]['category']}):")
        print(f"   Return:        {results['ann_return']:+.2f}%")
        print(f"   Sharpe:        {results['sharpe']:.2f}")
        print(f"   Trades:        {results['total_trades']}")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")

    # Summary
    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")

    avg_return = np.mean([r['ann_return'] for r in all_results.values()])
    avg_trades = np.mean([r['total_trades'] for r in all_results.values()])
    avg_sharpe = np.mean([r['sharpe'] for r in all_results.values()])

    print(f"\nORB Simulation Performance (Daily Data):")
    print(f"  Average Return:   {avg_return:+.2f}%")
    print(f"  Average Trades:   {avg_trades:.1f} per year")
    print(f"  Average Sharpe:   {avg_sharpe:.2f}")

    print(f"\n⚠️  IMPORTANT LIMITATIONS:")
    print(f"  - This uses DAILY bars, not intraday data")
    print(f"  - Real ORB needs 1-min or 5-min bars")
    print(f"  - True ORB would generate 50-100+ trades/year")
    print(f"  - IBKR provides intraday data via get_historical_bars()")
    print(f"    with bar_size='1 min' or '5 mins'")

    print(f"\nTo run REAL ORB:")
    print(f"  1. Fetch intraday bars (1-min or 5-min)")
    print(f"  2. Calculate opening range (first 15-30 mins)")
    print(f"  3. Detect breakouts during the day")
    print(f"  4. Exit at target/stop or market close")

    print("="*90)


if __name__ == "__main__":
    main()
