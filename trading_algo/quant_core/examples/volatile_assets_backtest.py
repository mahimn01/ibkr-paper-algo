#!/usr/bin/env python3
"""
Mean Reversion RSI on Volatile Assets

Tests strategy on:
1. FX currency pairs (via ETFs)
2. Volatile day trading stocks (TSLA, meme stocks, crypto-related)

Hypothesis: Higher volatility = more trading opportunities = higher returns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import numpy as np
from datetime import datetime
import time
import logging

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec

logging.basicConfig(level=logging.WARNING)


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return np.array([50.0] * len(prices))

    returns = np.diff(prices)
    rsi_values = []

    for i in range(len(prices)):
        if i < period:
            rsi_values.append(50.0)
            continue

        recent_returns = returns[max(0, i-period):i]
        gains = recent_returns[recent_returns > 0]
        losses = -recent_returns[recent_returns < 0]

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi_values.append(rsi)

    return np.array(rsi_values)


def backtest_mean_reversion_rsi(
    prices,
    dates,
    initial_capital=100000.0,
    rsi_period=14,
    oversold_threshold=30,
    overbought_threshold=70,
    position_size=1.0,
    stop_loss_pct=0.10,
):
    """Backtest with detailed trade tracking."""

    capital = initial_capital
    position = 0
    equity_curve = [initial_capital]
    trades = []

    rsi = calculate_rsi(prices, period=rsi_period)
    entry_price = 0

    for t in range(rsi_period + 10, len(prices)):
        current_price = prices[t]
        current_rsi = rsi[t]

        # Stop loss check
        if position > 0 and entry_price > 0:
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct < -stop_loss_pct:
                capital = position * current_price
                pnl = capital - initial_capital
                position = 0

                trades.append({
                    'type': 'sell',
                    'reason': 'stop_loss',
                    'price': current_price,
                    'rsi': current_rsi,
                    'date': dates[t],
                    'pnl': pnl,
                    'return_pct': (current_price / entry_price - 1) * 100
                })
                entry_price = 0

        # Buy signal
        if current_rsi < oversold_threshold and position == 0 and capital > 0:
            buy_amount = capital * position_size
            shares = buy_amount / current_price
            position = shares
            capital -= buy_amount
            entry_price = current_price

            trades.append({
                'type': 'buy',
                'reason': 'oversold',
                'price': current_price,
                'rsi': current_rsi,
                'date': dates[t],
                'shares': shares
            })

        # Sell signal
        elif current_rsi > overbought_threshold and position > 0:
            capital += position * current_price
            pnl = capital - initial_capital
            position = 0

            trades.append({
                'type': 'sell',
                'reason': 'overbought',
                'price': current_price,
                'rsi': current_rsi,
                'date': dates[t],
                'pnl': pnl,
                'return_pct': (current_price / entry_price - 1) * 100
            })
            entry_price = 0

        equity = capital + (position * current_price)
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

    sell_trades = [t for t in trades if t['type'] == 'sell']
    winning_trades = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
    total_trades = len(sell_trades)
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
    """Run backtest on volatile assets."""

    print("="*90)
    print(" "*20 + "MEAN REVERSION RSI - VOLATILE ASSETS TEST")
    print("="*90)

    # Connect to IBKR
    print("\nConnecting to IBKR TWS...")
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=30)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Define test universe
    test_symbols = {
        'FX ETFs': ['UUP', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA'],
        'Volatile Tech': ['TSLA', 'AMD', 'NVDA'],
        'Crypto-Related': ['COIN', 'MSTR', 'MARA'],
        'Meme Stocks': ['GME', 'AMC'],
        'High Volatility': ['PLTR', 'SNAP', 'UBER']
    }

    all_data = {}

    for category, symbols in test_symbols.items():
        print(f"\nFetching {category}...")
        for symbol in symbols:
            print(f"  {symbol}...", end=" ")
            try:
                instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")

                # Fetch 1 year of data
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

                time.sleep(1.5)

            except Exception as e:
                print(f"✗ {e}")

    broker.disconnect()

    if not all_data:
        print("\n✗ No data loaded")
        return

    print(f"\n✓ Loaded {len(all_data)} symbols")

    # Run backtests
    print(f"\n{'='*90}")
    print("RUNNING BACKTESTS")
    print(f"{'='*90}")

    all_results = {}

    for symbol, data in all_data.items():
        print(f"\n{symbol} ({data['category']}):")
        prices = data['prices']
        dates = data['dates']

        print("  Testing...", end=" ")
        results = backtest_mean_reversion_rsi(prices, dates)
        all_results[symbol] = results

        print(f"✓ {results['ann_return']:+.2f}% | {results['total_trades']} trades | Sharpe {results['sharpe']:.2f}")

    # Print comprehensive results
    print(f"\n{'='*90}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*90}")

    print(f"\n{'Symbol':<8} {'Category':<18} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'WinRate':>8} {'Vol':>8}")
    print("-"*90)

    for symbol, results in all_results.items():
        category = all_data[symbol]['category']
        ret = results['ann_return']
        sharpe = results['sharpe']
        trades = results['total_trades']
        win_rate = results['win_rate']
        vol = results['volatility']

        print(f"{symbol:<8} {category:<18} {ret:>9.2f}% {sharpe:>8.2f} {trades:>8} {win_rate:>7.1f}% {vol:>7.1f}%")

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

    # Top performers by trades
    print(f"\n{'='*90}")
    print("MOST ACTIVE (By Number of Trades)")
    print(f"{'='*90}")

    sorted_by_trades = sorted(all_results.items(), key=lambda x: x[1]['total_trades'], reverse=True)

    for i, (symbol, results) in enumerate(sorted_by_trades[:10]):
        print(f"\n{i+1}. {symbol} ({all_data[symbol]['category']}):")
        print(f"   Trades:        {results['total_trades']}")
        print(f"   Return:        {results['ann_return']:+.2f}%")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")
        print(f"   Sharpe:        {results['sharpe']:.2f}")
        print(f"   Volatility:    {results['volatility']:.1f}%")

    # Top performers by return
    print(f"\n{'='*90}")
    print("BEST PERFORMERS (By Return)")
    print(f"{'='*90}")

    sorted_by_return = sorted(all_results.items(), key=lambda x: x[1]['ann_return'], reverse=True)

    for i, (symbol, results) in enumerate(sorted_by_return[:10]):
        print(f"\n{i+1}. {symbol} ({all_data[symbol]['category']}):")
        print(f"   Return:        {results['ann_return']:+.2f}%")
        print(f"   Sharpe:        {results['sharpe']:.2f}")
        print(f"   Trades:        {results['total_trades']}")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")
        print(f"   Max DD:        {results['max_dd']:.2f}%")

    # Volatility vs Returns analysis
    print(f"\n{'='*90}")
    print("VOLATILITY vs PERFORMANCE")
    print(f"{'='*90}")

    vols = [r['volatility'] for r in all_results.values()]
    returns = [r['ann_return'] for r in all_results.values()]
    trades_list = [r['total_trades'] for r in all_results.values()]

    print(f"\nCorrelation Analysis:")
    print(f"  Volatility vs Return:  {np.corrcoef(vols, returns)[0,1]:.3f}")
    print(f"  Volatility vs Trades:  {np.corrcoef(vols, trades_list)[0,1]:.3f}")
    print(f"  Trades vs Return:      {np.corrcoef(trades_list, returns)[0,1]:.3f}")

    # Summary
    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")

    avg_return = np.mean([r['ann_return'] for r in all_results.values()])
    avg_trades = np.mean([r['total_trades'] for r in all_results.values()])
    avg_sharpe = np.mean([r['sharpe'] for r in all_results.values()])

    print(f"\nVolatile Assets Performance:")
    print(f"  Average Return:   {avg_return:+.2f}%")
    print(f"  Average Trades:   {avg_trades:.1f} per year")
    print(f"  Average Sharpe:   {avg_sharpe:.2f}")

    # Compare to stable stocks
    print(f"\nKey Findings:")

    if avg_trades > 5:
        print(f"  ✓ More trading activity ({avg_trades:.1f} trades vs 1-2 for stable stocks)")

    if avg_return > 10:
        print(f"  ✓ Higher returns on volatile assets ({avg_return:.2f}%)")
    else:
        print(f"  ⚠️ Returns similar or lower than stable stocks")

    # Best category
    best_category = max(categories.items(), key=lambda x: np.mean([r['ann_return'] for r in x[1]]))
    print(f"\n✓ BEST CATEGORY: {best_category[0]}")
    print(f"  Avg Return: {np.mean([r['ann_return'] for r in best_category[1]]):.2f}%")

    print("="*90)


if __name__ == "__main__":
    main()
