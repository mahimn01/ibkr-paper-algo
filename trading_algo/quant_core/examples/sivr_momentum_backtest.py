#!/usr/bin/env python3
"""
SIVR (Silver ETF) Momentum Backtest

Detailed backtest of Pure Momentum strategy on SIVR over recent months.
Shows trade-by-trade execution and month-by-month performance.
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
from collections import defaultdict

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig, PureMomentumStrategy

logging.basicConfig(level=logging.WARNING)


def run_detailed_sivr_backtest(ohlcv, timestamps, config):
    """Run detailed backtest with trade logging."""

    strategy = PureMomentumStrategy(config)
    close_prices = ohlcv[:, 3]

    cash = 100000.0
    shares = 0

    equity_curve = []
    dates = []
    trades = []

    warmup = max(config.trend_ma, config.momentum_lookback) + 10

    for t in range(warmup, len(timestamps)):
        current_price = close_prices[t]
        price_history = {'SIVR': close_prices[:t+1]}

        # Calculate equity
        position_value = shares * current_price
        equity = cash + position_value
        equity_curve.append(equity)
        dates.append(timestamps[t])

        # Get target weight
        target_weights = strategy.get_target_weights(['SIVR'], price_history)
        target_weight = target_weights.get('SIVR', 0)
        target_value = equity * target_weight

        # Current position value
        current_value = shares * current_price
        delta_value = target_value - current_value

        # Execute trade if significant
        if abs(delta_value) > 1000:  # $1000 minimum trade
            delta_shares = delta_value / current_price

            slippage = 0.001
            exec_price = current_price * (1 + slippage if delta_shares > 0 else 1 - slippage)
            cost = abs(delta_shares) * exec_price
            commission = cost * 0.0005

            trade_type = "BUY" if delta_shares > 0 else "SELL"

            if delta_shares > 0:
                cash -= cost + commission
            else:
                cash += cost - commission

            shares += delta_shares

            # Log trade
            trades.append({
                'date': timestamps[t],
                'type': trade_type,
                'shares': abs(delta_shares),
                'price': exec_price,
                'value': cost,
                'commission': commission,
                'new_shares': shares,
                'equity': equity,
                'weight': target_weight,
            })

    return {
        'equity_curve': np.array(equity_curve),
        'dates': dates,
        'trades': trades,
        'final_shares': shares,
        'final_cash': cash,
    }


def main():
    """Run SIVR backtest."""

    print("="*80)
    print(" "*25 + "SIVR MOMENTUM BACKTEST")
    print("="*80)

    print("\nFetching SIVR data (last 12 months)...")

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=24)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Fetch SIVR
    instrument = InstrumentSpec(kind="STK", symbol="SIVR", exchange="SMART", currency="USD")
    bars = broker.get_historical_bars(
        instrument,
        duration="1 Y",
        bar_size="1 day",
        what_to_show="TRADES",
        use_rth=True
    )

    broker.disconnect()

    if not bars or len(bars) < 100:
        print("✗ Not enough data")
        return

    ohlcv = np.zeros((len(bars), 5))
    timestamps = []

    for i, bar in enumerate(bars):
        ohlcv[i] = [bar.open, bar.high, bar.low, bar.close, bar.volume or 0]
        timestamps.append(datetime.fromtimestamp(bar.timestamp_epoch_s))

    print(f"✓ Loaded {len(bars)} bars: {timestamps[0].date()} to {timestamps[-1].date()}")

    # Run backtest with Aggressive config
    print("\nRunning Pure Momentum Aggressive backtest...")

    aggressive_config = MomentumConfig(
        fast_ma=5,
        slow_ma=20,
        trend_ma=50,
        momentum_lookback=20,
        max_position=0.40,
        target_exposure=2.0,
        vol_scale=True,
        target_vol=0.30,
    )

    result = run_detailed_sivr_backtest(ohlcv, timestamps, aggressive_config)

    equity_curve = result['equity_curve']
    dates = result['dates']
    trades = result['trades']

    # Calculate performance
    close_prices = ohlcv[:, 3]

    # Buy & Hold
    warmup = max(aggressive_config.trend_ma, aggressive_config.momentum_lookback) + 10
    bh_start_price = close_prices[warmup]
    bh_end_price = close_prices[-1]
    bh_return = (bh_end_price / bh_start_price - 1) * 100
    bh_value = 100000 * (bh_end_price / bh_start_price)

    # Strategy
    strategy_return = (equity_curve[-1] / 100000 - 1) * 100
    n_days = len(dates)
    n_years = n_days / 252
    ann_return = ((equity_curve[-1] / 100000) ** (1 / n_years) - 1) * 100

    # Returns
    daily_rets = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = (np.mean(daily_rets) * 252 - 0.02) / (np.std(daily_rets) * np.sqrt(252))

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_dd = np.max(drawdown)

    # Month-by-month
    monthly_data = defaultdict(lambda: {'equity': [], 'prices': []})
    for i, date in enumerate(dates):
        key = (date.year, date.month)
        monthly_data[key]['equity'].append(equity_curve[i])

        # Find corresponding price
        close_idx = warmup + i
        if close_idx < len(close_prices):
            monthly_data[key]['prices'].append(close_prices[close_idx])

    monthly_performance = {}
    for (year, month), data in monthly_data.items():
        if len(data['equity']) < 2:
            continue

        strat_ret = (data['equity'][-1] / data['equity'][0] - 1) * 100

        if len(data['prices']) >= 2:
            bh_ret = (data['prices'][-1] / data['prices'][0] - 1) * 100
        else:
            bh_ret = 0

        monthly_performance[(year, month)] = {
            'strategy': strat_ret,
            'buy_hold': bh_ret,
            'alpha': strat_ret - bh_ret,
        }

    # Print results
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Period: {dates[0].date()} to {dates[-1].date()} ({n_days} days)")
    print(f"\n{'Metric':<25} {'Strategy':>15} {'Buy & Hold':>15} {'Difference':>15}")
    print("-"*80)
    print(f"{'Initial Capital':<25} ${'100,000.00':>14} ${'100,000.00':>14} ${'0.00':>14}")
    print(f"{'Final Value':<25} ${equity_curve[-1]:>14,.2f} ${bh_value:>14,.2f} ${equity_curve[-1] - bh_value:>14,.2f}")
    print(f"{'Total Return':<25} {strategy_return:>14.2f}% {bh_return:>14.2f}% {strategy_return - bh_return:>14.2f}%")
    print(f"{'Annualized Return':<25} {ann_return:>14.2f}% {(bh_return / n_years):>14.2f}% {ann_return - (bh_return / n_years):>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.2f} {'N/A':>15} {'N/A':>15}")
    print(f"{'Max Drawdown':<25} {max_dd:>14.2f}% {'N/A':>15} {'N/A':>15}")
    print(f"{'Total Trades':<25} {len(trades):>15} {'0':>15} {len(trades):>15}")

    # Trade analysis
    print(f"\n{'='*80}")
    print("TRADE SUMMARY")
    print(f"{'='*80}")

    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']

    print(f"Total Trades: {len(trades)}")
    print(f"  Buys: {len(buys)}")
    print(f"  Sells: {len(sells)}")

    if trades:
        total_commission = sum(t['commission'] for t in trades)
        avg_trade_size = np.mean([t['value'] for t in trades])

        print(f"  Total Commissions: ${total_commission:,.2f}")
        print(f"  Avg Trade Size: ${avg_trade_size:,.2f}")

    # Recent trades (last 10)
    print(f"\n{'='*80}")
    print("RECENT TRADES (Last 10)")
    print(f"{'='*80}")
    print(f"{'Date':<12} {'Type':<6} {'Shares':>10} {'Price':>10} {'Value':>12} {'Weight':>8}")
    print("-"*80)

    for trade in trades[-10:]:
        print(f"{trade['date'].date()} {trade['type']:<6} {trade['shares']:>10.0f} "
              f"${trade['price']:>9.2f} ${trade['value']:>11,.0f} {trade['weight']:>7.1%}")

    # Month-by-month comparison
    print(f"\n{'='*80}")
    print("MONTH-BY-MONTH PERFORMANCE")
    print(f"{'='*80}")

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Group by year
    years = sorted(set(ym[0] for ym in monthly_performance.keys()))

    for year in years:
        print(f"\n{year}:")
        print(f"{'Month':<6} {'Strategy':>12} {'Buy&Hold':>12} {'Alpha':>12}")
        print("-"*45)

        for month in range(1, 13):
            if (year, month) in monthly_performance:
                perf = monthly_performance[(year, month)]
                month_name = month_names[month - 1]

                print(f"{month_name:<6} {perf['strategy']:>11.2f}% {perf['buy_hold']:>11.2f}% "
                      f"{perf['alpha']:>11.2f}%")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if strategy_return > bh_return:
        print(f"✅ Strategy OUTPERFORMED buy-and-hold by {strategy_return - bh_return:.2f}%")
    else:
        print(f"❌ Strategy UNDERPERFORMED buy-and-hold by {bh_return - strategy_return:.2f}%")

    # Calculate monthly win rate
    winning_months = sum(1 for p in monthly_performance.values() if p['alpha'] > 0)
    total_months = len(monthly_performance)
    win_rate = winning_months / total_months * 100 if total_months > 0 else 0

    print(f"\nKey Metrics:")
    print(f"  - Strategy returned {strategy_return:.2f}% vs Buy&Hold {bh_return:.2f}%")
    print(f"  - Executed {len(trades)} trades with ${sum(t['commission'] for t in trades):,.2f} in commissions")
    print(f"  - Beat buy&hold in {winning_months}/{total_months} months ({win_rate:.1f}% win rate)")
    print(f"  - Sharpe Ratio: {sharpe:.2f}")
    print(f"  - Max Drawdown: {max_dd:.2f}%")

    # Current position
    print(f"\nCurrent Position:")
    print(f"  - Shares: {result['final_shares']:.0f}")
    print(f"  - Cash: ${result['final_cash']:,.2f}")
    print(f"  - Position Value: ${result['final_shares'] * close_prices[-1]:,.2f}")
    print(f"  - Total Equity: ${equity_curve[-1]:,.2f}")

    print("="*80)


if __name__ == "__main__":
    main()
