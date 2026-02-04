#!/usr/bin/env python3
"""
Forex Momentum vs SPY Comparison

Tests Pure Momentum Aggressive strategy on major FX pairs/ETFs.
Compares to S&P 500 buy-and-hold over the same period.
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


def calculate_buy_hold_performance(prices, dates):
    """Calculate buy-and-hold performance with period breakdowns."""

    # Overall performance
    total_return = (prices[-1] / prices[0] - 1) * 100
    n_years = len(dates) / 252
    ann_return = ((prices[-1] / prices[0]) ** (1 / n_years) - 1) * 100

    # Daily returns for stats
    daily_rets = np.diff(prices) / prices[:-1]
    sharpe = (np.mean(daily_rets) * 252 - 0.02) / (np.std(daily_rets) * np.sqrt(252))

    # Max drawdown
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak * 100
    max_dd = np.max(drawdown)

    # Year-by-year
    yearly_stats = defaultdict(lambda: {'prices': [], 'dates': []})
    for i, date in enumerate(dates):
        yearly_stats[date.year]['prices'].append(prices[i])
        yearly_stats[date.year]['dates'].append(date)

    yearly_returns = {}
    for year, data in yearly_stats.items():
        if len(data['prices']) < 2:
            continue
        start_price = data['prices'][0]
        end_price = data['prices'][-1]
        ret = (end_price / start_price - 1) * 100

        peak_price = max(data['prices'])
        min_price = min(data['prices'])
        dd = ((peak_price - min_price) / peak_price) * 100

        yearly_returns[year] = {
            'return': ret,
            'start': start_price,
            'end': end_price,
            'max_dd': dd,
            'days': len(data['prices'])
        }

    # Month-by-month
    monthly_stats = defaultdict(lambda: {'prices': [], 'dates': []})
    for i, date in enumerate(dates):
        key = (date.year, date.month)
        monthly_stats[key]['prices'].append(prices[i])
        monthly_stats[key]['dates'].append(date)

    monthly_returns = {}
    for (year, month), data in monthly_stats.items():
        if len(data['prices']) < 2:
            continue
        start_price = data['prices'][0]
        end_price = data['prices'][-1]
        ret = (end_price / start_price - 1) * 100

        monthly_returns[(year, month)] = {
            'return': ret,
            'days': len(data['prices'])
        }

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'yearly': yearly_returns,
        'monthly': monthly_returns,
    }


def run_strategy_backtest(data, timestamps, config):
    """Run momentum strategy backtest."""
    strategy = PureMomentumStrategy(config)
    symbols = list(data.keys())
    close_prices = {s: data[s][:, 3] for s in symbols}

    cash = 100000.0
    positions = {}
    equity_curve = []
    dates = []

    warmup = max(config.trend_ma, config.momentum_lookback) + 10

    for t in range(warmup, len(timestamps)):
        current_prices = {s: close_prices[s][t] for s in symbols}
        price_history = {s: close_prices[s][:t+1] for s in symbols}

        # Calculate equity
        position_value = sum(positions.get(s, 0) * current_prices[s] for s in symbols)
        equity = cash + position_value
        equity_curve.append(equity)
        dates.append(timestamps[t])

        # Get targets
        target_weights = strategy.get_target_weights(symbols, price_history)
        target_values = {s: equity * w for s, w in target_weights.items()}

        # Rebalance
        for symbol in symbols:
            current_shares = positions.get(symbol, 0)
            current_value = current_shares * current_prices[symbol]
            target_value = target_values.get(symbol, 0)
            delta_value = target_value - current_value

            if abs(delta_value) > 100:
                delta_shares = delta_value / current_prices[symbol]
                slippage = 0.001
                exec_price = current_prices[symbol] * (1 + slippage if delta_shares > 0 else 1 - slippage)
                cost = abs(delta_shares) * exec_price
                commission = cost * 0.0005

                if delta_shares > 0:
                    cash -= cost + commission
                else:
                    cash += cost - commission

                positions[symbol] = current_shares + delta_shares

    return np.array(equity_curve), dates


def main():
    """Run comparison backtest."""

    print("="*80)
    print(" "*20 + "FOREX MOMENTUM vs S&P 500 COMPARISON")
    print("="*80)

    # Major FX ETFs
    forex_universe = [
        "UUP",   # US Dollar Index
        "FXE",   # Euro
        "FXY",   # Japanese Yen
        "FXB",   # British Pound
        "FXC",   # Canadian Dollar
        "FXA",   # Australian Dollar
        "FXF",   # Swiss Franc
    ]

    print(f"\nFetching 3 years of data...")
    print(f"  - SPY (S&P 500)")
    print(f"  - {len(forex_universe)} major FX ETFs")

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=22)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Fetch SPY
    spy_instrument = InstrumentSpec(kind="STK", symbol="SPY", exchange="SMART", currency="USD")
    spy_bars = broker.get_historical_bars(spy_instrument, duration="3 Y", bar_size="1 day", what_to_show="TRADES", use_rth=True)

    spy_prices = np.array([bar.close for bar in spy_bars])
    spy_dates = [datetime.fromtimestamp(bar.timestamp_epoch_s) for bar in spy_bars]

    print(f"\n✓ SPY: {len(spy_bars)} bars ({spy_dates[0].date()} to {spy_dates[-1].date()})")

    time.sleep(2)

    # Fetch FX ETFs
    all_data = {}
    all_timestamps = {}

    for symbol in forex_universe:
        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            bars = broker.get_historical_bars(instrument, duration="3 Y", bar_size="1 day", what_to_show="TRADES", use_rth=True)

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
            print(f"  ✗ {symbol}: {e}")

    broker.disconnect()

    print(f"✓ Forex ETFs: {len(all_data)} loaded")

    if len(all_data) < 3:
        print("\n✗ Not enough FX data loaded")
        return

    # Align forex data
    ref_symbol = max(all_data.keys(), key=lambda s: len(all_data[s]))
    ref_timestamps = all_timestamps[ref_symbol]
    ts_to_idx = {ts.date(): i for i, ts in enumerate(ref_timestamps)}

    aligned_data = {}
    for symbol, ohlcv in all_data.items():
        if symbol == ref_symbol:
            aligned_data[symbol] = ohlcv
            continue

        aligned = np.full((len(ref_timestamps), 5), np.nan)
        for i, ts in enumerate(all_timestamps[symbol]):
            if ts.date() in ts_to_idx:
                aligned[ts_to_idx[ts.date()]] = ohlcv[i]

        for col in range(5):
            mask = np.isnan(aligned[:, col])
            if mask.any() and not mask.all():
                idx = np.where(~mask, np.arange(len(aligned)), 0)
                np.maximum.accumulate(idx, out=idx)
                aligned[:, col] = aligned[idx, col]

        if not np.isnan(aligned).all():
            aligned_data[symbol] = aligned

    # Run strategy backtest
    print(f"\nRunning forex momentum backtest...")
    aggressive_config = MomentumConfig(
        fast_ma=5, slow_ma=20, trend_ma=50,
        momentum_lookback=20,
        max_position=0.40,
        target_exposure=2.0,
        vol_scale=True,
        target_vol=0.30,
    )

    strategy_equity, strategy_dates = run_strategy_backtest(aligned_data, ref_timestamps, aggressive_config)

    # Align SPY to strategy dates
    spy_date_to_idx = {d.date(): i for i, d in enumerate(spy_dates)}
    aligned_spy = []
    aligned_dates = []

    for date in strategy_dates:
        if date.date() in spy_date_to_idx:
            aligned_spy.append(spy_prices[spy_date_to_idx[date.date()]])
            aligned_dates.append(date)

    aligned_spy = np.array(aligned_spy)

    # Scale SPY to $100k starting capital
    spy_scaled = aligned_spy / aligned_spy[0] * 100000

    # Calculate statistics
    print(f"\nCalculating statistics...")

    spy_stats = calculate_buy_hold_performance(spy_scaled, aligned_dates)

    # Strategy stats
    strategy_returns = np.diff(strategy_equity) / strategy_equity[:-1]
    strategy_total = (strategy_equity[-1] / 100000 - 1) * 100
    strategy_ann = ((strategy_equity[-1] / 100000) ** (1 / (len(aligned_dates) / 252)) - 1) * 100
    strategy_sharpe = (np.mean(strategy_returns) * 252 - 0.02) / (np.std(strategy_returns) * np.sqrt(252))

    peak = np.maximum.accumulate(strategy_equity)
    drawdown = (peak - strategy_equity) / peak * 100
    strategy_max_dd = np.max(drawdown)

    # Year-by-year comparison
    yearly_strategy = defaultdict(lambda: {'equity': [], 'dates': []})
    for i, date in enumerate(aligned_dates):
        yearly_strategy[date.year]['equity'].append(strategy_equity[i])
        yearly_strategy[date.year]['dates'].append(date)

    strategy_yearly = {}
    for year, data in yearly_strategy.items():
        if len(data['equity']) < 2:
            continue
        start = data['equity'][0]
        end = data['equity'][-1]
        ret = (end / start - 1) * 100
        strategy_yearly[year] = {'return': ret}

    # Print results
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE COMPARISON (3 Years)")
    print(f"{'='*80}")
    print(f"Period: {aligned_dates[0].date()} to {aligned_dates[-1].date()}")
    print(f"\n{'Metric':<25} {'Forex Momentum':>20} {'SPY (S&P 500)':>20} {'Difference':>15}")
    print("-"*80)
    print(f"{'Total Return':<25} {strategy_total:>19.2f}% {spy_stats['total_return']:>19.2f}% {strategy_total - spy_stats['total_return']:>14.2f}%")
    print(f"{'Annualized Return':<25} {strategy_ann:>19.2f}% {spy_stats['ann_return']:>19.2f}% {strategy_ann - spy_stats['ann_return']:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {strategy_sharpe:>20.2f} {spy_stats['sharpe']:>20.2f} {strategy_sharpe - spy_stats['sharpe']:>15.2f}")
    print(f"{'Max Drawdown':<25} {strategy_max_dd:>19.2f}% {spy_stats['max_dd']:>19.2f}% {strategy_max_dd - spy_stats['max_dd']:>14.2f}%")
    print(f"{'Final Value':<25} ${strategy_equity[-1]:>18,.0f} ${spy_scaled[-1]:>18,.0f} ${strategy_equity[-1] - spy_scaled[-1]:>13,.0f}")

    # Year-by-year comparison
    print(f"\n{'='*80}")
    print("YEAR-BY-YEAR COMPARISON")
    print(f"{'='*80}")
    print(f"{'Year':<6} {'Forex Momentum':>20} {'SPY (S&P 500)':>20} {'Outperformance':>20}")
    print("-"*80)

    for year in sorted(spy_stats['yearly'].keys()):
        if year in strategy_yearly:
            strat_ret = strategy_yearly[year]['return']
            spy_ret = spy_stats['yearly'][year]['return']
            diff = strat_ret - spy_ret

            strat_str = f"{strat_ret:>18.2f}%"
            spy_str = f"{spy_ret:>18.2f}%"
            diff_str = f"{diff:>18.2f}% {'✓' if diff > 0 else '✗'}"

            print(f"{year:<6} {strat_str} {spy_str} {diff_str}")

    # Month-by-month for recent year
    print(f"\n{'='*80}")
    print("MONTH-BY-MONTH COMPARISON (2025)")
    print(f"{'='*80}")

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    monthly_strategy = defaultdict(lambda: {'equity': []})
    for i, date in enumerate(aligned_dates):
        key = (date.year, date.month)
        monthly_strategy[key]['equity'].append(strategy_equity[i])

    strategy_monthly = {}
    for (year, month), data in monthly_strategy.items():
        if len(data['equity']) < 2:
            continue
        ret = (data['equity'][-1] / data['equity'][0] - 1) * 100
        strategy_monthly[(year, month)] = {'return': ret}

    print(f"{'Month':<6} {'Forex':>12} {'SPY':>12} {'Diff':>12}")
    print("-"*45)

    for month in range(1, 13):
        if (2025, month) in strategy_monthly and (2025, month) in spy_stats['monthly']:
            strat_ret = strategy_monthly[(2025, month)]['return']
            spy_ret = spy_stats['monthly'][(2025, month)]['return']
            diff = strat_ret - spy_ret

            print(f"{month_names[month-1]:<6} {strat_ret:>11.2f}% {spy_ret:>11.2f}% {diff:>11.2f}%")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    alpha = strategy_ann - spy_stats['ann_return']

    if alpha > 0:
        print(f"✓ Forex Momentum OUTPERFORMED SPY by {alpha:.2f}% annually")
    else:
        print(f"✗ Forex Momentum UNDERPERFORMED SPY by {abs(alpha):.2f}% annually")

    print(f"\nKey Insights:")
    print(f"  - SPY had {spy_stats['total_return']:.2f}% total return over period")
    print(f"  - Forex strategy had {strategy_total:.2f}% total return")
    print(f"  - Forex strategy had {strategy_max_dd:.2f}% max drawdown vs {spy_stats['max_dd']:.2f}% for SPY")

    # Count winning/losing years
    years_beat = sum(1 for y in strategy_yearly if y in spy_stats['yearly'] and
                     strategy_yearly[y]['return'] > spy_stats['yearly'][y]['return'])
    total_years = len([y for y in strategy_yearly if y in spy_stats['yearly']])

    print(f"  - Beat SPY in {years_beat}/{total_years} years")

    print("="*80)


if __name__ == "__main__":
    main()
