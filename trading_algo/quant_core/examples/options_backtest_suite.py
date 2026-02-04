#!/usr/bin/env python3
"""
Comprehensive Options Strategy Backtest Suite

Tests multiple options strategies on historical data:
1. Variance Risk Premium (VRP) Harvesting
2. Theta Collection
3. 0DTE Iron Condors

Compares performance metrics:
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Average trade P&L
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import numpy as np
from datetime import datetime, timedelta
import time
import logging

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.quant_core.strategies.options.vrp_harvester import VRPHarvester, backtest_vrp_strategy
from trading_algo.quant_core.strategies.options.theta_collector import ThetaCollector
from trading_algo.quant_core.strategies.options.iron_condor import IronCondorStrategy
from trading_algo.quant_core.models.greeks import OptionSpec, BlackScholesCalculator

logging.basicConfig(level=logging.WARNING)


def fetch_historical_data(broker, symbol, duration="1 Y"):
    """Fetch historical data for backtesting."""
    try:
        instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
        bars = broker.get_historical_bars(
            instrument,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True
        )

        if not bars or len(bars) < 100:
            return None

        prices = np.array([bar.close for bar in bars])
        dates = [datetime.fromtimestamp(bar.timestamp_epoch_s) for bar in bars]
        returns = np.diff(prices) / prices[:-1]

        # Estimate implied volatility (simplified - real version would fetch from market)
        # Using rolling realized vol as proxy
        window = 20
        implied_vols = np.zeros(len(prices))
        for i in range(window, len(returns)):
            implied_vols[i] = np.std(returns[i-window:i]) * np.sqrt(252) * 1.2  # 20% premium

        # Fill initial values
        implied_vols[:window] = implied_vols[window]

        return {
            'symbol': symbol,
            'prices': prices,
            'dates': dates,
            'returns': returns,
            'implied_vols': implied_vols
        }

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def run_strategy_backtest(strategy_class, data, **params):
    """Run backtest for a specific strategy."""
    strategy = strategy_class(initial_capital=100000.0, **params)

    symbol = data['symbol']
    prices = data['prices']
    returns = data['returns']
    implied_vols = data['implied_vols']
    dates = data['dates']

    # Warm-up period
    start_idx = 60

    for t in range(start_idx, len(dates)):
        current_date = dates[t]
        current_price = prices[t]
        current_iv = implied_vols[t]

        # Historical data up to current point
        hist_data = {
            'returns': returns[:t],
            'prices': prices[:t],
            'implied_vol': current_iv,
            'vix': 15.0  # Simplified
        }

        # Generate signals
        signals = strategy.generate_signals(symbol, current_price, hist_data, current_date)

        # Execute signals
        for signal in signals:
            expiry_days = signal.get('expiry_days', 30)
            time_to_expiry = expiry_days / 365.0

            # Calculate option price
            spec = OptionSpec(
                spot=current_price,
                strike=signal['strike'],
                time_to_expiry=time_to_expiry,
                volatility=current_iv,
                risk_free_rate=strategy.risk_free_rate,
                option_type=signal['option_type']
            )

            option_price = BlackScholesCalculator.price(spec)
            expiry_date = current_date + timedelta(days=expiry_days)

            # Determine quantity (negative for sell)
            quantity = signal['quantity']
            if signal['action'] == 'sell':
                quantity = -abs(quantity)
            else:
                quantity = abs(quantity)

            try:
                strategy.open_position(
                    symbol=symbol,
                    option_type=signal['option_type'],
                    strike=signal['strike'],
                    expiry=expiry_date,
                    quantity=quantity,
                    price=option_price,
                    underlying_price=current_price,
                    current_date=current_date,
                    implied_vol=current_iv
                )
            except Exception as e:
                pass  # Skip if position fails

        # Update positions
        strategy.update_positions(symbol, current_price, current_iv, current_date)

        # Risk management
        to_close = strategy.manage_risk(current_date)
        for position, reason in to_close:
            # Calculate current option price
            time_left = max((position.expiry - current_date).days / 365.0, 0.0)

            if time_left > 0:
                close_spec = OptionSpec(
                    spot=current_price,
                    strike=position.strike,
                    time_to_expiry=time_left,
                    volatility=current_iv,
                    risk_free_rate=strategy.risk_free_rate,
                    option_type=position.option_type
                )
                close_price = BlackScholesCalculator.price(close_spec)
            else:
                # Expired - intrinsic value
                if position.option_type == 'call':
                    close_price = max(current_price - position.strike, 0)
                else:
                    close_price = max(position.strike - current_price, 0)

            strategy.close_position(position, close_price, current_date)

    return strategy.get_portfolio_stats()


def main():
    """Run comprehensive backtest suite."""

    print("="*80)
    print(" "*20 + "OPTIONS STRATEGY BACKTEST SUITE")
    print("="*80)

    # Connect to IBKR
    print("\nConnecting to IBKR TWS...")
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=25)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Test symbols
    symbols = ["SPY", "QQQ", "GLD"]

    # Fetch data
    print(f"\nFetching historical data for {len(symbols)} symbols...")
    all_data = {}

    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        data = fetch_historical_data(broker, symbol, duration="1 Y")
        if data:
            all_data[symbol] = data
            print(f"    ✓ {len(data['dates'])} bars")
        time.sleep(2)

    broker.disconnect()

    if not all_data:
        print("\n✗ No data loaded")
        return

    print(f"\n✓ Loaded {len(all_data)} symbols")

    # Define strategies to test
    strategies = {
        'VRP Harvester': {
            'class': VRPHarvester,
            'params': {
                'min_vrp_threshold': 2.0,
                'target_delta': 0.30,
                'days_to_expiry': 30
            }
        },
        'Theta Collector': {
            'class': ThetaCollector,
            'params': {
                'target_delta': 0.20,
                'days_to_expiry': 45
            }
        },
        'Iron Condor 0DTE': {
            'class': IronCondorStrategy,
            'params': {
                'short_strike_distance': 0.02,
                'long_strike_distance': 0.03
            }
        }
    }

    # Run backtests
    print(f"\n{'='*80}")
    print("RUNNING BACKTESTS")
    print(f"{'='*80}")

    all_results = {}

    for symbol, data in all_data.items():
        print(f"\n{symbol}:")
        symbol_results = {}

        for strat_name, strat_config in strategies.items():
            print(f"  Testing {strat_name}...", end=" ")

            try:
                results = run_strategy_backtest(
                    strat_config['class'],
                    data,
                    **strat_config['params']
                )

                symbol_results[strat_name] = results
                print(f"✓ Return: {results['total_pnl']/100000*100:.2f}%")

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()

        all_results[symbol] = symbol_results

    # Print comprehensive results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*80}")

    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>8}")
        print("-"*80)

        for strat_name, metrics in results.items():
            ret = metrics['total_pnl'] / 100000 * 100
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 0)
            trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0) * 100

            print(f"{strat_name:<20} {ret:>9.2f}% {sharpe:>8.2f} {max_dd:>9.2f}% {trades:>8} {win_rate:>7.1f}%")

    # Strategy comparison summary
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")

    for strat_name in strategies.keys():
        returns = []
        sharpes = []
        for symbol_results in all_results.values():
            if strat_name in symbol_results:
                ret = symbol_results[strat_name]['total_pnl'] / 100000 * 100
                sharpe = symbol_results[strat_name].get('sharpe_ratio', 0)
                returns.append(ret)
                sharpes.append(sharpe)

        if returns:
            print(f"\n{strat_name}:")
            print(f"  Avg Return:    {np.mean(returns):>8.2f}%")
            print(f"  Best Return:   {max(returns):>8.2f}%")
            print(f"  Worst Return:  {min(returns):>8.2f}%")
            print(f"  Avg Sharpe:    {np.mean(sharpes):>8.2f}")
            print(f"  Consistency:   {sum(1 for r in returns if r > 0)}/{len(returns)} positive")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    # Find best strategy
    avg_returns = {}
    for strat_name in strategies.keys():
        returns = [
            all_results[sym][strat_name]['total_pnl'] / 100000 * 100
            for sym in all_results
            if strat_name in all_results[sym]
        ]
        if returns:
            avg_returns[strat_name] = np.mean(returns)

    if avg_returns:
        best = max(avg_returns.items(), key=lambda x: x[1])
        print(f"\n✓ BEST STRATEGY: {best[0]} ({best[1]:.2f}% avg return)")

        print(f"\nFor live trading:")
        print(f"  1. Start with paper trading")
        print(f"  2. Use strict position sizing (5% max per position)")
        print(f"  3. Implement stop losses religiously")
        print(f"  4. Diversify across multiple underlyings")
        print(f"  5. Monitor Greeks daily (delta, gamma, theta, vega)")

    print("="*80)


if __name__ == "__main__":
    main()
