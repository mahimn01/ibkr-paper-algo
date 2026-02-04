#!/usr/bin/env python3
"""
Check Current Market Signals

Shows what positions the Pure Momentum strategy would take RIGHT NOW
in commodities and tech markets.
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
from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig, PureMomentumStrategy

logging.basicConfig(level=logging.WARNING)


def analyze_current_positions(broker, universe, universe_name):
    """Analyze what the strategy would do RIGHT NOW."""

    print(f"\n{'='*80}")
    print(f"{universe_name.upper()} - CURRENT SIGNALS")
    print(f"{'='*80}")

    # Fetch recent data
    all_data = {}
    all_timestamps = {}

    for symbol in universe:
        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            bars = broker.get_historical_bars(
                instrument,
                duration="1 Y",  # 1 year for signal calculation
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True
            )

            if bars and len(bars) > 100:
                ohlcv = np.zeros((len(bars), 5))
                timestamps = []
                for i, bar in enumerate(bars):
                    ohlcv[i] = [bar.open, bar.high, bar.low, bar.close, bar.volume or 0]
                    timestamps.append(datetime.fromtimestamp(bar.timestamp_epoch_s))

                all_data[symbol] = ohlcv
                all_timestamps[symbol] = timestamps

            time.sleep(1.2)
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    if len(all_data) < 3:
        print("  ✗ Not enough data")
        return None

    # Align data
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

    # Get current signals
    config = MomentumConfig(
        fast_ma=5, slow_ma=20, trend_ma=50,
        momentum_lookback=20,
        max_position=0.40,
        target_exposure=2.0,
        vol_scale=True,
        target_vol=0.30,
    )

    strategy = PureMomentumStrategy(config)
    symbols = list(aligned_data.keys())
    close_prices = {s: aligned_data[s][:, 3] for s in symbols}

    # Calculate current weights
    price_history = {s: close_prices[s] for s in symbols}
    target_weights = strategy.get_target_weights(symbols, price_history)

    # Calculate recent returns (last 30 days)
    recent_returns = {}
    for symbol in symbols:
        prices = close_prices[symbol]
        if len(prices) >= 30:
            ret_30d = (prices[-1] / prices[-30] - 1) * 100
            ret_90d = (prices[-1] / prices[-90] - 1) * 100 if len(prices) >= 90 else 0
            recent_returns[symbol] = {'30d': ret_30d, '90d': ret_90d}

    # Print current positions
    print(f"\nAs of: {ref_timestamps[-1].date()}")
    print(f"\n{'Symbol':<8} {'Weight':>8} {'30d Ret':>10} {'90d Ret':>10} {'Signal'}")
    print("-"*50)

    total_long = 0
    total_short = 0

    for symbol in sorted(symbols):
        weight = target_weights.get(symbol, 0)
        if weight > 0:
            total_long += weight
        elif weight < 0:
            total_short += abs(weight)

        ret_30 = recent_returns.get(symbol, {}).get('30d', 0)
        ret_90 = recent_returns.get(symbol, {}).get('90d', 0)

        signal = "LONG" if weight > 0.05 else ("SHORT" if weight < -0.05 else "FLAT")

        print(f"{symbol:<8} {weight:>7.1%} {ret_30:>9.2f}% {ret_90:>9.2f}% {signal}")

    print("-"*50)
    print(f"{'TOTAL':<8} {sum(target_weights.values()):>7.1%}")
    print(f"\nGross Exposure: {total_long + total_short:.1%}")
    print(f"Net Exposure: {total_long - total_short:.1%}")
    print(f"Long: {total_long:.1%} | Short: {total_short:.1%}")

    # Overall market assessment
    avg_30d = np.mean([r['30d'] for r in recent_returns.values()])
    avg_90d = np.mean([r['90d'] for r in recent_returns.values()])

    print(f"\nMarket Assessment:")
    print(f"  Avg 30-day return: {avg_30d:+.2f}%")
    print(f"  Avg 90-day return: {avg_90d:+.2f}%")

    if avg_30d > 0 and avg_90d > 0:
        print(f"  Status: ✅ TRENDING UP - Strategy favorable")
    elif avg_30d < 0 and avg_90d < 0:
        print(f"  Status: ⚠️ TRENDING DOWN - Strategy may struggle")
    else:
        print(f"  Status: ⚠️ CHOPPY - Strategy may whipsaw")

    return {
        'exposure': total_long + total_short,
        'avg_30d': avg_30d,
        'avg_90d': avg_90d
    }


def main():
    """Check current signals."""

    print("="*80)
    print(" "*25 + "CURRENT MARKET SIGNALS")
    print("="*80)
    print("\nConnecting to IBKR TWS...")

    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=23)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Test Commodities
    commodities = ["GLD", "SLV", "PPLT", "USO", "UNG", "DBA"]
    comm_stats = analyze_current_positions(broker, commodities, "Commodities")

    time.sleep(3)

    # Test Tech
    tech = ["QQQ", "AAPL", "MSFT", "NVDA", "META", "GOOGL"]
    tech_stats = analyze_current_positions(broker, tech, "Growth Tech")

    broker.disconnect()

    # Summary
    print(f"\n{'='*80}")
    print("RECOMMENDATION FOR LIVE TRADING")
    print(f"{'='*80}")

    if comm_stats and comm_stats['avg_30d'] > 0:
        print(f"\n✅ COMMODITIES: Strategy is working (avg +{comm_stats['avg_30d']:.2f}% last 30d)")
        print(f"   Recommended: START or CONTINUE trading commodities")
    else:
        print(f"\n⚠️ COMMODITIES: Weak momentum (avg {comm_stats['avg_30d']:.2f}% last 30d)")
        print(f"   Recommended: Use caution or wait for trend")

    if tech_stats and tech_stats['avg_30d'] > 0:
        print(f"\n✅ TECH: Strategy is working (avg +{tech_stats['avg_30d']:.2f}% last 30d)")
        print(f"   Recommended: START or CONTINUE trading tech")
    else:
        print(f"\n⚠️ TECH: Weak momentum (avg {tech_stats['avg_30d']:.2f}% last 30d)")
        print(f"   Recommended: Use caution or wait for trend")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
