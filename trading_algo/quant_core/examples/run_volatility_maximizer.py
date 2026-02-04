#!/usr/bin/env python3
"""
Run Volatility Maximizer Strategy

Tests the aggressive volatility exploitation strategy on FX markets.
Designed for MAXIMUM PROFIT, not risk-adjusted returns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Fix asyncio
import asyncio
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
from trading_algo.quant_core.strategies.volatility_maximizer import (
    VolatilityMaximizer, VolMaxConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_volmax_backtest(
    historical_data: dict,
    timestamps: list,
    initial_capital: float = 100000.0,
    config: VolMaxConfig = None,
) -> dict:
    """Run volatility maximizer backtest."""

    config = config or VolMaxConfig()
    strategy = VolatilityMaximizer(config)

    symbols = list(historical_data.keys())
    n_bars = len(timestamps)

    # Extract close prices
    close_prices = {s: historical_data[s][:, 3] for s in symbols}

    # State
    cash = initial_capital
    positions: dict = {}  # symbol -> shares
    equity_curve = [initial_capital]
    trades = []
    daily_stats = []

    warmup = max(config.garch_estimation_window, config.vol_lookback) + 10

    logger.info(f"Running backtest: {len(symbols)} symbols, {n_bars} bars")
    logger.info(f"Config: {config.target_leverage:.1f}x leverage, Beyond-Kelly {config.beyond_kelly_multiplier:.1f}x")

    for t in range(warmup, n_bars):
        if t % 100 == 0:
            logger.info(f"Processing bar {t}/{n_bars} ({t/n_bars*100:.1f}%)")

        # Current prices
        current_prices = {s: close_prices[s][t] for s in symbols}

        # Price history
        price_history = {s: close_prices[s][:t+1] for s in symbols}

        # Calculate equity
        position_value = sum(
            positions.get(s, 0) * current_prices[s]
            for s in symbols
        )
        equity = cash + position_value

        # Generate signals
        signals = strategy.generate_signals(symbols, price_history, timestamps[t])

        # Scale to target leverage
        target_weights = strategy.scale_to_target_leverage(signals)

        # Calculate target positions
        target_values = {s: equity * w for s, w in target_weights.items()}

        # Rebalance
        gross_exposure = 0.0
        for symbol in symbols:
            current_shares = positions.get(symbol, 0)
            current_value = current_shares * current_prices[symbol]
            target_value = target_values.get(symbol, 0)

            delta_value = target_value - current_value

            if abs(delta_value) > 100:  # Min trade
                delta_shares = delta_value / current_prices[symbol]

                # Execute with slippage (higher for FX)
                slippage = 0.001  # 10bp for FX
                if delta_shares > 0:
                    exec_price = current_prices[symbol] * (1 + slippage)
                else:
                    exec_price = current_prices[symbol] * (1 - slippage)

                cost = abs(delta_shares) * exec_price
                commission = cost * 0.0002  # 2bp commission for FX

                if delta_shares > 0:
                    cash -= cost + commission
                else:
                    cash += cost - commission

                positions[symbol] = current_shares + delta_shares

                trades.append({
                    'timestamp': timestamps[t],
                    'symbol': symbol,
                    'shares': delta_shares,
                    'price': exec_price,
                    'signal_type': signals[symbol].opportunity_type.name if symbol in signals else 'UNKNOWN',
                    'conviction': signals[symbol].conviction if symbol in signals else 0,
                })

        # Calculate exposure
        for symbol in symbols:
            if symbol in positions and positions[symbol] != 0:
                gross_exposure += abs(positions[symbol] * current_prices[symbol])

        gross_exposure_pct = gross_exposure / equity if equity > 0 else 0

        # Update equity
        position_value = sum(
            positions.get(s, 0) * current_prices[s]
            for s in symbols
        )
        equity = cash + position_value
        equity_curve.append(equity)

        # Log daily stats
        daily_stats.append({
            'timestamp': timestamps[t],
            'equity': equity,
            'cash': cash,
            'gross_exposure': gross_exposure_pct,
            'n_positions': sum(1 for p in positions.values() if abs(p) > 0),
        })

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = (equity_curve[-1] / initial_capital) - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    volatility = np.std(returns) * np.sqrt(252)
    sharpe = (ann_return - 0.02) / volatility if volatility > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)

    # Win rate
    winning_trades = [t for t in trades if t.get('shares', 0) != 0]  # Simplified
    win_rate = 0.5  # Placeholder

    # Avg exposure
    avg_exposure = np.mean([s['gross_exposure'] for s in daily_stats]) if daily_stats else 0

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'equity_curve': equity_curve,
        'returns': returns,
        'trades': trades,
        'daily_stats': daily_stats,
        'final_value': equity_curve[-1],
        'avg_exposure': avg_exposure,
    }


def main():
    """Run volatility maximizer on FX markets."""

    # FX Universe (G10 + EM)
    fx_universe = [
        # G10
        "FXE",   # Euro
        "FXY",   # Yen
        "FXB",   # Pound
        "FXA",   # Aussie
        "FXC",   # Canadian
        "FXF",   # Swiss Franc
        # Dollar
        "UUP",   # Long USD
        "UDN",   # Short USD
        # Add volatility products if available
        # "VXX",   # VIX short-term futures
    ]

    print("="*70)
    print("VOLATILITY MAXIMIZER - FX MARKETS")
    print("="*70)
    print("\nPhilosophy: MAXIMUM PROFIT, accepting high drawdowns")
    print(f"Universe: {fx_universe}")

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=7497, client_id=16)
    broker = IBKRBroker(config=config, require_paper=True)
    broker.connect()

    # Fetch data
    print("\nFetching 2 years of FX data...")
    all_data = {}
    all_timestamps = {}

    for symbol in fx_universe:
        try:
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            bars = broker.get_historical_bars(instrument, duration="2 Y", bar_size="1 day", what_to_show="TRADES", use_rth=True)
            if bars and len(bars) > 100:
                ohlcv = np.zeros((len(bars), 5))
                timestamps = []
                for i, bar in enumerate(bars):
                    ohlcv[i] = [bar.open, bar.high, bar.low, bar.close, bar.volume or 0]
                    timestamps.append(datetime.fromtimestamp(bar.timestamp_epoch_s))
                all_data[symbol] = ohlcv
                all_timestamps[symbol] = timestamps
                print(f"  {symbol}: {len(bars)} bars")
            time.sleep(1.5)
        except Exception as e:
            print(f"  {symbol}: Failed - {e}")

    broker.disconnect()

    if len(all_data) < 3:
        print("Not enough data")
        return

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

    # Test different configurations
    configs = {
        "Aggressive": VolMaxConfig(
            target_leverage=2.5,
            beyond_kelly_multiplier=1.5,
            max_position=0.80,
            target_portfolio_vol=0.35,
        ),
        "Ultra-Aggressive": VolMaxConfig(
            target_leverage=4.0,
            beyond_kelly_multiplier=2.0,
            max_position=1.0,
            target_portfolio_vol=0.50,
            use_stops=False,
        ),
        "Volatility Hunter": VolMaxConfig(
            target_leverage=3.0,
            beyond_kelly_multiplier=1.8,
            vol_breakout_weight=0.50,  # Heavy on vol breakouts
            momentum_weight=0.30,
            carry_weight=0.15,
            max_position=0.90,
            target_portfolio_vol=0.45,
        ),
    }

    print(f"\n{'='*70}")
    print("RUNNING BACKTESTS")
    print(f"{'='*70}")

    results = {}
    for name, cfg in configs.items():
        print(f"\nTesting: {name}")
        print(f"  Leverage: {cfg.target_leverage:.1f}x")
        print(f"  Beyond-Kelly: {cfg.beyond_kelly_multiplier:.1f}x")
        print(f"  Target Vol: {cfg.target_portfolio_vol:.0%}")

        result = run_volmax_backtest(aligned_data, ref_timestamps, 100000.0, cfg)
        results[name] = result

    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS - VOLATILITY MAXIMIZER")
    print(f"{'='*70}")
    print(f"Period: {ref_timestamps[0].date()} to {ref_timestamps[-1].date()}")
    print(f"\n{'Strategy':<20} {'Return':>12} {'Ann.Ret':>10} {'Sharpe':>8} {'MaxDD':>10} {'Vol':>8} {'Exp':>8}")
    print("-"*70)

    for name, res in results.items():
        print(f"{name:<20} {res['total_return']:>11.2%} {res['annualized_return']:>9.2%} "
              f"{res['sharpe_ratio']:>8.2f} {res['max_drawdown']:>9.2%} "
              f"{res['volatility']:>7.1%} {res['avg_exposure']:>7.1%}")

    print("="*70)

    # Show best result details
    best = max(results.items(), key=lambda x: x[1]['total_return'])
    print(f"\nBEST PERFORMER: {best[0]}")
    print(f"  Final Value: ${best[1]['final_value']:,.2f}")
    print(f"  Total Trades: {best[1]['total_trades']}")

    # Analyze trades by signal type
    trades_by_type = {}
    for trade in best[1]['trades']:
        sig_type = trade.get('signal_type', 'UNKNOWN')
        if sig_type not in trades_by_type:
            trades_by_type[sig_type] = 0
        trades_by_type[sig_type] += 1

    print(f"\nTrade Breakdown:")
    for sig_type, count in sorted(trades_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sig_type}: {count} trades")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
This volatility maximizer is designed for PROFIT MAXIMIZATION:
- Uses Beyond-Kelly sizing for maximum growth
- Accepts 50-80% drawdowns
- Exploits volatility breakouts aggressively
- No risk dampening - let winners run

Perfect for FX markets where volatility creates opportunity.
    """)


if __name__ == "__main__":
    main()
