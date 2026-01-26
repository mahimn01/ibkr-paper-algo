#!/usr/bin/env python3
"""
Run Day Trading Backtest with Real IBKR Data.

Pulls recent 5-minute intraday data and backtests the ChameleonDayTrader.

Usage:
    python run_daytrader_backtest.py NVDA TSLA AAPL
    python run_daytrader_backtest.py NVDA --days 2
"""

import argparse
import sys
from datetime import datetime

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.rat.daytrader_backtest import (
    IntradayBacktester,
    IntradayBar,
    print_backtest_results,
)


def pull_intraday_data(
    broker: IBKRBroker,
    symbol: str,
    days: int = 1,
) -> list[IntradayBar]:
    """Pull 5-minute intraday data from IBKR."""
    instrument = InstrumentSpec(
        kind="STK",
        symbol=symbol,
        exchange="SMART",
        currency="USD",
    )

    print(f"  Pulling {days} day(s) of 5-min data for {symbol}...")

    bars = broker.get_historical_bars(
        instrument,
        duration=f"{days} D",
        bar_size="5 mins",
        what_to_show="TRADES",
        use_rth=True,  # Regular trading hours only
    )

    # Convert to IntradayBar format
    intraday_bars = []
    for bar in bars:
        ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
        intraday_bars.append(IntradayBar(
            timestamp=ts,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume or 0,
        ))

    print(f"  Retrieved {len(intraday_bars)} bars")
    return intraday_bars


def calculate_volatility_metrics(bars: list[IntradayBar]) -> dict:
    """Calculate volatility metrics for the data."""
    if len(bars) < 2:
        return {}

    # Returns
    returns = []
    for i in range(1, len(bars)):
        r = (bars[i].close - bars[i-1].close) / bars[i-1].close
        returns.append(r)

    # Volatility (annualized from 5-min)
    avg_ret = sum(returns) / len(returns)
    variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
    std = variance ** 0.5
    annualized_vol = std * (78 * 252) ** 0.5  # 78 bars/day, 252 days/year

    # Average True Range
    atrs = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        atrs.append(tr)
    avg_atr = sum(atrs) / len(atrs) if atrs else 0
    atr_pct = avg_atr / bars[-1].close * 100

    # Intraday range
    ranges = [(b.high - b.low) / b.close * 100 for b in bars]
    avg_range = sum(ranges) / len(ranges)

    # Volume
    avg_volume = sum(b.volume for b in bars) / len(bars)

    return {
        'annualized_volatility': annualized_vol,
        'avg_atr_pct': atr_pct,
        'avg_bar_range_pct': avg_range,
        'avg_volume': avg_volume,
        'total_return': (bars[-1].close - bars[0].close) / bars[0].close,
    }


def main():
    parser = argparse.ArgumentParser(description="Day Trading Backtest")
    parser.add_argument("symbols", nargs="+", help="Symbols to backtest")
    parser.add_argument("--days", type=int, default=2, help="Days of data (default: 2)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--max-position", type=float, default=10000, help="Max position $")
    parser.add_argument("--conservative", action="store_true", help="Conservative mode")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port")

    args = parser.parse_args()

    print("=" * 70)
    print("DAY TRADING BACKTEST - CHAMELEON DAY TRADER v2")
    print("=" * 70)
    print(f"Symbols:      {', '.join(args.symbols)}")
    print(f"Days:         {args.days}")
    print(f"Capital:      ${args.capital:,.0f}")
    print(f"Max Position: ${args.max_position:,.0f}")
    print(f"Mode:         {'Conservative' if args.conservative else 'Aggressive'}")
    print(f"Features:     ATR stops, trailing stops, VWAP entries, RSI filters")
    print("=" * 70)

    # Connect to IBKR
    config = IBKRConfig(host="127.0.0.1", port=args.port, client_id=30)
    broker = IBKRBroker(config=config, require_paper=True)

    print("\nConnecting to IBKR...")
    try:
        broker.connect()
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    all_results = {}

    try:
        for symbol in args.symbols:
            symbol = symbol.upper()
            print(f"\n{'='*60}")
            print(f"BACKTESTING: {symbol}")
            print(f"{'='*60}")

            # Pull data
            try:
                bars = pull_intraday_data(broker, symbol, days=args.days)
            except Exception as e:
                print(f"  Error pulling data: {e}")
                continue

            if len(bars) < 50:
                print(f"  Not enough bars ({len(bars)}), need at least 50")
                continue

            # Show volatility metrics
            vol_metrics = calculate_volatility_metrics(bars)
            print(f"\n  Volatility Metrics:")
            print(f"    Annualized Vol:  {vol_metrics['annualized_volatility']*100:.1f}%")
            print(f"    Avg ATR:         {vol_metrics['avg_atr_pct']:.3f}%")
            print(f"    Avg Bar Range:   {vol_metrics['avg_bar_range_pct']:.3f}%")
            print(f"    Period Return:   {vol_metrics['total_return']*100:+.2f}%")

            # Run backtest
            backtester = IntradayBacktester(
                initial_capital=args.capital,
                max_position_dollars=args.max_position,
            )

            result = backtester.run(
                bars=bars,
                symbol=symbol,
                aggressive=not args.conservative,
                warmup_bars=35,
            )

            all_results[symbol] = result

            # Print results
            print_backtest_results(result)

    finally:
        broker.disconnect()
        print("\nDisconnected from IBKR")

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY - ALL SYMBOLS")
        print("=" * 70)
        print(f"{'Symbol':<8} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'PF':>7} {'Sharpe':>8} {'MaxDD':>8}")
        print("-" * 68)

        total_pnl = 0
        total_trades = 0
        total_tp = 0
        total_trail = 0
        total_sl = 0
        total_rev = 0
        total_eod = 0

        for symbol, res in all_results.items():
            pf_str = f"{res.profit_factor:.2f}" if res.profit_factor < 100 else "inf"
            print(f"{symbol:<8} {res.num_trades:>7} {res.win_rate*100:>6.1f}% "
                  f"${res.total_pnl:>+10,.2f} {pf_str:>7} {res.sharpe_ratio:>8.2f} "
                  f"{res.max_drawdown_pct*100:>7.2f}%")
            total_pnl += res.total_pnl
            total_trades += res.num_trades
            total_tp += res.num_take_profit_exits
            total_trail += res.num_trailing_stop_exits
            total_sl += res.num_stop_loss_exits
            total_rev += res.num_reversal_exits
            total_eod += res.num_eod_exits

        print("-" * 68)
        print(f"{'TOTAL':<8} {total_trades:>7} {'':>7} ${total_pnl:>+10,.2f}")
        print()
        print(f"EXIT TYPES: TP={total_tp}  Trail={total_trail}  SL={total_sl}  Rev={total_rev}  EOD={total_eod}")
        print("=" * 70)


if __name__ == "__main__":
    main()
