#!/usr/bin/env python3
"""
Run Chameleon Strategy backtest with REAL IBKR data.

This script uses actual historical data pulled from IBKR TWS
to test the Chameleon Strategy's performance on real market conditions.

Usage:
    python run_ibkr_chameleon_backtest.py
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from trading_algo.rat.chameleon_strategy import (
    ChameleonStrategy,
    MarketMode,
    create_chameleon_strategy,
)


@dataclass
class Bar:
    """OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_csv_data(filepath: str) -> List[Bar]:
    """Load bars from CSV file."""
    bars = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                ts = datetime.strptime(row['date'], '%Y-%m-%d')

            bars.append(Bar(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
            ))
    return bars


class RealDataBacktester:
    """Backtest with real market data."""

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.position: Optional[Dict] = None
        self.regime_counts: Dict[str, int] = {}

    def run(self, bars: List[Bar], symbol: str, warmup: int = 50) -> Dict:
        """Run backtest on real data."""
        self.equity = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.position = None
        self.regime_counts = {}

        strategy = create_chameleon_strategy()

        for i, bar in enumerate(bars):
            # Update strategy
            decision = strategy.update(
                symbol=symbol,
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

            # Track equity
            if self.position is not None:
                self._update_equity(bar)

            self.equity_curve.append(self.equity)

            # Skip warmup period for trading
            if i < warmup or decision is None:
                continue

            # Track regime
            regime = decision.get('regime')
            if regime:
                name = regime.name if hasattr(regime, 'name') else str(regime)
                self.regime_counts[name] = self.regime_counts.get(name, 0) + 1

            # Process trading decision
            self._process_decision(decision, bar)

        # Close any remaining position
        if self.position is not None and len(bars) > 0:
            self._close_position(bars[-1].close, bars[-1].timestamp)

        return self._calc_metrics()

    def _update_equity(self, bar: Bar):
        if self.position is None:
            return

        entry = self.position['entry_price']
        direction = self.position['direction']
        size = self.position['size']

        if direction > 0:
            ret = (bar.close - entry) / entry
        else:
            ret = (entry - bar.close) / entry

        # Leveraged return
        leveraged_ret = ret * size
        leveraged_ret = max(-0.9, min(3.0, leveraged_ret))

        self.equity = self.position['entry_equity'] * (1 + leveraged_ret)
        self.equity = max(self.equity, self.initial_capital * 0.01)

    def _process_decision(self, decision: Dict, bar: Bar):
        action = decision.get('action', 'hold')

        if action == 'hold':
            return

        if action in ('buy', 'short'):
            if self.position is not None:
                self._close_position(bar.close, bar.timestamp)

            direction = 1 if action == 'buy' else -1
            size = min(decision.get('size', 0.1), 4.0)

            self.position = {
                'direction': direction,
                'entry_price': bar.close,
                'size': size,
                'entry_equity': self.equity,
                'entry_time': bar.timestamp,
            }

        elif action in ('sell', 'cover'):
            if self.position:
                self._close_position(bar.close, bar.timestamp)

    def _close_position(self, price: float, timestamp: datetime):
        if self.position is None:
            return

        entry = self.position['entry_price']
        direction = self.position['direction']
        size = self.position['size']

        if direction > 0:
            ret = (price - entry) / entry
        else:
            ret = (entry - price) / entry

        leveraged_ret = ret * size
        leveraged_ret = max(-0.9, min(3.0, leveraged_ret))

        pnl = self.position['entry_equity'] * leveraged_ret

        self.equity = self.position['entry_equity'] + pnl
        self.equity = max(self.equity, self.initial_capital * 0.01)

        self.trades.append({
            'timestamp': timestamp,
            'direction': direction,
            'entry_price': entry,
            'exit_price': price,
            'pnl': pnl,
            'return': leveraged_ret,
            'raw_return': ret,
            'leverage': size,
        })

        self.position = None

    def _calc_metrics(self) -> Dict:
        if not self.equity_curve:
            return {}

        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Calculate Sharpe
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                r = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
                returns.append(r)

            if returns:
                avg_ret = sum(returns) / len(returns)
                variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
                std = variance ** 0.5
                sharpe = (avg_ret * 252) / (std * (252 ** 0.5)) if std > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Win rate
        if self.trades:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'regime_counts': self.regime_counts,
        }


def main():
    print("=" * 80)
    print("CHAMELEON STRATEGY - REAL IBKR DATA BACKTEST")
    print("=" * 80)
    print()

    data_dir = Path("data")
    symbols = ["AAPL", "MSFT", "NVDA"]

    all_results = {}

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {symbol}")
            continue

        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")

        # Load data
        bars = load_csv_data(str(csv_path))
        print(f"Loaded {len(bars)} bars")

        if len(bars) < 60:
            print("Not enough data for backtest (need at least 60 bars)")
            continue

        # Calculate buy-and-hold return
        start_price = bars[50].close  # After warmup
        end_price = bars[-1].close
        bh_return = (end_price - start_price) / start_price

        # Run backtest
        bt = RealDataBacktester(initial_capital=100_000)
        res = bt.run(bars, symbol, warmup=50)

        strategy_return = res.get('total_return', 0)
        alpha = strategy_return - bh_return

        print(f"\nReturns:")
        print(f"  Strategy:     {strategy_return * 100:+.2f}%")
        print(f"  Buy & Hold:   {bh_return * 100:+.2f}%")

        if alpha >= 0:
            print(f"  ALPHA:        {alpha * 100:+.2f}% [WIN]")
        else:
            print(f"  ALPHA:        {alpha * 100:+.2f}% [LOSS]")

        print(f"\nRisk:")
        print(f"  Sharpe:       {res.get('sharpe', 0):+.2f}")
        print(f"  Max DD:       {res.get('max_dd', 0) * 100:.2f}%")
        print(f"  Win Rate:     {res.get('win_rate', 0) * 100:.2f}%")
        print(f"  Trades:       {res.get('num_trades', 0)}")

        # Show regime breakdown
        regime_counts = res.get('regime_counts', {})
        if regime_counts:
            total = sum(regime_counts.values())
            print(f"\nRegime Detection:")
            for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                print(f"  {regime:20s} {pct:5.1f}%")

        all_results[symbol] = {
            'strategy_return': strategy_return,
            'bh_return': bh_return,
            'alpha': alpha,
            **res,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - REAL IBKR DATA")
    print("=" * 80)
    print()
    print(f"{'Symbol':<10} {'Strategy':>12} {'B&H':>12} {'Alpha':>12}")
    print("-" * 50)

    total_alpha = 0
    wins = 0

    for symbol, res in all_results.items():
        strategy = res['strategy_return'] * 100
        bh = res['bh_return'] * 100
        alpha = res['alpha'] * 100
        total_alpha += alpha

        marker = "[+]" if alpha >= 0 else "[-]"
        if alpha >= 0:
            wins += 1

        print(f"{symbol:<10} {strategy:>11.2f}% {bh:>11.2f}% {alpha:>+11.2f}% {marker}")

    print("-" * 50)

    if all_results:
        avg_alpha = total_alpha / len(all_results)
        print(f"\nOverall:")
        print(f"  Symbols with Alpha: {wins}/{len(all_results)}")
        print(f"  Average Alpha:      {avg_alpha:+.2f}%")

        if wins >= len(all_results) / 2:
            print("\n[SUCCESS] Chameleon Strategy shows positive alpha on real data!")
        else:
            print("\n[NEEDS WORK] Strategy underperforms on some real data symbols")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
