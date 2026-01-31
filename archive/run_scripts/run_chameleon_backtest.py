#!/usr/bin/env python3
"""
Chameleon Strategy Backtest - Testing Adaptive Alpha Across All Regimes

Tests the strategy across multiple market scenarios:
- Bull Markets (strong and moderate)
- Bear Markets (strong and moderate)
- Ranging/Choppy Markets
- Crisis/High Volatility
- Recovery Rally
"""

import math
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from trading_algo.rat.chameleon_strategy import (
    ChameleonStrategy,
    MarketMode,
    create_chameleon_strategy,
)


@dataclass
class Bar:
    """OHLCV bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def generate_market_scenario(
    scenario: str,
    symbol: str,
    days: int = 252,
    start_price: float = 100.0,
    seed: Optional[int] = None,
) -> Tuple[List[Bar], float]:
    """Generate bars for a specific market scenario."""
    if seed is not None:
        random.seed(seed)

    bars = []
    price = start_price
    start_date = datetime(2024, 1, 1)

    scenarios = {
        'strong_bull': {'annual_return': 0.40, 'annual_vol': 0.20},
        'moderate_bull': {'annual_return': 0.15, 'annual_vol': 0.18},
        'strong_bear': {'annual_return': -0.35, 'annual_vol': 0.30},
        'moderate_bear': {'annual_return': -0.15, 'annual_vol': 0.25},
        'ranging': {'annual_return': 0.02, 'annual_vol': 0.22},
        'crisis': {'annual_return': -0.20, 'annual_vol': 0.50},
        'recovery': {'annual_return': 0.50, 'annual_vol': 0.35},
    }

    params = scenarios.get(scenario, scenarios['ranging'])
    daily_return = params['annual_return'] / 252
    daily_vol = params['annual_vol'] / math.sqrt(252)

    for day in range(days):
        current_date = start_date + timedelta(days=day)
        if current_date.weekday() >= 5:
            continue

        noise = random.gauss(0, daily_vol)
        ret = daily_return + noise

        open_price = price
        close_price = price * (1 + ret)
        range_mult = abs(ret) * 0.5 + daily_vol * 0.5
        high_price = max(open_price, close_price) * (1 + random.uniform(0, range_mult))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, range_mult))
        volume = 10_000_000 * (1 + abs(ret) * 20) * random.uniform(0.7, 1.3)

        bars.append(Bar(
            symbol=symbol,
            timestamp=current_date,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=int(volume),
        ))
        price = close_price

    bh_return = (bars[-1].close - bars[0].open) / bars[0].open if bars else 0
    return bars, bh_return


class SimpleBacktester:
    """
    Simple percentage-based backtester.

    Tracks positions as percentage of equity to avoid numeric issues.
    """

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.strategy = create_chameleon_strategy()
        self.equity = initial_capital
        self.position = None  # {direction, entry_price, size, entry_equity}
        self.trades = []
        self.equity_curve = []
        self.regime_history = []

    def reset(self):
        self.strategy = create_chameleon_strategy()
        self.equity = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.regime_history = []

    def run(self, bars: List[Bar], warmup: int = 50) -> Dict:
        self.reset()
        bars = sorted(bars, key=lambda b: b.timestamp)

        for i, bar in enumerate(bars):
            self._update_equity(bar)
            self.equity_curve.append((bar.timestamp, self.equity))

            if i < warmup:
                continue

            decision = self.strategy.update(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

            if decision:
                self._process_decision(decision, bar)
                if 'regime' in decision:
                    self.regime_history.append((bar.timestamp, decision['regime']))

        # Close at end
        if self.position:
            self._close_position(bars[-1].close, bars[-1].timestamp)

        return self._calc_metrics()

    def _update_equity(self, bar: Bar):
        if self.position is None:
            return

        entry = self.position['entry_price']
        direction = self.position['direction']
        size = self.position['size']  # Allow full leverage (can be >1.0)

        if direction > 0:
            ret = (bar.close - entry) / entry
        else:
            ret = (entry - bar.close) / entry

        # Apply leverage to return
        leveraged_ret = ret * size

        # Cap leveraged return to prevent blowup but allow meaningful gains
        leveraged_ret = max(-0.9, min(3.0, leveraged_ret))  # Max 300% gain, 90% loss

        self.equity = self.position['entry_equity'] * (1 + leveraged_ret)
        self.equity = max(self.equity, self.initial_capital * 0.01)  # Floor at 1%

    def _process_decision(self, decision: Dict, bar: Bar):
        action = decision.get('action', 'hold')

        if action == 'hold':
            return

        if action in ('buy', 'short'):
            if self.position is not None:
                self._close_position(bar.close, bar.timestamp)

            direction = 1 if action == 'buy' else -1
            size = min(decision.get('size', 0.1), 4.0)  # Allow up to 400% leverage

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
        size = self.position['size']  # Allow full leverage

        if direction > 0:
            ret = (price - entry) / entry
        else:
            ret = (entry - price) / entry

        # Leveraged return
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
            'return': leveraged_ret,  # Store leveraged return
            'raw_return': ret,
            'leverage': size,
        })

        self.position = None

    def _calc_metrics(self) -> Dict:
        if not self.equity_curve:
            return {}

        total_return = (self.equity - self.initial_capital) / self.initial_capital

        # Daily returns
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i-1][1]
            curr = self.equity_curve[i][1]
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        # Sharpe
        if daily_returns:
            avg = sum(daily_returns) / len(daily_returns)
            std = math.sqrt(sum((r - avg)**2 for r in daily_returns) / max(1, len(daily_returns)))
            sharpe = (avg * 252) / (std * math.sqrt(252)) if std > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Win rate
        winners = [t for t in self.trades if t['pnl'] > 0]
        losers = [t for t in self.trades if t['pnl'] <= 0]
        total = len(winners) + len(losers)
        win_rate = len(winners) / total if total > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Regime distribution
        regime_counts = {}
        for _, regime in self.regime_history:
            name = regime.name
            regime_counts[name] = regime_counts.get(name, 0) + 1

        return {
            'total_return': total_return,
            'final_equity': self.equity,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': total,
            'win_rate': win_rate,
            'profit_factor': pf,
            'regime_distribution': regime_counts,
        }


def run_comprehensive_test():
    """Run comprehensive test across all market scenarios."""

    print("=" * 80)
    print("CHAMELEON STRATEGY - ADAPTIVE ALPHA TEST")
    print("=" * 80)
    print("\nGoal: Outperform buy-and-hold in BOTH bull AND bear markets\n")

    scenarios = [
        ('strong_bull', 'Strong Bull (+40% drift)'),
        ('moderate_bull', 'Moderate Bull (+15% drift)'),
        ('strong_bear', 'Strong Bear (-35% drift)'),
        ('moderate_bear', 'Moderate Bear (-15% drift)'),
        ('ranging', 'Ranging/Flat (0% drift)'),
        ('crisis', 'Crisis (-20%, high vol)'),
        ('recovery', 'Recovery Rally (+50% drift)'),
    ]

    results = []

    for scenario_id, name in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"{'='*60}")

        # Use deterministic seed based on scenario name (not hash which varies)
        seed_map = {
            'strong_bull': 100,
            'moderate_bull': 200,
            'strong_bear': 300,
            'moderate_bear': 400,
            'ranging': 500,
            'crisis': 600,
            'recovery': 700,
        }
        bars, bh_return = generate_market_scenario(
            scenario=scenario_id,
            symbol="TEST",
            days=252,
            seed=seed_map.get(scenario_id, 42),
        )

        bt = SimpleBacktester(initial_capital=100_000)
        res = bt.run(bars, warmup=50)

        strat_return = res.get('total_return', 0)
        alpha = strat_return - bh_return

        print(f"\nReturns:")
        print(f"  Strategy:    {strat_return:>10.2%}")
        print(f"  Buy & Hold:  {bh_return:>10.2%}")
        print(f"  ALPHA:       {alpha:>10.2%} {'[WIN]' if alpha > 0 else '[LOSS]'}")

        print(f"\nRisk:")
        print(f"  Sharpe:      {res.get('sharpe_ratio', 0):>10.2f}")
        print(f"  Max DD:      {res.get('max_drawdown', 0):>10.2%}")
        print(f"  Win Rate:    {res.get('win_rate', 0):>10.2%}")
        print(f"  Trades:      {res.get('total_trades', 0):>10}")

        if res.get('regime_distribution'):
            print(f"\nRegime Detection:")
            total = sum(res['regime_distribution'].values())
            for regime, count in sorted(res['regime_distribution'].items(), key=lambda x: -x[1])[:5]:
                pct = count / total * 100 if total > 0 else 0
                print(f"  {regime:<20} {pct:>5.1f}%")

        results.append({
            'scenario': name,
            'strategy': strat_return,
            'bh': bh_return,
            'alpha': alpha,
            'sharpe': res.get('sharpe_ratio', 0),
            'max_dd': res.get('max_drawdown', 0),
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Scenario':<30} {'Strategy':>10} {'B&H':>10} {'Alpha':>10}")
    print("-" * 65)

    for r in results:
        marker = "[+]" if r['alpha'] > 0 else "[-]"
        print(f"{r['scenario']:<30} {r['strategy']:>9.2%} {r['bh']:>9.2%} {r['alpha']:>9.2%} {marker}")

    # Aggregate
    bull_results = [r for r in results if 'bull' in r['scenario'].lower() or 'recovery' in r['scenario'].lower()]
    bear_results = [r for r in results if 'bear' in r['scenario'].lower() or 'crisis' in r['scenario'].lower()]

    bull_wins = sum(1 for r in bull_results if r['alpha'] > 0)
    bear_wins = sum(1 for r in bear_results if r['alpha'] > 0)

    print("-" * 65)
    print(f"\nBull Markets: {bull_wins}/{len(bull_results)} outperformance")
    print(f"Bear Markets: {bear_wins}/{len(bear_results)} outperformance")

    avg_alpha = sum(r['alpha'] for r in results) / len(results)
    avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
    avg_dd = sum(r['max_dd'] for r in results) / len(results)

    print(f"\nOverall:")
    print(f"  Average Alpha:       {avg_alpha:>10.2%}")
    print(f"  Average Sharpe:      {avg_sharpe:>10.2f}")
    print(f"  Average Max DD:      {avg_dd:>10.2%}")

    if bull_wins > 0 and bear_wins > 0:
        print("\n[SUCCESS] Strategy shows alpha in BOTH bull and bear markets!")
    elif bull_wins > len(bull_results) // 2 and bear_wins > len(bear_results) // 2:
        print("\n[PARTIAL SUCCESS] Strategy shows majority alpha across regimes")
    else:
        print("\n[NEEDS WORK] Strategy not consistently outperforming")

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    run_comprehensive_test()
