#!/usr/bin/env python3
"""
Real Data Backtest for RAT Framework

Fetches actual historical stock data from Yahoo Finance and runs
a comprehensive backtest with the enhanced RAT framework.
"""

import json
import time
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional

# Import enhanced components
from trading_algo.rat.enhanced_signals import (
    EnhancedSignalGenerator,
    MarketRegime,
    PARAMS,
)
from trading_algo.rat.risk_manager import (
    RiskManager,
    RiskLimits,
)
from trading_algo.rat.backtest.analytics import PerformanceAnalytics


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


def fetch_yahoo_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d"
) -> List[Bar]:
    """
    Fetch historical data from Yahoo Finance.

    Uses the chart API which doesn't require authentication.
    """
    # Convert dates to Unix timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    # Yahoo Finance chart API
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,split",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Parse response
        result = data.get("chart", {}).get("result", [])
        if not result:
            print(f"  Warning: No data returned for {symbol}")
            return []

        chart_data = result[0]
        timestamps = chart_data.get("timestamp", [])
        quote = chart_data.get("indicators", {}).get("quote", [{}])[0]

        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        bars = []
        for i, ts in enumerate(timestamps):
            # Skip if any value is None
            if any(x is None or (i < len(x) and x[i] is None) for x in [opens, highs, lows, closes]):
                continue
            if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes):
                continue
            if opens[i] is None or highs[i] is None or lows[i] is None or closes[i] is None:
                continue

            bars.append(Bar(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ts),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(volumes[i]) if i < len(volumes) and volumes[i] else 0,
            ))

        return bars

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {symbol}: {e}")
        return []
    except (KeyError, IndexError, TypeError) as e:
        print(f"  Error parsing {symbol} data: {e}")
        return []


class RealDataBacktester:
    """
    Backtester using real market data.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5,
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

        # Components
        self.signal_generator = EnhancedSignalGenerator()
        self.risk_manager = RiskManager(initial_capital, RiskLimits())
        self.analytics = PerformanceAnalytics(initial_capital)

        # State
        self.equity = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []

    def reset(self):
        """Reset backtester state."""
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.signal_generator = EnhancedSignalGenerator()
        self.risk_manager = RiskManager(self.initial_capital, RiskLimits())
        self.analytics = PerformanceAnalytics(self.initial_capital)

    def run(
        self,
        bars: List[Bar],
        warmup_bars: int = 30,
        confidence_threshold: float = 0.25,
    ) -> Dict:
        """Run backtest on bar data."""
        self.reset()

        # Sort by timestamp
        bars = sorted(bars, key=lambda b: b.timestamp)

        # Track metrics
        signals_generated = 0
        regime_counts: Dict[str, int] = {}
        strategy_counts: Dict[str, int] = {}

        for i, bar in enumerate(bars):
            # Update position values
            self._update_positions(bar)

            # Update risk manager
            self.risk_manager.update_equity(self.equity, bar.timestamp)

            # Skip warmup
            if i < warmup_bars:
                continue

            # Generate signal
            enhanced_signal = self.signal_generator.update(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

            # Track regime
            if enhanced_signal:
                regime_name = enhanced_signal.regime.name
                regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

                # Track strategy
                strategy = enhanced_signal.signal.metadata.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            # Check if we should trade
            if enhanced_signal and enhanced_signal.signal.confidence > confidence_threshold:
                signals_generated += 1
                self._process_signal(enhanced_signal, bar)

            # Record equity
            self.analytics.record_equity(bar.timestamp, self.equity)

        # Close all positions at end
        self._close_all_positions(bars[-1] if bars else None)

        # Calculate metrics
        metrics = self.analytics.calculate_metrics()

        return {
            'total_return': metrics.total_return_pct,
            'annualized_return': metrics.annualized_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'total_trades': metrics.total_trades,
            'signals_generated': signals_generated,
            'regime_distribution': regime_counts,
            'strategy_distribution': strategy_counts,
            'final_equity': self.equity,
        }

    def _update_positions(self, bar: Bar):
        """Update position values with new prices."""
        for symbol, pos in self.positions.items():
            if symbol == bar.symbol:
                pos['market_value'] = pos['quantity'] * bar.close
                pos['unrealized_pnl'] = pos['quantity'] * (bar.close - pos['avg_cost'])

                self.risk_manager.update_position(
                    symbol=symbol,
                    quantity=pos['quantity'],
                    current_price=bar.close,
                    avg_cost=pos['avg_cost'],
                )

        total_position_value = sum(p['market_value'] for p in self.positions.values())
        self.equity = self.cash + total_position_value

    def _process_signal(self, enhanced_signal, bar: Bar):
        """Process trading signal."""
        signal = enhanced_signal.signal
        symbol = signal.symbol

        can_trade, reason, adjusted_size = self.risk_manager.can_open_position(
            symbol=symbol,
            size_pct=enhanced_signal.position_size,
            estimated_value=self.equity * enhanced_signal.position_size,
        )

        if not can_trade:
            return

        if signal.direction > 0:  # Buy signal
            if symbol not in self.positions:
                self._open_position(
                    symbol=symbol,
                    direction=1,
                    size_pct=adjusted_size,
                    price=bar.close,
                    timestamp=bar.timestamp,
                    stop_loss=enhanced_signal.stop_loss,
                    take_profit=enhanced_signal.take_profit,
                )
        else:  # Sell signal
            if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                self._close_position(
                    symbol=symbol,
                    price=bar.close,
                    timestamp=bar.timestamp,
                )

    def _open_position(
        self,
        symbol: str,
        direction: int,
        size_pct: float,
        price: float,
        timestamp: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Open a new position."""
        position_value = self.equity * size_pct
        slippage = price * self.slippage_bps / 10000 * direction
        exec_price = price + slippage
        quantity = position_value / exec_price
        commission = quantity * self.commission_per_share

        cost = quantity * exec_price + commission
        if cost > self.cash:
            quantity = (self.cash - commission) / exec_price
            cost = quantity * exec_price + commission

        if quantity <= 0:
            return

        self.cash -= cost
        self.positions[symbol] = {
            'quantity': quantity * direction,
            'avg_cost': exec_price,
            'market_value': quantity * price,
            'unrealized_pnl': 0,
            'entry_time': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
        }

        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'BUY' if direction > 0 else 'SELL',
            'quantity': quantity,
            'price': exec_price,
            'commission': commission,
        })

    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
    ):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        quantity = abs(pos['quantity'])
        direction = 1 if pos['quantity'] > 0 else -1

        slippage = price * self.slippage_bps / 10000 * (-direction)
        exec_price = price + slippage
        commission = quantity * self.commission_per_share

        proceeds = quantity * exec_price - commission
        pnl = proceeds - quantity * pos['avg_cost']

        self.cash += proceeds

        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'SELL' if direction > 0 else 'BUY',
            'quantity': quantity,
            'price': exec_price,
            'commission': commission,
            'pnl': pnl,
        })

        # Record trade for analytics
        holding_period = timestamp - pos['entry_time'] if pos.get('entry_time') else timedelta(days=1)
        self.analytics.record_trade(type('Trade', (), {
            'pnl': pnl,
            'entry_price': pos['avg_cost'],
            'exit_price': exec_price,
            'quantity': quantity,
            'holding_period': holding_period,
        })())

        del self.positions[symbol]

    def _close_all_positions(self, bar: Optional[Bar]):
        """Close all open positions."""
        if bar is None:
            return

        for symbol in list(self.positions.keys()):
            self._close_position(symbol, bar.close, bar.timestamp)


def run_real_data_backtest(
    symbols: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    years: int = 2,
    initial_capital: float = 100_000,
) -> Dict:
    """
    Run backtest with real market data from Yahoo Finance.
    """
    print("=" * 70)
    print("RAT FRAMEWORK BACKTEST - REAL MARKET DATA")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {years} years")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 70)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    # Fetch data
    print(f"\nFetching historical data from Yahoo Finance...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    all_bars = []
    for symbol in symbols:
        print(f"  Fetching {symbol}...", end=" ")
        bars = fetch_yahoo_data(symbol, start_date, end_date)
        if bars:
            all_bars.extend(bars)
            print(f"{len(bars)} bars")
        else:
            print("failed")
        time.sleep(0.5)  # Rate limiting

    if not all_bars:
        print("\nError: No data fetched. Cannot run backtest.")
        return {}

    print(f"\nTotal bars fetched: {len(all_bars)}")

    # Sort by timestamp
    all_bars.sort(key=lambda b: b.timestamp)

    # Show date range of actual data
    print(f"Actual data range: {all_bars[0].timestamp.strftime('%Y-%m-%d')} to {all_bars[-1].timestamp.strftime('%Y-%m-%d')}")

    # Run backtest
    print("\nRunning backtest...")
    backtester = RealDataBacktester(initial_capital=initial_capital)
    results = backtester.run(all_bars, warmup_bars=30, confidence_threshold=0.25)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - REAL DATA")
    print("=" * 70)

    print("\nPERFORMANCE METRICS")
    print("-" * 40)
    print(f"  Initial Capital:    ${initial_capital:>15,.2f}")
    print(f"  Final Equity:       ${results['final_equity']:>15,.2f}")
    print(f"  Total Return:       {results['total_return']:>15.2%}")
    print(f"  Annualized Return:  {results['annualized_return']:>15.2%}")

    print("\nRISK METRICS")
    print("-" * 40)
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:>15.2f}")
    print(f"  Sortino Ratio:      {results['sortino_ratio']:>15.2f}")
    print(f"  Max Drawdown:       {results['max_drawdown']:>15.2%}")

    print("\nTRADING ACTIVITY")
    print("-" * 40)
    print(f"  Total Trades:       {results['total_trades']:>15}")
    print(f"  Signals Generated:  {results['signals_generated']:>15}")
    print(f"  Win Rate:           {results['win_rate']:>15.2%}")
    print(f"  Profit Factor:      {results['profit_factor']:>15.2f}")

    print("\nREGIME DISTRIBUTION")
    print("-" * 40)
    total_regimes = sum(results['regime_distribution'].values())
    for regime, count in sorted(results['regime_distribution'].items(), key=lambda x: -x[1]):
        pct = count / total_regimes * 100 if total_regimes > 0 else 0
        print(f"  {regime:<20} {count:>6} ({pct:>5.1f}%)")

    print("\nSTRATEGY DISTRIBUTION")
    print("-" * 40)
    total_strategies = sum(results['strategy_distribution'].values())
    for strategy, count in sorted(results['strategy_distribution'].items(), key=lambda x: -x[1]):
        pct = count / total_strategies * 100 if total_strategies > 0 else 0
        print(f"  {strategy:<25} {count:>6} ({pct:>5.1f}%)")

    # Performance assessment
    print("\nPERFORMANCE ASSESSMENT")
    print("-" * 40)

    if results['sharpe_ratio'] > 1.0:
        print("  Excellent risk-adjusted returns (Sharpe > 1.0)")
    elif results['sharpe_ratio'] > 0.5:
        print("  Good risk-adjusted returns (Sharpe > 0.5)")
    elif results['sharpe_ratio'] > 0:
        print("  Positive but modest Sharpe ratio")
    else:
        print("  Negative Sharpe ratio - strategy needs optimization")

    if results['max_drawdown'] < 0.10:
        print("  Low drawdown risk (< 10%)")
    elif results['max_drawdown'] < 0.20:
        print("  Moderate drawdown risk (10-20%)")
    else:
        print("  High drawdown risk (> 20%)")

    if results['win_rate'] > 0.50:
        print("  Positive win rate (> 50%)")
    elif results['win_rate'] > 0.40:
        print("  Marginal win rate (40-50%)")
    else:
        print("  Low win rate (< 40%) - relies on larger winners")

    if results['profit_factor'] > 1.5:
        print("  Strong profit factor (> 1.5)")
    elif results['profit_factor'] > 1.0:
        print("  Positive but weak profit factor (1.0-1.5)")
    else:
        print("  Negative profit factor (< 1.0)")

    # Comparison to buy-and-hold
    if all_bars:
        # Calculate simple buy-and-hold for comparison
        symbol_first_close = {}
        symbol_last_close = {}
        for bar in all_bars:
            if bar.symbol not in symbol_first_close:
                symbol_first_close[bar.symbol] = bar.close
            symbol_last_close[bar.symbol] = bar.close

        bh_returns = []
        for sym in symbol_first_close:
            if sym in symbol_last_close and symbol_first_close[sym] > 0:
                ret = (symbol_last_close[sym] - symbol_first_close[sym]) / symbol_first_close[sym]
                bh_returns.append(ret)

        if bh_returns:
            avg_bh_return = sum(bh_returns) / len(bh_returns)
            print(f"\nBENCHMARK COMPARISON")
            print("-" * 40)
            print(f"  Buy & Hold Return:  {avg_bh_return:>15.2%}")
            print(f"  Strategy Return:    {results['total_return']:>15.2%}")
            alpha = results['total_return'] - avg_bh_return
            print(f"  Alpha:              {alpha:>15.2%}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = run_real_data_backtest(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        years=2,
        initial_capital=100_000,
    )
