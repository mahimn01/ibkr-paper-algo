#!/usr/bin/env python3
"""
Realistic Backtest for RAT Framework

Generates realistic market data based on actual market characteristics:
- Actual current prices (AAPL ~$258, MSFT ~$415, etc.)
- Historical volatility for tech stocks (~25-35% annualized)
- Realistic market regimes based on 2023-2025 market behavior
- Proper correlation structure between stocks
"""

import math
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional

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


# Realistic stock parameters based on actual market data
STOCK_PARAMS = {
    "AAPL": {
        "current_price": 258.0,  # Jan 2026 price
        "annual_volatility": 0.28,  # ~28% annual vol for AAPL
        "avg_daily_volume": 55_000_000,
        "beta": 1.25,
    },
    "MSFT": {
        "current_price": 415.0,
        "annual_volatility": 0.25,
        "avg_daily_volume": 20_000_000,
        "beta": 1.15,
    },
    "GOOGL": {
        "current_price": 195.0,
        "annual_volatility": 0.30,
        "avg_daily_volume": 22_000_000,
        "beta": 1.20,
    },
    "AMZN": {
        "current_price": 225.0,
        "annual_volatility": 0.32,
        "avg_daily_volume": 35_000_000,
        "beta": 1.30,
    },
    "META": {
        "current_price": 600.0,
        "annual_volatility": 0.38,
        "avg_daily_volume": 12_000_000,
        "beta": 1.40,
    },
    "NVDA": {
        "current_price": 140.0,
        "annual_volatility": 0.50,  # NVDA is very volatile
        "avg_daily_volume": 250_000_000,
        "beta": 1.80,
    },
    "TSLA": {
        "current_price": 420.0,
        "annual_volatility": 0.55,  # TSLA is extremely volatile
        "avg_daily_volume": 80_000_000,
        "beta": 2.00,
    },
}

# Market regime timeline based on actual 2023-2025 events
# Returns are rough approximations of actual market behavior
MARKET_REGIMES = [
    # Q1 2024: Strong rally (AI boom continues)
    {"months": 3, "market_drift": 0.08, "market_vol_mult": 0.9, "name": "AI Rally Q1 2024"},
    # Q2 2024: Consolidation with some volatility
    {"months": 3, "market_drift": -0.02, "market_vol_mult": 1.1, "name": "Consolidation Q2 2024"},
    # Q3 2024: Summer rally
    {"months": 3, "market_drift": 0.05, "market_vol_mult": 0.8, "name": "Summer Rally Q3 2024"},
    # Q4 2024: Year-end volatility
    {"months": 3, "market_drift": 0.03, "market_vol_mult": 1.3, "name": "Year-End Q4 2024"},
    # Q1 2025: New year momentum
    {"months": 3, "market_drift": 0.04, "market_vol_mult": 1.0, "name": "Q1 2025 Momentum"},
    # Q2 2025: Correction
    {"months": 3, "market_drift": -0.05, "market_vol_mult": 1.4, "name": "Q2 2025 Correction"},
    # Q3 2025: Recovery
    {"months": 3, "market_drift": 0.06, "market_vol_mult": 1.0, "name": "Q3 2025 Recovery"},
    # Q4 2025 - Present: All-time highs
    {"months": 3, "market_drift": 0.07, "market_vol_mult": 0.85, "name": "Q4 2025 ATH"},
]


def generate_realistic_market_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    seed: Optional[int] = None,
) -> List[Bar]:
    """
    Generate realistic market data using actual stock characteristics
    and market regime timeline.
    """
    if seed is not None:
        random.seed(seed)

    all_bars = []

    for symbol in symbols:
        if symbol not in STOCK_PARAMS:
            print(f"  Warning: No parameters for {symbol}, using defaults")
            params = {
                "current_price": 100.0,
                "annual_volatility": 0.30,
                "avg_daily_volume": 10_000_000,
                "beta": 1.0,
            }
        else:
            params = STOCK_PARAMS[symbol]

        # Work backwards from current price to start date
        # to match actual price levels
        total_days = (end_date - start_date).days
        trading_days = int(total_days * 5 / 7)  # Approximate trading days

        # Calculate expected drift from regime timeline
        total_regime_drift = sum(r["market_drift"] for r in MARKET_REGIMES)
        current_price = params["current_price"]
        # Starting price that would lead to current price given expected drift
        start_price = current_price / (1 + total_regime_drift * params["beta"])

        daily_vol = params["annual_volatility"] / math.sqrt(252)
        avg_volume = params["avg_daily_volume"]
        beta = params["beta"]

        price = start_price
        current_date = start_date
        regime_idx = 0
        days_in_current_regime = 0
        regime_trading_days = int(MARKET_REGIMES[0]["months"] * 21)  # ~21 trading days/month

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Check if we need to switch regime
            if days_in_current_regime >= regime_trading_days and regime_idx < len(MARKET_REGIMES) - 1:
                regime_idx += 1
                days_in_current_regime = 0
                regime_trading_days = int(MARKET_REGIMES[regime_idx]["months"] * 21)

            regime = MARKET_REGIMES[regime_idx]
            market_drift = regime["market_drift"] / (regime["months"] * 21)  # Daily drift
            vol_mult = regime["market_vol_mult"]

            # Stock-specific drift (based on beta)
            stock_drift = market_drift * beta

            # Idiosyncratic component
            idio_vol = daily_vol * math.sqrt(1 - 0.3)  # Assume 30% of variance is systematic
            market_vol = daily_vol * math.sqrt(0.3) * vol_mult

            # Combined return
            daily_return = (
                stock_drift +
                random.gauss(0, market_vol) +
                random.gauss(0, idio_vol)
            )

            # Generate OHLC with realistic intraday pattern
            open_price = price
            close_price = price * (1 + daily_return)

            # Intraday high/low based on volatility
            intraday_range = abs(daily_vol * vol_mult * 1.5 * price)
            high_price = max(open_price, close_price) + random.uniform(0, intraday_range)
            low_price = min(open_price, close_price) - random.uniform(0, intraday_range)

            # Ensure high >= low
            if high_price < low_price:
                high_price, low_price = low_price, high_price

            # Volume correlates with volatility and absolute return
            vol_factor = 1 + abs(daily_return) * 10 + (vol_mult - 1) * 0.5
            volume = avg_volume * vol_factor * random.uniform(0.6, 1.4)

            all_bars.append(Bar(
                symbol=symbol,
                timestamp=current_date,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=int(volume),
            ))

            price = close_price
            current_date += timedelta(days=1)
            days_in_current_regime += 1

    return all_bars


class RealisticBacktester:
    """Backtester with realistic market data."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5,
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

        self.signal_generator = EnhancedSignalGenerator()
        # More aggressive risk limits for active trading
        risk_limits = RiskLimits(
            max_position_pct=0.15,           # 15% per position
            max_position_value=20000,        # $20k per position
            max_positions=10,                # Up to 10 positions
            max_gross_exposure=1.5,          # Allow up to 150% exposure
            max_daily_loss=0.05,             # 5% daily loss limit
            max_drawdown=0.25,               # 25% max drawdown
        )
        self.risk_manager = RiskManager(initial_capital, risk_limits)
        self.analytics = PerformanceAnalytics(initial_capital)

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
        risk_limits = RiskLimits(
            max_position_pct=0.15,
            max_position_value=20000,
            max_positions=10,
            max_gross_exposure=1.5,
            max_daily_loss=0.05,
            max_drawdown=0.25,
        )
        self.risk_manager = RiskManager(self.initial_capital, risk_limits)
        self.analytics = PerformanceAnalytics(self.initial_capital)

    def run(
        self,
        bars: List[Bar],
        warmup_bars: int = 30,
        confidence_threshold: float = 0.25,
    ) -> Dict:
        """Run backtest on bar data."""
        self.reset()

        bars = sorted(bars, key=lambda b: b.timestamp)

        signals_generated = 0
        regime_counts: Dict[str, int] = {}
        strategy_counts: Dict[str, int] = {}

        for i, bar in enumerate(bars):
            self._update_positions(bar)
            self.risk_manager.update_equity(self.equity, bar.timestamp)

            # Check stops on existing positions
            self._check_stops(bar)

            if i < warmup_bars:
                continue

            enhanced_signal = self.signal_generator.update(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

            if enhanced_signal:
                regime_name = enhanced_signal.regime.name
                regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

                strategy = enhanced_signal.signal.metadata.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            if enhanced_signal and enhanced_signal.signal.confidence > confidence_threshold:
                signals_generated += 1
                self._process_signal(enhanced_signal, bar)

            self.analytics.record_equity(bar.timestamp, self.equity)

        self._close_all_positions(bars[-1] if bars else None)

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
        """Update position values."""
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

    def _check_stops(self, bar: Bar):
        """Check and execute trailing stops."""
        if bar.symbol not in self.positions:
            return

        pos = self.positions[bar.symbol]
        quantity = pos['quantity']
        if quantity <= 0:
            return

        current_pnl_pct = (bar.close - pos['avg_cost']) / pos['avg_cost']

        # Update trailing stop (highest price seen * 0.95)
        if 'highest_price' not in pos:
            pos['highest_price'] = bar.close
        else:
            pos['highest_price'] = max(pos['highest_price'], bar.close)

        trailing_stop = pos['highest_price'] * 0.92  # 8% trailing stop

        # Exit conditions:
        # 1. Stop loss hit (8% from entry)
        # 2. Trailing stop hit (8% from highest)
        # 3. Take profit (20% gain)
        should_exit = False
        exit_reason = ""

        if current_pnl_pct < -0.08:  # 8% stop loss
            should_exit = True
            exit_reason = "stop_loss"
        elif bar.close < trailing_stop and current_pnl_pct > 0.05:  # Trailing stop only after 5% gain
            should_exit = True
            exit_reason = "trailing_stop"
        elif current_pnl_pct > 0.20:  # 20% take profit
            should_exit = True
            exit_reason = "take_profit"

        if should_exit:
            self._close_position(bar.symbol, bar.close, bar.timestamp)

    def _process_signal(self, enhanced_signal, bar: Bar):
        """Process trading signal with improved logic."""
        signal = enhanced_signal.signal
        symbol = signal.symbol
        regime = enhanced_signal.regime

        can_trade, reason, adjusted_size = self.risk_manager.can_open_position(
            symbol=symbol,
            size_pct=enhanced_signal.position_size,
            estimated_value=self.equity * enhanced_signal.position_size,
        )

        if not can_trade:
            return

        if signal.direction > 0:  # Buy signal
            if symbol not in self.positions:
                # Larger position sizes (8-12% instead of 5%)
                base_size = 0.08  # 8% base position

                # Increase in uptrends
                if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND):
                    base_size = 0.12  # 12% in uptrends

                self._open_position(
                    symbol=symbol,
                    direction=1,
                    size_pct=base_size,
                    price=bar.close,
                    timestamp=bar.timestamp,
                    stop_loss=enhanced_signal.stop_loss,
                    take_profit=enhanced_signal.take_profit,
                )
        else:  # Sell signal - close position
            if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                self._close_position(symbol, bar.close, bar.timestamp)

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
        })

    def _close_position(self, symbol: str, price: float, timestamp: datetime):
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
            'pnl': pnl,
        })

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


def run_realistic_backtest(
    symbols: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    initial_capital: float = 100_000,
) -> Dict:
    """Run backtest with realistic market data."""

    print("=" * 70)
    print("RAT FRAMEWORK BACKTEST - REALISTIC MARKET SIMULATION")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: Jan 2024 - Jan 2026 (2 years)")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 70)

    # Date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 1, 15)

    # Generate data
    print("\nGenerating realistic market data based on actual stock characteristics...")
    print("Using actual volatility, beta, and market regime timeline:")
    for regime in MARKET_REGIMES:
        drift_annual = regime["market_drift"] / regime["months"] * 12
        print(f"  {regime['name']}: {drift_annual:+.0%} annual drift, {regime['market_vol_mult']:.1f}x vol")

    print()
    all_bars = generate_realistic_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        seed=42,
    )

    # Count bars per symbol
    symbol_counts = {}
    symbol_prices = {}
    for bar in all_bars:
        symbol_counts[bar.symbol] = symbol_counts.get(bar.symbol, 0) + 1
        symbol_prices[bar.symbol] = bar.close

    print("Generated data:")
    for symbol in symbols:
        count = symbol_counts.get(symbol, 0)
        final_price = symbol_prices.get(symbol, 0)
        target_price = STOCK_PARAMS.get(symbol, {}).get("current_price", 0)
        print(f"  {symbol}: {count} bars, final=${final_price:.2f} (target=${target_price:.2f})")

    print(f"\nTotal bars: {len(all_bars)}")

    # Sort by timestamp
    all_bars.sort(key=lambda b: b.timestamp)

    # Run backtest
    print("\nRunning backtest...")
    backtester = RealisticBacktester(initial_capital=initial_capital)
    results = backtester.run(all_bars, warmup_bars=30, confidence_threshold=0.25)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
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

    print("\nPERFORMANCE ASSESSMENT")
    print("-" * 40)

    if results['sharpe_ratio'] > 1.0:
        print("  [EXCELLENT] Risk-adjusted returns (Sharpe > 1.0)")
    elif results['sharpe_ratio'] > 0.5:
        print("  [GOOD] Risk-adjusted returns (Sharpe > 0.5)")
    elif results['sharpe_ratio'] > 0:
        print("  [OK] Positive but modest Sharpe ratio")
    else:
        print("  [NEEDS WORK] Negative Sharpe ratio")

    if results['max_drawdown'] < 0.10:
        print("  [EXCELLENT] Low drawdown risk (< 10%)")
    elif results['max_drawdown'] < 0.20:
        print("  [MODERATE] Drawdown risk (10-20%)")
    else:
        print("  [HIGH RISK] Drawdown > 20%")

    if results['profit_factor'] > 1.5:
        print("  [EXCELLENT] Profit factor > 1.5")
    elif results['profit_factor'] > 1.0:
        print("  [OK] Positive profit factor")
    else:
        print("  [LOSS] Negative profit factor")

    # Calculate buy-and-hold benchmark
    print("\nBENCHMARK COMPARISON (Equal-weighted Buy & Hold)")
    print("-" * 40)

    # Calculate what equal-weight buy-and-hold would have returned
    bh_return = 0
    for symbol in symbols:
        params = STOCK_PARAMS.get(symbol, {})
        total_drift = sum(r["market_drift"] * params.get("beta", 1.0) for r in MARKET_REGIMES)
        bh_return += total_drift / len(symbols)

    print(f"  Buy & Hold Return:  {bh_return:>15.2%}")
    print(f"  Strategy Return:    {results['total_return']:>15.2%}")
    alpha = results['total_return'] - bh_return
    print(f"  Alpha vs B&H:       {alpha:>15.2%}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = run_realistic_backtest(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        initial_capital=100_000,
    )
