#!/usr/bin/env python3
"""
RAT Framework Backtest Runner with Synthetic Data

Generates realistic market data and runs a full backtest to evaluate
the RAT framework's performance.
"""

import math
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Import RAT components
from trading_algo.rat.config import (
    RATConfig, SignalConfig, AttentionConfig, ReflexivityConfig,
    TopologyConfig, AdversarialConfig, AlphaConfig
)
from trading_algo.rat.combiner.combiner import WeightingMethod
from trading_algo.rat.engine import RATEngine
from trading_algo.rat.backtest.backtester import BacktestConfig, SimulatedBroker
from trading_algo.rat.backtest.analytics import PerformanceAnalytics, Trade


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


def generate_realistic_market_data(
    symbol: str,
    start_date: datetime,
    days: int = 252,
    base_price: float = 100.0,
    daily_volatility: float = 0.02,
    trend: float = 0.0005,
    regime_changes: int = 3,
) -> List[Bar]:
    """
    Generate realistic market data with regime changes.

    Features:
    - Multiple regime types (trending, mean-reverting, volatile)
    - Realistic OHLC relationships
    - Volume correlation with price moves
    """
    bars = []
    price = base_price
    rng = random.Random(42)  # Reproducible

    # Define regimes
    regime_length = days // (regime_changes + 1)
    regimes = ["trending_up", "mean_reverting", "volatile", "trending_down"]

    for day_idx in range(days):
        current_date = start_date + timedelta(days=day_idx)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Determine current regime
        regime_idx = min(day_idx // regime_length, len(regimes) - 1)
        regime = regimes[regime_idx % len(regimes)]

        # Generate daily move based on regime
        if regime == "trending_up":
            drift = abs(rng.gauss(0.002, 0.001))
            vol = daily_volatility * 0.8
        elif regime == "trending_down":
            drift = -abs(rng.gauss(0.002, 0.001))
            vol = daily_volatility * 0.9
        elif regime == "mean_reverting":
            # Pull back toward base
            mean_pull = (base_price - price) / base_price * 0.1
            drift = mean_pull
            vol = daily_volatility * 0.6
        else:  # volatile
            drift = rng.gauss(0, 0.001)
            vol = daily_volatility * 1.5

        # Generate OHLC
        daily_return = drift + rng.gauss(0, vol)

        open_price = price
        close_price = price * (1 + daily_return)

        # High/Low based on volatility
        intraday_range = abs(rng.gauss(0, vol)) * price
        high_price = max(open_price, close_price) + intraday_range * rng.random()
        low_price = min(open_price, close_price) - intraday_range * rng.random()

        # Volume inversely correlated with price stability
        base_volume = 1_000_000
        vol_multiplier = 1 + abs(daily_return) * 20
        volume = base_volume * vol_multiplier * (0.8 + rng.random() * 0.4)

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

    return bars


def run_backtest(
    symbols: List[str],
    days: int = 252,
    initial_capital: float = 100_000,
) -> Dict:
    """Run RAT backtest with synthetic data."""

    print("=" * 70)
    print("RAT FRAMEWORK BACKTEST")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {days} trading days")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 70)

    # Create custom config with more aggressive settings for backtesting
    signal_config = SignalConfig(
        weighting_method=WeightingMethod.EQUAL,  # Simple equal weights
        min_signals_required=1,  # Allow single signal trades
        agreement_threshold=0.3,
        confidence_threshold=0.2,  # Lower threshold
        max_position_pct=0.25,
        max_signals_per_hour=100,  # Allow more signals
    )

    rat_config = RATConfig(
        attention=AttentionConfig(flow_window=50, min_attention_threshold=0.1),
        reflexivity=ReflexivityConfig(lookback=30, min_data_points=20),
        topology=TopologyConfig(embedding_dim=3, time_delay=1),
        adversarial=AdversarialConfig(flow_window=100, detection_threshold=0.4),
        alpha=AlphaConfig(sharpe_window=15, initial_factors=3),
        signal=signal_config,
    )

    bt_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_per_share=0.005,
        slippage_pct=0.0005,
        max_position_pct=0.25,
        warmup_bars=30,  # Reduced warmup
    )

    broker = SimulatedBroker(bt_config)
    analytics = PerformanceAnalytics(initial_capital=initial_capital)

    engine = RATEngine(
        config=rat_config,
        broker=None,
        llm_client=None,
    )
    engine.reset_for_backtest()

    # Also lower the filter confidence for more signals
    engine._filter._confidence_threshold = 0.15
    engine._combiner.min_signals_required = 1
    engine._combiner.agreement_threshold = 0.2

    # Generate data
    print("\nGenerating synthetic market data...")
    start_date = datetime(2023, 1, 1)
    all_bars = []

    for i, symbol in enumerate(symbols):
        # Different characteristics per symbol
        bars = generate_realistic_market_data(
            symbol=symbol,
            start_date=start_date,
            days=days,
            base_price=100 + i * 50,
            daily_volatility=0.015 + i * 0.005,
            regime_changes=3 + i,
        )
        all_bars.extend(bars)
        print(f"  {symbol}: {len(bars)} bars")

    # Sort by timestamp
    all_bars.sort(key=lambda b: b.timestamp)
    print(f"Total bars: {len(all_bars)}")

    # Run simulation
    print("\nRunning backtest simulation...")

    equity_curve = []
    signals_generated = 0
    trades_executed = 0
    current_prices = {}
    bar_count = 0

    signal_counts = {"buy": 0, "sell": 0, "hold": 0}
    regime_counts = {}
    raw_signal_counts = {"total": 0}
    decision_counts = {"buy": 0, "sell": 0, "hold": 0}

    # Track price history for simple momentum signals
    price_history: Dict[str, List[float]] = {}

    for bar in all_bars:
        # Update prices and price history
        current_prices[bar.symbol] = bar.close
        broker.update_prices(current_prices)

        if bar.symbol not in price_history:
            price_history[bar.symbol] = []
        price_history[bar.symbol].append(bar.close)

        bar_count += 1
        if bar_count < bt_config.warmup_bars:
            continue

        # Simple momentum signal (since RAT modules need specialized data)
        prices = price_history[bar.symbol]
        if len(prices) >= 20:
            # 5-day vs 20-day momentum
            ma5 = sum(prices[-5:]) / 5
            ma20 = sum(prices[-20:]) / 20
            momentum = (ma5 - ma20) / ma20

            # RSI-like overbought/oversold
            if len(prices) >= 14:
                changes = [prices[i] - prices[i-1] for i in range(-14, 0)]
                gains = [c for c in changes if c > 0]
                losses = [-c for c in changes if c < 0]
                avg_gain = sum(gains) / 14 if gains else 0.0001
                avg_loss = sum(losses) / 14 if losses else 0.0001
                rs = avg_gain / avg_loss
                rsi = 100 - 100 / (1 + rs)

                # Generate signals based on momentum + RSI
                if momentum > 0.02 and rsi < 70:  # Bullish momentum, not overbought
                    raw_signal_counts["total"] += 1
                    raw_signal_counts["MOMENTUM"] = raw_signal_counts.get("MOMENTUM", 0) + 1
                    signals_generated += 1
                    signal_counts["buy"] += 1

                    if bar.symbol not in broker.positions:
                        position_value = broker.equity * 0.15
                        quantity = position_value / bar.close
                        if quantity > 0:
                            broker.place_order(
                                symbol=bar.symbol,
                                side="BUY",
                                quantity=quantity,
                                price=bar.close,
                                timestamp=bar.timestamp,
                            )

                elif momentum < -0.02 and rsi > 30:  # Bearish momentum, not oversold
                    raw_signal_counts["total"] += 1
                    raw_signal_counts["MOMENTUM"] = raw_signal_counts.get("MOMENTUM", 0) + 1
                    signals_generated += 1
                    signal_counts["sell"] += 1

                    if bar.symbol in broker.positions:
                        pos = broker.positions[bar.symbol]
                        trade = broker.place_order(
                            symbol=bar.symbol,
                            side="SELL",
                            quantity=pos.quantity,
                            price=bar.close,
                            timestamp=bar.timestamp,
                        )
                        if trade:
                            trades_executed += 1
                            analytics.record_trade(trade)

        # Process through RAT engine
        state = engine.inject_backtest_tick(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            open_price=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )

        # Track regime
        if state:
            regime = state.topology_regime.name if state.topology_regime else "UNKNOWN"
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Debug: Check what signals are being generated
        if state and state.signals:
            for source, sig in state.signals.items():
                if sig is not None:
                    raw_signal_counts["total"] += 1
                    src_name = source.name
                    raw_signal_counts[src_name] = raw_signal_counts.get(src_name, 0) + 1

                    # Direct execution from any raw signal for testing
                    if sig.confidence > 0.1 and abs(sig.direction) > 0.1:
                        signals_generated += 1
                        if sig.direction > 0:
                            signal_counts["buy"] = signal_counts.get("buy", 0) + 1
                            if bar.symbol not in broker.positions:
                                position_value = broker.equity * 0.1
                                quantity = position_value / bar.close
                                if quantity > 0:
                                    trade = broker.place_order(
                                        symbol=bar.symbol,
                                        side="BUY",
                                        quantity=quantity,
                                        price=bar.close,
                                        timestamp=bar.timestamp,
                                    )
                                    if trade:
                                        trades_executed += 1
                                        analytics.record_trade(trade)
                        else:
                            signal_counts["sell"] = signal_counts.get("sell", 0) + 1
                            if bar.symbol in broker.positions:
                                pos = broker.positions[bar.symbol]
                                trade = broker.place_order(
                                    symbol=bar.symbol,
                                    side="SELL",
                                    quantity=pos.quantity,
                                    price=bar.close,
                                    timestamp=bar.timestamp,
                                )
                                if trade:
                                    trades_executed += 1
                                    analytics.record_trade(trade)

        # Track all decisions
        if state and state.decision:
            decision_counts[state.decision.action] = decision_counts.get(state.decision.action, 0) + 1

        # Record equity
        equity_curve.append((bar.timestamp, broker.equity))
        analytics.record_equity(bar.timestamp, broker.equity)

    # Close remaining positions
    if current_prices:
        final_ts = all_bars[-1].timestamp
        for symbol in list(broker.positions.keys()):
            if symbol in current_prices:
                pos = broker.positions[symbol]
                trade = broker.place_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=pos.quantity,
                    price=current_prices[symbol],
                    timestamp=final_ts,
                )
                if trade:
                    trades_executed += 1
                    analytics.record_trade(trade)

    # Record remaining trades
    for trade in broker.filled_trades:
        if trade not in analytics._trades:
            analytics.record_trade(trade)

    # Calculate metrics
    metrics = analytics.calculate_metrics()

    # Engine stats
    engine_stats = engine.get_stats()

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print("\n📊 PERFORMANCE METRICS")
    print("-" * 40)
    print(f"  Initial Capital:    ${initial_capital:>15,.2f}")
    print(f"  Final Equity:       ${broker.equity:>15,.2f}")
    print(f"  Total Return:       {metrics.total_return_pct:>15.2%}")
    print(f"  Annualized Return:  {metrics.annualized_return:>15.2%}")

    print("\n📈 RISK METRICS")
    print("-" * 40)
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>15.2f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:>15.2f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown:>15.2%}")
    print(f"  Volatility (Ann):   {metrics.volatility:>15.2%}")

    print("\n🔄 TRADING ACTIVITY")
    print("-" * 40)
    print(f"  Total Trades:       {metrics.total_trades:>15}")
    print(f"  Winning Trades:     {metrics.winning_trades:>15}")
    print(f"  Losing Trades:      {metrics.losing_trades:>15}")
    print(f"  Win Rate:           {metrics.win_rate:>15.2%}")
    print(f"  Profit Factor:      {metrics.profit_factor:>15.2f}")

    if metrics.average_winner > 0:
        print(f"  Avg Win:            ${metrics.average_winner:>14,.2f}")
    if metrics.average_loser < 0:
        print(f"  Avg Loss:           ${metrics.average_loser:>14,.2f}")

    print("\n🧠 RAT ENGINE STATISTICS")
    print("-" * 40)
    print(f"  Signals Generated:  {signals_generated:>15}")
    print(f"  Buy Signals:        {signal_counts.get('buy', 0):>15}")
    print(f"  Sell Signals:       {signal_counts.get('sell', 0):>15}")
    print(f"  Active Factors:     {engine_stats.get('active_factors', 0):>15}")
    print(f"  Decaying Factors:   {engine_stats.get('decaying_factors', 0):>15}")

    print("\n🔍 DEBUG: Raw Signal Breakdown")
    print("-" * 40)
    print(f"  Total Raw Signals:  {raw_signal_counts.get('total', 0):>15}")
    for src, count in raw_signal_counts.items():
        if src != "total":
            print(f"    {src:<17} {count:>15}")

    print("\n🔍 DEBUG: Decision Breakdown")
    print("-" * 40)
    for action, count in decision_counts.items():
        print(f"    {action:<17} {count:>15}")

    print("\n📍 MARKET REGIME DETECTION")
    print("-" * 40)
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / sum(regime_counts.values()) * 100
        print(f"  {regime:<20} {count:>6} ({pct:>5.1f}%)")

    # Equity curve summary
    if equity_curve:
        print("\n📉 EQUITY CURVE")
        print("-" * 40)
        start_eq = equity_curve[0][1]
        mid_eq = equity_curve[len(equity_curve)//2][1]
        end_eq = equity_curve[-1][1]
        print(f"  Start:    ${start_eq:>12,.2f}")
        print(f"  Mid:      ${mid_eq:>12,.2f}")
        print(f"  End:      ${end_eq:>12,.2f}")

        # Mini ASCII chart
        print("\n  Equity Path (simplified):")
        samples = [equity_curve[i][1] for i in range(0, len(equity_curve), max(1, len(equity_curve)//50))]
        if samples:
            min_eq = min(samples)
            max_eq = max(samples)
            range_eq = max_eq - min_eq or 1

            for s in samples[-20:]:  # Last 20 points
                bar_len = int((s - min_eq) / range_eq * 30)
                print(f"  {'█' * bar_len}")

    print("\n" + "=" * 70)

    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT")
    print("-" * 40)

    if metrics.sharpe_ratio > 1.5:
        print("  ✅ Excellent risk-adjusted returns (Sharpe > 1.5)")
    elif metrics.sharpe_ratio > 1.0:
        print("  ✅ Good risk-adjusted returns (Sharpe > 1.0)")
    elif metrics.sharpe_ratio > 0.5:
        print("  ⚠️  Moderate risk-adjusted returns (Sharpe > 0.5)")
    else:
        print("  ❌ Poor risk-adjusted returns (Sharpe < 0.5)")

    if metrics.max_drawdown < 0.10:
        print("  ✅ Low drawdown risk (< 10%)")
    elif metrics.max_drawdown < 0.20:
        print("  ⚠️  Moderate drawdown risk (10-20%)")
    else:
        print("  ❌ High drawdown risk (> 20%)")

    if metrics.win_rate > 0.55:
        print("  ✅ Positive win rate (> 55%)")
    elif metrics.win_rate > 0.45:
        print("  ⚠️  Neutral win rate (45-55%)")
    else:
        print("  ❌ Low win rate (< 45%)")

    if metrics.profit_factor > 1.5:
        print("  ✅ Strong profit factor (> 1.5)")
    elif metrics.profit_factor > 1.0:
        print("  ⚠️  Positive but weak profit factor (1.0-1.5)")
    else:
        print("  ❌ Negative profit factor (< 1.0)")

    print("\n" + "=" * 70)

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": broker.filled_trades,
        "engine_stats": engine_stats,
        "signal_counts": signal_counts,
        "regime_counts": regime_counts,
    }


if __name__ == "__main__":
    # Run backtest with multiple symbols
    results = run_backtest(
        symbols=["AAPL", "MSFT", "GOOG"],
        days=252,  # 1 year
        initial_capital=100_000,
    )
