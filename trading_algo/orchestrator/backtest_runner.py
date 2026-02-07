"""
Orchestrator Backtest Runner

Generates realistic synthetic 5-minute intraday data and runs the
Orchestrator through the backtest_v2 engine to validate performance.

Synthetic data includes:
  - Trending days (drift + noise)
  - Range-bound days (mean-reverting)
  - Volatile days (fat tails)
  - Regime switches within a day

This lets us measure the Orchestrator's behavior without requiring
live IBKR historical data, and provides a deterministic harness for
parameter sensitivity analysis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from trading_algo.backtest_v2.engine import BacktestEngine
from trading_algo.backtest_v2.models import BacktestConfig, BacktestResults, Bar
from trading_algo.orchestrator.config import OrchestratorConfig
from trading_algo.orchestrator.strategy import Orchestrator


# ── Synthetic data generation ─────────────────────────────────────────

@dataclass
class DaySpec:
    """Specification for a synthetic trading day."""
    day_type: str           # "trend_up", "trend_down", "range", "volatile"
    open_price: float
    daily_drift: float      # e.g. +0.015 for a +1.5% up day
    intraday_vol: float     # per-bar volatility (e.g. 0.001)
    volume_base: int = 50_000


def generate_day_bars(
    spec: DaySpec,
    day_date: date,
    bar_minutes: int = 5,
) -> List[Bar]:
    """
    Generate intraday bars for one trading day (9:30 → 16:00 ET).
    """
    market_open = datetime(day_date.year, day_date.month, day_date.day, 9, 30)
    n_bars = 390 // bar_minutes  # 78 bars for 5-min

    bars: List[Bar] = []
    price = spec.open_price
    cumulative_drift = 0.0

    for i in range(n_bars):
        ts = market_open + timedelta(minutes=i * bar_minutes)

        # Drift is distributed evenly across the day
        bar_drift = spec.daily_drift / n_bars

        # Noise depends on day type
        if spec.day_type == "volatile":
            # Fat tails via student-t approximation
            noise = random.gauss(0, spec.intraday_vol) * (1.0 + 0.5 * abs(random.gauss(0, 1)))
        elif spec.day_type == "range":
            # Mean-reverting: noise pulls back toward open
            reversion = -0.3 * (price - spec.open_price) / spec.open_price
            noise = random.gauss(0, spec.intraday_vol) + reversion / n_bars
        else:
            noise = random.gauss(0, spec.intraday_vol)

        ret = bar_drift + noise
        cumulative_drift += bar_drift

        new_price = price * (1 + ret)
        high = max(price, new_price) * (1 + abs(random.gauss(0, spec.intraday_vol * 0.3)))
        low = min(price, new_price) * (1 - abs(random.gauss(0, spec.intraday_vol * 0.3)))

        # Volume spikes at open and close
        vol_mult = 1.0
        if i < 6:         # First 30 min
            vol_mult = 2.0
        elif i >= n_bars - 6:  # Last 30 min
            vol_mult = 1.8

        bars.append(Bar(
            timestamp=ts,
            open=round(price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(new_price, 2),
            volume=int(spec.volume_base * vol_mult * (0.8 + 0.4 * random.random())),
        ))
        price = new_price

    return bars


def generate_reference_bars(
    day_specs: List[DaySpec],
    trading_dates: List[date],
    bar_minutes: int = 5,
) -> Dict[str, List[Bar]]:
    """
    Generate bars for reference ETFs (SPY, QQQ, IWM, SMH, XLK, XLF, XLE)
    that are correlated with (but not identical to) the primary symbol.
    """
    refs: Dict[str, List[Bar]] = {}
    etfs = ["SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE"]
    etf_base_prices = [450.0, 380.0, 210.0, 250.0, 195.0, 40.0, 82.0]

    for etf, base_price in zip(etfs, etf_base_prices):
        all_bars: List[Bar] = []
        price = base_price
        for day_idx, d in enumerate(trading_dates):
            spec = day_specs[day_idx % len(day_specs)]
            # ETFs correlate with the primary but with their own noise
            correlation = 0.6 if etf in ("SPY", "QQQ") else 0.3
            etf_drift = spec.daily_drift * correlation + random.gauss(0, 0.002)
            etf_spec = DaySpec(
                day_type=spec.day_type,
                open_price=price,
                daily_drift=etf_drift,
                intraday_vol=spec.intraday_vol * 0.8,
                volume_base=spec.volume_base * 3,
            )
            day_bars = generate_day_bars(etf_spec, d, bar_minutes)
            all_bars.extend(day_bars)
            price = day_bars[-1].close
        refs[etf] = all_bars

    return refs


def generate_synthetic_dataset(
    n_days: int = 20,
    symbol: str = "TEST",
    base_price: float = 150.0,
    seed: int = 42,
    bar_minutes: int = 5,
) -> Dict[str, List[Bar]]:
    """
    Generate a complete synthetic dataset: primary symbol + reference ETFs.

    Returns dict of symbol → list[Bar] suitable for BacktestEngine.run().
    """
    random.seed(seed)

    # Build a sequence of day types
    day_types = []
    for _ in range(n_days):
        r = random.random()
        if r < 0.3:
            day_types.append("trend_up")
        elif r < 0.55:
            day_types.append("trend_down")
        elif r < 0.8:
            day_types.append("range")
        else:
            day_types.append("volatile")

    # Generate trading dates (weekdays)
    start = date(2025, 6, 2)  # A Monday
    trading_dates: List[date] = []
    d = start
    while len(trading_dates) < n_days:
        if d.weekday() < 5:
            trading_dates.append(d)
        d += timedelta(days=1)

    # Build day specs
    price = base_price
    day_specs: List[DaySpec] = []
    for dt in day_types:
        if dt == "trend_up":
            drift = random.uniform(0.005, 0.02)
            vol = random.uniform(0.0008, 0.0015)
        elif dt == "trend_down":
            drift = random.uniform(-0.02, -0.005)
            vol = random.uniform(0.0008, 0.0015)
        elif dt == "range":
            drift = random.uniform(-0.003, 0.003)
            vol = random.uniform(0.0005, 0.001)
        else:  # volatile
            drift = random.uniform(-0.01, 0.01)
            vol = random.uniform(0.002, 0.004)

        day_specs.append(DaySpec(
            day_type=dt,
            open_price=price,
            daily_drift=drift,
            intraday_vol=vol,
        ))
        price *= (1 + drift)

    # Generate bars
    all_bars: List[Bar] = []
    price = base_price
    for idx, d in enumerate(trading_dates):
        spec = day_specs[idx]
        spec.open_price = price  # chain from previous close
        day_bars = generate_day_bars(spec, d, bar_minutes)
        all_bars.extend(day_bars)
        price = day_bars[-1].close

    data: Dict[str, List[Bar]] = {symbol: all_bars}

    # Add reference ETFs
    ref_data = generate_reference_bars(day_specs, trading_dates, bar_minutes)
    data.update(ref_data)

    return data


# ── Backtest execution ────────────────────────────────────────────────

@dataclass
class OrchestratorBacktestResult:
    """Summary of an Orchestrator backtest run."""
    config: OrchestratorConfig
    backtest_results: BacktestResults
    day_types_used: List[str] = field(default_factory=list)


def run_orchestrator_backtest(
    config: Optional[OrchestratorConfig] = None,
    n_days: int = 20,
    symbol: str = "TEST",
    base_price: float = 150.0,
    seed: int = 42,
    initial_capital: float = 100_000.0,
) -> OrchestratorBacktestResult:
    """
    Run the Orchestrator through the backtest_v2 engine on synthetic data.

    Parameters
    ----------
    config : OrchestratorConfig | None
        Strategy configuration (defaults used if None).
    n_days : int
        Number of simulated trading days.
    symbol : str
        Primary tradeable symbol name.
    base_price : float
        Starting price for the primary symbol.
    seed : int
        Random seed for reproducibility.
    initial_capital : float
        Starting account equity.

    Returns
    -------
    OrchestratorBacktestResult
        Contains the full BacktestResults and the config used.
    """
    cfg = config or OrchestratorConfig()

    # Generate data
    data = generate_synthetic_dataset(
        n_days=n_days,
        symbol=symbol,
        base_price=base_price,
        seed=seed,
    )

    # All symbols in data
    symbols = list(data.keys())

    # Build backtest config
    bt_config = BacktestConfig(
        strategy_name="Orchestrator",
        strategy_version="2.0.0",
        symbols=symbols,
        initial_capital=initial_capital,
        position_size_pct=cfg.max_position_pct,
        commission_per_share=0.005,
        slippage_pct=0.0005,
        allow_shorting=True,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20,
    )

    engine = BacktestEngine(bt_config)
    orchestrator = Orchestrator(cfg)
    results = engine.run(orchestrator, data)

    return OrchestratorBacktestResult(
        config=cfg,
        backtest_results=results,
    )


def print_results(result: OrchestratorBacktestResult) -> None:
    """Print a human-readable summary of backtest results."""
    r = result.backtest_results
    m = r.metrics

    print("=" * 60)
    print("  ORCHESTRATOR BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Strategy:       {r.config.strategy_name} v{r.config.strategy_version}")
    print(f"  Period:         {m.start_date} → {m.end_date}")
    print(f"  Trading Days:   {m.trading_days}")
    print(f"  Bars Processed: {r.bars_processed}")
    print()
    print("  RETURNS")
    print(f"    Total Return:      {m.total_return_pct:+.2f}%")
    print(f"    Annualized Return: {m.annualized_return:.2f}%")
    print(f"    CAGR:              {m.cagr:.2f}%")
    print()
    print("  RISK")
    print(f"    Sharpe Ratio:  {m.sharpe_ratio:.2f}")
    print(f"    Sortino Ratio: {m.sortino_ratio:.2f}")
    print(f"    Calmar Ratio:  {m.calmar_ratio:.2f}")
    print(f"    Max Drawdown:  {m.max_drawdown_pct:.2f}%")
    print()
    print("  TRADES")
    print(f"    Total Trades:  {m.total_trades}")
    print(f"    Win Rate:      {m.win_rate:.1f}%")
    print(f"    Profit Factor: {m.profit_factor:.2f}")
    print(f"    Avg Trade:     ${m.avg_trade:.2f}")
    print(f"    Expectancy:    ${m.expectancy:.2f}")
    print(f"    Best Trade:    ${m.largest_win:.2f}")
    print(f"    Worst Trade:   ${m.largest_loss:.2f}")
    print()
    print("  TIME")
    print(f"    Avg Bars in Winner: {m.avg_bars_in_winner:.1f}")
    print(f"    Avg Bars in Loser:  {m.avg_bars_in_loser:.1f}")
    print(f"    Time in Market:     {m.time_in_market_pct:.1f}%")
    print()

    if r.errors:
        print(f"  ERRORS: {len(r.errors)}")
        for e in r.errors[:5]:
            print(f"    - {e}")
    if r.warnings:
        print(f"  WARNINGS: {len(r.warnings)}")
        for w in r.warnings[:5]:
            print(f"    - {w}")

    print("=" * 60)
