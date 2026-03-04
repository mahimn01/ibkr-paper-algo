#!/usr/bin/env python3
"""
Phantom Alpha Engine — Deep Analysis & Stress Testing

Comprehensive robustness analysis of the 9-edge system:
  1. Per-edge solo performance (isolate each edge's contribution)
  2. Edge correlation matrix (verify diversification)
  3. Crisis period stress tests (May 2021, Luna, FTX, 2022 bear)
  4. Rolling Sharpe stability (is alpha decaying?)
  5. Drawdown deep-dive (duration, recovery, worst sequences)
  6. Daily return distribution (tail risk, skewness)
  7. Capital sensitivity ($1K, $10K, $30K, $100K)
  8. Walk-forward analysis (train on first half, test on second)
  9. 4-edge vs 9-edge comparison in each regime
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DeepAnalysis")

CACHE_DIR = project_root / "crypto_data_cache"


# ── Reuse data loaders from 9-edge script ──────────────────────

from crypto_alpha.scripts.run_9edge_backtest import (
    download_ohlcv_binance,
    download_funding_rates_binance,
    download_open_interest_binance,
    build_crypto_bars,
)


def load_all_data(start="2020-10-01", end="2026-03-01"):
    """Load all data (uses cache)."""
    symbols_perp = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    symbols_spot = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    symbol_names = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    all_data = {}
    for perp_sym, spot_sym, name in zip(symbols_perp, symbols_spot, symbol_names):
        perp_ts, perp_ohlcv = download_ohlcv_binance(perp_sym, "swap", "1h", start, end)
        spot_ts, spot_ohlcv = download_ohlcv_binance(spot_sym, "spot", "1h", start, end)
        fund_ts, fund_rates = download_funding_rates_binance(perp_sym, start, end)
        oi_ts, oi_values = download_open_interest_binance(perp_sym, "1h", start, end)
        bars = build_crypto_bars(perp_ts, perp_ohlcv, spot_ts, spot_ohlcv,
                                 fund_ts, fund_rates, oi_ts, oi_values)
        all_data[name] = bars

    return all_data


def build_9edge_controller():
    """Build the full 9-edge controller."""
    from trading_algo.multi_strategy.controller import (
        ControllerConfig, MultiStrategyController, StrategyAllocation,
    )
    from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
    from crypto_alpha.adapters.frm_adapter import FRMAdapter
    from crypto_alpha.adapters.radl_adapter import RADLAdapter
    from crypto_alpha.adapters.imc_adapter import IMCAdapter
    from crypto_alpha.adapters.lcp_adapter import LCPAdapter
    from crypto_alpha.adapters.ced_adapter import CEDAdapter
    from crypto_alpha.adapters.vts_adapter import VTSAdapter
    from crypto_alpha.adapters.vf_adapter import VFAdapter
    from crypto_alpha.adapters.vv_adapter import VVAdapter

    config = ControllerConfig(
        allocations={
            "PerpBasisMeanReversion":    StrategyAllocation(weight=0.14, max_positions=6),
            "FundingRateMomentum":       StrategyAllocation(weight=0.13, max_positions=6),
            "RegimeAdaptiveLeverage":    StrategyAllocation(weight=0.11, max_positions=6),
            "IntermarketCascade":        StrategyAllocation(weight=0.11, max_positions=6),
            "LiquidationCascade":        StrategyAllocation(weight=0.12, max_positions=6),
            "CrossExchangeDivergence":   StrategyAllocation(weight=0.09, max_positions=6),
            "VolTermStructure":          StrategyAllocation(weight=0.11, max_positions=6),
            "VolumeFlowDetector":        StrategyAllocation(weight=0.10, max_positions=6),
            "VolumeVelocityBreakout":    StrategyAllocation(weight=0.09, max_positions=6),
        },
        max_gross_exposure=3.0,
        max_net_exposure=2.0,
        max_single_symbol_weight=0.40,
        max_portfolio_positions=20,
        conflict_resolution="weighted_confidence",
        enable_vol_management=True,
        vol_target=0.30,
        vol_lookback=20,
        vol_scale_min=0.20,
        vol_scale_max=3.0,
        max_drawdown=0.25,
        daily_loss_limit=0.05,
        enable_entropy_filter=False,
    )

    controller = MultiStrategyController(config)
    controller.register(PBMRAdapter(base_weight=0.14))
    controller.register(FRMAdapter(base_weight=0.13))
    controller.register(RADLAdapter(base_weight=0.11))
    controller.register(IMCAdapter(base_weight=0.11))
    controller.register(LCPAdapter(base_weight=0.12))
    controller.register(CEDAdapter(base_weight=0.09))
    controller.register(VTSAdapter(base_weight=0.11))
    controller.register(VFAdapter(base_weight=0.10))
    controller.register(VVAdapter(base_weight=0.09))
    return controller


def build_4edge_controller():
    """Build the original 4-edge controller for comparison."""
    from trading_algo.multi_strategy.controller import (
        ControllerConfig, MultiStrategyController, StrategyAllocation,
    )
    from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
    from crypto_alpha.adapters.frm_adapter import FRMAdapter
    from crypto_alpha.adapters.radl_adapter import RADLAdapter
    from crypto_alpha.adapters.imc_adapter import IMCAdapter

    config = ControllerConfig(
        allocations={
            "PerpBasisMeanReversion":    StrategyAllocation(weight=0.25, max_positions=6),
            "FundingRateMomentum":       StrategyAllocation(weight=0.25, max_positions=6),
            "RegimeAdaptiveLeverage":    StrategyAllocation(weight=0.25, max_positions=6),
            "IntermarketCascade":        StrategyAllocation(weight=0.25, max_positions=6),
        },
        max_gross_exposure=3.0,
        max_net_exposure=2.0,
        max_single_symbol_weight=0.40,
        max_portfolio_positions=20,
        conflict_resolution="weighted_confidence",
        enable_vol_management=True,
        vol_target=0.30,
        vol_lookback=20,
        vol_scale_min=0.20,
        vol_scale_max=3.0,
        max_drawdown=0.25,
        daily_loss_limit=0.05,
        enable_entropy_filter=False,
    )

    controller = MultiStrategyController(config)
    controller.register(PBMRAdapter(base_weight=0.15))
    controller.register(FRMAdapter(base_weight=0.15))
    controller.register(RADLAdapter(base_weight=0.12))
    controller.register(IMCAdapter(base_weight=0.12))
    return controller


def run_backtest(controller, data, initial_capital=10_000.0):
    """Run a backtest and return results.

    Uses natural equity compounding (position sizes scale with equity).
    Returns are stationary percentage returns — Sharpe is meaningful.
    """
    from crypto_alpha.backtest.crypto_runner import CryptoBacktestConfig, CryptoBacktestRunner

    symbol_names = list(data.keys())

    bt_config = CryptoBacktestConfig(
        initial_capital=initial_capital,
        symbols=symbol_names,
        commission_bps_maker=2.0,
        commission_bps_taker=5.0,
        slippage_bps=5.0,
        max_leverage=3.0,
        max_position_pct=0.25,
        max_gross_exposure=2.0,
        signal_interval_bars=24,
        funding_interval_hours=8,
    )

    runner = CryptoBacktestRunner(controller, bt_config)
    results = runner.run(data)
    return results


def slice_data(all_data, start_date, end_date):
    """Slice data to a date range."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    sliced = {}
    for sym, bars in all_data.items():
        filtered = [b for b in bars if start_dt <= b.timestamp <= end_dt]
        if filtered:
            sliced[sym] = filtered
    return sliced


def compute_sharpe(daily_returns, rf_annual=0.045):
    """Compute annualized Sharpe from daily returns."""
    dr = np.array(daily_returns)
    if len(dr) < 2 or np.std(dr, ddof=1) < 1e-10:
        return 0.0
    daily_rf = (1 + rf_annual) ** (1 / 365) - 1
    excess = dr - daily_rf
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(365))


def compute_sortino(daily_returns, rf_annual=0.045):
    """Compute annualized Sortino from daily returns."""
    dr = np.array(daily_returns)
    if len(dr) < 2:
        return 0.0
    daily_rf = (1 + rf_annual) ** (1 / 365) - 1
    excess = dr - daily_rf
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd < 1e-10:
        return 99.0  # No downside
    return float(np.mean(excess) * np.sqrt(365) / dd)


def compute_max_dd(equity_curve):
    """Compute max drawdown from equity curve."""
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (peak - ec) / np.where(peak > 0, peak, 1)
    return float(np.max(dd))


# ══════════════════════════════════════════════════════════════════
#  ANALYSIS MODULES
# ══════════════════════════════════════════════════════════════════

def analysis_1_crisis_periods(all_data):
    """Run the 9-edge AND 4-edge systems through every major crypto crisis."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 1: CRISIS PERIOD STRESS TESTS (9-Edge vs 4-Edge)")
    print("=" * 80)

    crises = [
        ("2021 Bull Run Peak",     "2021-01-01", "2021-05-01", "BTC +100%"),
        ("May 2021 Crash",         "2021-05-01", "2021-07-20", "BTC -55%"),
        ("Recovery Rally",         "2021-07-20", "2021-11-10", "BTC +100%"),
        ("Luna/UST Collapse",      "2022-04-01", "2022-06-30", "BTC -45%"),
        ("FTX Collapse",           "2022-10-01", "2022-12-31", "BTC -18%"),
        ("Full 2022 Bear",         "2022-01-01", "2023-01-01", "BTC -65%"),
        ("2023 Recovery",          "2023-01-01", "2024-01-01", "BTC +157%"),
        ("2024 ETF Rally",         "2024-01-01", "2025-01-01", "BTC +118%"),
        ("2025 Maturation",        "2025-01-01", "2026-03-01", "BTC mixed"),
    ]

    print(f"\n  {'Period':.<28} {'Context':.<14} {'9E SR':>7} {'4E SR':>7} {'9E Ret':>9} {'4E Ret':>9} {'9E DD':>7} {'4E DD':>7} {'9E Win':>7}")
    print("  " + "-" * 110)

    for name, start, end, context in crises:
        sliced = slice_data(all_data, start, end)
        if not sliced or max(len(v) for v in sliced.values()) < 48:
            print(f"  {name:.<28} {context:.<14}  [insufficient data]")
            continue

        # 9-edge
        c9 = build_9edge_controller()
        r9 = run_backtest(c9, sliced, 10_000.0)

        # 4-edge
        c4 = build_4edge_controller()
        r4 = run_backtest(c4, sliced, 10_000.0)

        sr9 = r9.sharpe_ratio
        sr4 = r4.sharpe_ratio
        ret9 = r9.total_return
        ret4 = r4.total_return
        dd9 = r9.max_drawdown
        dd4 = r4.max_drawdown
        wr9 = r9.win_rate

        print(f"  {name:.<28} {context:.<14} {sr9:>+7.2f} {sr4:>+7.2f} {ret9:>+8.1%} {ret4:>+8.1%} {dd9:>6.1%} {dd4:>6.1%} {wr9:>6.1%}")

    print()


def analysis_2_rolling_sharpe(all_data):
    """Compute rolling 90-day Sharpe to check alpha stability."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 2: ROLLING SHARPE STABILITY (Is alpha decaying?)")
    print("=" * 80)

    c9 = build_9edge_controller()
    r9 = run_backtest(c9, all_data, 10_000.0)
    dr = np.array(r9.daily_returns)

    if len(dr) < 180:
        print("  Not enough data for rolling analysis")
        return

    windows = [90, 180, 365]
    for w in windows:
        rolling_srs = []
        for i in range(w, len(dr)):
            window_dr = dr[i - w:i]
            sr = compute_sharpe(window_dr)
            rolling_srs.append(sr)

        rolling_srs = np.array(rolling_srs)
        n_quarters = len(rolling_srs) // 90

        print(f"\n  Rolling {w}-day Sharpe ({len(rolling_srs)} observations):")
        print(f"    Mean: {np.mean(rolling_srs):>+7.3f}  |  Median: {np.median(rolling_srs):>+7.3f}")
        print(f"    Min:  {np.min(rolling_srs):>+7.3f}  |  Max:    {np.max(rolling_srs):>+7.3f}")
        print(f"    Std:  {np.std(rolling_srs):>7.3f}  |  % > 0:  {np.mean(rolling_srs > 0):>6.1%}")
        print(f"    % > 1: {np.mean(rolling_srs > 1):>5.1%}  |  % > 2:  {np.mean(rolling_srs > 2):>6.1%}  |  % > 3:  {np.mean(rolling_srs > 3):>6.1%}")

        # Trend: compare first half vs second half
        mid = len(rolling_srs) // 2
        first_half = np.mean(rolling_srs[:mid])
        second_half = np.mean(rolling_srs[mid:])
        trend = "IMPROVING" if second_half > first_half else "DECLINING"
        print(f"    1st half avg: {first_half:>+7.3f}  |  2nd half avg: {second_half:>+7.3f}  →  {trend}")

        # Quarterly breakdown
        if n_quarters >= 2:
            print(f"    Quarterly SR: ", end="")
            for q in range(min(n_quarters, 20)):
                q_sr = compute_sharpe(dr[q * 90:(q + 1) * 90])
                print(f"Q{q+1}={q_sr:>+.1f} ", end="")
            print()


def analysis_3_drawdown_deepdive(all_data):
    """Deep analysis of all drawdowns."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 3: DRAWDOWN DEEP DIVE")
    print("=" * 80)

    c9 = build_9edge_controller()
    r9 = run_backtest(c9, all_data, 10_000.0)

    ec = np.array(r9.equity_curve)
    dr = np.array(r9.daily_returns)

    peak = np.maximum.accumulate(ec)
    dd = (peak - ec) / np.where(peak > 0, peak, 1)

    # Find all drawdown episodes > 1%
    in_dd = False
    dd_start = 0
    episodes = []

    for i in range(len(dd)):
        if dd[i] > 0.01 and not in_dd:
            dd_start = i
            in_dd = True
        elif dd[i] == 0 and in_dd:
            max_dd_in_episode = float(np.max(dd[dd_start:i]))
            duration = i - dd_start
            episodes.append((dd_start, i, max_dd_in_episode, duration))
            in_dd = False

    if in_dd:  # Still in drawdown at end
        max_dd_in_episode = float(np.max(dd[dd_start:]))
        episodes.append((dd_start, len(dd) - 1, max_dd_in_episode, len(dd) - dd_start))

    # Sort by severity
    episodes.sort(key=lambda x: -x[2])

    print(f"\n  Total drawdown episodes (>1%): {len(episodes)}")
    print(f"  Max drawdown: {r9.max_drawdown:.2%}")
    print(f"  Max DD duration: {r9.max_drawdown_duration_days} days")

    print(f"\n  Top 10 Drawdown Episodes:")
    print(f"  {'#':>3} {'Depth':>7} {'Duration':>9} {'Start Day':>10} {'Recovery':>10}")
    print(f"  " + "-" * 50)
    for rank, (start, end, depth, dur) in enumerate(episodes[:10], 1):
        recovery = end - start - int(np.argmax(dd[start:end+1]))
        print(f"  {rank:>3} {depth:>6.2%} {dur:>7}d {start:>10} {recovery:>8}d")

    # Consecutive losing days
    losing_streaks = []
    current_streak = 0
    for d in dr:
        if d < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        losing_streaks.append(current_streak)

    if losing_streaks:
        print(f"\n  Losing Streaks:")
        print(f"    Max consecutive losing days: {max(losing_streaks)}")
        print(f"    Mean losing streak: {np.mean(losing_streaks):.1f} days")
        print(f"    Total losing streaks: {len(losing_streaks)}")
        print(f"    Streaks > 3 days: {sum(1 for s in losing_streaks if s > 3)}")
        print(f"    Streaks > 5 days: {sum(1 for s in losing_streaks if s > 5)}")
        print(f"    Streaks > 7 days: {sum(1 for s in losing_streaks if s > 7)}")


def analysis_4_return_distribution(all_data):
    """Analyze daily return distribution and tail risk."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 4: DAILY RETURN DISTRIBUTION & TAIL RISK")
    print("=" * 80)

    c9 = build_9edge_controller()
    r9 = run_backtest(c9, all_data, 10_000.0)
    dr = np.array(r9.daily_returns)

    from scipy import stats as sp_stats

    print(f"\n  Observations: {len(dr)} daily returns")
    print(f"\n  Central Tendency:")
    print(f"    Mean:   {np.mean(dr):>+10.4%}  ({np.mean(dr)*365:>+.1%} annualized)")
    print(f"    Median: {np.median(dr):>+10.4%}")
    print(f"    Std:    {np.std(dr, ddof=1):>10.4%}  ({np.std(dr, ddof=1)*np.sqrt(365):>.1%} annualized)")

    print(f"\n  Tail Risk:")
    print(f"    Skewness:     {sp_stats.skew(dr):>+8.3f}  {'(positive = right tail = good)' if sp_stats.skew(dr) > 0 else '(negative = left tail = concerning)'}")
    print(f"    Kurtosis:     {sp_stats.kurtosis(dr):>+8.3f}  {'(leptokurtic = fat tails)' if sp_stats.kurtosis(dr) > 0 else '(platykurtic)'}")

    percentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
    print(f"\n  Percentile Distribution:")
    for p in percentiles:
        val = np.percentile(dr, p)
        print(f"    P{p:>5.1f}:  {val:>+10.4%}")

    print(f"\n  Extreme Days:")
    print(f"    Worst day:    {np.min(dr):>+10.4%}")
    print(f"    Best day:     {np.max(dr):>+10.4%}")
    print(f"    Days < -3%:   {np.sum(dr < -0.03):>4} ({np.mean(dr < -0.03):.2%})")
    print(f"    Days < -5%:   {np.sum(dr < -0.05):>4} ({np.mean(dr < -0.05):.2%})")
    print(f"    Days > +3%:   {np.sum(dr > 0.03):>4} ({np.mean(dr > 0.03):.2%})")
    print(f"    Days > +5%:   {np.sum(dr > 0.05):>4} ({np.mean(dr > 0.05):.2%})")

    # Win/loss asymmetry
    wins = dr[dr > 0]
    losses = dr[dr < 0]
    print(f"\n  Win/Loss Asymmetry:")
    print(f"    Positive days:  {len(wins):>5} ({len(wins)/len(dr):.1%})")
    print(f"    Negative days:  {len(losses):>5} ({len(losses)/len(dr):.1%})")
    print(f"    Avg win:        {np.mean(wins):>+10.4%}")
    print(f"    Avg loss:       {np.mean(losses):>+10.4%}")
    print(f"    Win/Loss ratio: {abs(np.mean(wins)/np.mean(losses)):>8.2f}x")
    print(f"    Expectancy:     {np.mean(dr):>+10.4%} per day")

    # VaR analysis
    print(f"\n  Value at Risk:")
    for conf in [90, 95, 99]:
        var = float(-np.percentile(dr, 100 - conf))
        tail = dr[dr <= np.percentile(dr, 100 - conf)]
        cvar = float(-np.mean(tail)) if len(tail) > 0 else var
        print(f"    VaR {conf}%:  {var:>8.4%}  |  CVaR {conf}%:  {cvar:>8.4%}")


def analysis_5_capital_sensitivity(all_data):
    """Test different starting capitals to check sizing effects."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 5: CAPITAL SENSITIVITY ANALYSIS")
    print("=" * 80)

    capitals = [1_000, 5_000, 10_000, 30_000, 100_000]

    print(f"\n  {'Capital':>10} {'SR':>7} {'Return':>9} {'Max DD':>7} {'Trades':>7} {'WR':>6} {'Sortino':>8} {'Calmar':>8} {'Final Equity':>14}")
    print(f"  " + "-" * 90)

    for cap in capitals:
        # Fixed sizing (cap at initial capital to prevent compounding artifacts)
        c9 = build_9edge_controller()
        r9 = run_backtest(c9, all_data, cap)

        print(f"  ${cap:>9,} {r9.sharpe_ratio:>+7.3f} {r9.total_return:>+8.1%} {r9.max_drawdown:>6.2%} "
              f"{r9.total_trades:>7,} {r9.win_rate:>5.1%} {r9.sortino_ratio:>+8.2f} {r9.calmar_ratio:>+8.2f} "
              f"${np.array(r9.equity_curve)[-1]:>13,.0f}")


def analysis_6_walk_forward(all_data):
    """Walk-forward validation: train on first half, test on second."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 6: WALK-FORWARD VALIDATION")
    print("=" * 80)

    # Split into periods
    periods = [
        ("Period 1 (IS)",  "2020-10-01", "2022-04-01"),  # In-sample: bull + crash
        ("Period 2 (OOS)", "2022-04-01", "2023-10-01"),  # Out-of-sample: bear
        ("Period 3 (OOS)", "2023-10-01", "2025-01-01"),  # Out-of-sample: recovery + rally
        ("Period 4 (OOS)", "2025-01-01", "2026-03-01"),  # Out-of-sample: latest
    ]

    print(f"\n  Walk-forward periods (edge params fixed, no re-optimization):")
    print(f"\n  {'Period':.<30} {'SR':>7} {'Return':>9} {'Max DD':>7} {'Trades':>7} {'WR':>6} {'Sortino':>8}")
    print(f"  " + "-" * 80)

    srs_oos = []
    for name, start, end in periods:
        sliced = slice_data(all_data, start, end)
        if not sliced or max(len(v) for v in sliced.values()) < 100:
            print(f"  {name:.<30} [insufficient data]")
            continue

        c9 = build_9edge_controller()
        r9 = run_backtest(c9, sliced, 10_000.0)

        is_oos = "OOS" in name
        if is_oos:
            srs_oos.append(r9.sharpe_ratio)

        marker = " ←IS" if not is_oos else ""
        print(f"  {name:.<30} {r9.sharpe_ratio:>+7.3f} {r9.total_return:>+8.1%} {r9.max_drawdown:>6.2%} "
              f"{r9.total_trades:>7,} {r9.win_rate:>5.1%} {r9.sortino_ratio:>+8.2f}{marker}")

    if srs_oos:
        print(f"\n  OOS Summary:")
        print(f"    Mean OOS SR: {np.mean(srs_oos):>+7.3f}")
        print(f"    Min OOS SR:  {np.min(srs_oos):>+7.3f}")
        print(f"    Max OOS SR:  {np.max(srs_oos):>+7.3f}")
        print(f"    All OOS > 0: {'YES' if all(s > 0 for s in srs_oos) else 'NO'}")
        print(f"    All OOS > 1: {'YES' if all(s > 1 for s in srs_oos) else 'NO'}")


def analysis_7_monthly_consistency(all_data):
    """Analyze month-by-month consistency."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 7: MONTHLY CONSISTENCY & INCOME PROJECTION")
    print("=" * 80)

    c9 = build_9edge_controller()
    r9 = run_backtest(c9, all_data, 10_000.0)
    dr = np.array(r9.daily_returns)

    n_months = len(dr) // 30
    monthly_rets = []
    for m in range(n_months):
        month_dr = dr[m * 30:(m + 1) * 30]
        monthly_rets.append(float(np.prod(1 + month_dr) - 1))

    monthly_rets = np.array(monthly_rets)

    print(f"\n  {n_months} months analyzed:")
    print(f"    Mean monthly:   {np.mean(monthly_rets):>+8.2%}")
    print(f"    Median monthly: {np.median(monthly_rets):>+8.2%}")
    print(f"    Std monthly:    {np.std(monthly_rets, ddof=1):>8.2%}")
    print(f"    Monthly Sharpe: {np.mean(monthly_rets)/np.std(monthly_rets, ddof=1)*np.sqrt(12):>+8.3f} (annualized)")
    print(f"    Best:           {np.max(monthly_rets):>+8.2%}")
    print(f"    Worst:          {np.min(monthly_rets):>+8.2%}")
    print(f"    % Positive:     {np.mean(monthly_rets > 0):>7.1%}")
    print(f"    % > +5%:        {np.mean(monthly_rets > 0.05):>7.1%}")
    print(f"    % > +10%:       {np.mean(monthly_rets > 0.10):>7.1%}")

    # Monthly PnL on $10K
    print(f"\n  Monthly P&L on $10K fixed capital:")
    print(f"  {'Month':>5} {'Return':>8} {'P&L':>9}  |  {'Month':>5} {'Return':>8} {'P&L':>9}  |  {'Month':>5} {'Return':>8} {'P&L':>9}")
    print(f"  " + "-" * 80)

    cols = 3
    rows = (n_months + cols - 1) // cols
    for row in range(min(rows, 25)):
        line = ""
        for col in range(cols):
            idx = row + col * rows
            if idx < n_months:
                ret = monthly_rets[idx]
                pnl = 10_000 * ret
                line += f"  M{idx+1:>3} {ret:>+7.1%} ${pnl:>+8,.0f}  |"
        print(line.rstrip("|"))

    # Income projections at different capital levels
    print(f"\n  Income Projection (using median monthly return of {np.median(monthly_rets):.2%}):")
    med_ret = np.median(monthly_rets)
    for cap in [1_000, 5_000, 10_000, 30_000, 50_000, 100_000]:
        monthly_income = cap * med_ret
        annual_income = monthly_income * 12
        print(f"    ${cap:>8,} → ${monthly_income:>+8,.0f}/mo  (${annual_income:>+10,.0f}/yr)")


def analysis_8_edge_correlation(all_data):
    """Estimate inter-edge signal correlation."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 8: EDGE INDEPENDENCE VERIFICATION")
    print("=" * 80)

    # Run each edge solo and collect daily returns
    from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
    from crypto_alpha.adapters.frm_adapter import FRMAdapter
    from crypto_alpha.adapters.radl_adapter import RADLAdapter
    from crypto_alpha.adapters.imc_adapter import IMCAdapter
    from crypto_alpha.adapters.lcp_adapter import LCPAdapter
    from crypto_alpha.adapters.ced_adapter import CEDAdapter
    from crypto_alpha.adapters.vts_adapter import VTSAdapter
    from crypto_alpha.adapters.vf_adapter import VFAdapter
    from crypto_alpha.adapters.vv_adapter import VVAdapter
    from trading_algo.multi_strategy.controller import (
        ControllerConfig, MultiStrategyController, StrategyAllocation,
    )

    edge_configs = [
        ("PBMR", PBMRAdapter, 0.14),
        ("FRM",  FRMAdapter, 0.13),
        ("RADL", RADLAdapter, 0.11),
        ("IMC",  IMCAdapter, 0.11),
        ("CED",  CEDAdapter, 0.09),
        ("VTS",  VTSAdapter, 0.11),
        ("VF",   VFAdapter, 0.10),
        ("VV",   VVAdapter, 0.09),
    ]
    # Skip LCP — only 5 signals, not enough for correlation

    edge_returns = {}
    edge_srs = {}

    print(f"\n  Running each edge solo...")

    for name, adapter_cls, weight in edge_configs:
        adapter = adapter_cls(base_weight=weight)
        edge_name = adapter.name

        config = ControllerConfig(
            allocations={edge_name: StrategyAllocation(weight=1.0, max_positions=6)},
            max_gross_exposure=3.0, max_net_exposure=2.0,
            max_single_symbol_weight=0.40, max_portfolio_positions=20,
            conflict_resolution="weighted_confidence",
            enable_vol_management=True, vol_target=0.30,
            vol_lookback=20, vol_scale_min=0.20, vol_scale_max=3.0,
            max_drawdown=0.25, daily_loss_limit=0.05,
            enable_entropy_filter=False,
        )

        controller = MultiStrategyController(config)
        controller.register(adapter)

        r = run_backtest(controller, all_data, 10_000.0)
        edge_returns[name] = np.array(r.daily_returns)
        edge_srs[name] = r.sharpe_ratio

        sig_count = sum(a.n_signals for a in r.strategy_attribution.values()) if r.strategy_attribution else 0
        print(f"    {name:.<8} SR={r.sharpe_ratio:>+7.3f}  Ret={r.total_return:>+8.1%}  DD={r.max_drawdown:>5.1%}  Trades={r.total_trades:>5}  Signals={sig_count:>6}")

    # Compute correlation matrix
    names = list(edge_returns.keys())
    n = len(names)
    min_len = min(len(v) for v in edge_returns.values())

    if min_len < 30:
        print("\n  Not enough overlapping returns for correlation analysis")
        return

    ret_matrix = np.column_stack([edge_returns[name][:min_len] for name in names])
    corr = np.corrcoef(ret_matrix.T)

    print(f"\n  Inter-Edge Return Correlation Matrix:")
    header = "         " + "".join(f"{n:>7}" for n in names)
    print(f"  {header}")
    for i, name in enumerate(names):
        row = f"  {name:.<8}"
        for j in range(n):
            val = corr[i, j]
            row += f" {val:>+6.3f}"
        print(row)

    # Average off-diagonal correlation
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(corr[i, j])

    print(f"\n  Average pairwise correlation: {np.mean(off_diag):>+.4f}")
    print(f"  Max pairwise correlation:     {np.max(off_diag):>+.4f}")
    print(f"  Min pairwise correlation:     {np.min(off_diag):>+.4f}")
    print(f"  % pairs with |corr| < 0.1:   {np.mean(np.abs(off_diag) < 0.1):.1%}")
    print(f"  % pairs with |corr| < 0.3:   {np.mean(np.abs(off_diag) < 0.3):.1%}")

    # Theoretical vs actual combined SR
    avg_solo_sr = np.mean(list(edge_srs.values()))
    avg_corr = np.mean(off_diag)
    n_edges = len(names)
    theoretical_sr = np.sqrt(n_edges) * avg_solo_sr / np.sqrt(1 + (n_edges - 1) * max(avg_corr, 0))
    print(f"\n  Solo Edge Stats:")
    print(f"    Avg solo SR:       {avg_solo_sr:>+7.3f}")
    print(f"    Theoretical combined SR: {theoretical_sr:>+7.3f} (using sqrt(N)*avg_SR/sqrt(1+(N-1)*corr))")
    # Run combined to get actual SR
    c_combined = build_9edge_controller()
    r_combined = run_backtest(c_combined, all_data, 10_000.0)
    print(f"    Actual combined SR:      {r_combined.sharpe_ratio:>+7.3f}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("  PHANTOM ALPHA ENGINE — COMPREHENSIVE DEEP ANALYSIS")
    print("  9-Edge System | Real Binance Data | Oct 2020 → Mar 2026")
    print("█" * 80)

    t0 = time.time()

    print("\nLoading data (from cache)...")
    all_data = load_all_data()
    print(f"Loaded {sum(len(v) for v in all_data.values()):,} total bars\n")

    # Run all analyses
    analysis_1_crisis_periods(all_data)
    analysis_2_rolling_sharpe(all_data)
    analysis_3_drawdown_deepdive(all_data)
    analysis_4_return_distribution(all_data)
    analysis_5_capital_sensitivity(all_data)
    analysis_6_walk_forward(all_data)
    analysis_7_monthly_consistency(all_data)
    analysis_8_edge_correlation(all_data)

    elapsed = time.time() - t0
    print(f"\n{'█' * 80}")
    print(f"  DEEP ANALYSIS COMPLETE — {elapsed:.0f}s total")
    print(f"{'█' * 80}\n")
