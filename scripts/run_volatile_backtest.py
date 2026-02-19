#!/usr/bin/env python3
"""
High-Volatility Backtest Suite

Runs the full trading algorithm against highly volatile instruments
(SMCI, NVDA) to stress-test alpha generation in extreme conditions.

SMCI: 102.7% annualised vol, 6.6x SPY, 50 days with >10% moves
NVDA: ~60% annualised vol, 3.8x SPY, AI-driven momentum

Usage:
    .venv/bin/python scripts/run_volatile_backtest.py
"""

from __future__ import annotations

import sys
import time
import logging
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)

from trading_algo.quant_core.data.ibkr_data_loader import (
    load_ibkr_bars, resample_to_daily, get_available_symbols,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"

def fmt_ratio(v: float) -> str:
    return f"{v:+.3f}"

def print_header(title: str) -> None:
    w = 78
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)

def print_metrics(metrics: Dict[str, Any], name: str) -> None:
    tr = metrics.get("total_return", 0)
    ar = metrics.get("annualized_return", 0)
    sr = metrics.get("sharpe_ratio", 0)
    so = metrics.get("sortino_ratio", 0)
    md = metrics.get("max_drawdown", 0)
    vol = metrics.get("volatility", 0)
    nt = metrics.get("total_trades", 0)
    wr = metrics.get("win_rate", 0)

    print(f"\n  {name}")
    print(f"  {'─' * 50}")
    print(f"  {'Total Return:':<25} {fmt_pct(tr):>12}")
    print(f"  {'Annualized Return:':<25} {fmt_pct(ar):>12}")
    print(f"  {'Sharpe Ratio:':<25} {fmt_ratio(sr):>12}")
    print(f"  {'Sortino Ratio:':<25} {fmt_ratio(so):>12}")
    print(f"  {'Max Drawdown:':<25} {fmt_pct(-abs(md)):>12}")
    print(f"  {'Volatility:':<25} {fmt_pct(vol):>12}")
    print(f"  {'Total Trades:':<25} {nt:>12}")
    print(f"  {'Win Rate:':<25} {fmt_pct(wr):>12}")


def compute_buy_hold(closes: NDArray) -> Dict:
    """Buy-and-hold benchmark for comparison."""
    total_ret = (closes[-1] / closes[0]) - 1.0
    daily_rets = np.diff(closes) / closes[:-1]
    n_years = max(len(daily_rets) / 252.0, 1 / 252)
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1.0 if (1 + total_ret) > 0 else -1.0
    vol = float(np.std(daily_rets) * np.sqrt(252))
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
    downside = daily_rets[daily_rets < 0]
    ds_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 1 else vol
    sortino = (ann_ret - 0.02) / ds_vol if ds_vol > 0 else 0
    peak = np.maximum.accumulate(closes)
    dd = (peak - closes) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))
    return {
        "total_return": total_ret, "annualized_return": ann_ret,
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "max_drawdown": max_dd, "volatility": vol,
        "total_trades": 1, "win_rate": 1.0 if total_ret > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Individual strategy backtests (high-vol focused)
# ---------------------------------------------------------------------------

def backtest_regime_transition_volatile(
    data: Dict[str, NDArray], timestamps: List[datetime],
) -> Dict:
    from trading_algo.quant_core.strategies.regime_transition import (
        run_regime_transition_backtest, TransitionConfig,
    )
    # Calibrated for high-vol: more permissive thresholds since regime
    # transitions should be MORE frequent and more profitable in SMCI
    config = TransitionConfig(
        min_signal_strength=0.0005,
        velocity_threshold=0.003,
        transition_threshold=0.15,
    )
    return run_regime_transition_backtest(data, timestamps, config=config)


def backtest_cross_asset_volatile(
    data: Dict[str, NDArray], timestamps: List[datetime],
) -> Dict:
    from trading_algo.quant_core.strategies.cross_asset_divergence import (
        run_cross_asset_backtest, DivergenceConfig,
    )
    close_data = {sym: ohlcv[:, 3] for sym, ohlcv in data.items()}
    available = list(close_data.keys())
    pairs = []
    # Pair volatile stocks against SPY as anchor
    for sym in available:
        if sym != "SPY" and "SPY" in available:
            pairs.append((sym, "SPY"))
    # Also pair volatile stocks against each other
    vol_syms = [s for s in available if s not in ("SPY", "QQQ", "IWM")]
    for i in range(len(vol_syms)):
        for j in range(i + 1, len(vol_syms)):
            pairs.append((vol_syms[i], vol_syms[j]))
    if not pairs and len(available) >= 2:
        pairs = [(available[0], available[1])]

    config = DivergenceConfig(
        asset_pairs=pairs,
        inverse_pairs=[],
        entry_threshold=1.5,  # Lower threshold for more volatile divergences
    )
    return run_cross_asset_backtest(close_data, timestamps, config=config)


def backtest_flow_pressure_volatile(
    data: Dict[str, NDArray], timestamps: List[datetime],
) -> Dict:
    from trading_algo.quant_core.strategies.flow_pressure import (
        run_flow_pressure_backtest,
    )
    return run_flow_pressure_backtest(data, timestamps)


def backtest_liquidity_cycles_volatile(
    ohlcv_5m: NDArray, timestamps_5m: List[datetime], symbol: str,
) -> Dict:
    from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
        run_liquidity_cycle_backtest,
    )
    ts_arr = np.array(timestamps_5m, dtype="datetime64[ns]")
    result = run_liquidity_cycle_backtest(
        timestamps=ts_arr,
        opens=ohlcv_5m[:, 0], highs=ohlcv_5m[:, 1],
        lows=ohlcv_5m[:, 2], closes=ohlcv_5m[:, 3],
        volumes=ohlcv_5m[:, 4], symbol=symbol,
    )
    return {
        "total_return": result.total_return,
        "annualized_return": getattr(result, "annualized_return", 0)
            or getattr(result, "annualised_return", 0),
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "volatility": getattr(result, "volatility", 0),
        "total_trades": getattr(result, "total_trades", 0)
            or getattr(result, "n_trades", 0),
        "win_rate": result.win_rate,
    }


# ---------------------------------------------------------------------------
# Multi-strategy controller backtest (high-vol universe)
# ---------------------------------------------------------------------------

def backtest_multi_strategy_volatile(
    data_5m: Dict[str, NDArray], timestamps_5m: List[datetime],
) -> Dict:
    from trading_algo.multi_strategy.controller import (
        MultiStrategyController, ControllerConfig,
    )
    from trading_algo.multi_strategy.backtest_runner import (
        MultiStrategyBacktestRunner, MultiStrategyBacktestConfig,
    )

    ctrl_config = ControllerConfig(
        enable_regime_adaptation=True,
        regime_blend_factor=0.3,
        enable_vol_management=True,
    )
    controller = MultiStrategyController(ctrl_config)

    registered = []
    adapter_imports = [
        ("OrchestratorStrategyAdapter", "Orchestrator"),
        ("ORBStrategyAdapter", "ORB"),
        ("MomentumStrategyAdapter", "PureMomentum"),
        ("RegimeTransitionAdapter", "RegimeTransition"),
        ("CrossAssetDivergenceAdapter", "CrossAssetDivergence"),
        ("FlowPressureAdapter", "FlowPressure"),
        ("LiquidityCycleAdapter", "LiquidityCycles"),
    ]
    for adapter_name, strategy_name in adapter_imports:
        try:
            mod = __import__(
                "trading_algo.multi_strategy.adapters",
                fromlist=[adapter_name],
            )
            adapter_cls = getattr(mod, adapter_name)
            controller.register(adapter_cls())
            registered.append(strategy_name)
        except Exception as e:
            print(f"  [SKIP] {strategy_name}: {e}")

    print(f"  Registered {len(registered)} strategies: {', '.join(registered)}")

    symbols = list(data_5m.keys())

    class SimpleBar:
        __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
        def __init__(self, ts, o, h, l, c, v):
            self.timestamp = ts; self.open = o; self.high = h
            self.low = l; self.close = c; self.volume = v

    bar_data: Dict[str, List] = {sym: [] for sym in symbols}
    ref_sym = symbols[0]
    n_bars = len(data_5m[ref_sym])

    for sym in symbols:
        ohlcv = data_5m[sym]
        for i in range(n_bars):
            ts = timestamps_5m[i] if i < len(timestamps_5m) else timestamps_5m[-1]
            bar_data[sym].append(
                SimpleBar(ts, ohlcv[i, 0], ohlcv[i, 1], ohlcv[i, 2],
                          ohlcv[i, 3], ohlcv[i, 4])
            )

    bt_config = MultiStrategyBacktestConfig(
        initial_capital=100_000,
        symbols=symbols,
    )
    runner = MultiStrategyBacktestRunner(controller, bt_config)

    def progress(pct, msg):
        print(f"\r  [{pct*100:5.1f}%] {msg:<60}", end="", flush=True)

    results = runner.run(bar_data, progress_callback=progress)
    print()

    return {
        "total_return": results.total_return,
        "annualized_return": results.annualized_return,
        "sharpe_ratio": results.sharpe_ratio,
        "sortino_ratio": results.sortino_ratio,
        "max_drawdown": results.max_drawdown,
        "volatility": results.volatility,
        "total_trades": results.total_trades,
        "win_rate": results.win_rate,
        "strategy_attribution": {
            name: attr.n_signals
            for name, attr in results.strategy_attribution.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    print_header("HIGH-VOLATILITY BACKTEST SUITE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    available = get_available_symbols()
    print(f"  Available symbols: {', '.join(available)}")

    # ──── Volatility Profile ──────────────────────────────────────────────
    print_header("VOLATILITY PROFILE")
    vol_symbols = ["SMCI", "NVDA"]
    anchor_symbols = ["SPY", "QQQ"]
    all_daily = {}
    all_daily_ts = {}

    for sym in vol_symbols + anchor_symbols:
        if sym not in available:
            print(f"  [SKIP] {sym}: no data")
            continue
        try:
            ohlcv, ts = load_ibkr_bars(sym, "2024-01-01", "2026-01-31")
            d, dt = resample_to_daily(ohlcv, ts)
            all_daily[sym] = d
            all_daily_ts[sym] = dt
            closes = d[:, 3]
            rets = np.diff(closes) / closes[:-1]
            ann_vol = np.std(rets) * np.sqrt(252) * 100
            max_move = np.max(np.abs(rets)) * 100
            big_days = np.sum(np.abs(rets) > 0.05)
            print(f"  {sym}: {len(d)} bars, ${closes[0]:.2f}->${closes[-1]:.2f}, "
                  f"vol={ann_vol:.0f}%, max_day={max_move:.0f}%, "
                  f"days>5%={big_days}")
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")

    # Align daily data — use common symbols with sufficient data
    ref_sym = max(all_daily, key=lambda s: len(all_daily[s]))
    ref_n = len(all_daily[ref_sym])
    min_ok = max(ref_n // 2, 100)
    for sym in list(all_daily):
        if len(all_daily[sym]) < min_ok:
            print(f"  [DROP] {sym}: only {len(all_daily[sym])} bars (need {min_ok})")
            del all_daily[sym]
            del all_daily_ts[sym]
    n = min(len(all_daily[s]) for s in all_daily)
    daily_data = {s: all_daily[s][:n] for s in all_daily}
    daily_ts = all_daily_ts[ref_sym][:n]
    print(f"\n  Final daily universe: {', '.join(daily_data.keys())} "
          f"({n} bars, {daily_ts[0].date()} to {daily_ts[-1].date()})")

    # Load 5-min data for multi-strategy
    intraday_syms = [s for s in ["SMCI", "SPY", "QQQ"] if s in available]
    print(f"  Loading 5-min data for: {', '.join(intraday_syms)}")
    data_5m = {}
    ts_5m_all = {}
    for sym in intraday_syms:
        try:
            ohlcv, ts = load_ibkr_bars(sym, "2024-01-01", "2026-01-31")
            data_5m[sym] = ohlcv
            ts_5m_all[sym] = ts
        except Exception as e:
            print(f"  [WARN] {sym} 5m: {e}")

    if data_5m:
        ref = min(data_5m, key=lambda s: len(data_5m[s]))
        n5 = len(data_5m[ref])
        for sym in list(data_5m):
            data_5m[sym] = data_5m[sym][:n5]
        ts_5m = ts_5m_all[ref][:n5]
        print(f"  5-min bars: {n5} bars")

    all_results = {}

    # ──── Buy-and-Hold Benchmarks ─────────────────────────────────────────
    print_header("BUY-AND-HOLD BENCHMARKS")
    for sym in daily_data:
        bh = compute_buy_hold(daily_data[sym][:, 3])
        print_metrics(bh, f"Buy & Hold {sym}")
        all_results[f"BuyHold_{sym}"] = bh

    # ──── Regime Transition on Volatile Universe ──────────────────────────
    print_header("BACKTEST 1: REGIME TRANSITION (HIGH-VOL UNIVERSE)")
    try:
        t0 = time.time()
        result = backtest_regime_transition_volatile(daily_data, daily_ts)
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(result, "Regime Transition (SMCI/NVDA/SPY/QQQ)")
        all_results["RegimeTransition"] = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

    # ──── Cross-Asset Divergence ──────────────────────────────────────────
    print_header("BACKTEST 2: CROSS-ASSET DIVERGENCE (SMCI vs SPY/QQQ)")
    try:
        t0 = time.time()
        result = backtest_cross_asset_volatile(daily_data, daily_ts)
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(result, "Cross-Asset Divergence (SMCI/NVDA vs anchors)")
        all_results["CrossAssetDiv"] = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

    # ──── Flow Pressure ───────────────────────────────────────────────────
    print_header("BACKTEST 3: FLOW PRESSURE (HIGH-VOL)")
    try:
        t0 = time.time()
        result = backtest_flow_pressure_volatile(daily_data, daily_ts)
        print(f"  Completed in {time.time() - t0:.1f}s")
        print_metrics(result, "Flow Pressure (SMCI/NVDA/SPY/QQQ)")
        all_results["FlowPressure"] = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

    # ──── Liquidity Cycles on SMCI 5-min ──────────────────────────────────
    print_header("BACKTEST 4: INTRADAY LIQUIDITY CYCLES (SMCI)")
    if data_5m and "SMCI" in data_5m:
        try:
            t0 = time.time()
            result = backtest_liquidity_cycles_volatile(
                data_5m["SMCI"], ts_5m, "SMCI",
            )
            print(f"  Completed in {time.time() - t0:.1f}s")
            print_metrics(result, "Liquidity Cycles (SMCI 5-min)")
            all_results["LiquidityCycles_SMCI"] = result
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
    else:
        print("  [SKIP] No 5-min SMCI data")

    # ──── Multi-Strategy Controller (High-Vol) ────────────────────────────
    print_header("BACKTEST 5: FULL MULTI-STRATEGY CONTROLLER (HIGH-VOL)")
    if data_5m:
        try:
            t0 = time.time()
            result = backtest_multi_strategy_volatile(data_5m, ts_5m)
            print(f"  Completed in {time.time() - t0:.1f}s")
            print_metrics(result, "Combined Portfolio (SMCI/SPY/QQQ)")
            attr = result.get("strategy_attribution", {})
            if attr:
                print(f"\n  Strategy Signal Attribution:")
                print(f"  {'─' * 50}")
                for name, count in sorted(attr.items(), key=lambda x: -x[1]):
                    print(f"    {name:<30} {count:>6} signals")
            all_results["MultiStrategy"] = result
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    # ──── Summary ─────────────────────────────────────────────────────────
    print_header("PERFORMANCE SUMMARY — HIGH VOLATILITY")

    # Separate benchmarks and strategies
    benchmarks = {k: v for k, v in all_results.items() if k.startswith("BuyHold")}
    strategies = {k: v for k, v in all_results.items() if not k.startswith("BuyHold")}

    if benchmarks:
        print(f"\n  {'BENCHMARKS':<28} {'Return':>9} {'Ann.Ret':>9} {'Sharpe':>8} "
              f"{'MaxDD':>9} {'Vol':>8}")
        print(f"  {'─' * 76}")
        for name, m in benchmarks.items():
            print(f"  {name:<28} {fmt_pct(m['total_return']):>9} "
                  f"{fmt_pct(m['annualized_return']):>9} "
                  f"{fmt_ratio(m['sharpe_ratio']):>8} "
                  f"{fmt_pct(-abs(m['max_drawdown'])):>9} "
                  f"{fmt_pct(m['volatility']):>8}")

    if strategies:
        print(f"\n  {'STRATEGIES':<28} {'Return':>9} {'Ann.Ret':>9} {'Sharpe':>8} "
              f"{'MaxDD':>9} {'Trades':>7} {'WinRate':>8}")
        print(f"  {'─' * 88}")
        indiv_sharpes = []
        for name, m in strategies.items():
            if name == "MultiStrategy":
                continue
            tr = m.get("total_return", 0)
            ar = m.get("annualized_return", 0)
            sr = m.get("sharpe_ratio", 0)
            md = m.get("max_drawdown", 0)
            nt = m.get("total_trades", 0)
            wr = m.get("win_rate", 0)
            indiv_sharpes.append(sr)
            print(f"  {name:<28} {fmt_pct(tr):>9} {fmt_pct(ar):>9} "
                  f"{fmt_ratio(sr):>8} {fmt_pct(-abs(md)):>9} "
                  f"{nt:>7} {fmt_pct(wr):>8}")

        if "MultiStrategy" in strategies:
            m = strategies["MultiStrategy"]
            print(f"  {'─' * 88}")
            print(f"  {'COMBINED PORTFOLIO':<28} {fmt_pct(m['total_return']):>9} "
                  f"{fmt_pct(m['annualized_return']):>9} "
                  f"{fmt_ratio(m['sharpe_ratio']):>8} "
                  f"{fmt_pct(-abs(m['max_drawdown'])):>9} "
                  f"{m['total_trades']:>7} {fmt_pct(m['win_rate']):>8}")

            # Alpha vs buy-and-hold
            if "BuyHold_SMCI" in benchmarks:
                bh_ret = benchmarks["BuyHold_SMCI"]["total_return"]
                algo_ret = m["total_return"]
                alpha = algo_ret - bh_ret
                print(f"\n  Alpha vs SMCI Buy&Hold: {fmt_pct(alpha)}")
            if "BuyHold_SPY" in benchmarks:
                bh_ret = benchmarks["BuyHold_SPY"]["total_return"]
                algo_ret = m["total_return"]
                alpha = algo_ret - bh_ret
                print(f"  Alpha vs SPY Buy&Hold:  {fmt_pct(alpha)}")

    elapsed = time.time() - start_time
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
