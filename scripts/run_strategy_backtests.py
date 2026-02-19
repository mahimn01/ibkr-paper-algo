#!/usr/bin/env python3
"""
Comprehensive Strategy Backtest Suite

Runs all trading strategies (existing + novel) against cached IBKR historical
data and prints a detailed performance report.

Usage:
    .venv/bin/python scripts/run_strategy_backtests.py
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy warnings during backtest
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

from trading_algo.quant_core.data.ibkr_data_loader import (
    load_ibkr_bars,
    resample_to_daily,
    load_universe_data,
    get_available_symbols,
)


def load_daily_data(symbols: List[str]) -> tuple[Dict[str, NDArray], List[datetime]]:
    """Load and resample cached 5-min data to daily bars."""
    all_data = {}
    all_ts = {}
    for sym in symbols:
        try:
            ohlcv_5m, ts_5m = load_ibkr_bars(sym, "2024-01-01", "2026-01-31")
            daily_ohlcv, daily_ts = resample_to_daily(ohlcv_5m, ts_5m)
            all_data[sym] = daily_ohlcv
            all_ts[sym] = daily_ts
            print(f"    {sym}: {len(daily_ohlcv)} daily bars")
        except Exception as e:
            print(f"    [WARN] {sym}: {e}")
    if not all_data:
        return {}, []
    # Use the symbol with the MOST data as reference length, and drop
    # symbols that have less than 50% of the reference (avoids truncating
    # ~500-bar datasets to 20 because of a short fragment).
    ref = max(all_data, key=lambda s: len(all_data[s]))
    ref_n = len(all_data[ref])
    min_acceptable = max(ref_n // 2, 100)
    for sym in list(all_data):
        if len(all_data[sym]) < min_acceptable:
            print(f"    [DROP] {sym}: only {len(all_data[sym])} bars "
                  f"(need {min_acceptable})")
            del all_data[sym]
            del all_ts[sym]
    if not all_data:
        return {}, []
    # Truncate to shortest remaining length
    n = min(len(all_data[s]) for s in all_data)
    for sym in list(all_data):
        all_data[sym] = all_data[sym][:n]
    ref = list(all_data.keys())[0]
    return all_data, all_ts[ref][:n]


def load_5min_data(symbols: List[str]) -> tuple[Dict[str, NDArray], List[datetime]]:
    """Load cached 5-min IBKR data."""
    try:
        return load_universe_data(symbols, "2024-01-01", "2026-01-31")
    except Exception:
        pass
    # Fallback: load individually with explicit date range
    all_data = {}
    all_ts = {}
    for sym in symbols:
        for date_range in [("2024-01-01", "2026-01-31"), ("2024-01-01", "2025-01-01")]:
            try:
                ohlcv, ts = load_ibkr_bars(sym, date_range[0], date_range[1])
                all_data[sym] = ohlcv
                all_ts[sym] = ts
                break
            except Exception:
                continue
    if not all_data:
        return {}, []
    ref = min(all_data, key=lambda s: len(all_data[s]))
    n = len(all_data[ref])
    for sym in list(all_data):
        all_data[sym] = all_data[sym][:n]
    return all_data, all_ts[ref][:n]


# ---------------------------------------------------------------------------
# Metric formatting
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
    """Print a strategy's backtest metrics in a formatted table."""
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


# ---------------------------------------------------------------------------
# 1. Individual Strategy Backtests
# ---------------------------------------------------------------------------

def backtest_regime_transition(data: Dict[str, NDArray], timestamps: List[datetime]) -> Dict:
    """Run Regime Transition strategy backtest."""
    from trading_algo.quant_core.strategies.regime_transition import (
        run_regime_transition_backtest,
        TransitionConfig,
    )
    # Use empirically-calibrated thresholds: default min_signal_strength=0.05
    # is ~12.6% annualised expected return — too high for regime-transition alpha.
    config = TransitionConfig(
        min_signal_strength=0.001,  # ~0.25% annualised
        velocity_threshold=0.005,    # detectable transition velocity
    )
    return run_regime_transition_backtest(data, timestamps, config=config)


def backtest_cross_asset(data: Dict[str, NDArray], timestamps: List[datetime]) -> Dict:
    """Run Cross-Asset Divergence strategy backtest."""
    from trading_algo.quant_core.strategies.cross_asset_divergence import (
        run_cross_asset_backtest,
        DivergenceConfig,
    )
    # Needs dict of symbol -> 1D close prices
    close_data = {}
    for sym, ohlcv in data.items():
        close_data[sym] = ohlcv[:, 3]  # close column
    # Configure pairs from available symbols
    available = list(close_data.keys())
    pairs = []
    if "SPY" in available and "QQQ" in available:
        pairs.append(("SPY", "QQQ"))
    if "SPY" in available and "IWM" in available:
        pairs.append(("SPY", "IWM"))
    if "AAPL" in available and "NVDA" in available:
        pairs.append(("AAPL", "NVDA"))
    if "SPY" in available and "AAPL" in available:
        pairs.append(("SPY", "AAPL"))
    if not pairs:
        pairs = [(available[0], available[1])] if len(available) >= 2 else []
    config = DivergenceConfig(asset_pairs=pairs, inverse_pairs=[])
    return run_cross_asset_backtest(close_data, timestamps, config=config)


def backtest_flow_pressure(data: Dict[str, NDArray], timestamps: List[datetime]) -> Dict:
    """Run Flow Pressure strategy backtest."""
    from trading_algo.quant_core.strategies.flow_pressure import (
        run_flow_pressure_backtest,
    )
    return run_flow_pressure_backtest(data, timestamps)


def backtest_liquidity_cycles(ohlcv_5m: NDArray, timestamps_5m: List[datetime]) -> Dict:
    """Run Liquidity Cycles strategy backtest on 5-min data."""
    from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
        run_liquidity_cycle_backtest,
    )
    ts_arr = np.array(timestamps_5m, dtype="datetime64[ns]")
    result = run_liquidity_cycle_backtest(
        timestamps=ts_arr,
        opens=ohlcv_5m[:, 0],
        highs=ohlcv_5m[:, 1],
        lows=ohlcv_5m[:, 2],
        closes=ohlcv_5m[:, 3],
        volumes=ohlcv_5m[:, 4],
        symbol="SPY",
    )
    # Convert dataclass to dict (handle British/American spelling)
    return {
        "total_return": result.total_return,
        "annualized_return": getattr(result, "annualized_return", 0) or getattr(result, "annualised_return", 0),
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "volatility": getattr(result, "volatility", 0),
        "total_trades": getattr(result, "total_trades", 0) or getattr(result, "n_trades", 0),
        "win_rate": result.win_rate,
    }


# ---------------------------------------------------------------------------
# 2. Multi-Strategy Controller Backtest
# ---------------------------------------------------------------------------

def backtest_multi_strategy(
    data_5m: Dict[str, NDArray],
    timestamps_5m: List[datetime],
) -> Dict:
    """Run the full multi-strategy controller backtest."""
    from trading_algo.multi_strategy.controller import (
        MultiStrategyController,
        ControllerConfig,
    )
    from trading_algo.multi_strategy.backtest_runner import (
        MultiStrategyBacktestRunner,
        MultiStrategyBacktestConfig,
    )

    # Set up controller with default (updated) allocations
    ctrl_config = ControllerConfig(
        enable_regime_adaptation=True,
        regime_blend_factor=0.3,
        enable_vol_management=True,
    )
    controller = MultiStrategyController(ctrl_config)

    # Register available adapters
    registered = []

    # Orchestrator
    try:
        from trading_algo.multi_strategy.adapters import OrchestratorStrategyAdapter
        controller.register(OrchestratorStrategyAdapter())
        registered.append("Orchestrator")
    except Exception as e:
        print(f"  [SKIP] Orchestrator: {e}")

    # ORB
    try:
        from trading_algo.multi_strategy.adapters import ORBStrategyAdapter
        controller.register(ORBStrategyAdapter())
        registered.append("ORB")
    except Exception as e:
        print(f"  [SKIP] ORB: {e}")

    # Pairs Trading (skip - known array size bug with 5-min data)
    # try:
    #     from trading_algo.multi_strategy.adapters import PairsStrategyAdapter
    #     controller.register(PairsStrategyAdapter())
    #     registered.append("PairsTrading")
    # except Exception as e:
    #     print(f"  [SKIP] PairsTrading: {e}")

    # Pure Momentum
    try:
        from trading_algo.multi_strategy.adapters import MomentumStrategyAdapter
        controller.register(MomentumStrategyAdapter())
        registered.append("PureMomentum")
    except Exception as e:
        print(f"  [SKIP] PureMomentum: {e}")

    # New strategies
    try:
        from trading_algo.multi_strategy.adapters import RegimeTransitionAdapter
        controller.register(RegimeTransitionAdapter())
        registered.append("RegimeTransition")
    except Exception as e:
        print(f"  [SKIP] RegimeTransition: {e}")

    try:
        from trading_algo.multi_strategy.adapters import CrossAssetDivergenceAdapter
        controller.register(CrossAssetDivergenceAdapter())
        registered.append("CrossAssetDivergence")
    except Exception as e:
        print(f"  [SKIP] CrossAssetDivergence: {e}")

    try:
        from trading_algo.multi_strategy.adapters import FlowPressureAdapter
        controller.register(FlowPressureAdapter())
        registered.append("FlowPressure")
    except Exception as e:
        print(f"  [SKIP] FlowPressure: {e}")

    try:
        from trading_algo.multi_strategy.adapters import LiquidityCycleAdapter
        controller.register(LiquidityCycleAdapter())
        registered.append("LiquidityCycles")
    except Exception as e:
        print(f"  [SKIP] LiquidityCycles: {e}")

    print(f"  Registered {len(registered)} strategies: {', '.join(registered)}")

    # Build bar objects for the runner
    symbols = list(data_5m.keys())

    class SimpleBar:
        def __init__(self, ts, o, h, l, c, v):
            self.timestamp = ts
            self.open = o
            self.high = h
            self.low = l
            self.close = c
            self.volume = v

    bar_data: Dict[str, List] = {sym: [] for sym in symbols}
    ref_sym = symbols[0]
    n_bars = len(data_5m[ref_sym])

    for sym in symbols:
        ohlcv = data_5m[sym]
        for i in range(n_bars):
            ts = timestamps_5m[i] if i < len(timestamps_5m) else timestamps_5m[-1]
            bar_data[sym].append(SimpleBar(
                ts, ohlcv[i, 0], ohlcv[i, 1], ohlcv[i, 2], ohlcv[i, 3], ohlcv[i, 4]
            ))

    bt_config = MultiStrategyBacktestConfig(
        initial_capital=100_000,
        symbols=symbols,
    )
    runner = MultiStrategyBacktestRunner(controller, bt_config)

    def progress(pct, msg):
        print(f"\r  [{pct*100:5.1f}%] {msg:<60}", end="", flush=True)

    results = runner.run(bar_data, progress_callback=progress)
    print()  # newline after progress

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

    print_header("TRADING ALGORITHM BACKTEST SUITE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")

    # Check available data
    available = get_available_symbols()
    print(f"  Available symbols: {', '.join(available)}")

    # ─────────────────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────────────────
    print_header("LOADING DATA")

    daily_symbols = [s for s in ["SPY", "QQQ", "IWM", "AAPL", "NVDA"] if s in available]
    print(f"  Loading daily data for: {', '.join(daily_symbols)}")
    daily_data, daily_ts = load_daily_data(daily_symbols)
    if daily_data:
        ref = list(daily_data.keys())[0]
        print(f"  Daily bars: {len(daily_data[ref])} days, "
              f"{daily_ts[0].date()} to {daily_ts[-1].date()}")

    intraday_symbols = [s for s in ["SPY", "QQQ", "AAPL"] if s in available]
    print(f"  Loading 5-min data for: {', '.join(intraday_symbols)}")
    data_5m, ts_5m = load_5min_data(intraday_symbols)
    if data_5m:
        ref = list(data_5m.keys())[0]
        print(f"  5-min bars: {len(data_5m[ref])} bars, "
              f"{ts_5m[0].date()} to {ts_5m[-1].date()}")

    # ─────────────────────────────────────────────────────────────────────
    # Individual strategy backtests
    # ─────────────────────────────────────────────────────────────────────
    all_results = {}

    # --- Regime Transition ---
    print_header("BACKTEST 1: REGIME TRANSITION STRATEGY")
    try:
        t0 = time.time()
        result = backtest_regime_transition(daily_data, daily_ts)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        print_metrics(result, "Regime Transition (SPY/QQQ/IWM)")
        all_results["RegimeTransition"] = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

    # --- Cross-Asset Divergence ---
    print_header("BACKTEST 2: CROSS-ASSET DIVERGENCE STRATEGY")
    try:
        t0 = time.time()
        result = backtest_cross_asset(daily_data, daily_ts)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        print_metrics(result, "Cross-Asset Divergence")
        all_results["CrossAssetDivergence"] = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

    # --- Flow Pressure ---
    print_header("BACKTEST 3: FLOW PRESSURE STRATEGY")
    try:
        t0 = time.time()
        result = backtest_flow_pressure(daily_data, daily_ts)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        print_metrics(result, "Flow Pressure (Turn-of-Month, Quarter-End, Expiry)")
        all_results["FlowPressure"] = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()

    # --- Liquidity Cycles (5-min) ---
    print_header("BACKTEST 4: INTRADAY LIQUIDITY CYCLES")
    if data_5m and "SPY" in data_5m:
        try:
            t0 = time.time()
            result = backtest_liquidity_cycles(data_5m["SPY"], ts_5m)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            print_metrics(result, "Liquidity Cycles (SPY 5-min)")
            all_results["LiquidityCycles"] = result
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
    else:
        print("  [SKIP] No 5-min SPY data available")

    # ─────────────────────────────────────────────────────────────────────
    # Multi-strategy controller backtest
    # ─────────────────────────────────────────────────────────────────────
    print_header("BACKTEST 5: FULL MULTI-STRATEGY CONTROLLER")
    if data_5m:
        try:
            t0 = time.time()
            result = backtest_multi_strategy(data_5m, ts_5m)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            print_metrics(result, "Combined Multi-Strategy Portfolio")

            # Strategy attribution
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
    else:
        print("  [SKIP] No 5-min data available")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    print_header("PERFORMANCE SUMMARY")

    if all_results:
        # Header
        print(f"\n  {'Strategy':<28} {'Return':>9} {'Ann.Ret':>9} {'Sharpe':>8} "
              f"{'MaxDD':>9} {'Trades':>7} {'WinRate':>8}")
        print(f"  {'─' * 88}")

        sharpes = []
        for name, m in all_results.items():
            if name == "MultiStrategy":
                continue
            tr = m.get("total_return", 0)
            ar = m.get("annualized_return", 0)
            sr = m.get("sharpe_ratio", 0)
            md = m.get("max_drawdown", 0)
            nt = m.get("total_trades", 0)
            wr = m.get("win_rate", 0)
            sharpes.append(sr)
            print(f"  {name:<28} {fmt_pct(tr):>9} {fmt_pct(ar):>9} {fmt_ratio(sr):>8} "
                  f"{fmt_pct(-abs(md)):>9} {nt:>7} {fmt_pct(wr):>8}")

        # Multi-strategy line
        if "MultiStrategy" in all_results:
            m = all_results["MultiStrategy"]
            tr = m.get("total_return", 0)
            ar = m.get("annualized_return", 0)
            sr = m.get("sharpe_ratio", 0)
            md = m.get("max_drawdown", 0)
            nt = m.get("total_trades", 0)
            wr = m.get("win_rate", 0)
            print(f"  {'─' * 88}")
            print(f"  {'COMBINED PORTFOLIO':<28} {fmt_pct(tr):>9} {fmt_pct(ar):>9} {fmt_ratio(sr):>8} "
                  f"{fmt_pct(-abs(md)):>9} {nt:>7} {fmt_pct(wr):>8}")

            # Diversification benefit
            if sharpes:
                avg_sharpe = np.mean(sharpes)
                if avg_sharpe != 0:
                    div_ratio = sr / avg_sharpe
                    print(f"\n  Diversification Ratio: {div_ratio:.2f}x "
                          f"(Combined Sharpe / Avg Individual Sharpe)")

    elapsed_total = time.time() - start_time
    print(f"\n  Total runtime: {elapsed_total:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
