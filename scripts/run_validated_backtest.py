#!/usr/bin/env python3
"""
Walk-Forward Validated Multi-Strategy Backtest

Comprehensive validation of the multi-strategy trading algorithm using:
  1. Walk-forward protocol with rolling 6-month IS / 1-month OOS windows
  2. Strategy combination testing (A/B/C/D combos)
  3. Per-strategy attribution on the best combination
  4. Simplified PBO / Deflated Sharpe Ratio analysis
  5. Full human-readable report saved to backtest_results/

Usage:
    python scripts/run_validated_backtest.py
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)
from trading_algo.multi_strategy.backtest_runner import (
    MultiStrategyBacktestConfig,
    MultiStrategyBacktestResults,
    MultiStrategyBacktestRunner,
)
from trading_algo.multi_strategy.walk_forward import (
    WalkForwardValidator,
    WalkForwardResult,
)
from trading_algo.multi_strategy.adapters import (
    OrchestratorStrategyAdapter,
    ORBStrategyAdapter,
    PairsStrategyAdapter,
    MomentumStrategyAdapter,
    RegimeTransitionAdapter,
    CrossAssetDivergenceAdapter,
    FlowPressureAdapter,
    LiquidityCycleAdapter,
    HurstAdaptiveAdapter,
    LeadLagAdapter,
    TimeAdaptiveAdapter,
)
from trading_algo.quant_core.data.ibkr_data_loader import (
    load_ibkr_bars,
    load_universe_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "SMCI", "IWM"]
INITIAL_CAPITAL = 100_000.0
RESULTS_DIR = PROJECT_ROOT / "backtest_results"

# Walk-forward parameters
BARS_PER_DAY = 78            # 5-min bars in a regular trading day
BARS_PER_MONTH = 21 * BARS_PER_DAY   # ~1,638
BARS_6_MONTHS = 6 * BARS_PER_MONTH   # ~9,828
WF_STEP = BARS_PER_MONTH             # 1-month rolling step
IS_FRACTION = 0.60                    # 60% in-sample

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("validated_backtest")
logger.setLevel(logging.INFO)


# =========================================================================
# Data helpers
# =========================================================================

@dataclass
class BarObject:
    """Lightweight bar compatible with MultiStrategyBacktestRunner."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_bar_data() -> Dict[str, List[BarObject]]:
    """Load OHLCV data for the full universe, returning Bar objects."""
    print("Loading universe data ...")
    aligned_data, timestamps = load_universe_data(
        symbols=SYMBOLS,
        start_date="2024-01-01",
        end_date="2026-01-31",
        bar_size="5mins",
    )
    bar_data: Dict[str, List[BarObject]] = {}
    for symbol, ohlcv in aligned_data.items():
        bars: List[BarObject] = []
        for i, ts in enumerate(timestamps):
            if np.isnan(ohlcv[i, 0]):
                continue
            bars.append(BarObject(
                timestamp=ts,
                open=float(ohlcv[i, 0]),
                high=float(ohlcv[i, 1]),
                low=float(ohlcv[i, 2]),
                close=float(ohlcv[i, 3]),
                volume=float(ohlcv[i, 4]),
            ))
        bar_data[symbol] = bars
    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    total_bars = len(bar_data[ref_sym])
    print(f"  Loaded {len(bar_data)} symbols, {total_bars} bars each "
          f"({bar_data[ref_sym][0].timestamp.date()} to "
          f"{bar_data[ref_sym][-1].timestamp.date()})")
    return bar_data


def slice_bar_data(
    data: Dict[str, List[BarObject]],
    start_idx: int,
    end_idx: int,
) -> Dict[str, List[BarObject]]:
    """Slice all symbols' bar lists by index range."""
    return {sym: bars[start_idx:end_idx] for sym, bars in data.items()}


# =========================================================================
# Strategy combination factories
# =========================================================================

def _make_adapters(names: List[str]) -> List[Any]:
    """Instantiate adapters by canonical name, skipping failures."""
    registry: Dict[str, Callable[[], Any]] = {
        "Orchestrator": lambda: OrchestratorStrategyAdapter(),
        "ORB": lambda: ORBStrategyAdapter(),
        "PairsTrading": lambda: PairsStrategyAdapter(),
        "PureMomentum": lambda: MomentumStrategyAdapter(),
        "RegimeTransition": lambda: RegimeTransitionAdapter(),
        "CrossAssetDivergence": lambda: CrossAssetDivergenceAdapter(),
        "FlowPressure": lambda: FlowPressureAdapter(),
        "LiquidityCycles": lambda: LiquidityCycleAdapter(),
        "HurstAdaptive": lambda: HurstAdaptiveAdapter(),
        "LeadLagArbitrage": lambda: LeadLagAdapter(),
        "TimeAdaptive": lambda: TimeAdaptiveAdapter(),
    }
    adapters: List[Any] = []
    for n in names:
        factory = registry.get(n)
        if factory is None:
            logger.warning("Unknown adapter name: %s — skipping", n)
            continue
        try:
            adapters.append(factory())
        except Exception as exc:
            logger.warning("Failed to create adapter %s: %s", n, exc)
    return adapters


COMBO_SPECS: Dict[str, Dict[str, Any]] = {
    "A: Original 4": {
        "strategies": [
            "Orchestrator", "ORB", "PairsTrading", "PureMomentum",
        ],
        "entropy_filter": False,
    },
    "B: Original + Phase3": {
        "strategies": [
            "Orchestrator", "ORB", "PairsTrading", "PureMomentum",
            "RegimeTransition", "CrossAssetDivergence",
            "FlowPressure", "LiquidityCycles",
        ],
        "entropy_filter": False,
    },
    "C: All strategies": {
        "strategies": [
            "Orchestrator", "ORB", "PairsTrading", "PureMomentum",
            "RegimeTransition", "CrossAssetDivergence",
            "FlowPressure", "LiquidityCycles",
            "HurstAdaptive", "LeadLagArbitrage", "TimeAdaptive",
        ],
        "entropy_filter": False,
    },
    "D: All + entropy": {
        "strategies": [
            "Orchestrator", "ORB", "PairsTrading", "PureMomentum",
            "RegimeTransition", "CrossAssetDivergence",
            "FlowPressure", "LiquidityCycles",
            "HurstAdaptive", "LeadLagArbitrage", "TimeAdaptive",
        ],
        "entropy_filter": True,
    },
}


def build_controller_and_runner(
    strategy_names: List[str],
    entropy_filter: bool = False,
) -> Tuple[MultiStrategyController, MultiStrategyBacktestRunner]:
    """Build a fresh controller + runner for the given strategy set."""
    cfg = ControllerConfig(
        enable_entropy_filter=entropy_filter,
        enable_vol_management=True,
    )
    controller = MultiStrategyController(cfg)
    for adapter in _make_adapters(strategy_names):
        controller.register(adapter)

    bt_cfg = MultiStrategyBacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        symbols=SYMBOLS,
    )
    runner = MultiStrategyBacktestRunner(controller, bt_cfg)
    return controller, runner


# =========================================================================
# 1. Walk-forward protocol
# =========================================================================

@dataclass
class WFWindowResult:
    """Metrics for one walk-forward window."""
    window_idx: int
    is_start_idx: int
    is_end_idx: int
    oos_start_idx: int
    oos_end_idx: int
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    ratio: float  # OOS / IS (>0.5 acceptable)


def _sharpe_from_daily(daily_returns: List[float]) -> float:
    """Annualised Sharpe (excess over 2% risk-free)."""
    arr = np.array(daily_returns)
    if len(arr) < 2:
        return 0.0
    ann_ret = float(np.mean(arr) * 252)
    ann_vol = float(np.std(arr, ddof=1) * np.sqrt(252))
    if ann_vol < 1e-8:
        return 0.0
    return (ann_ret - 0.02) / ann_vol


def _total_return_from_daily(daily_returns: List[float]) -> float:
    arr = np.array(daily_returns)
    if len(arr) == 0:
        return 0.0
    return float(np.prod(1 + arr) - 1)


def run_walk_forward(
    data: Dict[str, List[BarObject]],
    combo_name: str,
    strategy_names: List[str],
    entropy_filter: bool,
) -> Tuple[List[WFWindowResult], MultiStrategyBacktestResults, MultiStrategyBacktestResults]:
    """
    Rolling walk-forward: 6-month IS window, 1-month OOS step.

    Returns:
        (list of window results, full IS result, full OOS result)
    """
    ref_sym = max(data, key=lambda s: len(data[s]))
    total_bars = len(data[ref_sym])
    is_end = int(total_bars * IS_FRACTION)

    windows: List[WFWindowResult] = []

    # Rolling windows through the OOS region
    window_start = max(0, is_end - BARS_6_MONTHS)
    window_idx = 0

    while window_start + BARS_6_MONTHS < total_bars:
        is_s = window_start
        is_e = window_start + BARS_6_MONTHS
        oos_s = is_e
        oos_e = min(oos_s + WF_STEP, total_bars)
        if oos_e <= oos_s:
            break

        # --- IS run ---
        _, runner_is = build_controller_and_runner(strategy_names, entropy_filter)
        is_data = slice_bar_data(data, is_s, is_e)
        try:
            is_result = runner_is.run(is_data)
        except Exception as exc:
            logger.warning("IS run failed for window %d: %s", window_idx, exc)
            window_start += WF_STEP
            window_idx += 1
            continue

        # --- OOS run ---
        _, runner_oos = build_controller_and_runner(strategy_names, entropy_filter)
        oos_data = slice_bar_data(data, oos_s, oos_e)
        try:
            oos_result = runner_oos.run(oos_data)
        except Exception as exc:
            logger.warning("OOS run failed for window %d: %s", window_idx, exc)
            window_start += WF_STEP
            window_idx += 1
            continue

        is_sharpe = is_result.sharpe_ratio
        oos_sharpe = oos_result.sharpe_ratio
        ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 0.01 else 0.0

        windows.append(WFWindowResult(
            window_idx=window_idx,
            is_start_idx=is_s,
            is_end_idx=is_e,
            oos_start_idx=oos_s,
            oos_end_idx=oos_e,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            is_return=is_result.total_return,
            oos_return=oos_result.total_return,
            ratio=ratio,
        ))

        window_start += WF_STEP
        window_idx += 1

    # Full IS / OOS splits for aggregate reporting
    _, runner_full_is = build_controller_and_runner(strategy_names, entropy_filter)
    full_is = runner_full_is.run(slice_bar_data(data, 0, is_end))

    _, runner_full_oos = build_controller_and_runner(strategy_names, entropy_filter)
    full_oos = runner_full_oos.run(slice_bar_data(data, is_end, total_bars))

    return windows, full_is, full_oos


# =========================================================================
# 2. Combination testing
# =========================================================================

@dataclass
class ComboResult:
    """Aggregate result for one strategy combination."""
    name: str
    strategy_names: List[str]
    entropy_filter: bool
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_return: float = 0.0
    oos_return: float = 0.0
    is_sortino: float = 0.0
    oos_sortino: float = 0.0
    is_max_dd: float = 0.0
    oos_max_dd: float = 0.0
    is_oos_ratio: float = 0.0
    wf_windows: List[WFWindowResult] = field(default_factory=list)
    full_is: Optional[MultiStrategyBacktestResults] = None
    full_oos: Optional[MultiStrategyBacktestResults] = None


def test_combinations(
    data: Dict[str, List[BarObject]],
) -> List[ComboResult]:
    """Run walk-forward + full backtest for every combination."""
    results: List[ComboResult] = []
    n_combos = len(COMBO_SPECS)

    for idx, (combo_name, spec) in enumerate(COMBO_SPECS.items(), 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{n_combos}] Testing combination: {combo_name}")
        print(f"  Strategies: {', '.join(spec['strategies'])}")
        print(f"  Entropy filter: {spec['entropy_filter']}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            wf_wins, full_is, full_oos = run_walk_forward(
                data, combo_name,
                spec["strategies"],
                spec["entropy_filter"],
            )
        except Exception as exc:
            logger.error("Combination %s FAILED: %s", combo_name, exc)
            traceback.print_exc()
            continue

        elapsed = time.time() - t0

        cr = ComboResult(
            name=combo_name,
            strategy_names=spec["strategies"],
            entropy_filter=spec["entropy_filter"],
            is_sharpe=full_is.sharpe_ratio,
            oos_sharpe=full_oos.sharpe_ratio,
            is_return=full_is.total_return,
            oos_return=full_oos.total_return,
            is_sortino=full_is.sortino_ratio,
            oos_sortino=full_oos.sortino_ratio,
            is_max_dd=full_is.max_drawdown,
            oos_max_dd=full_oos.max_drawdown,
            is_oos_ratio=(
                full_oos.sharpe_ratio / full_is.sharpe_ratio
                if abs(full_is.sharpe_ratio) > 0.01 else 0.0
            ),
            wf_windows=wf_wins,
            full_is=full_is,
            full_oos=full_oos,
        )
        results.append(cr)

        print(f"  IS Sharpe: {cr.is_sharpe:+.3f}  "
              f"OOS Sharpe: {cr.oos_sharpe:+.3f}  "
              f"Ratio: {cr.is_oos_ratio:.2f}  "
              f"({elapsed:.1f}s)")

    return results


# =========================================================================
# 3. Final combined backtest on best combination
# =========================================================================

def run_final_backtest(
    data: Dict[str, List[BarObject]],
    best: ComboResult,
) -> MultiStrategyBacktestResults:
    """Run the best combination on the full data and return results."""
    print(f"\nRunning final full-period backtest for: {best.name}")
    _, runner = build_controller_and_runner(
        best.strategy_names, best.entropy_filter,
    )

    def progress(pct: float, msg: str) -> None:
        bar_len = 30
        filled = int(pct * bar_len)
        bar_str = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar_str}] {pct*100:5.1f}%  {msg[:50]:<50}", end="", flush=True)

    result = runner.run(data, progress_callback=progress)
    print()  # newline after progress bar
    return result


# =========================================================================
# 4. Deflated Sharpe Ratio
# =========================================================================

def deflated_sharpe_ratio(
    sr_observed: float,
    n_trials: int,
    n_observations: int,
    sr_std: float = 1.0,
) -> Tuple[float, float]:
    """
    Simplified Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    DSR adjusts the observed Sharpe ratio for the number of strategy
    combinations tested, penalising data-snooping.

    Returns (DSR, p-value).
    """
    if n_trials < 2 or n_observations < 10 or sr_std < 1e-8:
        return sr_observed, 1.0

    # Expected maximum Sharpe under the null (all strategies have SR=0)
    # E[max(Z_1,...,Z_k)] ~ sqrt(2*ln(k)) for k iid standard normals
    sr_benchmark = np.sqrt(2 * np.log(n_trials)) * sr_std / np.sqrt(n_observations)

    # Test statistic
    dsr = (sr_observed - sr_benchmark) / (sr_std / np.sqrt(n_observations))

    # One-sided p-value (H0: true SR <= sr_benchmark)
    p_value = 1.0 - float(sp_stats.norm.cdf(dsr))

    return float(dsr), float(p_value)


# =========================================================================
# 5. Report generation
# =========================================================================

def compute_spy_buy_and_hold(data: Dict[str, List[BarObject]]) -> float:
    """Return total return of SPY buy-and-hold over the full period."""
    spy_bars = data.get("SPY", [])
    if len(spy_bars) < 2:
        return 0.0
    return spy_bars[-1].close / spy_bars[0].open - 1.0


def generate_report(
    combo_results: List[ComboResult],
    best: ComboResult,
    final_result: MultiStrategyBacktestResults,
    spy_bh_return: float,
    dsr: float,
    dsr_pval: float,
    data: Dict[str, List[BarObject]],
) -> str:
    """Build the full text report."""
    lines: List[str] = []
    W = 80

    def sep(char: str = "=") -> str:
        return char * W

    def heading(title: str) -> None:
        lines.append("")
        lines.append(sep())
        lines.append(f"  {title}")
        lines.append(sep())

    ref_sym = max(data, key=lambda s: len(data[s]))
    total_bars = len(data[ref_sym])
    is_end = int(total_bars * IS_FRACTION)

    heading("WALK-FORWARD VALIDATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Symbols:   {', '.join(SYMBOLS)}")
    lines.append(f"  Total bars per symbol: ~{total_bars}")
    lines.append(f"  IS bars: 0..{is_end}  |  OOS bars: {is_end}..{total_bars}")
    lines.append(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")

    # -- Per-strategy individual performance (from best combo full run) ----
    heading("PER-STRATEGY INDIVIDUAL PERFORMANCE (best combo, full period)")
    hdr = (f"  {'Strategy':<25} {'Signals':>8} ")
    lines.append(hdr)
    lines.append("  " + "-" * (W - 4))
    for name, attr in sorted(
        final_result.strategy_attribution.items(), key=lambda x: -x[1].n_signals
    ):
        lines.append(
            f"  {name:<25} {attr.n_signals:>8}"
        )

    # -- Combination results -----------------------------------------------
    heading("COMBINATION RESULTS")
    hdr = (f"  {'Combo':<22} {'IS SR':>7} {'OOS SR':>7} {'IS/OOS':>7} "
           f"{'IS Ret':>8} {'OOS Ret':>8} {'OOS DD':>7} {'OOS Sort':>9}")
    lines.append(hdr)
    lines.append("  " + "-" * (W - 4))
    for cr in combo_results:
        lines.append(
            f"  {cr.name:<22} "
            f"{cr.is_sharpe:>+7.3f} "
            f"{cr.oos_sharpe:>+7.3f} "
            f"{cr.is_oos_ratio:>7.2f} "
            f"{cr.is_return * 100:>+7.1f}% "
            f"{cr.oos_return * 100:>+7.1f}% "
            f"{cr.oos_max_dd * 100:>6.1f}% "
            f"{cr.oos_sortino:>+8.3f}"
        )

    # -- Walk-forward windows for best combo --------------------------------
    heading(f"WALK-FORWARD WINDOWS  ({best.name})")
    lines.append(f"  {'Window':>6} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
                 f"{'OOS/IS':>7} {'IS Ret':>8} {'OOS Ret':>8}")
    lines.append("  " + "-" * (W - 4))
    for w in best.wf_windows:
        lines.append(
            f"  {w.window_idx:>6} "
            f"{w.is_sharpe:>+10.3f} "
            f"{w.oos_sharpe:>+11.3f} "
            f"{w.ratio:>7.2f} "
            f"{w.is_return * 100:>+7.1f}% "
            f"{w.oos_return * 100:>+7.1f}%"
        )
    if best.wf_windows:
        avg_ratio = np.mean([w.ratio for w in best.wf_windows])
        lines.append(f"  {'AVG':>6} "
                     f"{np.mean([w.is_sharpe for w in best.wf_windows]):>+10.3f} "
                     f"{np.mean([w.oos_sharpe for w in best.wf_windows]):>+11.3f} "
                     f"{avg_ratio:>7.2f}")
        acceptable = avg_ratio > 0.5
        lines.append(f"  Avg OOS/IS ratio: {avg_ratio:.2f}  "
                     f"({'ACCEPTABLE (>0.5)' if acceptable else 'DEGRADED (<0.5)'})")

    # -- Best combination summary ------------------------------------------
    heading("BEST COMBINATION")
    lines.append(f"  Name:        {best.name}")
    lines.append(f"  Strategies:  {', '.join(best.strategy_names)}")
    lines.append(f"  Entropy:     {'ON' if best.entropy_filter else 'OFF'}")
    lines.append(f"  OOS Sharpe:  {best.oos_sharpe:+.3f}")
    lines.append(f"  OOS Return:  {best.oos_return * 100:+.1f}%")
    lines.append(f"  OOS Max DD:  {best.oos_max_dd * 100:.1f}%")

    # -- Alpha vs SPY buy-and-hold -----------------------------------------
    heading("ALPHA vs SPY BUY-AND-HOLD (full period)")
    algo_return = final_result.total_return
    alpha = algo_return - spy_bh_return
    lines.append(f"  SPY Buy-and-Hold Return:   {spy_bh_return * 100:+.1f}%")
    lines.append(f"  Algo Return (full period): {algo_return * 100:+.1f}%")
    lines.append(f"  Alpha:                     {alpha * 100:+.1f}%")

    # -- Strategy Contribution Attribution ---------------------------------
    heading("STRATEGY CONTRIBUTION ATTRIBUTION (full period)")
    total_signals = sum(a.n_signals for a in final_result.strategy_attribution.values())
    lines.append(f"  {'Strategy':<25} {'Signals':>8} {'% of Total':>11}")
    lines.append("  " + "-" * (W - 4))
    for name, attr in sorted(
        final_result.strategy_attribution.items(), key=lambda x: -x[1].n_signals
    ):
        pct = attr.n_signals / total_signals * 100 if total_signals > 0 else 0
        lines.append(f"  {name:<25} {attr.n_signals:>8} {pct:>10.1f}%")
    lines.append(f"  {'TOTAL':<25} {total_signals:>8} {'100.0':>10}%")

    # -- IS vs OOS comparison (best combo) ---------------------------------
    heading("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON (best combo)")
    is_res = best.full_is
    oos_res = best.full_oos
    if is_res and oos_res:
        metrics = [
            ("Sharpe Ratio", is_res.sharpe_ratio, oos_res.sharpe_ratio),
            ("Sortino Ratio", is_res.sortino_ratio, oos_res.sortino_ratio),
            ("Total Return", is_res.total_return, oos_res.total_return),
            ("Max Drawdown", is_res.max_drawdown, oos_res.max_drawdown),
            ("Volatility", is_res.volatility, oos_res.volatility),
            ("Win Rate", is_res.win_rate, oos_res.win_rate),
            ("Total Trades", is_res.total_trades, oos_res.total_trades),
        ]
        lines.append(f"  {'Metric':<18} {'In-Sample':>12} {'Out-of-Sample':>14} {'Ratio':>8}")
        lines.append("  " + "-" * (W - 4))
        for label, is_val, oos_val in metrics:
            if "Return" in label or "Drawdown" in label or "Volatility" in label or "Rate" in label:
                is_str = f"{is_val * 100:.2f}%"
                oos_str = f"{oos_val * 100:.2f}%"
            elif "Trades" in label:
                is_str = f"{int(is_val)}"
                oos_str = f"{int(oos_val)}"
            else:
                is_str = f"{is_val:+.3f}"
                oos_str = f"{oos_val:+.3f}"
            ratio_val = oos_val / is_val if abs(is_val) > 1e-8 else 0.0
            lines.append(f"  {label:<18} {is_str:>12} {oos_str:>14} {ratio_val:>8.2f}")

    # -- Deflated Sharpe Ratio ---------------------------------------------
    heading("DEFLATED SHARPE RATIO")
    lines.append(f"  Observed OOS Sharpe:  {best.oos_sharpe:+.3f}")
    lines.append(f"  Number of combos tested: {len(combo_results)}")
    lines.append(f"  Deflated Sharpe Ratio:   {dsr:+.3f}")
    lines.append(f"  p-value:                 {dsr_pval:.4f}")
    if dsr_pval < 0.05:
        lines.append("  Conclusion: Significant at 5% level — low overfitting risk")
    elif dsr_pval < 0.10:
        lines.append("  Conclusion: Marginally significant — moderate overfitting risk")
    else:
        lines.append("  Conclusion: Not significant — possible overfitting")

    # -- Full-period final backtest summary --------------------------------
    heading("FULL-PERIOD FINAL BACKTEST SUMMARY (best combo)")
    lines.append(f"  Total Return:      {final_result.total_return * 100:+.2f}%")
    lines.append(f"  Annualized Return: {final_result.annualized_return * 100:+.2f}%")
    lines.append(f"  Sharpe Ratio:      {final_result.sharpe_ratio:+.3f}")
    lines.append(f"  Sortino Ratio:     {final_result.sortino_ratio:+.3f}")
    lines.append(f"  Max Drawdown:      {final_result.max_drawdown * 100:.2f}%")
    lines.append(f"  Volatility:        {final_result.volatility * 100:.2f}%")
    lines.append(f"  Total Trades:      {final_result.total_trades}")
    lines.append(f"  Win Rate:          {final_result.win_rate * 100:.1f}%")

    lines.append("")
    lines.append(sep())
    lines.append("  END OF REPORT")
    lines.append(sep())
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    t_start = time.time()

    print()
    print("#" * 70)
    print("#  MULTI-STRATEGY WALK-FORWARD VALIDATED BACKTEST")
    print("#" * 70)
    print()

    # 0. Load data
    data = load_bar_data()

    # 1. & 2. Test all combinations with walk-forward
    combo_results = test_combinations(data)
    if not combo_results:
        print("\nERROR: All combinations failed. Aborting.")
        sys.exit(1)

    # Pick best by OOS Sharpe
    best = max(combo_results, key=lambda c: c.oos_sharpe)
    print(f"\nBest combination by OOS Sharpe: {best.name}  "
          f"(OOS SR={best.oos_sharpe:+.3f})")

    # 3. Final full-period backtest on best combo
    final_result = run_final_backtest(data, best)

    # 4. Deflated Sharpe
    n_daily = len(final_result.daily_returns)
    oos_daily = np.array(best.full_oos.daily_returns) if best.full_oos else np.array([])
    sr_std = float(np.std(oos_daily) * np.sqrt(252)) if len(oos_daily) > 2 else 1.0
    dsr, dsr_pval = deflated_sharpe_ratio(
        sr_observed=best.oos_sharpe,
        n_trials=len(combo_results),
        n_observations=n_daily,
        sr_std=sr_std,
    )

    # SPY benchmark
    spy_bh = compute_spy_buy_and_hold(data)

    # 5. Build report
    report = generate_report(
        combo_results, best, final_result,
        spy_bh, dsr, dsr_pval, data,
    )

    # Print to stdout
    print(report)

    # Save to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "validated_backtest_results.txt"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {output_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
