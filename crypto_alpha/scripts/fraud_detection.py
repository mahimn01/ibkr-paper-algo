#!/usr/bin/env python3
"""
Phantom Alpha Engine — Fraud Detection & Bias Quantification (v2)

Enterprise edition — tests run against the fixed runner which already uses:
  - Next-bar OPEN execution (signals on bar N fill at bar N+1 OPEN)
  - Natural equity compounding (no max_sizing_equity cap)
  - Backward-only data lookups (no forward-looking funding/spot/OI)
  - VWAP position tracking
  - Settlement-aligned funding

Tests:
  1. RANDOM SIGNALS (null test) — replace all edges with coin flips
     MUST produce SR ~0. If positive, infrastructure still has bias.
  2. REVERSED SIGNALS — flip every long/short; should lose money
  3. DOUBLED COSTS — 10bps slippage + 10bps commission each way
  4. NO FUNDING PAYMENTS — isolate signal alpha from funding alpha
  5. BUY-AND-HOLD COMPARISON — is the system actually beating HODL?
  6. SINGLE SYMBOL — BTC only, isolate edge alpha from diversification
  7. REDUCED VOL SCALING — cap vol_scale_max at 1.5 instead of 3.0
  8. COMBINED CONSERVATIVE — higher costs + reduced vol scaling
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)

# ── Reuse data loaders ────────────────────────────────────────
from crypto_alpha.scripts.deep_analysis import (
    load_all_data,
    build_9edge_controller,
    compute_sharpe,
)
from crypto_alpha.backtest.crypto_runner import (
    CryptoBacktestConfig,
    CryptoBacktestRunner,
)
from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)
from trading_algo.multi_strategy.protocol import StrategySignal


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def make_config(data, **overrides):
    """Build a standard CryptoBacktestConfig with optional overrides."""
    defaults = dict(
        initial_capital=10_000.0,
        symbols=list(data.keys()),
        commission_bps_maker=2.0,
        commission_bps_taker=5.0,
        slippage_bps=5.0,
        max_leverage=3.0,
        max_position_pct=0.25,
        max_gross_exposure=2.0,
        signal_interval_bars=24,
        funding_interval_hours=8,
    )
    defaults.update(overrides)
    return CryptoBacktestConfig(**defaults)


def print_result(label, results):
    ec = np.array(results.equity_curve)
    print(f"\n  {label}")
    print(f"    SR={results.sharpe_ratio:>+7.3f}  "
          f"Return={results.total_return:>+9.1%}  "
          f"MaxDD={results.max_drawdown:>6.2%}  "
          f"Trades={results.total_trades:>5}  "
          f"WR={results.win_rate:>5.1%}  "
          f"Sortino={results.sortino_ratio:>+7.2f}  "
          f"Final=${ec[-1]:>12,.0f}")


# ══════════════════════════════════════════════════════════════
#  BASELINE: Normal run (for comparison)
# ══════════════════════════════════════════════════════════════

def run_baseline(data, label="BASELINE (next-bar-open, natural compounding)"):
    """Run normal 9-edge backtest with enterprise runner."""
    controller = build_9edge_controller()
    config = make_config(data)
    runner = CryptoBacktestRunner(controller, config)
    results = runner.run(data)
    print_result(label, results)
    return results


# ══════════════════════════════════════════════════════════════
#  TEST 1: RANDOM SIGNALS (Null Test)
# ══════════════════════════════════════════════════════════════

class RandomSignalRunner(CryptoBacktestRunner):
    """Replace all signal directions with random coin flips.

    Overrides _process_signals which is called by run() whenever new
    signals are generated.  Signals are then queued and executed at
    the NEXT bar's open by the base runner — no look-ahead.
    """

    def _process_signals(self, signals, timestamp):
        randomized = []
        for sig in signals:
            new_sig = StrategySignal(
                strategy_name=sig.strategy_name,
                symbol=sig.symbol,
                direction=random.choice([-1, 0, 1]),
                target_weight=sig.target_weight,
                confidence=sig.confidence,
                metadata=sig.metadata,
            )
            randomized.append(new_sig)
        super()._process_signals(randomized, timestamp)


def test_random_signals(data, n_trials=10):
    """THE CRITICAL TEST: random signals must produce SR ~0."""
    print("\n  Running random-signal trials...")
    random_srs = []
    for trial in range(n_trials):
        random.seed(trial * 42)
        controller = build_9edge_controller()
        config = make_config(data)
        runner = RandomSignalRunner(controller, config)
        r = runner.run(data)
        random_srs.append(r.sharpe_ratio)
        if n_trials <= 10:
            print(f"    Trial {trial+1:>2}: SR={r.sharpe_ratio:>+7.3f}  "
                  f"Return={r.total_return:>+8.1%}  DD={r.max_drawdown:>6.2%}")

    arr = np.array(random_srs)
    print(f"\n    Mean random SR:   {np.mean(arr):>+7.3f}")
    print(f"    Median random SR: {np.median(arr):>+7.3f}")
    print(f"    Std random SR:    {np.std(arr):>7.3f}")
    print(f"    Min:              {np.min(arr):>+7.3f}")
    print(f"    Max:              {np.max(arr):>+7.3f}")

    passed = abs(np.mean(arr)) < 0.5
    print(f"    VERDICT: {'PASS — random SR near 0 (infrastructure is honest)' if passed else 'FAIL — infrastructure has positive bias!'}")
    return np.mean(arr)


# ══════════════════════════════════════════════════════════════
#  TEST 2: REVERSED SIGNALS
# ══════════════════════════════════════════════════════════════

class ReversedSignalRunner(CryptoBacktestRunner):
    """Flip every long to short and vice versa."""

    def _process_signals(self, signals, timestamp):
        reversed_sigs = []
        for sig in signals:
            new_sig = StrategySignal(
                strategy_name=sig.strategy_name,
                symbol=sig.symbol,
                direction=-sig.direction if sig.direction != 0 else 0,
                target_weight=sig.target_weight,
                confidence=sig.confidence,
                metadata=sig.metadata,
            )
            reversed_sigs.append(new_sig)
        super()._process_signals(reversed_sigs, timestamp)


def test_reversed_signals(data):
    """If real alpha exists, reversed signals should lose."""
    controller = build_9edge_controller()
    config = make_config(data)
    runner = ReversedSignalRunner(controller, config)
    results = runner.run(data)
    print_result("REVERSED SIGNALS", results)
    passed = results.sharpe_ratio < 0
    print(f"    VERDICT: {'PASS — reversed signals lose' if passed else 'FAIL — reversed signals still profit!'}")
    return results


# ══════════════════════════════════════════════════════════════
#  TEST 3: DOUBLED COSTS
# ══════════════════════════════════════════════════════════════

def test_doubled_costs(data):
    """Run with 2x slippage and 2x commission."""
    controller = build_9edge_controller()
    config = make_config(data,
        commission_bps_maker=4.0,
        commission_bps_taker=10.0,
        slippage_bps=10.0,
    )
    runner = CryptoBacktestRunner(controller, config)
    results = runner.run(data)
    print_result("DOUBLED COSTS (10bps slip + 10bps comm)", results)
    return results


# ══════════════════════════════════════════════════════════════
#  TEST 4: NO FUNDING PAYMENTS
# ══════════════════════════════════════════════════════════════

class NoFundingRunner(CryptoBacktestRunner):
    """Disable all funding payments."""
    def _maybe_apply_funding(self, timestamp):
        pass


def test_no_funding(data):
    """Strip funding payments to isolate signal alpha."""
    controller = build_9edge_controller()
    config = make_config(data)
    runner = NoFundingRunner(controller, config)
    results = runner.run(data)
    print_result("NO FUNDING PAYMENTS", results)
    return results


# ══════════════════════════════════════════════════════════════
#  TEST 5: BUY-AND-HOLD COMPARISON
# ══════════════════════════════════════════════════════════════

def test_buy_and_hold(data):
    """Compare against simple buy-and-hold BTC/ETH/SOL."""
    print("\n  Buy-and-Hold Benchmarks:")

    for sym, bars in data.items():
        if len(bars) < 2:
            continue
        start_price = bars[0].close
        end_price = bars[-1].close
        total_ret = (end_price / start_price) - 1

        daily_closes = []
        last_day = None
        for b in bars:
            d = b.timestamp.date()
            if d != last_day:
                if last_day is not None:
                    daily_closes.append(b.close)
                last_day = d
        daily_closes.append(bars[-1].close)

        dr = np.diff(daily_closes) / daily_closes[:-1]
        sr = compute_sharpe(dr)

        n_years = len(dr) / 365
        peak = np.maximum.accumulate(daily_closes)
        dd = (peak - daily_closes) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd))

        print(f"    {sym}: Return={total_ret:>+8.1%}, SR={sr:>+6.2f}, MaxDD={max_dd:>6.1%}")

    print()


# ══════════════════════════════════════════════════════════════
#  TEST 6: SINGLE SYMBOL (BTC only)
# ══════════════════════════════════════════════════════════════

def test_single_symbol(data):
    """Run on BTC only — isolate edge alpha from cross-symbol diversification."""
    btc_only = {k: v for k, v in data.items() if "BTC" in k}
    if not btc_only:
        print("  No BTC data found")
        return None

    controller = build_9edge_controller()
    config = make_config(btc_only)
    runner = CryptoBacktestRunner(controller, config)
    results = runner.run(btc_only)
    print_result("BTC ONLY (single symbol)", results)
    return results


# ══════════════════════════════════════════════════════════════
#  TEST 7: REDUCED VOL SCALING
# ══════════════════════════════════════════════════════════════

def test_reduced_vol_scaling(data):
    """Cap vol_scale_max at 1.5 instead of 3.0."""
    ctrl_config = ControllerConfig(
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
        max_gross_exposure=2.0,
        max_net_exposure=1.5,
        max_single_symbol_weight=0.40,
        max_portfolio_positions=20,
        conflict_resolution="weighted_confidence",
        enable_vol_management=True,
        vol_target=0.30,
        vol_lookback=20,
        vol_scale_min=0.50,
        vol_scale_max=1.5,
        max_drawdown=0.25,
        daily_loss_limit=0.05,
        enable_entropy_filter=False,
    )

    controller = MultiStrategyController(ctrl_config)

    from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
    from crypto_alpha.adapters.frm_adapter import FRMAdapter
    from crypto_alpha.adapters.radl_adapter import RADLAdapter
    from crypto_alpha.adapters.imc_adapter import IMCAdapter
    from crypto_alpha.adapters.lcp_adapter import LCPAdapter
    from crypto_alpha.adapters.ced_adapter import CEDAdapter
    from crypto_alpha.adapters.vts_adapter import VTSAdapter
    from crypto_alpha.adapters.vf_adapter import VFAdapter
    from crypto_alpha.adapters.vv_adapter import VVAdapter

    controller.register(PBMRAdapter(base_weight=0.14))
    controller.register(FRMAdapter(base_weight=0.13))
    controller.register(RADLAdapter(base_weight=0.11))
    controller.register(IMCAdapter(base_weight=0.11))
    controller.register(LCPAdapter(base_weight=0.12))
    controller.register(CEDAdapter(base_weight=0.09))
    controller.register(VTSAdapter(base_weight=0.11))
    controller.register(VFAdapter(base_weight=0.10))
    controller.register(VVAdapter(base_weight=0.09))

    config = make_config(data)
    runner = CryptoBacktestRunner(controller, config)
    results = runner.run(data)
    print_result("REDUCED VOL SCALING (max 1.5x)", results)
    return results


# ══════════════════════════════════════════════════════════════
#  TEST 8: COMBINED CONSERVATIVE
# ══════════════════════════════════════════════════════════════

def test_combined_conservative(data):
    """The most conservative estimate: higher costs + reduced vol scaling."""
    ctrl_config = ControllerConfig(
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
        max_gross_exposure=2.0,
        max_net_exposure=1.5,
        max_single_symbol_weight=0.40,
        max_portfolio_positions=20,
        conflict_resolution="weighted_confidence",
        enable_vol_management=True,
        vol_target=0.30,
        vol_lookback=20,
        vol_scale_min=0.50,
        vol_scale_max=1.5,
        max_drawdown=0.25,
        daily_loss_limit=0.05,
        enable_entropy_filter=False,
    )

    controller = MultiStrategyController(ctrl_config)

    from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
    from crypto_alpha.adapters.frm_adapter import FRMAdapter
    from crypto_alpha.adapters.radl_adapter import RADLAdapter
    from crypto_alpha.adapters.imc_adapter import IMCAdapter
    from crypto_alpha.adapters.lcp_adapter import LCPAdapter
    from crypto_alpha.adapters.ced_adapter import CEDAdapter
    from crypto_alpha.adapters.vts_adapter import VTSAdapter
    from crypto_alpha.adapters.vf_adapter import VFAdapter
    from crypto_alpha.adapters.vv_adapter import VVAdapter

    controller.register(PBMRAdapter(base_weight=0.14))
    controller.register(FRMAdapter(base_weight=0.13))
    controller.register(RADLAdapter(base_weight=0.11))
    controller.register(IMCAdapter(base_weight=0.11))
    controller.register(LCPAdapter(base_weight=0.12))
    controller.register(CEDAdapter(base_weight=0.09))
    controller.register(VTSAdapter(base_weight=0.11))
    controller.register(VFAdapter(base_weight=0.10))
    controller.register(VVAdapter(base_weight=0.09))

    config = make_config(data,
        commission_bps_maker=4.0,
        commission_bps_taker=8.0,
        slippage_bps=8.0,
    )
    runner = CryptoBacktestRunner(controller, config)
    results = runner.run(data)
    print_result("COMBINED CONSERVATIVE (8bps costs + 1.5x vol)", results)
    print(f"    VERDICT: This is the most conservative estimate of live performance")
    return results


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    start = time.time()

    print("=" * 80)
    print("  PHANTOM ALPHA ENGINE — FRAUD DETECTION v2 (Enterprise Runner)")
    print("  Fixes: next-bar-open, natural compounding, backward-only data")
    print("=" * 80)
    print("\n  Loading data (uses cache)...")

    data = load_all_data()
    print(f"  Loaded {sum(len(v) for v in data.values())} bars across {len(data)} symbols")

    # ── Run all tests ──────────────────────────────────────────

    print("\n" + "=" * 80)
    print("  TEST 0: BASELINE (next-bar-open, natural compounding)")
    print("=" * 80)
    baseline = run_baseline(data)

    print("\n" + "=" * 80)
    print("  TEST 1: RANDOM SIGNALS (null hypothesis)")
    print("  If SR > 0.5, infrastructure still has bias.")
    print("=" * 80)
    random_sr = test_random_signals(data)

    print("\n" + "=" * 80)
    print("  TEST 2: REVERSED SIGNALS")
    print("=" * 80)
    reversed_r = test_reversed_signals(data)

    print("\n" + "=" * 80)
    print("  TEST 3: DOUBLED COSTS")
    print("=" * 80)
    doubled = test_doubled_costs(data)

    print("\n" + "=" * 80)
    print("  TEST 4: NO FUNDING PAYMENTS")
    print("=" * 80)
    no_fund = test_no_funding(data)

    print("\n" + "=" * 80)
    print("  TEST 5: BUY-AND-HOLD BENCHMARKS")
    print("=" * 80)
    test_buy_and_hold(data)

    print("\n" + "=" * 80)
    print("  TEST 6: BTC ONLY (single symbol)")
    print("=" * 80)
    btc_only = test_single_symbol(data)

    print("\n" + "=" * 80)
    print("  TEST 7: REDUCED VOL SCALING (1.5x max)")
    print("=" * 80)
    reduced_vol = test_reduced_vol_scaling(data)

    print("\n" + "=" * 80)
    print("  TEST 8: COMBINED CONSERVATIVE")
    print("  (8bps costs + 1.5x vol scaling)")
    print("=" * 80)
    combined = test_combined_conservative(data)

    # ── Summary Table ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY: FRAUD DETECTION RESULTS")
    print("=" * 80)

    results_table = [
        ("BASELINE (enterprise runner)", baseline.sharpe_ratio, baseline.total_return, baseline.max_drawdown),
        ("Random signals (null)", random_sr, None, None),
        ("Reversed signals", reversed_r.sharpe_ratio, reversed_r.total_return, reversed_r.max_drawdown),
        ("Doubled costs", doubled.sharpe_ratio, doubled.total_return, doubled.max_drawdown),
        ("No funding", no_fund.sharpe_ratio, no_fund.total_return, no_fund.max_drawdown),
        ("BTC only", btc_only.sharpe_ratio if btc_only else 0, btc_only.total_return if btc_only else 0, btc_only.max_drawdown if btc_only else 0),
        ("Reduced vol (1.5x)", reduced_vol.sharpe_ratio, reduced_vol.total_return, reduced_vol.max_drawdown),
        ("COMBINED CONSERVATIVE", combined.sharpe_ratio, combined.total_return, combined.max_drawdown),
    ]

    print(f"\n  {'Test':.<40} {'SR':>8} {'Return':>10} {'MaxDD':>8}")
    print("  " + "-" * 70)
    for name, sr, ret, dd in results_table:
        ret_str = f"{ret:>+9.1%}" if ret is not None else "    N/A"
        dd_str = f"{dd:>7.2%}" if dd is not None else "    N/A"
        sr_val = sr if sr is not None else 0
        print(f"  {name:.<40} {sr_val:>+8.3f} {ret_str} {dd_str}")

    # ── Integrity Checks ──────────────────────────────────────
    print(f"\n  INTEGRITY CHECKS:")
    random_ok = abs(random_sr) < 0.5
    reversed_ok = reversed_r.sharpe_ratio < baseline.sharpe_ratio * 0.5
    costs_ok = doubled.sharpe_ratio < baseline.sharpe_ratio
    print(f"    Random SR near 0:        {'PASS' if random_ok else 'FAIL'} (SR={random_sr:+.3f}, threshold <0.5)")
    print(f"    Reversed SR < baseline:  {'PASS' if reversed_ok else 'FAIL'} (SR={reversed_r.sharpe_ratio:+.3f})")
    print(f"    Doubled costs degrade:   {'PASS' if costs_ok else 'FAIL'} (SR={doubled.sharpe_ratio:+.3f})")

    if random_ok and reversed_ok:
        print(f"\n  CONCLUSION: Infrastructure appears honest.")
        print(f"  MOST CONSERVATIVE SR ESTIMATE: {combined.sharpe_ratio:+.3f}")
    else:
        print(f"\n  WARNING: Infrastructure integrity issues remain!")
        if not random_ok:
            print(f"    Random signals produce SR={random_sr:+.3f} — should be near 0")
        if not reversed_ok:
            print(f"    Reversed signals SR={reversed_r.sharpe_ratio:+.3f} — should be well below baseline")

    elapsed = time.time() - start
    print(f"\n{'=' * 80}")
    print(f"  FRAUD DETECTION v2 COMPLETE — {elapsed:.0f}s total")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
