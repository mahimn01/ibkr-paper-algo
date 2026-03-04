#!/usr/bin/env python3
"""
Phantom Alpha Engine — Crypto Backtest Runner

Runs the 4-edge crypto system on synthetic or historical data
and produces comprehensive performance metrics.

Usage:
    python -m crypto_alpha.scripts.run_crypto_backtest
    python -m crypto_alpha.scripts.run_crypto_backtest --real  # Use CCXT data
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)
from crypto_alpha.backtest.crypto_runner import CryptoBacktestConfig, CryptoBacktestRunner
from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
from crypto_alpha.adapters.frm_adapter import FRMAdapter
from crypto_alpha.adapters.radl_adapter import RADLAdapter
from crypto_alpha.adapters.imc_adapter import IMCAdapter
from crypto_alpha.data.synthetic import generate_synthetic_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PhantomAlpha")


def build_controller() -> MultiStrategyController:
    """Build the multi-strategy controller with 4 crypto edges."""
    config = ControllerConfig(
        allocations={
            "PerpBasisMeanReversion": StrategyAllocation(weight=0.25, max_positions=6),
            "FundingRateMomentum": StrategyAllocation(weight=0.25, max_positions=6),
            "RegimeAdaptiveLeverage": StrategyAllocation(weight=0.25, max_positions=6),
            "IntermarketCascade": StrategyAllocation(weight=0.25, max_positions=6),
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
        enable_entropy_filter=False,  # Start without entropy filter
    )

    controller = MultiStrategyController(config)

    # Register edges
    pbmr = PBMRAdapter(base_weight=0.15)
    frm = FRMAdapter(base_weight=0.15)
    radl = RADLAdapter(base_weight=0.12)
    imc = IMCAdapter(base_weight=0.12)

    controller.register(pbmr)
    controller.register(frm)
    controller.register(radl)
    controller.register(imc)

    return controller


def run_backtest(use_real_data: bool = False, n_days: int = 365 * 3):
    """Run the crypto backtest."""
    print("\n" + "=" * 70)
    print("  PHANTOM ALPHA ENGINE — Crypto Backtest")
    print("  4-Edge System: PBMR + FRM + RADL + IMC")
    print("=" * 70 + "\n")

    # ── Generate or load data ──
    if use_real_data:
        print("Loading real data via CCXT...")
        from crypto_alpha.data.ccxt_loader import download_ohlcv, download_funding_rates
        # TODO: implement real data loading
        raise NotImplementedError("Real data loading not yet implemented. Use synthetic data.")
    else:
        print(f"Generating {n_days}-day synthetic universe (BTC, ETH, SOL)...")
        t0 = time.time()
        data = generate_synthetic_universe(n_days=n_days, bars_per_day=288, seed=42)
        elapsed = time.time() - t0
        print(f"  Generated {sum(len(v) for v in data.values()):,} total bars in {elapsed:.1f}s")

    # Feed spot prices to adapters (needed for basis calculation)
    # The CryptoBar objects already have spot_price embedded

    # ── Build controller ──
    controller = build_controller()

    # ── Configure backtest ──
    bt_config = CryptoBacktestConfig(
        initial_capital=30_000.0,
        symbols=list(data.keys()),
        commission_bps_maker=2.0,
        commission_bps_taker=5.0,
        slippage_bps=5.0,           # Realistic slippage for crypto
        max_leverage=3.0,            # Conservative leverage
        max_position_pct=0.25,       # Max 25% per symbol
        max_gross_exposure=2.0,      # Max 2x total exposure
        max_sizing_equity=60_000.0,  # Cap at 2x initial (prevents unrealistic compounding)
        signal_interval_bars=288,    # Daily signals (1 bar = 5min, 288 = 1 day)
        funding_interval_hours=8,
    )

    # ── Run backtest ──
    runner = CryptoBacktestRunner(controller, bt_config)

    print(f"\nRunning backtest: ${bt_config.initial_capital:,.0f} initial capital")
    print(f"  Symbols: {bt_config.symbols}")
    print(f"  Commission: {bt_config.commission_bps_taker} bps taker")
    print(f"  Max leverage: {bt_config.max_leverage}x")
    print(f"  Signal interval: {bt_config.signal_interval_bars} bars")
    print()

    t0 = time.time()

    def progress(pct, msg):
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}% {msg}", end="", flush=True)

    results = runner.run(data, progress_callback=progress)
    elapsed = time.time() - t0
    print(f"\n\n  Backtest completed in {elapsed:.1f}s\n")

    # ── Print results ──
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  {'Total Return':.<35} {results.total_return:>+10.2%}")
    print(f"  {'Annualized Return':.<35} {results.annualized_return:>+10.2%}")
    print(f"  {'Sharpe Ratio':.<35} {results.sharpe_ratio:>+10.3f}")
    print(f"  {'Sortino Ratio':.<35} {results.sortino_ratio:>+10.3f}")
    print(f"  {'Max Drawdown':.<35} {results.max_drawdown:>10.2%}")
    print(f"  {'Volatility':.<35} {results.volatility:>10.2%}")
    print(f"  {'Calmar Ratio':.<35} {results.calmar_ratio:>+10.3f}")

    print(f"\n  {'Total Trades':.<35} {results.total_trades:>10,}")
    print(f"  {'Win Rate':.<35} {results.win_rate:>10.2%}")
    print(f"  {'Profit Factor':.<35} {results.profit_factor:>10.3f}")

    print(f"\n  {'VaR (95%)':.<35} {results.var_95:>10.4f}")
    print(f"  {'CVaR (95%)':.<35} {results.cvar_95:>10.4f}")
    print(f"  {'Skewness':.<35} {results.skewness:>+10.3f}")
    print(f"  {'Kurtosis':.<35} {results.kurtosis:>+10.3f}")
    print(f"  {'Max DD Duration (days)':.<35} {results.max_drawdown_duration_days:>10}")

    # Crypto-specific metrics
    if hasattr(results, 'metadata'):
        meta = results.metadata  # type: ignore
        print(f"\n  {'--- Crypto Metrics ---':.<35}")
        print(f"  {'Total Funding P&L':.<35} ${meta.get('total_funding_pnl', 0):>+10,.2f}")
        print(f"  {'Funding P&L (% of capital)':.<35} {meta.get('funding_pnl_pct', 0):>+10.2%}")
        print(f"  {'Liquidations':.<35} {meta.get('liquidations', 0):>10}")

    # Per-strategy attribution
    if results.strategy_attribution:
        print(f"\n  {'--- Per-Edge Attribution ---':.<35}")
        for name, attr in sorted(results.strategy_attribution.items()):
            print(f"    {name:.<30} {attr.n_signals:>6} signals")

    # Equity curve stats
    ec = np.array(results.equity_curve)
    if len(ec) > 2:
        peak = np.maximum.accumulate(ec)
        dd = (peak - ec) / np.where(peak > 0, peak, 1)

        print(f"\n  {'--- Equity Curve ---':.<35}")
        print(f"  {'Starting Equity':.<35} ${ec[0]:>10,.2f}")
        print(f"  {'Ending Equity':.<35} ${ec[-1]:>10,.2f}")
        print(f"  {'Peak Equity':.<35} ${np.max(ec):>10,.2f}")
        print(f"  {'Trough Equity':.<35} ${np.min(ec):>10,.2f}")

    print("\n" + "=" * 70)

    # ── Monthly returns ──
    if results.daily_returns:
        dr = np.array(results.daily_returns)
        n_months = len(dr) // 30
        if n_months > 0:
            print(f"\n  Monthly Returns (avg of {n_months} months):")
            monthly_rets = []
            for m in range(n_months):
                month_dr = dr[m * 30:(m + 1) * 30]
                monthly_ret = np.prod(1 + month_dr) - 1
                monthly_rets.append(monthly_ret)

            monthly_rets = np.array(monthly_rets)
            print(f"  {'Mean Monthly Return':.<35} {np.mean(monthly_rets):>+10.2%}")
            print(f"  {'Median Monthly Return':.<35} {np.median(monthly_rets):>+10.2%}")
            print(f"  {'Best Month':.<35} {np.max(monthly_rets):>+10.2%}")
            print(f"  {'Worst Month':.<35} {np.min(monthly_rets):>+10.2%}")
            print(f"  {'% Positive Months':.<35} {np.mean(monthly_rets > 0):>10.1%}")

            # Monthly Sharpe approximation
            monthly_sr = np.mean(monthly_rets) / np.std(monthly_rets, ddof=1) * np.sqrt(12) if np.std(monthly_rets) > 0 else 0
            print(f"  {'Monthly Sharpe (annualized)':.<35} {monthly_sr:>+10.3f}")

    print("\n" + "=" * 70)
    print("  PHANTOM ALPHA ENGINE — Backtest Complete")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phantom Alpha Engine Backtest")
    parser.add_argument("--real", action="store_true", help="Use real CCXT data")
    parser.add_argument("--days", type=int, default=365 * 3, help="Number of days")
    args = parser.parse_args()

    run_backtest(use_real_data=args.real, n_days=args.days)
