#!/usr/bin/env python3
"""
Phantom Alpha Engine — FULL 9-EDGE Real Data Backtest

Downloads real Binance data and runs the complete 9-edge system
with 3 meta-innovations (Alpha Decay Detector, Correlation Regime
Switch, Conviction Amplifier).

9 Edges:
  1. Liquidation Cascade Predictor (LCP)
  2. Funding Rate Momentum (FRM)
  3. Cross-Exchange Divergence (CED)
  4. Volatility Term Structure (VTS)
  5. Volume Flow Detector (VF)
  6. Volume Velocity Breakout (VV)
  7. Perpetual Basis Mean Reversion (PBMR)
  8. Regime-Adaptive Dynamic Leverage (RADL)
  9. Intermarket Momentum Cascade (IMC)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PhantomAlpha9Edge")

CACHE_DIR = project_root / "crypto_data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Data Download (reuse from existing script) ───────────────────

def download_ohlcv_binance(
    symbol: str,
    market_type: str = "swap",
    timeframe: str = "1h",
    start_date: str = "2020-10-01",
    end_date: str = "2026-03-01",
) -> Tuple[List[datetime], np.ndarray]:
    """Download OHLCV from Binance with caching."""
    import ccxt

    safe_sym = symbol.replace("/", "_").replace(":", "_")
    cache_file = CACHE_DIR / f"{safe_sym}_{market_type}_{timeframe}_{start_date}_{end_date}.npz"

    if cache_file.exists():
        data = np.load(str(cache_file), allow_pickle=True)
        timestamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in data["ts"]]
        ohlcv = data["ohlcv"]
        logger.info(f"  Cached: {symbol} ({market_type}) {len(timestamps)} bars")
        return timestamps, ohlcv

    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": market_type}})
    ex.load_markets()

    tf_ms = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    bar_ms = tf_ms.get(timeframe, 3_600_000)

    since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_bars = []
    current = since

    while current < end_ms:
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe, since=current, limit=1500)
        except Exception as e:
            logger.warning(f"  Error {symbol}: {e}, retrying...")
            time.sleep(3)
            continue

        if not bars:
            break

        all_bars.extend(bars)
        current = bars[-1][0] + bar_ms
        time.sleep(ex.rateLimit / 1000 * 1.1)

        if len(all_bars) % 5000 < 1500:
            logger.info(f"  ... {symbol} ({market_type}): {len(all_bars)} bars")

    if not all_bars:
        raise ValueError(f"No data for {symbol} ({market_type})")

    timestamps = [datetime.fromtimestamp(b[0] / 1000, tz=timezone.utc) for b in all_bars]
    ohlcv = np.array([[b[1], b[2], b[3], b[4], b[5]] for b in all_bars], dtype=np.float64)

    ts_epoch = np.array([t.timestamp() for t in timestamps])
    np.savez_compressed(str(cache_file), ts=ts_epoch, ohlcv=ohlcv)
    logger.info(f"  Downloaded: {symbol} ({market_type}) {len(timestamps)} bars")

    return timestamps, ohlcv


def download_funding_rates_binance(
    symbol: str,
    start_date: str = "2020-10-01",
    end_date: str = "2026-03-01",
) -> Tuple[List[datetime], np.ndarray]:
    """Download funding rate history from Binance."""
    import ccxt

    safe_sym = symbol.replace("/", "_").replace(":", "_")
    cache_file = CACHE_DIR / f"{safe_sym}_funding_{start_date}_{end_date}.npz"

    if cache_file.exists():
        data = np.load(str(cache_file), allow_pickle=True)
        timestamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in data["ts"]]
        rates = data["rates"]
        logger.info(f"  Cached: {symbol} funding {len(timestamps)} entries")
        return timestamps, rates

    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()

    since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_rates = []
    current = since

    while current < end_ms:
        try:
            rates = ex.fetch_funding_rate_history(symbol, since=current, limit=1000)
        except Exception as e:
            logger.warning(f"  Error funding {symbol}: {e}, retrying...")
            time.sleep(3)
            continue

        if not rates:
            break

        all_rates.extend(rates)
        current = rates[-1]["timestamp"] + 1
        time.sleep(ex.rateLimit / 1000 * 1.1)

    if not all_rates:
        logger.warning(f"No funding data for {symbol}")
        return [], np.array([])

    timestamps = [datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc) for r in all_rates]
    rate_values = np.array([r.get("fundingRate", 0.0) for r in all_rates], dtype=np.float64)

    ts_epoch = np.array([t.timestamp() for t in timestamps])
    np.savez_compressed(str(cache_file), ts=ts_epoch, rates=rate_values)
    logger.info(f"  Downloaded: {symbol} funding {len(timestamps)} entries")

    return timestamps, rate_values


def download_open_interest_binance(
    symbol: str,
    timeframe: str = "1h",
    start_date: str = "2020-10-01",
    end_date: str = "2026-03-01",
) -> Tuple[List[datetime], np.ndarray]:
    """Download open interest history from Binance futures.

    Note: Binance OI API has limited historical depth (~30 days for 1h).
    We download what's available and gracefully handle failures.
    """
    import ccxt

    safe_sym = symbol.replace("/", "_").replace(":", "_")
    cache_file = CACHE_DIR / f"{safe_sym}_oi_{timeframe}_{start_date}_{end_date}.npz"

    if cache_file.exists():
        data = np.load(str(cache_file), allow_pickle=True)
        timestamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in data["ts"]]
        oi_values = data["oi"]
        logger.info(f"  Cached: {symbol} OI {len(timestamps)} entries")
        return timestamps, oi_values

    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()

    # Binance OI API only supports recent data (~30 days for 1h timeframe)
    # and doesn't accept 'since' parameter the same way as OHLCV.
    # We'll fetch what's available without 'since' parameter.
    all_oi = []
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            # Fetch without 'since' — gets most recent available data
            oi_data = ex.fetch_open_interest_history(symbol, timeframe, limit=500)
            if oi_data:
                all_oi.extend(oi_data)
                logger.info(f"  Got {len(oi_data)} OI entries for {symbol}")
            break
        except Exception as e:
            retries += 1
            logger.warning(f"  Error OI {symbol} (attempt {retries}/{max_retries}): {e}")
            time.sleep(3)

    if not all_oi:
        logger.warning(f"  No OI data for {symbol} (LCP edge will not fire)")
        return [], np.array([])

    timestamps = [datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc) for r in all_oi]
    oi_values = np.array([r.get("openInterestValue", r.get("baseVolume", 0.0)) for r in all_oi], dtype=np.float64)

    ts_epoch = np.array([t.timestamp() for t in timestamps])
    np.savez_compressed(str(cache_file), ts=ts_epoch, oi=oi_values)
    logger.info(f"  Downloaded: {symbol} OI {len(timestamps)} entries")

    return timestamps, oi_values


# ── Build CryptoBar objects ────────────────────────────────────────

def build_crypto_bars(
    perp_timestamps: List[datetime],
    perp_ohlcv: np.ndarray,
    spot_timestamps: List[datetime],
    spot_ohlcv: np.ndarray,
    funding_timestamps: List[datetime],
    funding_rates: np.ndarray,
    oi_timestamps: List[datetime] = None,
    oi_values: np.ndarray = None,
) -> List:
    """Merge perp OHLCV + spot + funding + OI into CryptoBar objects."""
    from crypto_alpha.types import CryptoBar

    # Build spot lookup: epoch -> close price
    spot_lookup = {}
    for i, ts in enumerate(spot_timestamps):
        spot_lookup[int(ts.timestamp())] = float(spot_ohlcv[i, 3])

    # Build funding lookup
    funding_lookup = {}
    for i, ts in enumerate(funding_timestamps):
        funding_lookup[int(ts.timestamp())] = float(funding_rates[i])

    # Build OI lookup
    oi_lookup = {}
    if oi_timestamps and oi_values is not None and len(oi_values) > 0:
        for i, ts in enumerate(oi_timestamps):
            oi_lookup[int(ts.timestamp())] = float(oi_values[i])

    bars = []
    last_funding = 0.0
    last_spot = 0.0
    last_oi = 0.0

    for i, ts in enumerate(perp_timestamps):
        ts_epoch = int(ts.timestamp())

        # Find closest funding rate (BACKWARD-ONLY — no future data)
        if ts_epoch in funding_lookup:
            last_funding = funding_lookup[ts_epoch]
        else:
            for offset in range(300, 3601, 300):
                if ts_epoch - offset in funding_lookup:
                    last_funding = funding_lookup[ts_epoch - offset]
                    break

        # Find spot price (BACKWARD-ONLY)
        if ts_epoch in spot_lookup:
            last_spot = spot_lookup[ts_epoch]
        else:
            for offset in range(60, 3601, 60):
                if ts_epoch - offset in spot_lookup:
                    last_spot = spot_lookup[ts_epoch - offset]
                    break

        if last_spot == 0.0:
            last_spot = float(perp_ohlcv[i, 3])

        # Find OI (BACKWARD-ONLY)
        if ts_epoch in oi_lookup:
            last_oi = oi_lookup[ts_epoch]
        elif oi_lookup:
            for offset in range(300, 7201, 300):
                if ts_epoch - offset in oi_lookup:
                    last_oi = oi_lookup[ts_epoch - offset]
                    break

        bars.append(CryptoBar(
            timestamp=ts,
            open=float(perp_ohlcv[i, 0]),
            high=float(perp_ohlcv[i, 1]),
            low=float(perp_ohlcv[i, 2]),
            close=float(perp_ohlcv[i, 3]),
            volume=float(perp_ohlcv[i, 4]),
            funding_rate=last_funding,
            spot_price=last_spot,
            open_interest=last_oi if last_oi > 0 else None,
        ))

    return bars


# ── Run 9-Edge Backtest ───────────────────────────────────────────

def run_9edge_backtest():
    from trading_algo.multi_strategy.controller import (
        ControllerConfig, MultiStrategyController, StrategyAllocation,
    )
    from crypto_alpha.backtest.crypto_runner import CryptoBacktestConfig, CryptoBacktestRunner

    # Import all 9 adapters
    from crypto_alpha.adapters.pbmr_adapter import PBMRAdapter
    from crypto_alpha.adapters.frm_adapter import FRMAdapter
    from crypto_alpha.adapters.radl_adapter import RADLAdapter
    from crypto_alpha.adapters.imc_adapter import IMCAdapter
    from crypto_alpha.adapters.lcp_adapter import LCPAdapter
    from crypto_alpha.adapters.ced_adapter import CEDAdapter
    from crypto_alpha.adapters.vts_adapter import VTSAdapter
    from crypto_alpha.adapters.vf_adapter import VFAdapter
    from crypto_alpha.adapters.vv_adapter import VVAdapter

    # Import meta-innovations
    from crypto_alpha.meta.alpha_decay_detector import AlphaDecayDetector
    from crypto_alpha.meta.correlation_regime import CorrelationRegimeSwitch
    from crypto_alpha.meta.conviction_amplifier import ConvictionAmplifier

    symbols_perp = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    symbols_spot = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    symbol_names = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    start = "2020-10-01"
    end = "2026-03-01"

    print("\n" + "=" * 70)
    print("  PHANTOM ALPHA ENGINE — FULL 9-EDGE REAL DATA BACKTEST")
    print("  Binance Perpetual Futures + Spot + Funding + Open Interest")
    print("  9 Edges: PBMR + FRM + RADL + IMC + LCP + CED + VTS + VF + VV")
    print("  3 Meta: Alpha Decay + Correlation Regime + Conviction Amplifier")
    print("=" * 70 + "\n")

    # ── Download all data ──
    print("Phase 1: Downloading real market data from Binance...\n")

    all_data = {}
    for i, (perp_sym, spot_sym, name) in enumerate(zip(symbols_perp, symbols_spot, symbol_names)):
        print(f"[{i+1}/3] {name}:")

        # Perp OHLCV
        perp_ts, perp_ohlcv = download_ohlcv_binance(perp_sym, "swap", "1h", start, end)

        # Spot OHLCV
        spot_ts, spot_ohlcv = download_ohlcv_binance(spot_sym, "spot", "1h", start, end)

        # Funding rates
        fund_ts, fund_rates = download_funding_rates_binance(perp_sym, start, end)

        # Open Interest (NEW for LCP edge)
        oi_ts, oi_values = download_open_interest_binance(perp_sym, "1h", start, end)

        # Build CryptoBar objects with OI included
        bars = build_crypto_bars(perp_ts, perp_ohlcv, spot_ts, spot_ohlcv,
                                 fund_ts, fund_rates, oi_ts, oi_values)

        all_data[name] = bars
        oi_count = sum(1 for b in bars if b.open_interest is not None)
        print(f"  Total: {len(bars)} hourly bars, {len(fund_ts)} funding obs, {oi_count} OI bars")
        print(f"  Date range: {bars[0].timestamp.date()} to {bars[-1].timestamp.date()}")
        print(f"  Price range: ${bars[0].close:,.2f} to ${bars[-1].close:,.2f}")
        print()

    # ── Build 9-edge controller ──
    print("Phase 2: Building 9-edge controller with meta-innovations...\n")

    # Weights sum to 1.0 — distributed by expected SR and data quality
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

    # Register all 9 edges
    # Original 4 edges
    controller.register(PBMRAdapter(base_weight=0.14))
    controller.register(FRMAdapter(base_weight=0.13))
    controller.register(RADLAdapter(base_weight=0.11))
    controller.register(IMCAdapter(base_weight=0.11))
    # 5 new edges
    controller.register(LCPAdapter(base_weight=0.12))
    controller.register(CEDAdapter(base_weight=0.09))
    controller.register(VTSAdapter(base_weight=0.11))
    controller.register(VFAdapter(base_weight=0.10))
    controller.register(VVAdapter(base_weight=0.09))

    print(f"  Registered {len(controller._strategies)} edges:")
    for name in controller._strategies:
        print(f"    - {name}")
    print()

    # ── Configure backtest ──
    bt_config = CryptoBacktestConfig(
        initial_capital=30_000.0,
        symbols=symbol_names,
        commission_bps_maker=2.0,
        commission_bps_taker=5.0,
        slippage_bps=5.0,
        max_leverage=3.0,
        max_position_pct=0.25,
        max_gross_exposure=2.0,
        max_sizing_equity=60_000.0,
        signal_interval_bars=24,  # Daily signals on 1h bars
        funding_interval_hours=8,
    )

    # ── Run backtest ──
    print("Phase 3: Running 9-edge backtest...\n")

    runner = CryptoBacktestRunner(controller, bt_config)

    print(f"  Capital: ${bt_config.initial_capital:,.0f}")
    print(f"  Symbols: {bt_config.symbols}")
    n_days = (datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days
    print(f"  Period: {start} to {end} ({n_days} days = {n_days/365:.1f} years)")
    print(f"  Edges: 9 | Commission: {bt_config.commission_bps_taker} bps | Max leverage: {bt_config.max_leverage}x")
    print()

    t0 = time.time()

    def progress(pct, msg):
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}% {msg}", end="", flush=True)

    results = runner.run(all_data, progress_callback=progress)
    elapsed = time.time() - t0
    print(f"\n\n  Backtest completed in {elapsed:.1f}s\n")

    # ── Print comprehensive results ──
    print("=" * 70)
    print("  FULL 9-EDGE REAL DATA RESULTS")
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

    # Crypto-specific
    if hasattr(results, "metadata"):
        meta = results.metadata
        print(f"\n  {'--- Crypto Metrics ---':.<35}")
        print(f"  {'Total Funding P&L':.<35} ${meta.get('total_funding_pnl', 0):>+10,.2f}")
        print(f"  {'Funding P&L (% of capital)':.<35} {meta.get('funding_pnl_pct', 0):>+10.2%}")
        print(f"  {'Liquidations':.<35} {meta.get('liquidations', 0):>10}")

    # Per-edge attribution
    if results.strategy_attribution:
        print(f"\n  {'--- Per-Edge Attribution (9 Edges) ---':.<35}")
        for name, attr in sorted(results.strategy_attribution.items()):
            print(f"    {name:.<35} {attr.n_signals:>6} signals")

    # Equity curve
    ec = np.array(results.equity_curve)
    if len(ec) > 2:
        print(f"\n  {'--- Equity Curve ---':.<35}")
        print(f"  {'Starting Equity':.<35} ${ec[0]:>10,.2f}")
        print(f"  {'Ending Equity':.<35} ${ec[-1]:>10,.2f}")
        print(f"  {'Peak Equity':.<35} ${np.max(ec):>10,.2f}")
        print(f"  {'Trough Equity':.<35} ${np.min(ec):>10,.2f}")

    # ── 4-edge vs 9-edge comparison ──
    print(f"\n  {'--- 4-Edge vs 9-Edge Comparison ---':.<35}")
    print(f"  Previous 4-edge SR:  ~+3.56  (PBMR + FRM + RADL + IMC)")
    print(f"  Current 9-edge SR:   {results.sharpe_ratio:>+6.3f}  (+ LCP + CED + VTS + VF + VV)")
    improvement = ((results.sharpe_ratio / 3.56) - 1) * 100 if results.sharpe_ratio > 0 else 0
    print(f"  Improvement:         {improvement:>+5.1f}%")

    # Yearly breakdown
    dr = np.array(results.daily_returns) if results.daily_returns else np.array([])
    if len(dr) > 365:
        n_years = len(dr) // 365
        print(f"\n  {'--- Yearly Breakdown ---':.<35}")
        for y in range(n_years + 1):
            start_idx = y * 365
            end_idx = min((y + 1) * 365, len(dr))
            if start_idx >= len(dr):
                break
            year_dr = dr[start_idx:end_idx]
            if len(year_dr) < 30:
                continue
            year_ret = np.prod(1 + year_dr) - 1
            year_vol = np.std(year_dr, ddof=1) * np.sqrt(365)
            year_sr = np.mean(year_dr) / np.std(year_dr, ddof=1) * np.sqrt(365) if np.std(year_dr) > 0 else 0
            print(f"  Year {y+1}: Ret={year_ret:>+8.1%}  Vol={year_vol:>6.1%}  SR={year_sr:>+6.3f}")

    # Monthly stats
    if len(dr) > 30:
        n_months = len(dr) // 30
        monthly_rets = []
        for m in range(n_months):
            month_dr = dr[m * 30:(m + 1) * 30]
            monthly_rets.append(np.prod(1 + month_dr) - 1)
        monthly_rets = np.array(monthly_rets)

        print(f"\n  {'--- Monthly Stats ---':.<35}")
        print(f"  {'Mean Monthly Return':.<35} {np.mean(monthly_rets):>+10.2%}")
        print(f"  {'Median Monthly Return':.<35} {np.median(monthly_rets):>+10.2%}")
        print(f"  {'Best Month':.<35} {np.max(monthly_rets):>+10.2%}")
        print(f"  {'Worst Month':.<35} {np.min(monthly_rets):>+10.2%}")
        print(f"  {'% Positive Months':.<35} {np.mean(monthly_rets > 0):>10.1%}")

    # ── $10K Projection ──
    if len(dr) > 0:
        print(f"\n  {'--- $10K Capital Projection ---':.<35}")
        cap = 10_000.0
        monthly_pnl = []
        for m in range(len(dr) // 30):
            month_dr = dr[m * 30:(m + 1) * 30]
            month_ret = np.prod(1 + month_dr) - 1
            monthly_pnl.append(cap * month_ret)
            cap += cap * month_ret

        if monthly_pnl:
            avg_monthly = np.mean(monthly_pnl[:12])  # First year avg
            print(f"  {'Avg monthly P&L (year 1)':.<35} ${avg_monthly:>+10,.0f}")
            n_mo = len(monthly_pnl)
            print(f"  {f'Final equity ({n_mo}mo)':.<35} ${cap:>10,.0f}")

            # Find month where monthly PnL > $10K
            for i, pnl in enumerate(monthly_pnl):
                if pnl >= 10_000:
                    print(f"  {'First $10K/month':.<35} Month {i+1}")
                    break

    print("\n" + "=" * 70)
    print("  PHANTOM ALPHA ENGINE — 9-Edge Backtest Complete")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_9edge_backtest()
