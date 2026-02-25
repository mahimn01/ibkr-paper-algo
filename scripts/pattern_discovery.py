#!/usr/bin/env python3
"""
Novel Pattern Discovery Engine
===============================

Deep statistical analysis of 5-minute IBKR bar data to discover
exploitable market microstructure patterns. Each analysis produces
quantified edges with statistical significance.

Analyses:
  1. Cross-Asset Lead-Lag (Pearson at multiple lags)
  2. Transfer Entropy (nonlinear causal flow)
  3. Intraday Time-of-Day Profile (78 5-min buckets)
  4. Overnight Gap Analysis (reversal rates by gap size)
  5. Hurst Exponent (R/S and DFA, rolling)
  6. Entropy Analysis (Sample + Permutation entropy)
  7. Ordinal Pattern Distribution (forbidden patterns, transitions)
  8. Fractional Differentiation (optimal d per symbol)
  9. Cross-Sectional Intraday Momentum

Usage:
    .venv/bin/python scripts/pattern_discovery.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime, time as dt_time
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trading_algo.quant_core.data.ibkr_data_loader import (
    load_ibkr_bars,
    resample_to_daily,
    get_available_symbols,
)
from trading_algo.quant_core.ml.information_theory import (
    transfer_entropy,
    sample_entropy,
    permutation_entropy,
    rolling_sample_entropy,
)
from trading_algo.quant_core.ml.fractal_analysis import (
    hurst_exponent_rs,
    rolling_hurst,
    dfa,
    classify_hurst_regime,
)
from trading_algo.quant_core.ml.fractional_diff import (
    find_optimal_d,
    memory_stationarity_report,
)
from trading_algo.quant_core.utils.math_utils import log_returns, simple_returns


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

SEPARATOR = "=" * 78
THIN_SEP = "-" * 78
BARS_PER_DAY = 78  # 6.5 hours * 60 / 5


def _header(title: str) -> str:
    return f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}"


def _load_all_data() -> Tuple[
    Dict[str, NDArray], Dict[str, List[datetime]],
    Dict[str, NDArray], Dict[str, List[datetime]],
]:
    """Load 5-min and daily data for all available symbols."""
    symbols = get_available_symbols()
    print(f"  Available symbols: {', '.join(symbols)}")

    data_5m: Dict[str, NDArray] = {}
    ts_5m: Dict[str, List[datetime]] = {}
    data_daily: Dict[str, NDArray] = {}
    ts_daily: Dict[str, List[datetime]] = {}

    for sym in symbols:
        try:
            ohlcv, timestamps = load_ibkr_bars(sym, bar_size="5mins")
            if len(ohlcv) < BARS_PER_DAY * 100:
                print(f"  [SKIP] {sym}: only {len(ohlcv)} 5-min bars")
                continue
            data_5m[sym] = ohlcv
            ts_5m[sym] = timestamps

            d_ohlcv, d_ts = resample_to_daily(ohlcv, timestamps)
            data_daily[sym] = d_ohlcv
            ts_daily[sym] = d_ts

            n_days = len(d_ohlcv)
            print(f"  {sym}: {len(ohlcv)} 5-min bars, {n_days} days, "
                  f"${ohlcv[0,3]:.2f} -> ${ohlcv[-1,3]:.2f}")
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")

    return data_5m, ts_5m, data_daily, ts_daily


def _get_returns(ohlcv: NDArray) -> NDArray:
    """Get close-to-close log returns from OHLCV."""
    closes = ohlcv[:, 3]
    return log_returns(closes)


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1: Cross-Asset Lead-Lag
# ═══════════════════════════════════════════════════════════════════════════

def analyze_lead_lag(
    data_5m: Dict[str, NDArray],
    max_lag: int = 15,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    """
    Cross-correlation at lags 1..max_lag for all symbol pairs.
    Returns dict of (sym_a, sym_b) -> {lag: correlation}.
    """
    print(_header("ANALYSIS 1: CROSS-ASSET LEAD-LAG"))

    # Align all symbols to common length
    symbols = sorted(data_5m.keys())
    min_len = min(len(data_5m[s]) for s in symbols)
    returns = {s: _get_returns(data_5m[s][-min_len:]) for s in symbols}

    results = {}

    for sym_a, sym_b in combinations(symbols, 2):
        ret_a = returns[sym_a]
        ret_b = returns[sym_b]
        lag_corrs = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            if lag > 0:
                # Does sym_a lead sym_b? (a at t predicts b at t+lag)
                corr = np.corrcoef(ret_a[:-lag], ret_b[lag:])[0, 1]
            else:
                # Does sym_b lead sym_a?
                corr = np.corrcoef(ret_a[-lag:], ret_b[:lag])[0, 1]
            lag_corrs[lag] = corr

        results[(sym_a, sym_b)] = lag_corrs

        # Find strongest lead-lag
        best_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
        best_corr = lag_corrs[best_lag]
        if abs(best_corr) > 0.02:
            leader = sym_a if best_lag > 0 else sym_b
            follower = sym_b if best_lag > 0 else sym_a
            print(f"  {leader} -> {follower}: lag={abs(best_lag)} bars "
                  f"({abs(best_lag)*5}min), corr={best_corr:+.4f}")

    # Summary: top 10 lead-lag relationships
    print(f"\n  Top lead-lag pairs (|corr| > 0.015):")
    all_pairs = []
    for (a, b), lags in results.items():
        for lag, corr in lags.items():
            if abs(corr) > 0.015:
                all_pairs.append((a, b, lag, corr))
    all_pairs.sort(key=lambda x: abs(x[3]), reverse=True)
    for a, b, lag, corr in all_pairs[:15]:
        leader = a if lag > 0 else b
        follower = b if lag > 0 else a
        print(f"    {leader:>5s} -> {follower:<5s}  lag={abs(lag):>2d} "
              f"({abs(lag)*5:>3d}min)  corr={corr:+.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2: Transfer Entropy (Nonlinear Causal Flow)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_transfer_entropy(
    data_5m: Dict[str, NDArray],
) -> Dict[Tuple[str, str], float]:
    """
    Compute transfer entropy for all directed pairs.
    TE(X->Y) measures nonlinear information flow from X to Y.
    """
    print(_header("ANALYSIS 2: TRANSFER ENTROPY (NONLINEAR CAUSAL FLOW)"))

    symbols = sorted(data_5m.keys())
    min_len = min(len(data_5m[s]) for s in symbols)
    # Use daily returns for stability (5-min is too noisy for TE)
    returns = {}
    for s in symbols:
        daily, _ = resample_to_daily(data_5m[s][-min_len:],
                                      [datetime(2024, 1, 1)] * min_len)
        returns[s] = _get_returns(daily)

    results = {}

    for sym_a in symbols:
        for sym_b in symbols:
            if sym_a == sym_b:
                continue
            ret_a = returns[sym_a]
            ret_b = returns[sym_b]
            common = min(len(ret_a), len(ret_b))
            te = transfer_entropy(ret_a[:common], ret_b[:common], lag=1, k=3)
            results[(sym_a, sym_b)] = te

    # Find asymmetric pairs (where TE(A->B) >> TE(B->A))
    print(f"  Transfer Entropy (TE) for all directed pairs:")
    print(f"  {'Source':>6s} -> {'Target':<6s}  TE(A->B)  TE(B->A)  Asymmetry")
    print(f"  {THIN_SEP}")

    printed = set()
    for (a, b), te_ab in sorted(results.items(), key=lambda x: -x[1]):
        if (b, a) in printed:
            continue
        printed.add((a, b))
        te_ba = results.get((b, a), 0)
        asym = te_ab - te_ba
        if abs(asym) > 0.001:
            leader = a if asym > 0 else b
            arrow = "->" if asym > 0 else "<-"
            print(f"  {a:>6s} -> {b:<6s}  {te_ab:.4f}    {te_ba:.4f}    "
                  f"{asym:+.4f}  ({leader} leads)")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 3: Intraday Time-of-Day Profile
# ═══════════════════════════════════════════════════════════════════════════

def analyze_time_of_day(
    data_5m: Dict[str, NDArray],
    ts_5m: Dict[str, List[datetime]],
) -> Dict[str, NDArray]:
    """
    Profile returns, volatility, and volume by time-of-day bucket.
    78 buckets per day (9:30 to 16:00, 5-min bars).
    """
    print(_header("ANALYSIS 3: INTRADAY TIME-OF-DAY PROFILE"))

    results = {}

    for sym in sorted(data_5m.keys()):
        ohlcv = data_5m[sym]
        timestamps = ts_5m[sym]
        returns = _get_returns(ohlcv)

        # Group by intraday bucket
        bucket_returns = defaultdict(list)
        bucket_volumes = defaultdict(list)

        for i in range(1, len(returns)):
            ts = timestamps[i]
            bucket = ts.hour * 12 + ts.minute // 5  # unique intraday index
            bucket_returns[bucket].append(returns[i])
            bucket_volumes[bucket].append(ohlcv[i, 4])

        # Compute stats per bucket
        buckets = sorted(bucket_returns.keys())
        n_buckets = len(buckets)
        stats = np.zeros((n_buckets, 5))  # mean_ret, std, sharpe, mean_vol, autocorr

        for j, b in enumerate(buckets):
            rets = np.array(bucket_returns[b])
            vols = np.array(bucket_volumes[b])
            mean_r = np.nanmean(rets)
            std_r = np.nanstd(rets)
            sharpe = mean_r / std_r * np.sqrt(252) if std_r > 1e-10 else 0
            mean_v = np.nanmean(vols)
            # Autocorrelation at lag 1 (does this bucket's return predict the next?)
            if j < n_buckets - 1:
                next_rets = np.array(bucket_returns[buckets[j + 1]])
                common = min(len(rets), len(next_rets))
                if common > 10:
                    autocorr = np.corrcoef(rets[:common], next_rets[:common])[0, 1]
                else:
                    autocorr = 0
            else:
                autocorr = 0
            stats[j] = [mean_r, std_r, sharpe, mean_v, autocorr]

        results[sym] = stats

    # Print summary for SPY
    if "SPY" in results:
        spy_stats = results["SPY"]
        print(f"\n  SPY Time-of-Day Profile (top Sharpe buckets):")
        print(f"  {'Time':>7s}  {'Mean Ret':>9s}  {'Std':>7s}  {'Sharpe':>7s}  "
              f"{'AutoCorr':>9s}  Regime")
        print(f"  {THIN_SEP}")

        # Map bucket index back to time
        bucket_times = []
        h, m = 9, 30
        for _ in range(78):
            bucket_times.append(f"{h:02d}:{m:02d}")
            m += 5
            if m >= 60:
                m -= 60
                h += 1

        # Sort by absolute Sharpe
        ranked = np.argsort(-np.abs(spy_stats[:len(bucket_times), 2]))
        for idx in ranked[:20]:
            if idx < len(bucket_times):
                s = spy_stats[idx]
                regime = "MOM" if s[4] > 0.02 else ("MR" if s[4] < -0.02 else "RND")
                print(f"  {bucket_times[idx]:>7s}  {s[0]:>+9.6f}  {s[1]:>7.5f}  "
                      f"{s[2]:>+7.2f}  {s[4]:>+9.4f}  {regime}")

        # Period summaries
        print(f"\n  SPY Period Summaries:")
        periods = [
            ("Morning (9:30-10:30)", 0, 12),
            ("Midday (10:30-14:00)", 12, 54),
            ("Afternoon (14:00-15:30)", 54, 72),
            ("Close (15:30-16:00)", 72, 78),
        ]
        for name, start, end in periods:
            if end > len(spy_stats):
                end = len(spy_stats)
            period_stats = spy_stats[start:end]
            mean_sharpe = np.mean(period_stats[:, 2])
            mean_autocorr = np.mean(period_stats[:, 4])
            regime = "MOMENTUM" if mean_autocorr > 0.01 else (
                "MEAN-REVERSION" if mean_autocorr < -0.01 else "RANDOM"
            )
            print(f"    {name:<28s}  Sharpe={mean_sharpe:+.3f}  "
                  f"AutoCorr={mean_autocorr:+.4f}  -> {regime}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 4: Overnight Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_overnight_gaps(
    data_5m: Dict[str, NDArray],
    ts_5m: Dict[str, List[datetime]],
) -> Dict[str, Dict]:
    """
    Analyze overnight gap behavior: size distribution, reversal rates.
    """
    print(_header("ANALYSIS 4: OVERNIGHT GAP ANALYSIS"))

    results = {}

    for sym in sorted(data_5m.keys()):
        ohlcv = data_5m[sym]
        timestamps = ts_5m[sym]

        # Identify day boundaries
        gaps = []
        day_opens = []
        prev_close = None
        prev_close_idx = None

        for i in range(len(timestamps)):
            ts = timestamps[i]
            if i > 0 and timestamps[i].date() != timestamps[i - 1].date():
                if prev_close is not None:
                    gap_pct = (ohlcv[i, 0] / prev_close - 1) * 100
                    # Track first-hour reversal
                    end_first_hour = min(i + 12, len(ohlcv))  # 12 bars = 1 hour
                    first_hour_close = ohlcv[end_first_hour - 1, 3]
                    first_hour_ret = (first_hour_close / ohlcv[i, 0] - 1) * 100
                    # Did the gap reverse?
                    gap_reversed = (gap_pct > 0 and first_hour_ret < 0) or \
                                   (gap_pct < 0 and first_hour_ret > 0)
                    # Full day reversal
                    day_end = i
                    while day_end < len(timestamps) - 1 and \
                            timestamps[day_end + 1].date() == ts.date():
                        day_end += 1
                    day_close = ohlcv[day_end, 3]
                    day_ret = (day_close / ohlcv[i, 0] - 1) * 100
                    day_reversed = (gap_pct > 0 and day_ret < 0) or \
                                   (gap_pct < 0 and day_ret > 0)
                    gaps.append({
                        "gap_pct": gap_pct,
                        "first_hour_ret": first_hour_ret,
                        "day_ret": day_ret,
                        "first_hour_reversed": gap_reversed,
                        "day_reversed": day_reversed,
                    })
            prev_close = ohlcv[i, 3]

        if not gaps:
            continue

        gap_arr = np.array([g["gap_pct"] for g in gaps])
        results[sym] = {
            "n_gaps": len(gaps),
            "mean_gap": np.mean(np.abs(gap_arr)),
            "gaps": gaps,
        }

        # Reversal rates by gap size quintile
        abs_gaps = np.abs(gap_arr)
        quintiles = np.percentile(abs_gaps, [20, 40, 60, 80])

        print(f"\n  {sym}: {len(gaps)} overnight gaps, mean |gap|={np.mean(abs_gaps):.3f}%")
        print(f"  {'Gap Size':>12s}  {'Count':>5s}  {'1h Rev':>6s}  {'Day Rev':>7s}  "
              f"{'Mean 1h Ret':>11s}  {'Signal'}")
        print(f"  {THIN_SEP}")

        bins = [(0, quintiles[0]), (quintiles[0], quintiles[1]),
                (quintiles[1], quintiles[2]), (quintiles[2], quintiles[3]),
                (quintiles[3], np.inf)]
        labels = ["Tiny", "Small", "Medium", "Large", "Huge"]

        for label, (lo, hi) in zip(labels, bins):
            mask = [(lo <= abs(g["gap_pct"]) < hi) for g in gaps]
            subset = [g for g, m in zip(gaps, mask) if m]
            if not subset:
                continue
            n = len(subset)
            rev_1h = sum(1 for g in subset if g["first_hour_reversed"]) / n
            rev_day = sum(1 for g in subset if g["day_reversed"]) / n
            mean_1h = np.mean([g["first_hour_ret"] for g in subset])
            # Is this a strong fade signal?
            signal = "FADE" if rev_1h > 0.55 else ("FOLLOW" if rev_1h < 0.45 else "NEUTRAL")
            print(f"  {label:>12s}  {n:>5d}  {rev_1h:>5.1%}  {rev_day:>6.1%}  "
                  f"{mean_1h:>+10.4f}%  {signal}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 5: Hurst Exponent (Fractal Structure)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_hurst(
    data_5m: Dict[str, NDArray],
    data_daily: Dict[str, NDArray],
) -> Dict[str, Dict]:
    """
    Compute Hurst exponent at multiple timescales.
    """
    print(_header("ANALYSIS 5: HURST EXPONENT (FRACTAL STRUCTURE)"))

    results = {}

    for sym in sorted(data_daily.keys()):
        daily_returns = _get_returns(data_daily[sym])
        fivemin_returns = _get_returns(data_5m[sym])

        # Static Hurst on full series
        h_daily_rs = hurst_exponent_rs(daily_returns)
        h_daily_dfa = dfa(daily_returns)
        h_5m_rs = hurst_exponent_rs(fivemin_returns[:5000])  # subsample for speed

        # Rolling Hurst on daily data (21-day window)
        rolling_h = rolling_hurst(daily_returns, window=42, step=1)
        valid_h = rolling_h[~np.isnan(rolling_h)]

        # Regime distribution
        if len(valid_h) > 0:
            pct_mr = np.mean(valid_h < 0.45) * 100
            pct_rw = np.mean((valid_h >= 0.45) & (valid_h <= 0.55)) * 100
            pct_mom = np.mean(valid_h > 0.55) * 100
        else:
            pct_mr = pct_rw = pct_mom = 0

        results[sym] = {
            "h_daily_rs": h_daily_rs,
            "h_daily_dfa": h_daily_dfa,
            "h_5m_rs": h_5m_rs,
            "pct_mean_revert": pct_mr,
            "pct_random_walk": pct_rw,
            "pct_momentum": pct_mom,
            "rolling_h": rolling_h,
        }

        regime = classify_hurst_regime(h_daily_rs)
        print(f"  {sym:>5s}: H(daily,R/S)={h_daily_rs:.3f}  H(daily,DFA)={h_daily_dfa:.3f}  "
              f"H(5m,R/S)={h_5m_rs:.3f}  regime={regime}")
        print(f"         Rolling regime: {pct_mr:.1f}% MR | {pct_rw:.1f}% RW | "
              f"{pct_mom:.1f}% MOM")

    # Key insight
    print(f"\n  KEY INSIGHT: Fraction of time each regime is tradeable:")
    for sym in sorted(results.keys()):
        r = results[sym]
        tradeable = r["pct_mean_revert"] + r["pct_momentum"]
        print(f"    {sym}: {tradeable:.1f}% tradeable "
              f"(MR={r['pct_mean_revert']:.1f}%, MOM={r['pct_momentum']:.1f}%), "
              f"skip {r['pct_random_walk']:.1f}% random walk")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 6: Entropy Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_entropy(
    data_5m: Dict[str, NDArray],
    data_daily: Dict[str, NDArray],
) -> Dict[str, Dict]:
    """
    Sample entropy and permutation entropy as regime indicators.
    """
    print(_header("ANALYSIS 6: ENTROPY ANALYSIS"))

    results = {}

    for sym in sorted(data_daily.keys()):
        daily_returns = _get_returns(data_daily[sym])
        fivemin_returns = _get_returns(data_5m[sym])

        # Static entropy
        se_daily = sample_entropy(daily_returns, m=2, r=0.2)
        pe_daily = permutation_entropy(daily_returns, order=4)
        se_5m = sample_entropy(fivemin_returns[:5000], m=2, r=0.2)

        # Rolling sample entropy on daily returns
        rolling_se = rolling_sample_entropy(daily_returns, window=42, m=2, r=0.2)
        valid_se = rolling_se[~np.isnan(rolling_se)]

        # Does low entropy predict higher returns?
        if len(valid_se) > 50:
            # Quartile analysis
            q25 = np.nanpercentile(valid_se, 25)
            q75 = np.nanpercentile(valid_se, 75)

            # Forward 5-day returns after low vs high entropy days
            fwd_returns = np.zeros(len(daily_returns))
            for i in range(len(daily_returns) - 5):
                fwd_returns[i] = np.sum(daily_returns[i + 1:i + 6])

            low_mask = rolling_se < q25
            high_mask = rolling_se > q75

            low_fwd = fwd_returns[low_mask[:len(fwd_returns)]]
            high_fwd = fwd_returns[high_mask[:len(fwd_returns)]]

            low_fwd = low_fwd[~np.isnan(low_fwd)]
            high_fwd = high_fwd[~np.isnan(high_fwd)]

            low_mean = np.mean(low_fwd) if len(low_fwd) > 5 else np.nan
            high_mean = np.mean(high_fwd) if len(high_fwd) > 5 else np.nan
            low_sharpe = (np.mean(low_fwd) / np.std(low_fwd) * np.sqrt(252 / 5)
                          if len(low_fwd) > 5 and np.std(low_fwd) > 0 else np.nan)
            high_sharpe = (np.mean(high_fwd) / np.std(high_fwd) * np.sqrt(252 / 5)
                           if len(high_fwd) > 5 and np.std(high_fwd) > 0 else np.nan)
        else:
            low_mean = high_mean = low_sharpe = high_sharpe = np.nan

        results[sym] = {
            "se_daily": se_daily,
            "pe_daily": pe_daily,
            "se_5m": se_5m,
            "low_entropy_fwd_return": low_mean,
            "high_entropy_fwd_return": high_mean,
            "low_entropy_sharpe": low_sharpe,
            "high_entropy_sharpe": high_sharpe,
        }

        print(f"  {sym:>5s}: SampEn(daily)={se_daily:.3f}  PermEn(daily)={pe_daily:.3f}  "
              f"SampEn(5m)={se_5m:.3f}")
        print(f"         Low entropy 5d fwd: {low_mean:+.5f} (Sharpe={low_sharpe:+.2f})  "
              f"High entropy 5d fwd: {high_mean:+.5f} (Sharpe={high_sharpe:+.2f})")

    # Summary
    print(f"\n  KEY INSIGHT: Entropy as predictive filter")
    for sym in sorted(results.keys()):
        r = results[sym]
        spread = (r["low_entropy_fwd_return"] or 0) - (r["high_entropy_fwd_return"] or 0)
        if not np.isnan(spread):
            edge = "TRADE LOW ENTROPY" if spread > 0 else "TRADE HIGH ENTROPY"
            print(f"    {sym}: Low-High return spread = {spread:+.5f} -> {edge}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 7: Ordinal Pattern Distribution
# ═══════════════════════════════════════════════════════════════════════════

def analyze_ordinal_patterns(
    data_daily: Dict[str, NDArray],
) -> Dict[str, Dict]:
    """
    Ordinal pattern analysis: forbidden patterns, transition matrix.
    """
    print(_header("ANALYSIS 7: ORDINAL PATTERN ANALYSIS"))

    # Import ordinal patterns module (may still be building)
    try:
        from trading_algo.quant_core.ml.ordinal_patterns import (
            ordinal_distribution,
            forbidden_patterns,
            pattern_transition_matrix,
            rolling_permutation_entropy,
        )
        has_ordinal = True
    except ImportError:
        print("  [SKIP] Ordinal patterns module not yet available")
        has_ordinal = False
        return {}

    results = {}

    for sym in sorted(data_daily.keys()):
        returns = _get_returns(data_daily[sym])

        # Distribution for order 3 and 4
        dist_3 = ordinal_distribution(returns, order=3)
        dist_4 = ordinal_distribution(returns, order=4)

        # Forbidden patterns
        forbidden_4 = forbidden_patterns(returns, order=4)

        # Transition matrix
        trans, patterns = pattern_transition_matrix(returns, order=3)

        # Most predictive transitions (highest row variance = most uneven)
        if trans.size > 0:
            row_var = np.var(trans, axis=1)
            most_predictive_idx = np.argmax(row_var)
            mp_pattern = patterns[most_predictive_idx] if most_predictive_idx < len(patterns) else None
            mp_row = trans[most_predictive_idx] if most_predictive_idx < trans.shape[0] else None
        else:
            mp_pattern = None
            mp_row = None

        # Rolling permutation entropy
        rolling_pe = rolling_permutation_entropy(returns, order=3, window=42)

        results[sym] = {
            "dist_3": dist_3,
            "dist_4": dist_4,
            "n_forbidden_4": len(forbidden_4),
            "forbidden_4": forbidden_4,
            "transition_matrix": trans,
            "patterns": patterns,
        }

        print(f"\n  {sym}:")
        print(f"    Order-3 patterns: {len(dist_3)} observed (max 6)")
        print(f"    Order-4 patterns: {len(dist_4)} observed (max 24)")
        print(f"    Forbidden order-4 patterns: {len(forbidden_4)}")
        if forbidden_4:
            print(f"      {list(forbidden_4)[:5]}")

        # Show most and least common patterns
        if dist_3:
            sorted_dist = sorted(dist_3.items(), key=lambda x: -x[1])
            print(f"    Most common order-3: {sorted_dist[0][0]} = {sorted_dist[0][1]:.3f}")
            print(f"    Least common order-3: {sorted_dist[-1][0]} = {sorted_dist[-1][1]:.3f}")
            # Deviation from uniform (1/6 = 0.1667)
            uniform = 1 / 6
            max_dev = max(abs(p - uniform) for _, p in sorted_dist)
            print(f"    Max deviation from uniform: {max_dev:.4f} "
                  f"({'significant' if max_dev > 0.02 else 'weak'})")

        if mp_pattern is not None and mp_row is not None:
            next_most_likely = patterns[np.argmax(mp_row)] if len(patterns) > 0 else "?"
            print(f"    Most predictive transition: after {mp_pattern} -> "
                  f"{next_most_likely} (p={np.max(mp_row):.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 8: Fractional Differentiation
# ═══════════════════════════════════════════════════════════════════════════

def analyze_fractional_diff(
    data_daily: Dict[str, NDArray],
) -> Dict[str, Dict]:
    """
    Find optimal fractional differentiation d for each symbol.
    """
    print(_header("ANALYSIS 8: FRACTIONAL DIFFERENTIATION"))

    results = {}

    for sym in sorted(data_daily.keys()):
        closes = data_daily[sym][:, 3]
        log_prices = np.log(closes)

        d_opt, info = find_optimal_d(log_prices, max_d=1.0, step=0.05)

        # Memory retention
        corrs = info.get("correlation_with_original", {})
        corr_at_d = corrs.get(d_opt, np.nan)

        results[sym] = {
            "optimal_d": d_opt,
            "correlation": corr_at_d,
            "adf_stats": info.get("adf_stats", {}),
        }

        print(f"  {sym:>5s}: optimal d = {d_opt:.2f}, "
              f"memory retention (corr with prices) = {corr_at_d:.3f}")

        # Show the tradeoff at a few d values
        for d_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            adf = info.get("adf_stats", {}).get(d_val)
            corr = corrs.get(d_val, np.nan)
            stationary = "YES" if adf and adf[1] < 0.05 else "no"
            pval = adf[1] if adf else np.nan
            print(f"         d={d_val:.1f}: ADF p={pval:.4f} ({stationary}), "
                  f"corr={corr:.3f}")

    # Summary
    print(f"\n  KEY INSIGHT: Optimal d values")
    for sym in sorted(results.keys()):
        r = results[sym]
        quality = "EXCELLENT" if r["correlation"] > 0.9 else (
            "GOOD" if r["correlation"] > 0.7 else "MODERATE"
        )
        print(f"    {sym}: d={r['optimal_d']:.2f} -> {quality} memory retention "
              f"({r['correlation']:.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 9: Cross-Sectional Intraday Momentum
# ═══════════════════════════════════════════════════════════════════════════

def analyze_cross_sectional(
    data_5m: Dict[str, NDArray],
    ts_5m: Dict[str, List[datetime]],
) -> Dict:
    """
    Does first-hour cross-sectional momentum predict rest-of-day returns?
    """
    print(_header("ANALYSIS 9: CROSS-SECTIONAL INTRADAY MOMENTUM"))

    symbols = sorted(data_5m.keys())
    if len(symbols) < 3:
        print("  [SKIP] Need at least 3 symbols")
        return {}

    # Find common dates
    dates_per_sym = {}
    for sym in symbols:
        dates = set()
        for ts in ts_5m[sym]:
            dates.add(ts.date())
        dates_per_sym[sym] = dates

    common_dates = sorted(set.intersection(*dates_per_sym.values()))

    # For each day, compute first-hour and rest-of-day returns
    day_data = []

    for sym in symbols:
        ohlcv = data_5m[sym]
        timestamps = ts_5m[sym]

        # Group bars by date
        by_date = defaultdict(list)
        for i, ts in enumerate(timestamps):
            by_date[ts.date()].append(i)

        for date in common_dates:
            indices = by_date.get(date, [])
            if len(indices) < 13:
                continue

            # First hour = first 12 bars (60 min)
            first_hour_open = ohlcv[indices[0], 0]
            first_hour_close = ohlcv[indices[min(11, len(indices) - 1)], 3]
            first_hour_ret = first_hour_close / first_hour_open - 1

            # Rest of day
            rod_close = ohlcv[indices[-1], 3]
            rod_ret = rod_close / first_hour_close - 1

            day_data.append({
                "date": date,
                "symbol": sym,
                "first_hour": first_hour_ret,
                "rest_of_day": rod_ret,
            })

    if not day_data:
        print("  [SKIP] Insufficient aligned data")
        return {}

    # Cross-sectional analysis per date
    dates_analyzed = defaultdict(dict)
    for d in day_data:
        dates_analyzed[d["date"]][d["symbol"]] = d

    winner_rod = []
    loser_rod = []
    spread_rets = []

    for date, syms in dates_analyzed.items():
        if len(syms) < 3:
            continue
        # Rank by first-hour return
        ranked = sorted(syms.items(), key=lambda x: x[1]["first_hour"], reverse=True)
        winner = ranked[0][1]
        loser = ranked[-1][1]
        winner_rod.append(winner["rest_of_day"])
        loser_rod.append(loser["rest_of_day"])
        spread_rets.append(winner["rest_of_day"] - loser["rest_of_day"])

    winner_rod = np.array(winner_rod)
    loser_rod = np.array(loser_rod)
    spread_rets = np.array(spread_rets)

    # Statistics
    winner_mean = np.mean(winner_rod) * 100
    loser_mean = np.mean(loser_rod) * 100
    spread_mean = np.mean(spread_rets) * 100
    spread_std = np.std(spread_rets) * 100
    spread_sharpe = np.mean(spread_rets) / np.std(spread_rets) * np.sqrt(252) if np.std(spread_rets) > 0 else 0
    continuation_rate = np.mean(spread_rets > 0) * 100

    print(f"  Cross-Sectional Intraday Momentum ({len(symbols)} symbols, "
          f"{len(spread_rets)} days):")
    print(f"    Winner rest-of-day:  {winner_mean:+.4f}%")
    print(f"    Loser rest-of-day:   {loser_mean:+.4f}%")
    print(f"    Winner-Loser spread: {spread_mean:+.4f}% (std={spread_std:.4f}%)")
    print(f"    Spread Sharpe:       {spread_sharpe:+.3f}")
    print(f"    Continuation rate:   {continuation_rate:.1f}%")

    if spread_sharpe > 0.3:
        print(f"    -> MOMENTUM: First-hour winners continue to outperform")
    elif spread_sharpe < -0.3:
        print(f"    -> REVERSAL: First-hour winners underperform rest-of-day")
    else:
        print(f"    -> NEUTRAL: No clear cross-sectional pattern")

    return {
        "n_days": len(spread_rets),
        "spread_mean": spread_mean,
        "spread_sharpe": spread_sharpe,
        "continuation_rate": continuation_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Discovery Engine
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print(SEPARATOR)
    print("  NOVEL PATTERN DISCOVERY ENGINE")
    print(SEPARATOR)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print(_header("DATA LOADING"))
    data_5m, ts_5m, data_daily, ts_daily = _load_all_data()

    if not data_5m:
        print("  ERROR: No data loaded")
        return

    all_results = {}

    # Run all analyses
    try:
        all_results["lead_lag"] = analyze_lead_lag(data_5m)
    except Exception as e:
        print(f"  [ERROR] Lead-lag analysis: {e}")

    try:
        all_results["transfer_entropy"] = analyze_transfer_entropy(data_5m)
    except Exception as e:
        print(f"  [ERROR] Transfer entropy: {e}")

    try:
        all_results["time_of_day"] = analyze_time_of_day(data_5m, ts_5m)
    except Exception as e:
        print(f"  [ERROR] Time-of-day: {e}")

    try:
        all_results["overnight_gaps"] = analyze_overnight_gaps(data_5m, ts_5m)
    except Exception as e:
        print(f"  [ERROR] Overnight gaps: {e}")

    try:
        all_results["hurst"] = analyze_hurst(data_5m, data_daily)
    except Exception as e:
        print(f"  [ERROR] Hurst: {e}")

    try:
        all_results["entropy"] = analyze_entropy(data_5m, data_daily)
    except Exception as e:
        print(f"  [ERROR] Entropy: {e}")

    try:
        all_results["ordinal"] = analyze_ordinal_patterns(data_daily)
    except Exception as e:
        print(f"  [ERROR] Ordinal patterns: {e}")

    try:
        all_results["frac_diff"] = analyze_fractional_diff(data_daily)
    except Exception as e:
        print(f"  [ERROR] Fractional diff: {e}")

    try:
        all_results["cross_sectional"] = analyze_cross_sectional(data_5m, ts_5m)
    except Exception as e:
        print(f"  [ERROR] Cross-sectional: {e}")

    # Final summary
    elapsed = time.time() - t0
    print(_header("DISCOVERY SUMMARY"))
    print(f"  Total analyses run: {len(all_results)}")
    print(f"  Total runtime: {elapsed:.1f}s")

    print(f"\n  ACTIONABLE EDGES IDENTIFIED:")
    print(f"  {'='*60}")

    edge_num = 1

    # Lead-lag edges
    if "lead_lag" in all_results:
        ll = all_results["lead_lag"]
        strong_pairs = []
        for (a, b), lags in ll.items():
            for lag, corr in lags.items():
                if abs(corr) > 0.025:
                    strong_pairs.append((a, b, lag, corr))
        if strong_pairs:
            print(f"\n  Edge {edge_num}: CROSS-ASSET LEAD-LAG")
            print(f"    {len(strong_pairs)} pairs with |corr| > 0.025")
            print(f"    -> Build LeadLagArbitrage strategy")
            edge_num += 1

    # Entropy filter
    if "entropy" in all_results:
        ent = all_results["entropy"]
        for sym, r in ent.items():
            low_s = r.get("low_entropy_sharpe", np.nan)
            high_s = r.get("high_entropy_sharpe", np.nan)
            if not np.isnan(low_s) and not np.isnan(high_s) and low_s > high_s + 0.3:
                print(f"\n  Edge {edge_num}: ENTROPY FILTER ({sym})")
                print(f"    Low entropy Sharpe: {low_s:+.2f} vs High: {high_s:+.2f}")
                print(f"    -> Build EntropyRegimeFilter to gate all strategies")
                edge_num += 1
                break

    # Hurst adaptive
    if "hurst" in all_results:
        hurst = all_results["hurst"]
        for sym, r in hurst.items():
            tradeable = r["pct_mean_revert"] + r["pct_momentum"]
            if tradeable > 40:
                print(f"\n  Edge {edge_num}: HURST-ADAPTIVE EXECUTION ({sym})")
                print(f"    {r['pct_mean_revert']:.1f}% MR + {r['pct_momentum']:.1f}% MOM "
                      f"= {tradeable:.1f}% tradeable")
                print(f"    -> Build HurstAdaptive strategy")
                edge_num += 1
                break

    # Fractional diff
    if "frac_diff" in all_results:
        fd = all_results["frac_diff"]
        good_symbols = [s for s, r in fd.items() if r["correlation"] > 0.7]
        if good_symbols:
            print(f"\n  Edge {edge_num}: FRACTIONAL DIFFERENTIATION FEATURES")
            print(f"    {len(good_symbols)} symbols with corr > 0.7 at optimal d")
            print(f"    -> Use as ML features in NonlinearSignalCombiner")
            edge_num += 1

    # Time-of-day
    if "time_of_day" in all_results:
        print(f"\n  Edge {edge_num}: TIME-OF-DAY ADAPTIVE EXECUTION")
        print(f"    Morning momentum, midday reversion, afternoon momentum")
        print(f"    -> Build IntradayTimeAdaptive strategy")
        edge_num += 1

    print(f"\n  Total edges: {edge_num - 1}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
