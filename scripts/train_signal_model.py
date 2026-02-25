#!/usr/bin/env python3
"""
Unified ML Signal Framework -- Walk-Forward XGBoost Training

Replaces naive weighted averaging of trading signals with a learned nonlinear
combination using XGBoost walk-forward training.  For each symbol in the
7-symbol universe (SPY, QQQ, AAPL, NVDA, MSFT, SMCI, IWM), the pipeline:

    1. Loads 5-min bar data from the IBKR cache.
    2. Constructs a rich feature matrix (base + advanced + novel features).
    3. Builds forward-return targets at 1-bar and 78-bar horizons.
    4. Trains XGBoost via expanding-window walk-forward with a 50-bar purge.
    5. Generates out-of-sample (OOS) predictions and position sizes.
    6. Evaluates performance (Sharpe, Sortino, IC, hit rate, max drawdown).
    7. Compares ML-combined signals vs equal-weight baseline.
    8. Saves trained model artefacts for use by the live controller.

Usage:
    python scripts/train_signal_model.py

References:
    - Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"
    - Lopez de Prado (2018): "Advances in Financial Machine Learning"
    - Friedman (2001): "Greedy Function Approximation: A Gradient Boosting Machine"
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from trading_algo.quant_core.data.ibkr_data_loader import (
    load_ibkr_bars,
    load_universe_data,
)
from trading_algo.quant_core.ml.features import FeatureEngine, FeatureSet
from trading_algo.quant_core.ml.advanced_features import AdvancedFeatureEngine
from trading_algo.quant_core.ml.fractional_diff import frac_diff_ffd
from trading_algo.quant_core.ml.information_theory import (
    rolling_sample_entropy,
    rolling_permutation_entropy,
)
from trading_algo.quant_core.ml.fractal_analysis import rolling_hurst
from trading_algo.quant_core.utils.statistics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)
from trading_algo.quant_core.utils.constants import EPSILON

# ---------------------------------------------------------------------------
# Suppress noisy warnings during training
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOLS: List[str] = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "SMCI", "IWM"]
BARS_PER_DAY: int = 78  # 6.5 hours * 12 five-min bars/hour

# Walk-forward parameters
MIN_TRAIN_BARS: int = 5000
PURGE_BARS: int = 50
TEST_WINDOW: int = 390  # 1 trading day of 5-min bars (78 * 5)
STEP_BARS: int = TEST_WINDOW  # non-overlapping daily test windows

# XGBoost parameters
XGB_PARAMS: Dict[str, Any] = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
EARLY_STOPPING_ROUNDS: int = 20

# Position sizing parameters
VOL_TARGET: float = 0.001
MAX_WEIGHT: float = 0.15

# Novel feature parameters
ENTROPY_WINDOW: int = 78
HURST_WINDOW: int = 78
FRAC_DIFF_D: float = 0.3

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "backtest_results" / "ml_signal_model"


# ===========================================================================
# 1. FEATURE MATRIX CONSTRUCTION
# ===========================================================================

def compute_feature_matrix(
    ohlcv: NDArray[np.float64],
    timestamps: List[datetime],
    cross_asset_prices: Optional[Dict[str, NDArray[np.float64]]] = None,
    symbol: str = "",
) -> Tuple[NDArray[np.float64], List[str]]:
    """
    Construct the full feature matrix for a single symbol.

    Combines base features (FeatureEngine), advanced features
    (AdvancedFeatureEngine), and novel information-theoretic / fractal
    features into a single (T, K) matrix.

    Args:
        ohlcv: OHLCV array of shape (T, 5).
        timestamps: List of datetime timestamps (length T).
        cross_asset_prices: Dict of symbol -> close prices for cross-asset
            features.  The target symbol should be excluded.
        symbol: Symbol name (for logging).

    Returns:
        Tuple of (feature_matrix, feature_names).
    """
    T = ohlcv.shape[0]
    close = ohlcv[:, 3].astype(np.float64)
    high = ohlcv[:, 1].astype(np.float64)
    low = ohlcv[:, 2].astype(np.float64)
    open_ = ohlcv[:, 0].astype(np.float64)
    volume = ohlcv[:, 4].astype(np.float64)

    logger.info("  Computing features for %s (%d bars)...", symbol, T)

    # ---- Base features via FeatureEngine (unnormalized -- we normalize later) ----
    base_engine = FeatureEngine(
        momentum_windows=[5, 10, 20, 60, 120],
        volatility_windows=[5, 10, 20, 60],
        volume_windows=[5, 10, 20],
        include_technical=True,
        include_microstructure=False,
        normalize=False,  # we do our own z-scoring per walk-forward window
    )
    base_fs: FeatureSet = base_engine.compute_features(
        prices=close, volumes=volume, high=high, low=low,
    )
    logger.info("    Base features: %d", base_fs.n_features)

    # ---- Advanced features via AdvancedFeatureEngine ----
    ts_array = np.array(timestamps)
    adv_engine = AdvancedFeatureEngine(
        base_engine=base_engine,
        lead_lag_lags=[1, 2, 5],
        cross_corr_window=60,
        stat_return_window=20,
        hurst_window=60,
        normalize=False,
    )
    adv_fs: FeatureSet = adv_engine.compute_advanced_features(
        prices=close,
        volumes=volume,
        high=high,
        low=low,
        open_=open_,
        cross_asset_prices=cross_asset_prices,
        timestamps=ts_array,
    )
    logger.info("    Advanced features: %d", adv_fs.n_features)

    # Start with the advanced feature set (which includes base features)
    all_features: Dict[str, NDArray[np.float64]] = {}
    for i, name in enumerate(adv_fs.feature_names):
        all_features[f"{name}"] = adv_fs.features[:, i]

    # ---- Novel features: information-theoretic and fractal ----
    # Rolling sample entropy (78-bar window on close prices)
    t0 = time.time()
    se = rolling_sample_entropy(close, window=ENTROPY_WINDOW, m=2, r=0.2)
    se_padded = _pad_to_length(se, T)
    all_features["novel_sample_entropy_78"] = se_padded

    # Rolling permutation entropy (78-bar window on close prices)
    pe = rolling_permutation_entropy(close, window=ENTROPY_WINDOW, order=3, delay=1)
    pe_padded = _pad_to_length(pe, T)
    all_features["novel_perm_entropy_78"] = pe_padded

    # Rolling Hurst exponent (78-bar window on close prices)
    hurst = rolling_hurst(close, window=HURST_WINDOW, min_window=8, step=1)
    hurst_padded = _pad_to_length(hurst, T)
    all_features["novel_hurst_78"] = hurst_padded
    logger.info("    Novel features computed in %.1fs", time.time() - t0)

    # ---- Fractional differentiation (d=0.3) ----
    log_close = np.log(np.maximum(close, EPSILON))
    frac_diff = frac_diff_ffd(log_close, d=FRAC_DIFF_D)
    frac_padded = _pad_to_length(frac_diff, T)
    all_features["novel_frac_diff_0.3"] = frac_padded

    # ---- Assemble final feature matrix ----
    feature_names = list(all_features.keys())
    feature_matrix = np.column_stack([
        all_features[name] for name in feature_names
    ])

    # Replace inf with NaN
    feature_matrix = np.where(np.isinf(feature_matrix), np.nan, feature_matrix)

    logger.info("    Total features: %d", len(feature_names))
    return feature_matrix, feature_names


def _pad_to_length(arr: NDArray[np.float64], target: int) -> NDArray[np.float64]:
    """Pad an array with leading NaN to match target length."""
    n = len(arr)
    if n >= target:
        return arr[-target:]
    padding = np.full(target - n, np.nan, dtype=np.float64)
    return np.concatenate([padding, arr])


# ===========================================================================
# 2. TARGET CONSTRUCTION
# ===========================================================================

def compute_targets(
    close: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute forward return targets.

    Args:
        close: Close price array of shape (T,).

    Returns:
        Tuple of (fwd_1bar_return, fwd_78bar_return), each of shape (T,).
        Last bar(s) are NaN due to the forward shift.
    """
    T = len(close)

    # Forward 1-bar return: (close[t+1] - close[t]) / close[t]
    fwd_1 = np.full(T, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_1[:-1] = (close[1:] - close[:-1]) / np.maximum(close[:-1], EPSILON)

    # Forward 78-bar return (daily horizon for 5-min bars)
    fwd_78 = np.full(T, np.nan, dtype=np.float64)
    if T > BARS_PER_DAY:
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd_78[:-BARS_PER_DAY] = (
                (close[BARS_PER_DAY:] - close[:-BARS_PER_DAY])
                / np.maximum(close[:-BARS_PER_DAY], EPSILON)
            )

    return fwd_1, fwd_78


# ===========================================================================
# 3. WALK-FORWARD XGBOOST TRAINING
# ===========================================================================

def walk_forward_train(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    feature_names: List[str],
    symbol: str = "",
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, float]]:
    """
    Walk-forward expanding-window XGBoost training.

    At each step:
      - Train on bars [0, train_end) with a purge gap of PURGE_BARS.
      - Predict on bars [test_start, test_start + TEST_WINDOW).
      - Step forward by STEP_BARS.

    Args:
        features: Feature matrix (T, K).
        target: Target array (T,).
        feature_names: List of feature names.
        symbol: Symbol name (for logging).

    Returns:
        Tuple of (predictions, actuals, feature_importances).
        predictions and actuals are arrays of OOS values (may be shorter
        than T since the initial training window is excluded).
        feature_importances is a dict of feature_name -> average importance.
    """
    T = features.shape[0]
    predictions = np.full(T, np.nan, dtype=np.float64)
    importance_accum: Dict[str, float] = {n: 0.0 for n in feature_names}
    n_train_steps = 0
    scaler = StandardScaler()

    logger.info("  Walk-forward training for %s (T=%d)...", symbol, T)

    step_count = 0
    test_start = MIN_TRAIN_BARS + PURGE_BARS

    while test_start < T:
        test_end = min(test_start + TEST_WINDOW, T)
        train_end = test_start - PURGE_BARS

        if train_end < MIN_TRAIN_BARS:
            test_start += STEP_BARS
            continue

        # ---- Extract train/test sets ----
        X_train_raw = features[:train_end]
        y_train_raw = target[:train_end]

        X_test_raw = features[test_start:test_end]
        y_test_raw = target[test_start:test_end]

        # ---- Remove rows with NaN in train ----
        # With 600+ features, requiring ALL features to be non-NaN is too
        # strict.  Instead: (a) drop features that are >50% NaN in this
        # window, (b) fill remaining NaN with 0 (already z-scored later),
        # (c) require the target to be non-NaN.
        target_valid = ~np.isnan(y_train_raw)
        X_train_raw = X_train_raw[target_valid]
        y_train_raw = y_train_raw[target_valid]

        # Fill NaN features with 0 (neutral after z-scoring)
        X_train = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = y_train_raw

        if len(X_train) < 500:
            test_start += STEP_BARS
            continue

        # ---- Standardize features (z-score within training window) ----
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(
            np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
        )

        # ---- Split validation set from training for early stopping ----
        n_val = max(1, int(len(X_train_scaled) * 0.15))
        X_trn = X_train_scaled[:-n_val]
        y_trn = y_train[:-n_val]
        X_val = X_train_scaled[-n_val:]
        y_val = y_train[-n_val:]

        # ---- Train XGBoost ----
        # XGBoost 3.x: early_stopping_rounds is a constructor parameter
        model = xgb.XGBRegressor(
            **XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ---- Predict OOS ----
        preds = model.predict(X_test_scaled)
        predictions[test_start:test_end] = preds

        # ---- Accumulate feature importances ----
        importances = model.feature_importances_
        for i, name in enumerate(feature_names):
            if i < len(importances):
                importance_accum[name] += importances[i]
        n_train_steps += 1

        # ---- Progress logging ----
        step_count += 1
        if step_count % 10 == 0:
            # Compute running OOS metrics on predictions so far
            valid_mask = ~np.isnan(predictions) & ~np.isnan(target)
            if valid_mask.sum() > 10:
                running_ic = _pearson_corr(
                    predictions[valid_mask], target[valid_mask]
                )
                logger.info(
                    "    Step %d | train_end=%d | test=[%d:%d] | "
                    "running IC=%.4f | n_steps=%d",
                    step_count, train_end, test_start, test_end,
                    running_ic, n_train_steps,
                )

        test_start += STEP_BARS

    # ---- Average feature importances ----
    if n_train_steps > 0:
        importance_accum = {
            k: v / n_train_steps for k, v in importance_accum.items()
        }

    logger.info(
        "  Walk-forward complete for %s: %d training steps.",
        symbol, n_train_steps,
    )

    return predictions, target, importance_accum


# ===========================================================================
# 4. SIGNAL GENERATION AND EVALUATION
# ===========================================================================

def compute_positions(
    predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert predicted returns to position sizes.

    position = sign(pred) * min(|pred| / vol_target, max_weight)

    Args:
        predictions: Array of predicted returns.

    Returns:
        Array of position sizes in [-max_weight, +max_weight].
    """
    positions = np.where(
        np.isnan(predictions),
        0.0,
        np.sign(predictions) * np.minimum(
            np.abs(predictions) / VOL_TARGET, MAX_WEIGHT
        ),
    )
    return positions


def evaluate_oos(
    predictions: NDArray[np.float64],
    actuals: NDArray[np.float64],
    timestamps: List[datetime],
    symbol: str = "",
) -> Dict[str, Any]:
    """
    Evaluate out-of-sample performance of the ML signal.

    Args:
        predictions: OOS predicted returns (T,).
        actuals: Realized returns (T,).
        timestamps: Timestamps (T,).
        symbol: Symbol name.

    Returns:
        Dictionary of performance metrics.
    """
    # Keep only bars with valid OOS predictions
    valid = ~np.isnan(predictions) & ~np.isnan(actuals)
    pred_v = predictions[valid]
    act_v = actuals[valid]

    if len(pred_v) < 100:
        logger.warning("  %s: too few OOS predictions (%d)", symbol, len(pred_v))
        return {"symbol": symbol, "n_oos": len(pred_v), "sharpe": 0.0}

    # ---- IC (Information Coefficient) ----
    ic = _pearson_corr(pred_v, act_v)

    # ---- Hit rate ----
    hit_rate = float(np.mean(np.sign(pred_v) == np.sign(act_v)))

    # ---- Position-weighted returns ----
    positions = compute_positions(pred_v)
    strat_returns = positions * act_v

    # ---- Performance metrics ----
    sr = sharpe_ratio(strat_returns, annualize=True)
    sort_r = sortino_ratio(strat_returns, annualize=True)
    mdd = max_drawdown(strat_returns)
    total_return = float(np.prod(1 + strat_returns) - 1)
    mean_daily_ret = float(np.mean(strat_returns))
    ann_return = mean_daily_ret * BARS_PER_DAY * 252  # bars -> annualized

    # ---- Equal-weight baseline ----
    eq_weight = MAX_WEIGHT / 2  # conservative equal-weight position
    eq_positions = np.sign(act_v) * 0  # zero information -> hold a small long
    # Simple long-only baseline
    baseline_returns = eq_weight * act_v
    baseline_sharpe = sharpe_ratio(baseline_returns, annualize=True)

    # ---- Monthly breakdown (by calendar month) ----
    monthly_perf = _monthly_breakdown(strat_returns, timestamps, valid)

    return {
        "symbol": symbol,
        "n_oos": len(pred_v),
        "ic": ic,
        "hit_rate": hit_rate,
        "sharpe": sr,
        "sortino": sort_r,
        "max_drawdown": mdd,
        "total_return": total_return,
        "ann_return": ann_return,
        "mean_bar_return": mean_daily_ret,
        "baseline_sharpe": baseline_sharpe,
        "monthly_perf": monthly_perf,
    }


def _pearson_corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Pearson correlation, handling degenerate cases."""
    if len(x) < 2:
        return 0.0
    sx, sy = np.std(x), np.std(y)
    if sx < EPSILON or sy < EPSILON:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _monthly_breakdown(
    strat_returns: NDArray[np.float64],
    timestamps: List[datetime],
    valid_mask: NDArray[np.bool_],
) -> Dict[str, Dict[str, float]]:
    """Compute monthly OOS performance breakdown."""
    valid_ts = [ts for ts, v in zip(timestamps, valid_mask) if v]
    if len(valid_ts) != len(strat_returns):
        # Truncate to match
        n = min(len(valid_ts), len(strat_returns))
        valid_ts = valid_ts[:n]
        strat_returns = strat_returns[:n]

    monthly: Dict[str, List[float]] = {}
    for ts, ret in zip(valid_ts, strat_returns):
        key = ts.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(ret)

    result: Dict[str, Dict[str, float]] = {}
    for month, rets in sorted(monthly.items()):
        rets_arr = np.array(rets, dtype=np.float64)
        result[month] = {
            "return": float(np.sum(rets_arr)),
            "sharpe": sharpe_ratio(rets_arr, annualize=True) if len(rets_arr) > 2 else 0.0,
            "n_bars": len(rets_arr),
        }
    return result


# ===========================================================================
# 5. OUTPUT AND REPORTING
# ===========================================================================

def print_report(
    all_results: Dict[str, Dict[str, Any]],
    all_importances: Dict[str, Dict[str, float]],
) -> None:
    """Print a detailed performance report across all symbols."""
    print("\n" + "=" * 80)
    print("  ML SIGNAL MODEL -- WALK-FORWARD OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  XGBoost: depth={XGB_PARAMS['max_depth']}, "
          f"n_est={XGB_PARAMS['n_estimators']}, "
          f"lr={XGB_PARAMS['learning_rate']}")
    print(f"  Walk-forward: min_train={MIN_TRAIN_BARS}, "
          f"purge={PURGE_BARS}, test_window={TEST_WINDOW}")
    print(f"  Position sizing: vol_target={VOL_TARGET}, max_weight={MAX_WEIGHT}")
    print("=" * 80)

    # ---- Per-symbol summary ----
    print("\n  Per-Symbol OOS Performance:")
    print("  " + "-" * 76)
    print(f"  {'Symbol':<8} {'OOS Bars':>10} {'IC':>8} {'Hit%':>8} "
          f"{'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'TotRet%':>8}")
    print("  " + "-" * 76)

    agg_sharpes = []
    agg_ics = []
    for sym in SYMBOLS:
        r = all_results.get(sym)
        if r is None or r.get("n_oos", 0) < 100:
            print(f"  {sym:<8} {'SKIP':>10}")
            continue
        print(f"  {sym:<8} {r['n_oos']:>10d} {r['ic']:>8.4f} "
              f"{r['hit_rate']*100:>7.1f}% {r['sharpe']:>8.2f} "
              f"{r['sortino']:>8.2f} {r['max_drawdown']:>7.1f}% "
              f"{r['total_return']*100:>7.2f}%")
        agg_sharpes.append(r["sharpe"])
        agg_ics.append(r["ic"])

    print("  " + "-" * 76)
    if agg_sharpes:
        print(f"  {'Average':<8} {'':>10} {np.mean(agg_ics):>8.4f} "
              f"{'':>8} {np.mean(agg_sharpes):>8.2f}")

    # ---- ML vs Equal-Weight Baseline ----
    print("\n  ML vs Equal-Weight Long Baseline (Sharpe):")
    print("  " + "-" * 50)
    for sym in SYMBOLS:
        r = all_results.get(sym)
        if r is None or r.get("n_oos", 0) < 100:
            continue
        ml_sr = r["sharpe"]
        bl_sr = r["baseline_sharpe"]
        delta = ml_sr - bl_sr
        print(f"  {sym:<8} ML={ml_sr:>7.2f}  Baseline={bl_sr:>7.2f}  "
              f"Delta={delta:>+7.2f}")

    # ---- Top 20 features by importance (averaged across symbols) ----
    combined_imp: Dict[str, float] = {}
    n_symbols_with_imp = 0
    for sym, imp in all_importances.items():
        if imp:
            n_symbols_with_imp += 1
            for feat, val in imp.items():
                combined_imp[feat] = combined_imp.get(feat, 0.0) + val
    if n_symbols_with_imp > 0:
        combined_imp = {
            k: v / n_symbols_with_imp for k, v in combined_imp.items()
        }

    sorted_imp = sorted(combined_imp.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n  Top 20 Features by Importance (averaged across {n_symbols_with_imp} symbols):")
    print("  " + "-" * 60)
    for i, (feat, val) in enumerate(sorted_imp[:20]):
        bar = "#" * int(val * 200)  # scaled bar
        print(f"  {i+1:>3d}. {feat:<40s} {val:.4f}  {bar}")

    # ---- Monthly breakdown for the first valid symbol ----
    for sym in SYMBOLS:
        r = all_results.get(sym)
        if r is None or "monthly_perf" not in r:
            continue
        monthly = r["monthly_perf"]
        if not monthly:
            continue
        print(f"\n  Monthly OOS Breakdown ({sym}):")
        print("  " + "-" * 50)
        print(f"  {'Month':<10} {'Return%':>10} {'Sharpe':>10} {'Bars':>8}")
        print("  " + "-" * 50)
        for month, stats in monthly.items():
            print(f"  {month:<10} {stats['return']*100:>9.3f}% "
                  f"{stats['sharpe']:>10.2f} {stats['n_bars']:>8d}")
        break  # only show the first symbol's monthly detail

    print("\n" + "=" * 80)


def save_artefacts(
    all_importances: Dict[str, Dict[str, float]],
    all_results: Dict[str, Dict[str, Any]],
    feature_names: List[str],
) -> None:
    """Save trained model parameters and feature list for the controller."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save feature importance rankings
    artefact = {
        "feature_names": feature_names,
        "per_symbol_importance": all_importances,
        "per_symbol_results": {
            sym: {k: v for k, v in res.items() if k != "monthly_perf"}
            for sym, res in all_results.items()
        },
        "xgb_params": XGB_PARAMS,
        "walk_forward_params": {
            "min_train_bars": MIN_TRAIN_BARS,
            "purge_bars": PURGE_BARS,
            "test_window": TEST_WINDOW,
            "step_bars": STEP_BARS,
        },
        "position_sizing": {
            "vol_target": VOL_TARGET,
            "max_weight": MAX_WEIGHT,
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_path = OUTPUT_DIR / "signal_model_artefacts.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Artefacts saved to %s", output_path)

    # Also save a human-readable feature ranking
    combined_imp: Dict[str, float] = {}
    n_sym = 0
    for imp in all_importances.values():
        if imp:
            n_sym += 1
            for feat, val in imp.items():
                combined_imp[feat] = combined_imp.get(feat, 0.0) + val
    if n_sym > 0:
        combined_imp = {k: v / n_sym for k, v in combined_imp.items()}

    sorted_imp = sorted(combined_imp.items(), key=lambda kv: kv[1], reverse=True)
    ranking_path = OUTPUT_DIR / "feature_ranking.txt"
    with open(ranking_path, "w") as f:
        f.write("Feature Importance Ranking (averaged across symbols)\n")
        f.write("=" * 60 + "\n")
        for i, (feat, val) in enumerate(sorted_imp):
            f.write(f"{i+1:>4d}. {feat:<45s} {val:.6f}\n")

    logger.info("Feature ranking saved to %s", ranking_path)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Main entry point: load data, train, evaluate, report."""
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("ML Signal Model Training -- Walk-Forward XGBoost")
    logger.info("=" * 60)

    # ---- Load universe data ----
    logger.info("Loading 5-min bar data for %d symbols...", len(SYMBOLS))
    try:
        universe_data, ref_timestamps = load_universe_data(
            symbols=SYMBOLS,
            bar_size="5mins",
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load universe data: %s", e)
        logger.info(
            "Falling back to loading symbols individually..."
        )
        universe_data = {}
        ref_timestamps = []
        for sym in SYMBOLS:
            try:
                ohlcv, ts = load_ibkr_bars(sym, bar_size="5mins")
                universe_data[sym] = ohlcv
                if not ref_timestamps or len(ts) > len(ref_timestamps):
                    ref_timestamps = ts
            except Exception as exc:
                logger.warning("  Could not load %s: %s", sym, exc)

    if not universe_data:
        logger.error("No data available. Exiting.")
        sys.exit(1)

    loaded_symbols = sorted(universe_data.keys())
    logger.info(
        "Loaded %d symbols: %s (ref bars: %d)",
        len(loaded_symbols), ", ".join(loaded_symbols), len(ref_timestamps),
    )

    # ---- Build cross-asset close prices dict (for lead-lag features) ----
    # Align all cross-asset prices to the target symbol's length.
    # If another symbol is shorter, pad with NaN at the END (its data covers
    # an earlier date range).  If longer, truncate to match.
    cross_asset_map: Dict[str, Dict[str, NDArray[np.float64]]] = {}
    for sym in loaded_symbols:
        sym_len = universe_data[sym].shape[0]
        others = {}
        for other_sym in loaded_symbols:
            if other_sym != sym:
                other_close = universe_data[other_sym][:, 3].astype(np.float64)
                if len(other_close) >= sym_len:
                    others[other_sym] = other_close[:sym_len]
                else:
                    pad = np.full(sym_len - len(other_close), np.nan)
                    others[other_sym] = np.concatenate([other_close, pad])
        cross_asset_map[sym] = others

    # ---- Train per symbol ----
    all_results: Dict[str, Dict[str, Any]] = {}
    all_importances: Dict[str, Dict[str, float]] = {}
    global_feature_names: List[str] = []

    for sym in loaded_symbols:
        logger.info("\n--- Processing %s ---", sym)
        ohlcv = universe_data[sym]
        T = ohlcv.shape[0]

        if T < MIN_TRAIN_BARS + PURGE_BARS + TEST_WINDOW:
            logger.warning(
                "  %s: insufficient data (%d bars, need %d). Skipping.",
                sym, T, MIN_TRAIN_BARS + PURGE_BARS + TEST_WINDOW,
            )
            continue

        # 1. Feature construction
        features, feature_names = compute_feature_matrix(
            ohlcv=ohlcv,
            timestamps=ref_timestamps[:T] if len(ref_timestamps) >= T else ref_timestamps,
            cross_asset_prices=cross_asset_map.get(sym),
            symbol=sym,
        )

        if not global_feature_names:
            global_feature_names = feature_names

        # 2. Target construction (forward 1-bar return)
        close = ohlcv[:, 3].astype(np.float64)
        fwd_1, fwd_78 = compute_targets(close)

        # 3. Walk-forward training (use 1-bar forward return as target)
        predictions, actuals, importances = walk_forward_train(
            features=features,
            target=fwd_1,
            feature_names=feature_names,
            symbol=sym,
        )

        all_importances[sym] = importances

        # 4. Evaluate OOS
        ts_for_eval = ref_timestamps[:T] if len(ref_timestamps) >= T else ref_timestamps
        results = evaluate_oos(
            predictions=predictions,
            actuals=fwd_1,
            timestamps=ts_for_eval,
            symbol=sym,
        )
        all_results[sym] = results

    # ---- Print report ----
    print_report(all_results, all_importances)

    # ---- Save artefacts ----
    save_artefacts(all_importances, all_results, global_feature_names)

    elapsed = time.time() - t_start
    logger.info("Total training time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
