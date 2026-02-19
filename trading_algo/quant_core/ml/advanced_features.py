"""
Advanced Feature Engineering for Trading Signals

Extends the base FeatureEngine with higher-order feature categories:
    1. Cross-Asset Lead-Lag: inter-market predictive signals
    2. Interaction Features: pairwise feature combinations
    3. Calendar/Seasonal: cyclical time effects, FOMC drift
    4. Regime-Conditional: regime-masked feature variants
    5. Order Flow / Microstructure Extended: trade classification, pressure
    6. Higher-Order Statistical: skewness, kurtosis, Hurst, autocorrelation

References:
    - Lucca & Moench (2015): "The Pre-FOMC Announcement Drift"
    - Lee & Ready (1991): "Inferring Trade Direction from Intraday Data"
    - Lo (1991): "Long-Term Memory in Stock Market Prices" (Hurst exponent)
    - Amihud (2002): "Illiquidity and Stock Returns"
    - Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import datetime

from trading_algo.quant_core.ml.features import (
    FeatureSet,
    FeatureCategory,
    FeatureEngine,
)
from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252


# ---------------------------------------------------------------------------
# Extended Feature Categories
# ---------------------------------------------------------------------------

class AdvancedFeatureCategory(Enum):
    """Extended feature categories beyond the base FeatureCategory set."""
    CROSS_ASSET = auto()
    INTERACTION = auto()
    CALENDAR = auto()
    REGIME_CONDITIONAL = auto()
    ORDER_FLOW = auto()
    STATISTICAL = auto()


# Unified category type used throughout this module.
FeatureCategoryType = Union[FeatureCategory, AdvancedFeatureCategory]


# ---------------------------------------------------------------------------
# FOMC schedule heuristic
# ---------------------------------------------------------------------------

# Historical / projected FOMC meeting dates (month, day) for the announcement.
# This list covers 2015-2027. For dates outside this range the engine falls
# back to a "roughly every 6 weeks" heuristic.
_FOMC_DATES: List[Tuple[int, int, int]] = [
    # 2023
    (2023, 2, 1), (2023, 3, 22), (2023, 5, 3), (2023, 6, 14),
    (2023, 7, 26), (2023, 9, 20), (2023, 11, 1), (2023, 12, 13),
    # 2024
    (2024, 1, 31), (2024, 3, 20), (2024, 5, 1), (2024, 6, 12),
    (2024, 7, 31), (2024, 9, 18), (2024, 11, 7), (2024, 12, 18),
    # 2025
    (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18),
    (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 17),
    # 2026
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
    (2026, 7, 29), (2026, 9, 16), (2026, 11, 4), (2026, 12, 16),
]

_FOMC_ORDINALS: Optional[NDArray[np.int64]] = None


def _get_fomc_ordinals() -> NDArray[np.int64]:
    """Lazily build and cache the sorted array of FOMC date ordinals."""
    global _FOMC_ORDINALS
    if _FOMC_ORDINALS is None:
        ords = np.array(
            [datetime.date(y, m, d).toordinal() for y, m, d in _FOMC_DATES],
            dtype=np.int64,
        )
        ords.sort()
        _FOMC_ORDINALS = ords
    return _FOMC_ORDINALS


# ---------------------------------------------------------------------------
# Helper: vectorised rolling computations (pure numpy, no numba dependency)
# ---------------------------------------------------------------------------

def _rolling_apply(
    arr: NDArray[np.float64],
    window: int,
    func,
) -> NDArray[np.float64]:
    """
    Apply *func* over a rolling window, returning an array of the same length
    with NaN padding at the start where the window is incomplete.

    ``func`` receives a 1-D array of length *window* and returns a scalar.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    # Use stride tricks for a view of rolling windows
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    for i in range(windows.shape[0]):
        out[i + window - 1] = func(windows[i])
    return out


def _rolling_mean_np(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Rolling mean using cumulative sum (O(n))."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cs = np.cumsum(arr)
    out[window - 1] = cs[window - 1] / window
    out[window:] = (cs[window:] - cs[:-window]) / window
    return out


def _rolling_std_np(
    arr: NDArray[np.float64], window: int, ddof: int = 1
) -> NDArray[np.float64]:
    """Rolling standard deviation (two-pass via cumsum for numerical stability)."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    mean = _rolling_mean_np(arr, window)
    sq = arr ** 2
    cs_sq = np.cumsum(sq)
    roll_sq = np.full(n, np.nan, dtype=np.float64)
    roll_sq[window - 1] = cs_sq[window - 1] / window
    roll_sq[window:] = (cs_sq[window:] - cs_sq[:-window]) / window
    var = roll_sq - mean ** 2
    # Bessel correction
    if ddof != 0:
        var = var * window / (window - ddof)
    var = np.maximum(var, 0.0)  # guard against floating-point negatives
    valid = ~np.isnan(var)
    out[valid] = np.sqrt(var[valid])
    return out


def _rolling_corr_np(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Rolling Pearson correlation between *x* and *y*."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mx = _rolling_mean_np(x, window)
    my = _rolling_mean_np(y, window)
    cs_xy = np.cumsum(x * y)
    roll_xy = np.full(n, np.nan, dtype=np.float64)
    roll_xy[window - 1] = cs_xy[window - 1] / window
    roll_xy[window:] = (cs_xy[window:] - cs_xy[:-window]) / window
    cov = roll_xy - mx * my
    sx = _rolling_std_np(x, window, ddof=0)
    sy = _rolling_std_np(y, window, ddof=0)
    denom = sx * sy
    out = np.full(n, np.nan, dtype=np.float64)
    valid = (~np.isnan(denom)) & (denom > EPSILON)
    out[valid] = cov[valid] / denom[valid]
    return out


def _log_returns_np(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Log returns without numba dependency."""
    r = np.empty(len(prices) - 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = prices[1:] / prices[:-1]
        r = np.where(prices[:-1] > 0, np.log(np.maximum(ratio, EPSILON)), 0.0)
    return r


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdvancedFeatureEngine:
    """
    Extended feature engine that builds on the base :class:`FeatureEngine`.

    It wraps a ``FeatureEngine`` instance and augments its output with six
    additional feature families:

    1. **Cross-Asset Lead-Lag** -- rolling cross-correlations at multiple
       lags between a target asset and a set of reference assets.
    2. **Interaction Features** -- pairwise products, ratios, and
       differences among top-N base features, filtered by variance.
    3. **Calendar / Seasonal** -- day-of-week, month-of-year (sin/cos),
       options expiry proximity, quarter-end, FOMC cycle, end-of-month /
       turn-of-month flags.
    4. **Regime-Conditional** -- base features masked or weighted by
       regime labels / probabilities.
    5. **Order Flow (extended)** -- Lee-Ready proxy, volume-weighted
       price pressure, large-trade clustering, trade intensity, Amihud
       delta.
    6. **Higher-Order Statistical** -- rolling skewness, kurtosis, Hurst
       exponent proxy, autocorrelation at lags 1 and 5.

    Usage::

        engine = AdvancedFeatureEngine()

        # Full feature set (base + advanced)
        fs = engine.compute_advanced_features(
            prices=close,
            volumes=vol,
            high=high,
            low=low,
            open_=open_,
            cross_asset_prices={"SPY": spy, "TLT": tlt},
            regime_labels=hmm_labels,
            regime_probabilities=hmm_probs,
            timestamps=dates,
        )
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        base_engine: Optional[FeatureEngine] = None,
        # Cross-asset
        lead_lag_lags: List[int] = [1, 2, 5, 10],
        cross_corr_window: int = 60,
        # Interaction
        interaction_top_n: int = 20,
        interaction_min_variance: float = 1e-6,
        # Calendar
        calendar_encoding: str = "sincos",  # "sincos" or "onehot"
        # Statistical
        stat_return_window: int = 20,
        hurst_window: int = 60,
        # General
        normalize: bool = True,
        normalization_window: int = 252,
    ):
        """
        Initialize the advanced feature engine.

        Args:
            base_engine: Pre-configured FeatureEngine.  If *None* a default
                engine is created.
            lead_lag_lags: Lag offsets for cross-asset correlation.
            cross_corr_window: Rolling window for cross-correlation.
            interaction_top_n: Number of base features to use for
                interaction terms.
            interaction_min_variance: Minimum variance threshold to keep
                an interaction feature.
            calendar_encoding: ``"sincos"`` (default) or ``"onehot"`` for
                day-of-week encoding.
            stat_return_window: Window for rolling skewness, kurtosis and
                autocorrelation.
            hurst_window: Window for Hurst exponent estimation.
            normalize: Whether to z-score normalise the final feature
                matrix.
            normalization_window: Rolling window for z-score normalisation.
        """
        self.base_engine = base_engine or FeatureEngine()
        self.lead_lag_lags = lead_lag_lags
        self.cross_corr_window = cross_corr_window
        self.interaction_top_n = interaction_top_n
        self.interaction_min_variance = interaction_min_variance
        self.calendar_encoding = calendar_encoding
        self.stat_return_window = stat_return_window
        self.hurst_window = hurst_window
        self.normalize = normalize
        self.normalization_window = normalization_window

        # Incremental state for add_bar
        self._bar_buffer: Dict[str, List[float]] = {
            "close": [],
            "high": [],
            "low": [],
            "open": [],
            "volume": [],
        }
        self._cross_asset_buffers: Dict[str, List[float]] = {}
        self._regime_label_buffer: List[Optional[str]] = []
        self._regime_prob_buffer: List[Optional[NDArray]] = []
        self._timestamp_buffer: List[datetime.date] = []
        self._last_feature_set: Optional[FeatureSet] = None

    # ============================================================
    # PUBLIC API
    # ============================================================

    def compute_advanced_features(
        self,
        prices: NDArray[np.float64],
        volumes: Optional[NDArray[np.float64]] = None,
        high: Optional[NDArray[np.float64]] = None,
        low: Optional[NDArray[np.float64]] = None,
        open_: Optional[NDArray[np.float64]] = None,
        bid: Optional[NDArray[np.float64]] = None,
        ask: Optional[NDArray[np.float64]] = None,
        cross_asset_prices: Optional[Dict[str, NDArray[np.float64]]] = None,
        regime_labels: Optional[NDArray] = None,
        regime_probabilities: Optional[NDArray[np.float64]] = None,
        timestamps: Optional[NDArray] = None,
        base_feature_set: Optional[FeatureSet] = None,
    ) -> FeatureSet:
        """
        Compute all advanced features and merge with base features.

        Args:
            prices: Close prices for the target asset (length T).
            volumes: Trading volumes (length T).
            high: High prices (length T).
            low: Low prices (length T).
            open_: Open prices (length T).
            bid: Bid prices (length T) -- for base microstructure.
            ask: Ask prices (length T) -- for base microstructure.
            cross_asset_prices: ``{symbol: prices_array}`` for lead-lag.
            regime_labels: Array of regime labels (str or int, length T).
            regime_probabilities: Regime probability matrix (T x R) where
                R is the number of regimes.
            timestamps: Array of ``datetime.date`` or ``np.datetime64``
                (length T) for calendar features.
            base_feature_set: Pre-computed base features to extend instead
                of recomputing.  When *None* the base engine is invoked.

        Returns:
            :class:`FeatureSet` containing all base + advanced features.
        """
        n = len(prices)

        # ----- base features ------------------------------------------------
        if base_feature_set is not None:
            base_fs = base_feature_set
        else:
            base_fs = self.base_engine.compute_features(
                prices, volumes, high, low, bid, ask,
            )

        all_features: Dict[str, NDArray[np.float64]] = {}
        all_categories: Dict[str, FeatureCategoryType] = dict(base_fs.categories)

        # Start from base features dict
        base_dict = base_fs.to_dict()
        all_features.update(base_dict)

        # ----- 1. cross-asset lead-lag --------------------------------------
        if cross_asset_prices is not None and len(cross_asset_prices) > 0:
            ca = self._cross_asset_features(prices, cross_asset_prices)
            all_features.update(ca)
            for name in ca:
                all_categories[name] = AdvancedFeatureCategory.CROSS_ASSET

        # ----- 2. interaction features --------------------------------------
        interaction = self._interaction_features(base_fs)
        all_features.update(interaction)
        for name in interaction:
            all_categories[name] = AdvancedFeatureCategory.INTERACTION

        # ----- 3. calendar / seasonal --------------------------------------
        if timestamps is not None and len(timestamps) > 0:
            cal = self._calendar_features(timestamps)
            all_features.update(cal)
            for name in cal:
                all_categories[name] = AdvancedFeatureCategory.CALENDAR

        # ----- 4. regime-conditional ----------------------------------------
        if regime_labels is not None or regime_probabilities is not None:
            regime = self._regime_conditional_features(
                base_fs, regime_labels, regime_probabilities,
            )
            all_features.update(regime)
            for name in regime:
                all_categories[name] = AdvancedFeatureCategory.REGIME_CONDITIONAL

        # ----- 5. order flow / microstructure extended -----------------------
        of = self._order_flow_features(prices, volumes, high, low, open_)
        all_features.update(of)
        for name in of:
            all_categories[name] = AdvancedFeatureCategory.ORDER_FLOW

        # ----- 6. higher-order statistical -----------------------------------
        stat = self._statistical_features(prices)
        all_features.update(stat)
        for name in stat:
            all_categories[name] = AdvancedFeatureCategory.STATISTICAL

        # ----- assemble matrix -----------------------------------------------
        feature_names = list(all_features.keys())
        feature_matrix = np.column_stack([
            self._pad_feature(all_features[name], n) for name in feature_names
        ])

        if self.normalize:
            feature_matrix = self._normalize_features(
                feature_matrix, self.normalization_window,
            )

        self._last_feature_set = FeatureSet(
            features=feature_matrix,
            feature_names=feature_names,
            timestamps=timestamps,
            categories=all_categories,  # type: ignore[arg-type]
        )
        return self._last_feature_set

    # ---------------------------------------------------------------- add_bar
    def add_bar(
        self,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        open_: Optional[float] = None,
        volume: Optional[float] = None,
        cross_asset_closes: Optional[Dict[str, float]] = None,
        regime_label: Optional[str] = None,
        regime_prob: Optional[NDArray[np.float64]] = None,
        timestamp: Optional[datetime.date] = None,
    ) -> FeatureSet:
        """
        Incrementally append one bar and recompute features.

        This is designed for live-trading loops where bars arrive one at a
        time.  The method maintains internal buffers and recomputes the
        full feature matrix each call (efficient enough for daily bars;
        for sub-second use-cases consider a streaming approach).

        Args:
            close: Close price for the new bar.
            high: High price.
            low: Low price.
            open_: Open price.
            volume: Volume.
            cross_asset_closes: ``{symbol: close}`` for cross-asset data.
            regime_label: Current regime label (e.g. ``"bull"``).
            regime_prob: Current regime probability vector.
            timestamp: Date of the bar.

        Returns:
            Updated :class:`FeatureSet`.
        """
        self._bar_buffer["close"].append(close)
        self._bar_buffer["high"].append(high if high is not None else close)
        self._bar_buffer["low"].append(low if low is not None else close)
        self._bar_buffer["open"].append(open_ if open_ is not None else close)
        self._bar_buffer["volume"].append(volume if volume is not None else 0.0)

        if cross_asset_closes is not None:
            for sym, val in cross_asset_closes.items():
                if sym not in self._cross_asset_buffers:
                    # Back-fill with NaN for any bars that arrived before
                    # this symbol appeared.
                    self._cross_asset_buffers[sym] = [np.nan] * (
                        len(self._bar_buffer["close"]) - 1
                    )
                self._cross_asset_buffers[sym].append(val)
            # Ensure all existing symbols get a NaN if not provided this bar
            for sym in self._cross_asset_buffers:
                if cross_asset_closes is None or sym not in cross_asset_closes:
                    self._cross_asset_buffers[sym].append(np.nan)

        self._regime_label_buffer.append(regime_label)
        self._regime_prob_buffer.append(regime_prob)
        if timestamp is not None:
            self._timestamp_buffer.append(timestamp)

        # Convert buffers to arrays
        prices_arr = np.array(self._bar_buffer["close"], dtype=np.float64)
        vol_arr = np.array(self._bar_buffer["volume"], dtype=np.float64)
        high_arr = np.array(self._bar_buffer["high"], dtype=np.float64)
        low_arr = np.array(self._bar_buffer["low"], dtype=np.float64)
        open_arr = np.array(self._bar_buffer["open"], dtype=np.float64)

        ca_prices: Optional[Dict[str, NDArray[np.float64]]] = None
        if self._cross_asset_buffers:
            ca_prices = {
                sym: np.array(vals, dtype=np.float64)
                for sym, vals in self._cross_asset_buffers.items()
            }

        rl: Optional[NDArray] = None
        if any(x is not None for x in self._regime_label_buffer):
            rl = np.array(self._regime_label_buffer)

        rp: Optional[NDArray[np.float64]] = None
        if any(x is not None for x in self._regime_prob_buffer):
            rp = np.vstack([
                p if p is not None else np.full(
                    self._regime_prob_buffer[0].shape  # type: ignore[union-attr]
                    if self._regime_prob_buffer[0] is not None
                    else (1,),
                    np.nan,
                )
                for p in self._regime_prob_buffer
            ])

        ts: Optional[NDArray] = None
        if self._timestamp_buffer:
            ts = np.array(self._timestamp_buffer)

        return self.compute_advanced_features(
            prices=prices_arr,
            volumes=vol_arr,
            high=high_arr,
            low=low_arr,
            open_=open_arr,
            cross_asset_prices=ca_prices,
            regime_labels=rl,
            regime_probabilities=rp,
            timestamps=ts,
        )

    def reset_buffers(self) -> None:
        """Clear all incremental bar buffers."""
        for key in self._bar_buffer:
            self._bar_buffer[key] = []
        self._cross_asset_buffers.clear()
        self._regime_label_buffer.clear()
        self._regime_prob_buffer.clear()
        self._timestamp_buffer.clear()
        self._last_feature_set = None

    # ============================================================
    # 1. CROSS-ASSET LEAD-LAG FEATURES
    # ============================================================

    def _cross_asset_features(
        self,
        target_prices: NDArray[np.float64],
        cross_asset_prices: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Compute lead-lag cross-correlations and relative strength.

        For each reference asset *sym* and each lag *k* in
        ``self.lead_lag_lags``, we compute the rolling correlation between
        ``ret_target[t]`` and ``ret_sym[t-k]`` (i.e. whether *sym*'s past
        return predicts the target's current return).

        Also computes relative strength: target return minus universe mean
        return over a trailing window.

        Args:
            target_prices: Close prices for the target asset.
            cross_asset_prices: ``{symbol: close_prices}`` for reference
                assets.

        Returns:
            Dictionary mapping feature name to feature array.
        """
        features: Dict[str, NDArray[np.float64]] = {}
        target_ret = _log_returns_np(target_prices)
        n_ret = len(target_ret)

        if n_ret < self.cross_corr_window + max(self.lead_lag_lags):
            return features

        # Collect reference returns
        ref_returns: Dict[str, NDArray[np.float64]] = {}
        for sym, ref_prices in cross_asset_prices.items():
            if len(ref_prices) >= len(target_prices):
                ref_returns[sym] = _log_returns_np(ref_prices[:len(target_prices)])
            elif len(ref_prices) > 1:
                ref_returns[sym] = _log_returns_np(ref_prices)

        # Lead-lag correlations
        for sym, ref_ret in ref_returns.items():
            common_len = min(n_ret, len(ref_ret))
            for lag in self.lead_lag_lags:
                if common_len <= lag + self.cross_corr_window:
                    continue
                # ref at t-lag vs target at t
                x = ref_ret[: common_len - lag]
                y = target_ret[lag: common_len]
                corr = _rolling_corr_np(x, y, self.cross_corr_window)
                features[f"leadlag_{sym}_lag{lag}"] = corr

        # Relative strength vs universe mean
        if ref_returns:
            # Align all returns to common length
            min_len = min(n_ret, *(len(r) for r in ref_returns.values()))
            universe_rets = np.column_stack(
                [r[:min_len] for r in ref_returns.values()]
            )
            universe_mean = np.nanmean(universe_rets, axis=1)
            # Include target in the mean for proper cross-sectional calc
            all_mean = (universe_mean * universe_rets.shape[1] + target_ret[:min_len]) / (
                universe_rets.shape[1] + 1
            )
            rel_strength = target_ret[:min_len] - all_mean
            for window in [5, 20]:
                if min_len > window:
                    features[f"rel_strength_{window}d"] = _rolling_mean_np(
                        rel_strength, window,
                    )

        return features

    # ============================================================
    # 2. INTERACTION FEATURES
    # ============================================================

    def _interaction_features(
        self,
        base_fs: FeatureSet,
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Generate pairwise interaction features from the top-N base features.

        Interactions:
            - product:    f_i * f_j
            - ratio:      f_i / (|f_j| + eps)
            - difference: f_i - f_j

        Only features whose variance exceeds ``interaction_min_variance``
        are retained.

        Args:
            base_fs: Base feature set.

        Returns:
            Dictionary mapping feature name to feature array.
        """
        features: Dict[str, NDArray[np.float64]] = {}
        n_base = base_fs.n_features
        top_n = min(self.interaction_top_n, n_base)

        if top_n < 2:
            return features

        # Select top-N features by variance (most informative)
        mat = base_fs.features  # (T, K)
        variances = np.nanvar(mat, axis=0)
        top_indices = np.argsort(variances)[-top_n:][::-1]
        top_names = [base_fs.feature_names[i] for i in top_indices]
        top_cols = mat[:, top_indices]  # (T, top_n)

        for i in range(top_n):
            for j in range(i + 1, top_n):
                name_i = top_names[i]
                name_j = top_names[j]
                col_i = top_cols[:, i]
                col_j = top_cols[:, j]

                # Product
                prod = col_i * col_j
                if self._passes_variance_check(prod):
                    features[f"ix_{name_i}_x_{name_j}"] = prod

                # Ratio (i / j)
                ratio = col_i / (np.abs(col_j) + EPSILON)
                if self._passes_variance_check(ratio):
                    features[f"ix_{name_i}_div_{name_j}"] = ratio

                # Difference
                diff = col_i - col_j
                if self._passes_variance_check(diff):
                    features[f"ix_{name_i}_sub_{name_j}"] = diff

        return features

    def _passes_variance_check(self, arr: NDArray[np.float64]) -> bool:
        """Return *True* if the non-NaN variance exceeds the threshold."""
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return False
        return float(np.var(valid)) > self.interaction_min_variance

    # ============================================================
    # 3. CALENDAR / SEASONAL FEATURES
    # ============================================================

    def _calendar_features(
        self,
        timestamps: NDArray,
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Compute calendar and seasonal features.

        Features:
            - Day of week (sin/cos or one-hot)
            - Month of year (sin/cos)
            - Days to next options expiry (3rd Friday)
            - Days to quarter end
            - FOMC cycle position (days since last, days to next)
            - End-of-month flag (last 3 trading days)
            - Turn-of-month flag (last day + first 3 days)

        Args:
            timestamps: Array of ``datetime.date`` or ``np.datetime64``.

        Returns:
            Dictionary mapping feature name to feature array.
        """
        features: Dict[str, NDArray[np.float64]] = {}
        n = len(timestamps)

        dates = self._to_python_dates(timestamps)
        if dates is None or len(dates) == 0:
            return features

        # Pre-compute ordinals and weekdays
        ordinals = np.array([d.toordinal() for d in dates], dtype=np.int64)
        weekdays = np.array([d.weekday() for d in dates], dtype=np.float64)  # 0=Mon
        months = np.array([d.month for d in dates], dtype=np.float64)

        # --- Day of week ---
        if self.calendar_encoding == "onehot":
            for dow in range(5):
                features[f"dow_{dow}"] = (weekdays == dow).astype(np.float64)
        else:
            # sin/cos (period = 5 trading days)
            features["dow_sin"] = np.sin(2 * np.pi * weekdays / 5.0)
            features["dow_cos"] = np.cos(2 * np.pi * weekdays / 5.0)

        # --- Month of year (sin/cos, period = 12) ---
        features["moy_sin"] = np.sin(2 * np.pi * months / 12.0)
        features["moy_cos"] = np.cos(2 * np.pi * months / 12.0)

        # --- Days to next options expiry (3rd Friday) ---
        features["days_to_opex"] = np.array(
            [self._days_to_options_expiry(d) for d in dates], dtype=np.float64,
        )

        # --- Days to quarter end ---
        features["days_to_qtr_end"] = np.array(
            [self._days_to_quarter_end(d) for d in dates], dtype=np.float64,
        )

        # --- FOMC cycle ---
        fomc_since, fomc_until = self._fomc_cycle(ordinals)
        features["days_since_fomc"] = fomc_since.astype(np.float64)
        features["days_to_fomc"] = fomc_until.astype(np.float64)

        # --- End-of-month (last 3 trading days) ---
        features["eom_flag"] = self._end_of_month_flag(dates, last_n=3)

        # --- Turn-of-month (last 1 day + first 3 days) ---
        features["tom_flag"] = self._turn_of_month_flag(dates)

        return features

    # ---- Calendar helpers --------------------------------------------------

    @staticmethod
    def _to_python_dates(timestamps: NDArray) -> Optional[List[datetime.date]]:
        """Convert an array of timestamps to a list of ``datetime.date``."""
        if len(timestamps) == 0:
            return None
        sample = timestamps[0]
        if isinstance(sample, datetime.date):
            return list(timestamps)
        if isinstance(sample, datetime.datetime):
            return [t.date() for t in timestamps]
        if isinstance(sample, (np.datetime64,)):
            # Convert via pandas-free path
            epoch = np.datetime64("1970-01-01", "D")
            days_since_epoch = (timestamps.astype("datetime64[D]") - epoch).astype(int)
            base = datetime.date(1970, 1, 1)
            return [base + datetime.timedelta(days=int(d)) for d in days_since_epoch]
        # Last resort: try str conversion
        try:
            return [datetime.date.fromisoformat(str(t)) for t in timestamps]
        except Exception:
            return None

    @staticmethod
    def _days_to_options_expiry(d: datetime.date) -> int:
        """
        Days until the next monthly options expiry (3rd Friday of the month).

        If *d* is past this month's 3rd Friday, returns the distance to
        next month's 3rd Friday.
        """
        def third_friday(year: int, month: int) -> datetime.date:
            # First day of month
            first = datetime.date(year, month, 1)
            # weekday(): Monday=0 ... Friday=4
            # Days until first Friday
            days_until_fri = (4 - first.weekday()) % 7
            first_fri = first + datetime.timedelta(days=days_until_fri)
            return first_fri + datetime.timedelta(weeks=2)

        expiry = third_friday(d.year, d.month)
        if d > expiry:
            # Move to next month
            if d.month == 12:
                expiry = third_friday(d.year + 1, 1)
            else:
                expiry = third_friday(d.year, d.month + 1)
        return (expiry - d).days

    @staticmethod
    def _days_to_quarter_end(d: datetime.date) -> int:
        """Calendar days until the end of the current quarter."""
        q_month = ((d.month - 1) // 3 + 1) * 3
        if q_month == 12:
            q_end = datetime.date(d.year, 12, 31)
        else:
            q_end = datetime.date(d.year, q_month + 1, 1) - datetime.timedelta(days=1)
        delta = (q_end - d).days
        return max(delta, 0)

    @staticmethod
    def _fomc_cycle(ordinals: NDArray[np.int64]) -> Tuple[NDArray, NDArray]:
        """
        Compute days since last and days to next FOMC meeting.

        Falls back to a 42-day (6-week) cycle heuristic when dates are
        outside the known schedule.

        Returns:
            Tuple ``(days_since, days_until)`` each of shape ``(N,)``.
        """
        fomc_ords = _get_fomc_ordinals()
        n = len(ordinals)
        since = np.full(n, np.nan, dtype=np.float64)
        until = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            d = ordinals[i]
            idx = np.searchsorted(fomc_ords, d, side="right")
            if idx > 0 and idx < len(fomc_ords):
                since[i] = d - fomc_ords[idx - 1]
                until[i] = fomc_ords[idx] - d
            elif idx == 0:
                # Before all known dates -- heuristic
                until[i] = fomc_ords[0] - d
                since[i] = 42 - (until[i] % 42)
            else:
                # After all known dates -- heuristic
                since[i] = d - fomc_ords[-1]
                until[i] = 42 - (since[i] % 42)

        return since, until

    @staticmethod
    def _end_of_month_flag(
        dates: List[datetime.date], last_n: int = 3,
    ) -> NDArray[np.float64]:
        """
        Flag the last *last_n* trading days of each month.

        Trading days are inferred from the provided dates (only dates
        that appear in the series are considered trading days).
        """
        n = len(dates)
        flag = np.zeros(n, dtype=np.float64)
        if n == 0:
            return flag

        # Group indices by (year, month)
        month_groups: Dict[Tuple[int, int], List[int]] = {}
        for i, d in enumerate(dates):
            key = (d.year, d.month)
            if key not in month_groups:
                month_groups[key] = []
            month_groups[key].append(i)

        for indices in month_groups.values():
            for idx in indices[-last_n:]:
                flag[idx] = 1.0
        return flag

    @staticmethod
    def _turn_of_month_flag(dates: List[datetime.date]) -> NDArray[np.float64]:
        """
        Flag the last trading day of the month plus the first 3 trading
        days of the next month.
        """
        n = len(dates)
        flag = np.zeros(n, dtype=np.float64)
        if n == 0:
            return flag

        month_groups: Dict[Tuple[int, int], List[int]] = {}
        for i, d in enumerate(dates):
            key = (d.year, d.month)
            if key not in month_groups:
                month_groups[key] = []
            month_groups[key].append(i)

        sorted_keys = sorted(month_groups.keys())
        for ki, key in enumerate(sorted_keys):
            indices = month_groups[key]
            # Last trading day of this month
            flag[indices[-1]] = 1.0
            # First 3 trading days of next month
            if ki + 1 < len(sorted_keys):
                next_indices = month_groups[sorted_keys[ki + 1]]
                for idx in next_indices[:3]:
                    flag[idx] = 1.0

        return flag

    # ============================================================
    # 4. REGIME-CONDITIONAL FEATURES
    # ============================================================

    def _regime_conditional_features(
        self,
        base_fs: FeatureSet,
        regime_labels: Optional[NDArray],
        regime_probabilities: Optional[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Create regime-conditional variants of base features.

        Two modes:

        1. **Discrete labels** (``regime_labels``): for each unique label,
           create ``feature_<label> = feature if regime == label else 0``.
        2. **Continuous probabilities** (``regime_probabilities``): for
           each regime column *r*, create ``feature_regime{r} = feature *
           prob_r``.

        Args:
            base_fs: Base feature set.
            regime_labels: Array of regime labels (length T).
            regime_probabilities: Regime probabilities (T x R).

        Returns:
            Dictionary mapping feature name to feature array.
        """
        features: Dict[str, NDArray[np.float64]] = {}
        mat = base_fs.features  # (T, K)
        n_samples, n_feat = mat.shape

        # Select a subset of features to avoid massive explosion
        # Use top features by variance
        n_regime_features = min(10, n_feat)
        variances = np.nanvar(mat, axis=0)
        top_indices = np.argsort(variances)[-n_regime_features:]

        # --- Discrete labels ---
        if regime_labels is not None and len(regime_labels) == n_samples:
            unique_labels = sorted(set(
                str(lbl) for lbl in regime_labels if lbl is not None
            ))
            for label in unique_labels:
                mask = np.array(
                    [str(lbl) == label for lbl in regime_labels],
                    dtype=np.float64,
                )
                for idx in top_indices:
                    fname = base_fs.feature_names[idx]
                    masked = mat[:, idx] * mask
                    features[f"{fname}_{label}"] = masked

        # --- Continuous probabilities ---
        if regime_probabilities is not None and len(regime_probabilities) == n_samples:
            n_regimes = regime_probabilities.shape[1] if regime_probabilities.ndim > 1 else 1
            probs = (
                regime_probabilities
                if regime_probabilities.ndim > 1
                else regime_probabilities[:, np.newaxis]
            )
            for r in range(n_regimes):
                prob_col = probs[:, r]
                # Replace NaN with 0 for weighting
                prob_clean = np.where(np.isnan(prob_col), 0.0, prob_col)
                for idx in top_indices:
                    fname = base_fs.feature_names[idx]
                    weighted = mat[:, idx] * prob_clean
                    features[f"{fname}_rprob{r}"] = weighted

        return features

    # ============================================================
    # 5. ORDER FLOW / MICROSTRUCTURE EXTENDED
    # ============================================================

    def _order_flow_features(
        self,
        prices: NDArray[np.float64],
        volumes: Optional[NDArray[np.float64]],
        high: Optional[NDArray[np.float64]],
        low: Optional[NDArray[np.float64]],
        open_: Optional[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Extended order-flow and microstructure features.

        Features:
            - **Lee-Ready proxy** (close location value):
              CLV = (close - low) / (high - low) mapped to [-1, +1].
            - **Volume-weighted price pressure**:
              ``(close - open) / (high - low) * volume``.
            - **Large trade clustering**: rolling count of volume spikes
              exceeding 2x the 20-day average.
            - **Trade intensity**: ``volume / (high - low)`` -- a proxy
              for Kyle's lambda.
            - **Amihud illiquidity delta**: current Amihud ratio minus its
              20-day rolling average.

        Args:
            prices: Close prices.
            volumes: Volumes.
            high: High prices.
            low: Low prices.
            open_: Open prices.

        Returns:
            Dictionary mapping feature name to feature array.
        """
        features: Dict[str, NDArray[np.float64]] = {}
        n = len(prices)

        has_hl = high is not None and low is not None
        has_vol = volumes is not None

        # --- Lee-Ready close location value ---
        if has_hl:
            hl_range = high - low
            safe_range = np.where(hl_range > EPSILON, hl_range, np.nan)
            # CLV in [-1, +1]: (close - low) / (high - low) * 2 - 1
            clv = (prices - low) / safe_range * 2.0 - 1.0
            features["lee_ready_clv"] = clv

        # --- Volume-weighted price pressure ---
        if has_hl and has_vol and open_ is not None:
            hl_range = high - low
            safe_range = np.where(hl_range > EPSILON, hl_range, np.nan)
            pressure = (prices - open_) / safe_range * volumes
            features["vol_price_pressure"] = pressure

        # --- Large trade clustering ---
        if has_vol:
            vol_avg_20 = _rolling_mean_np(volumes, 20)
            # Boolean: volume > 2x average
            spikes = np.where(
                (~np.isnan(vol_avg_20)) & (vol_avg_20 > EPSILON),
                (volumes > 2.0 * vol_avg_20).astype(np.float64),
                np.nan,
            )
            # Rolling count of spikes over 20 bars
            if n > 20:
                spike_count = _rolling_mean_np(spikes, 20) * 20.0
                features["large_trade_cluster_20d"] = spike_count

        # --- Trade intensity ---
        if has_hl and has_vol:
            hl_range = high - low
            safe_range = np.where(hl_range > EPSILON, hl_range, np.nan)
            intensity = volumes / safe_range
            features["trade_intensity"] = intensity

        # --- Amihud illiquidity delta ---
        if has_vol and n > 21:
            returns = _log_returns_np(prices)
            min_len = min(len(returns), len(volumes) - 1)
            if min_len > 20:
                illiq = np.abs(returns[:min_len]) / (
                    volumes[1: min_len + 1] + EPSILON
                )
                illiq_avg = _rolling_mean_np(illiq, 20)
                delta = np.full(min_len, np.nan, dtype=np.float64)
                valid = ~np.isnan(illiq_avg)
                delta[valid] = illiq[valid] - illiq_avg[valid]
                features["amihud_delta_20d"] = delta

        return features

    # ============================================================
    # 6. HIGHER-ORDER STATISTICAL FEATURES
    # ============================================================

    def _statistical_features(
        self,
        prices: NDArray[np.float64],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Compute higher-order statistical features of the return series.

        Features:
            - **Rolling skewness** (20-day)
            - **Rolling kurtosis** (20-day, excess)
            - **Hurst exponent proxy** (rescaled range, 60-day)
            - **Autocorrelation at lag 1** (rolling 20-day)
            - **Autocorrelation at lag 5** (rolling 20-day)

        Args:
            prices: Close prices.

        Returns:
            Dictionary mapping feature name to feature array.
        """
        features: Dict[str, NDArray[np.float64]] = {}
        returns = _log_returns_np(prices)
        n = len(returns)
        w = self.stat_return_window

        # --- Skewness ---
        if n >= w:
            features["skew_20d"] = self._rolling_skewness(returns, w)

        # --- Kurtosis (excess) ---
        if n >= w:
            features["kurt_20d"] = self._rolling_kurtosis(returns, w)

        # --- Hurst exponent proxy ---
        hw = self.hurst_window
        if n >= hw:
            features["hurst_60d"] = self._rolling_hurst(returns, hw)

        # --- Autocorrelation lag 1 ---
        if n >= w + 1:
            features["autocorr_lag1_20d"] = self._rolling_autocorr(returns, w, lag=1)

        # --- Autocorrelation lag 5 ---
        if n >= w + 5:
            features["autocorr_lag5_20d"] = self._rolling_autocorr(returns, w, lag=5)

        return features

    # ---- Statistical helpers -----------------------------------------------

    @staticmethod
    def _rolling_skewness(
        arr: NDArray[np.float64], window: int,
    ) -> NDArray[np.float64]:
        """
        Rolling skewness using the adjusted Fisher-Pearson formula.

        skew = [n / ((n-1)(n-2))] * sum[((x - mean) / std)^3]
        """
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < window or window < 3:
            return out

        shape = (n - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1, ddof=1)

        for i in range(windows.shape[0]):
            s = stds[i]
            if s < EPSILON:
                out[i + window - 1] = 0.0
                continue
            z = (windows[i] - means[i]) / s
            w_ = window
            # Adjusted Fisher-Pearson
            out[i + window - 1] = (
                w_ / ((w_ - 1) * (w_ - 2)) * np.sum(z ** 3)
            )

        return out

    @staticmethod
    def _rolling_kurtosis(
        arr: NDArray[np.float64], window: int,
    ) -> NDArray[np.float64]:
        """
        Rolling excess kurtosis.

        kurt = [(n(n+1)) / ((n-1)(n-2)(n-3))] * sum[((x-mean)/std)^4]
               - 3(n-1)^2 / ((n-2)(n-3))
        """
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < window or window < 4:
            return out

        shape = (n - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1, ddof=1)

        for i in range(windows.shape[0]):
            s = stds[i]
            if s < EPSILON:
                out[i + window - 1] = 0.0
                continue
            z = (windows[i] - means[i]) / s
            w_ = window
            m4 = np.sum(z ** 4)
            # Excess kurtosis formula
            term1 = (w_ * (w_ + 1)) / ((w_ - 1) * (w_ - 2) * (w_ - 3)) * m4
            term2 = 3.0 * (w_ - 1) ** 2 / ((w_ - 2) * (w_ - 3))
            out[i + window - 1] = term1 - term2

        return out

    @staticmethod
    def _rolling_hurst(
        arr: NDArray[np.float64], window: int,
    ) -> NDArray[np.float64]:
        """
        Rolling Hurst exponent estimate via the rescaled range (R/S) method.

        H > 0.5 indicates trending (persistence).
        H < 0.5 indicates mean-reversion.
        H = 0.5 indicates a random walk.

        We subdivide the window into halves/quarters and regress
        log(R/S) on log(n) to get the slope H.  For speed we use a
        simplified 2-point estimator (full window vs half).
        """
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < window or window < 8:
            return out

        def rs_stat(series: NDArray[np.float64]) -> float:
            """Compute R/S statistic for a series."""
            m = len(series)
            if m < 2:
                return np.nan
            mean_val = np.mean(series)
            cumdev = np.cumsum(series - mean_val)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(series, ddof=1)
            if s < EPSILON:
                return np.nan
            return r / s

        shape = (n - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

        for i in range(windows.shape[0]):
            w_data = windows[i]
            # Full window R/S
            rs_full = rs_stat(w_data)
            # Half-window R/S (average of two halves)
            half = window // 2
            rs_h1 = rs_stat(w_data[:half])
            rs_h2 = rs_stat(w_data[half:])
            if (
                np.isnan(rs_full)
                or np.isnan(rs_h1)
                or np.isnan(rs_h2)
                or rs_h1 <= 0
                or rs_h2 <= 0
                or rs_full <= 0
            ):
                continue
            rs_half_avg = (rs_h1 + rs_h2) / 2.0
            # H = log(R/S_full / R/S_half) / log(full / half)
            log_n_ratio = np.log(window / half)
            log_rs_ratio = np.log(rs_full / rs_half_avg)
            if log_n_ratio > EPSILON:
                out[i + window - 1] = log_rs_ratio / log_n_ratio

        return out

    @staticmethod
    def _rolling_autocorr(
        arr: NDArray[np.float64], window: int, lag: int = 1,
    ) -> NDArray[np.float64]:
        """
        Rolling autocorrelation at the given lag.

        Uses Pearson correlation between ``arr[t-window+1 : t-lag+1]``
        and ``arr[t-window+1+lag : t+1]``.
        """
        n = len(arr)
        out = np.full(n, np.nan, dtype=np.float64)
        need = window + lag
        if n < need:
            return out

        for i in range(need - 1, n):
            start = i - window - lag + 1
            x = arr[start: start + window]
            y = arr[start + lag: start + lag + window]
            mx = np.mean(x)
            my = np.mean(y)
            cov = np.mean((x - mx) * (y - my))
            sx = np.std(x, ddof=0)
            sy = np.std(y, ddof=0)
            denom = sx * sy
            if denom > EPSILON:
                out[i] = cov / denom
            else:
                out[i] = 0.0
        return out

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    @staticmethod
    def _pad_feature(
        feature: NDArray[np.float64],
        target_length: int,
    ) -> NDArray[np.float64]:
        """Pad feature to *target_length* with NaN at the beginning."""
        flen = len(feature)
        if flen >= target_length:
            return feature[-target_length:]
        padding = np.full(target_length - flen, np.nan, dtype=np.float64)
        return np.concatenate([padding, feature])

    @staticmethod
    def _normalize_features(
        features: NDArray[np.float64],
        window: int = 252,
    ) -> NDArray[np.float64]:
        """
        Rolling z-score normalisation.

        Uses an expanding window initially, then a fixed rolling window.
        NaN values are preserved.
        """
        n, k = features.shape
        normalized = np.full_like(features, np.nan)

        for j in range(k):
            col = features[:, j]
            for i in range(n):
                if np.isnan(col[i]):
                    continue
                start = max(0, i - window + 1)
                history = col[start: i + 1]
                valid = history[~np.isnan(history)]
                if len(valid) < 2:
                    normalized[i, j] = 0.0
                else:
                    mean = np.mean(valid)
                    std = np.std(valid, ddof=1)
                    normalized[i, j] = (col[i] - mean) / (std + EPSILON)

        return normalized
