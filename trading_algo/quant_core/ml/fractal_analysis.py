"""
Fractal / Scaling Analysis of Financial Time Series

Production-quality implementations of classical fractal estimators for
detecting persistence, anti-persistence, and regime structure in price data.

Implements:
    1. Hurst Exponent via Rescaled Range (R/S) Analysis — Hurst (1951),
       Mandelbrot (1971)
    2. Detrended Fluctuation Analysis (DFA) — Peng et al. (1994)
    3. Multifractal DFA (MFDFA) — Kantelhardt et al. (2002)
    4. Rolling variants of all estimators for real-time regime detection
    5. Hurst regime classification for strategy switching

The Hurst exponent H is the key input for the Hurst-adaptive strategy:
    - H > 0.55  → momentum mode (ride trends)
    - H < 0.45  → mean-reversion mode (fade extremes)
    - H ≈ 0.50  → random walk, sit out (preserve capital)

References:
    - Hurst, H.E. (1951): "Long-term storage capacity of reservoirs"
    - Mandelbrot, B.B. & Wallis, J.R. (1969): "Robustness of the rescaled
      range R/S in the measurement of noncyclic long run statistical
      dependence"
    - Mandelbrot, B.B. (1971): "When can price be arbitraged efficiently?"
    - Peng, C.-K. et al. (1994): "Mosaic organization of DNA nucleotides"
    - Kantelhardt, J.W. et al. (2002): "Multifractal detrended fluctuation
      analysis of nonstationary time series"
    - Lo, A.W. (1991): "Long-Term Memory in Stock Market Prices"
    - Peters, E.E. (1994): "Fractal Market Analysis"
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "hurst_exponent_rs",
    "rolling_hurst",
    "dfa",
    "rolling_dfa",
    "classify_hurst_regime",
    "mfdfa",
]

# ---------------------------------------------------------------------------
# Numerical stability constants (local to avoid circular import risk)
# ---------------------------------------------------------------------------
_EPSILON: float = 1e-10
_MIN_BLOCKS: int = 2  # Minimum blocks required for a valid R/S or DFA estimate


# ===========================================================================
# 1. Hurst Exponent via Rescaled Range (R/S) Analysis
# ===========================================================================

def hurst_exponent_rs(
    x: NDArray[np.float64],
    min_window: int = 10,
    max_window: int | None = None,
) -> float:
    """
    Estimate the Hurst exponent using the classical Rescaled Range (R/S)
    method of Hurst (1951) / Mandelbrot & Wallis (1969).

    Algorithm
    ---------
    1. For each window size *n* in a geometric progression from
       ``min_window`` to ``max_window``:

       a. Divide the series into non-overlapping blocks of size *n*.
       b. For each block compute:
          - mean μ
          - cumulative deviations from μ
          - range R = max(cumdev) − min(cumdev)
          - standard deviation S (ddof=1)
          - R/S ratio = R / S  (skip if S ≈ 0)
       c. Average R/S across all blocks for this window size.

    2. Regress log(R/S) on log(n) via OLS.
    3. The slope is the Hurst exponent H.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input time series (e.g. log returns or prices).
    min_window : int
        Smallest block size.  Must be >= 4.
    max_window : int or None
        Largest block size.  Defaults to ``len(x) // 2``.

    Returns
    -------
    float
        Estimated Hurst exponent H.

        - H < 0.5  →  mean-reverting (anti-persistent)
        - H ≈ 0.5  →  random walk (uncorrelated)
        - H > 0.5  →  trending (persistent)

        Returns ``np.nan`` if the series is too short or degenerate.

    Notes
    -----
    The R/S method is known to be biased for short series and is
    sensitive to short-range correlations (Lo, 1991).  For non-stationary
    financial data the DFA estimator is generally preferred.  This
    implementation is included for completeness and for cross-validation.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # --- Input validation ---
    min_window = max(min_window, 4)
    if max_window is None:
        max_window = n // 2
    max_window = min(max_window, n)

    if n < min_window * _MIN_BLOCKS:
        return np.nan

    if max_window < min_window:
        return np.nan

    # Check for degenerate (constant) series
    if np.std(x) < _EPSILON:
        return np.nan

    # --- Generate logarithmically spaced window sizes ---
    n_sizes = max(10, int(np.log2(max_window / min_window) * 5))
    window_sizes = np.unique(
        np.geomspace(min_window, max_window, num=n_sizes).astype(np.int64)
    )
    # Keep only sizes that allow at least _MIN_BLOCKS blocks
    window_sizes = window_sizes[(window_sizes >= min_window) & (n // window_sizes >= _MIN_BLOCKS)]

    if len(window_sizes) < 2:
        return np.nan

    log_ns: List[float] = []
    log_rs: List[float] = []

    for w in window_sizes:
        w = int(w)
        n_blocks = n // w
        if n_blocks < _MIN_BLOCKS:
            continue

        rs_values: List[float] = []
        for b in range(n_blocks):
            block = x[b * w : (b + 1) * w]
            mean_b = np.mean(block)
            cumdev = np.cumsum(block - mean_b)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)

            if s < _EPSILON:
                continue

            rs_values.append(r / s)

        if len(rs_values) < 1:
            continue

        avg_rs = np.mean(rs_values)
        if avg_rs > _EPSILON:
            log_ns.append(math.log(w))
            log_rs.append(math.log(avg_rs))

    if len(log_ns) < 2:
        return np.nan

    # --- OLS regression: log(R/S) = H * log(n) + c ---
    log_ns_arr = np.array(log_ns, dtype=np.float64)
    log_rs_arr = np.array(log_rs, dtype=np.float64)

    h = _ols_slope(log_ns_arr, log_rs_arr)
    return h


# ===========================================================================
# 2. Rolling Hurst Exponent
# ===========================================================================

def rolling_hurst(
    x: NDArray[np.float64],
    window: int = 78,
    min_window: int = 8,
    step: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the Hurst exponent on a rolling basis using the R/S method.

    For large arrays (40 000+ bars), set ``step > 1`` to compute every
    *step*-th bar and forward-fill the rest.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input time series.
    window : int
        Rolling window size.  Default 78 (≈ one trading day of 5-min bars).
    min_window : int
        Minimum block size passed to :func:`hurst_exponent_rs`.
    step : int
        Compute every *step*-th bar.  Intermediate values are forward-filled
        from the last computed value.

    Returns
    -------
    NDArray[np.float64]
        Array of same length as *x*.  The first ``window - 1`` elements
        are ``np.nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < window or window < min_window * _MIN_BLOCKS:
        return out

    step = max(1, step)

    last_value = np.nan
    for i in range(window - 1, n):
        # Compute only at step intervals (or the very first valid index)
        if (i - (window - 1)) % step == 0:
            segment = x[i - window + 1 : i + 1]
            last_value = hurst_exponent_rs(segment, min_window=min_window)
        out[i] = last_value

    return out


# ===========================================================================
# 3. Detrended Fluctuation Analysis (DFA)
# ===========================================================================

def dfa(
    x: NDArray[np.float64],
    min_window: int = 4,
    max_window: int | None = None,
    order: int = 1,
) -> float:
    """
    Detrended Fluctuation Analysis (DFA) after Peng et al. (1994).

    DFA is more robust than R/S for non-stationary data — which financial
    time series invariably are — because the polynomial detrending removes
    local trends that would inflate the R/S statistic.

    Algorithm
    ---------
    1. Compute the profile (cumulative sum of mean-centred series):
       ``Y[k] = Σ_{i=1}^{k} (x[i] − <x>)``

    2. For each window size *n*:

       a. Divide Y into non-overlapping segments of size *n*.
       b. In each segment fit a polynomial of degree ``order`` and compute
          the variance of the residuals.
       c. Average the segment variances → F²(n).

    3. Regress log(F(n)) on log(n) via OLS.
    4. The slope α is the DFA exponent.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input time series.
    min_window : int
        Smallest segment size.  Must be >= ``order + 2``.
    max_window : int or None
        Largest segment size.  Defaults to ``len(x) // 4``.
    order : int
        Polynomial detrending order.
            - 1 = DFA-1 (linear detrend)
            - 2 = DFA-2 (quadratic detrend)

    Returns
    -------
    float
        DFA exponent α.

        - α < 0.5  →  anti-persistent
        - α ≈ 0.5  →  uncorrelated (white noise)
        - α > 0.5  →  persistent (long-range correlations)
        - α ≈ 1.0  →  1/f noise
        - α > 1.0  →  non-stationary, unbounded correlations

        Returns ``np.nan`` if the series is too short or degenerate.

    Notes
    -----
    For stationary processes the DFA exponent α equals the Hurst exponent H.
    For non-stationary integrated processes α = H + 1 for DFA-1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # --- Input validation ---
    min_window = max(min_window, order + 2)
    if max_window is None:
        max_window = n // 4
    max_window = min(max_window, n)

    if n < min_window * _MIN_BLOCKS:
        return np.nan

    if max_window < min_window:
        return np.nan

    # Check for degenerate series
    if np.std(x) < _EPSILON:
        return np.nan

    # --- Step 1: Compute profile ---
    profile = np.cumsum(x - np.mean(x))

    # --- Generate window sizes ---
    n_sizes = max(10, int(np.log2(max_window / min_window) * 5))
    window_sizes = np.unique(
        np.geomspace(min_window, max_window, num=n_sizes).astype(np.int64)
    )
    window_sizes = window_sizes[
        (window_sizes >= min_window) & (n // window_sizes >= _MIN_BLOCKS)
    ]

    if len(window_sizes) < 2:
        return np.nan

    log_ns: List[float] = []
    log_fs: List[float] = []

    for w in window_sizes:
        w = int(w)
        n_segments = n // w
        if n_segments < _MIN_BLOCKS:
            continue

        if order == 1:
            # Vectorised path for linear detrending — much faster
            f2 = _batch_linear_detrend_variance(profile, w)
            if f2 > _EPSILON:
                log_ns.append(math.log(w))
                log_fs.append(0.5 * math.log(f2))
        else:
            # General path for higher-order polynomial detrend
            variance_sum = 0.0
            count = 0

            # Forward segments
            for s in range(n_segments):
                segment = profile[s * w : (s + 1) * w]
                var = _segment_detrended_variance(segment, order)
                if not np.isnan(var):
                    variance_sum += var
                    count += 1

            # Backward segments (from the end) — captures remainder
            for s in range(n_segments):
                start = n - (s + 1) * w
                if start < 0:
                    break
                segment = profile[start : start + w]
                var = _segment_detrended_variance(segment, order)
                if not np.isnan(var):
                    variance_sum += var
                    count += 1

            if count == 0:
                continue

            f_n = math.sqrt(variance_sum / count)
            if f_n > _EPSILON:
                log_ns.append(math.log(w))
                log_fs.append(math.log(f_n))

    if len(log_ns) < 2:
        return np.nan

    # --- OLS regression: log(F) = α * log(n) + c ---
    log_ns_arr = np.array(log_ns, dtype=np.float64)
    log_fs_arr = np.array(log_fs, dtype=np.float64)

    alpha = _ols_slope(log_ns_arr, log_fs_arr)
    return alpha


# ===========================================================================
# 4. Rolling DFA
# ===========================================================================

def rolling_dfa(
    x: NDArray[np.float64],
    window: int = 390,
    step: int = 1,
    order: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the DFA exponent on a rolling basis.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input time series.
    window : int
        Rolling window size.  Default 390 (≈ one trading week of 5-min bars).
    step : int
        Compute every *step*-th bar, forward-fill the rest.
    order : int
        Polynomial detrending order for DFA.

    Returns
    -------
    NDArray[np.float64]
        Array of same length as *x*.  The first ``window - 1`` elements
        are ``np.nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)

    min_required = (order + 2) * _MIN_BLOCKS
    if n < window or window < min_required:
        return out

    step = max(1, step)

    last_value = np.nan
    for i in range(window - 1, n):
        if (i - (window - 1)) % step == 0:
            segment = x[i - window + 1 : i + 1]
            last_value = dfa(segment, order=order)
        out[i] = last_value

    return out


# ===========================================================================
# 5. Hurst Regime Classification
# ===========================================================================

def classify_hurst_regime(
    h: float,
    threshold_mr: float = 0.45,
    threshold_mom: float = 0.55,
) -> str:
    """
    Classify a Hurst exponent value into a market regime.

    This simple classifier is the gateway to strategy switching:

    - ``"mean_reverting"`` → fade extremes, pairs trading, Ornstein-Uhlenbeck
    - ``"trending"``       → momentum, breakout, trend-following
    - ``"random_walk"``    → sit out, reduce exposure, preserve capital

    Parameters
    ----------
    h : float
        Hurst exponent estimate (from R/S or DFA).
    threshold_mr : float
        Below this value the regime is classified as mean-reverting.
    threshold_mom : float
        Above this value the regime is classified as trending.

    Returns
    -------
    str
        One of ``"mean_reverting"``, ``"trending"``, or ``"random_walk"``.

    Examples
    --------
    >>> classify_hurst_regime(0.38)
    'mean_reverting'
    >>> classify_hurst_regime(0.50)
    'random_walk'
    >>> classify_hurst_regime(0.62)
    'trending'
    """
    if np.isnan(h):
        return "random_walk"

    if h < threshold_mr:
        return "mean_reverting"
    elif h > threshold_mom:
        return "trending"
    else:
        return "random_walk"


# ===========================================================================
# 6. Multifractal DFA (MFDFA)
# ===========================================================================

def mfdfa(
    x: NDArray[np.float64],
    q_range: tuple = (-5, 5),
    n_q: int = 11,
    min_window: int = 4,
    max_window: int | None = None,
    order: int = 1,
) -> Dict[str, NDArray[np.float64]]:
    """
    Multifractal Detrended Fluctuation Analysis after
    Kantelhardt et al. (2002).

    Extends standard DFA by computing generalised fluctuation functions
    F_q(n) for a range of moment orders q, yielding the generalised Hurst
    exponent h(q).  A single-valued h(q) ≡ H (constant for all q) indicates
    monofractal behaviour; varying h(q) indicates multifractality — complex
    dynamics with multiple interleaved scaling laws.

    Algorithm
    ---------
    1. Compute the profile Y as in standard DFA.
    2. For each window size *n*:
       a. Compute segment variances F²(ν, n) for all segments ν.
    3. For each q ≠ 0:
       ``F_q(n) = { (1/N_s) Σ [F²(ν,n)]^{q/2} }^{1/q}``
       For q = 0:
       ``F_0(n) = exp{ (1/2N_s) Σ ln[F²(ν,n)] }``
    4. Regress log(F_q(n)) on log(n) → slope h(q).
    5. Derive the multifractal spectrum:
       - τ(q) = q·h(q) − 1           (Rényi exponent)
       - α(q) = dτ/dq  = h(q) + q·h'(q)  (Hölder / singularity exponent)
       - f(α) = q·α − τ(q)           (singularity spectrum)

    Parameters
    ----------
    x : NDArray[np.float64]
        Input time series.
    q_range : tuple of (float, float)
        Range of moment orders.  Default (-5, 5).
    n_q : int
        Number of q values (linearly spaced).
    min_window : int
        Smallest segment size.
    max_window : int or None
        Largest segment size.  Defaults to ``len(x) // 4``.
    order : int
        Polynomial detrending order.

    Returns
    -------
    dict with keys:
        ``'q'``   : NDArray — moment orders
        ``'hq'``  : NDArray — generalised Hurst exponent h(q)
        ``'tauq'``: NDArray — Rényi exponent τ(q) = q·h(q) − 1
        ``'alpha'``: NDArray — singularity exponents (Hölder)
        ``'falpha'``: NDArray — singularity spectrum f(α)

        Returns ``{'q': ..., 'hq': array of NaN, ...}`` if estimation
        fails.

    Notes
    -----
    Multifractality in financial markets has been linked to heterogeneous
    trader horizons and intermittent volatility clustering.  A broad
    singularity spectrum (wide range of α) suggests richer dynamics.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # --- Generate q values, ensuring q=0 is included ---
    q_values = np.linspace(q_range[0], q_range[1], n_q)
    # Remove values very close to zero and add exact zero
    q_values = q_values[np.abs(q_values) > 0.01]
    q_values = np.sort(np.concatenate([q_values, [0.0]]))
    n_q_actual = len(q_values)

    nan_result = {
        "q": q_values,
        "hq": np.full(n_q_actual, np.nan, dtype=np.float64),
        "tauq": np.full(n_q_actual, np.nan, dtype=np.float64),
        "alpha": np.full(n_q_actual, np.nan, dtype=np.float64),
        "falpha": np.full(n_q_actual, np.nan, dtype=np.float64),
    }

    # --- Input validation ---
    min_window = max(min_window, order + 2)
    if max_window is None:
        max_window = n // 4
    max_window = min(max_window, n)

    if n < min_window * _MIN_BLOCKS or max_window < min_window:
        return nan_result

    if np.std(x) < _EPSILON:
        return nan_result

    # --- Step 1: Compute profile ---
    profile = np.cumsum(x - np.mean(x))

    # --- Generate window sizes ---
    n_sizes = max(10, int(np.log2(max_window / min_window) * 5))
    window_sizes = np.unique(
        np.geomspace(min_window, max_window, num=n_sizes).astype(np.int64)
    )
    window_sizes = window_sizes[
        (window_sizes >= min_window) & (n // window_sizes >= _MIN_BLOCKS)
    ]

    if len(window_sizes) < 2:
        return nan_result

    # --- Step 2: Compute segment variances for all window sizes ---
    # seg_variances[i] = list of F²(ν, n_i) for window size i
    seg_variances_by_scale: List[NDArray[np.float64]] = []
    valid_windows: List[int] = []

    for w in window_sizes:
        w = int(w)
        n_segments = n // w
        if n_segments < _MIN_BLOCKS:
            continue

        variances: List[float] = []

        # Forward segments
        for s in range(n_segments):
            segment = profile[s * w : (s + 1) * w]
            var = _segment_detrended_variance(segment, order)
            if not np.isnan(var) and var > _EPSILON:
                variances.append(var)

        # Backward segments
        for s in range(n_segments):
            start = n - (s + 1) * w
            if start < 0:
                break
            segment = profile[start : start + w]
            var = _segment_detrended_variance(segment, order)
            if not np.isnan(var) and var > _EPSILON:
                variances.append(var)

        if len(variances) >= _MIN_BLOCKS:
            seg_variances_by_scale.append(np.array(variances, dtype=np.float64))
            valid_windows.append(w)

    if len(valid_windows) < 2:
        return nan_result

    # --- Step 3: Compute F_q(n) for each q and n ---
    hq = np.full(n_q_actual, np.nan, dtype=np.float64)
    n_scales = len(valid_windows)

    for qi, q in enumerate(q_values):
        log_ns: List[float] = []
        log_fqs: List[float] = []

        for si in range(n_scales):
            vars_arr = seg_variances_by_scale[si]
            n_s = len(vars_arr)

            if abs(q) < 0.01:
                # q ≈ 0: F_0(n) = exp{ (1/(2*N_s)) * Σ ln[F²(ν,n)] }
                fq_n = math.exp(0.5 * np.mean(np.log(vars_arr)))
            else:
                # F_q(n) = { (1/N_s) * Σ [F²(ν,n)]^{q/2} }^{1/q}
                powered = np.mean(np.power(vars_arr, q / 2.0))
                if powered <= 0:
                    continue
                fq_n = math.pow(powered, 1.0 / q)

            if fq_n > _EPSILON:
                log_ns.append(math.log(valid_windows[si]))
                log_fqs.append(math.log(fq_n))

        if len(log_ns) >= 2:
            hq[qi] = _ols_slope(
                np.array(log_ns, dtype=np.float64),
                np.array(log_fqs, dtype=np.float64),
            )

    # --- Step 4: Derive multifractal spectrum ---
    tauq = q_values * hq - 1.0

    # α = dτ/dq (numerical derivative)
    alpha = np.full(n_q_actual, np.nan, dtype=np.float64)
    falpha = np.full(n_q_actual, np.nan, dtype=np.float64)

    valid_tau = ~np.isnan(tauq)
    if np.sum(valid_tau) >= 3:
        # Use central differences where possible
        q_valid = q_values[valid_tau]
        tau_valid = tauq[valid_tau]
        alpha_valid = np.gradient(tau_valid, q_valid)
        f_valid = q_valid * alpha_valid - tau_valid

        alpha[valid_tau] = alpha_valid
        falpha[valid_tau] = f_valid

    return {
        "q": q_values,
        "hq": hq,
        "tauq": tauq,
        "alpha": alpha,
        "falpha": falpha,
    }


# ===========================================================================
# Private Helpers
# ===========================================================================

def _batch_linear_detrend_variance(
    profile: NDArray[np.float64],
    w: int,
) -> float:
    """
    Fully vectorised computation of average detrended variance across
    non-overlapping segments of size *w* (forward + backward), using
    analytical linear detrending.

    All segments are processed as a single (N_seg, w) matrix via bulk
    numpy operations — no Python loops over segments.

    Parameters
    ----------
    profile : NDArray[np.float64]
        Cumulative profile array.
    w : int
        Segment (window) size.

    Returns
    -------
    float
        Average F²(n) across all segments.  Returns 0.0 if no valid
        segments.
    """
    n = len(profile)
    n_seg = n // w

    if n_seg < 1:
        return 0.0

    # Pre-compute constants for t = 0, 1, ..., w-1
    t = np.arange(w, dtype=np.float64)
    t_mean = (w - 1) * 0.5
    ss_tt = w * (w - 1) * (2 * w - 1) / 6.0 - w * t_mean * t_mean

    if abs(ss_tt) < _EPSILON:
        return 0.0

    # --- Forward segments: reshape contiguous block into (n_seg, w) ---
    fwd_block = profile[: n_seg * w].reshape(n_seg, w)

    # --- Backward segments: build from the tail ---
    remainder = n - n_seg * w
    if remainder > 0:
        bwd_block = profile[remainder : remainder + n_seg * w].reshape(n_seg, w)
        # Stack forward and backward
        blocks = np.vstack([fwd_block, bwd_block])
    else:
        # Forward and backward are identical — just use forward
        blocks = fwd_block

    # Vectorised linear detrend across all segments simultaneously
    # blocks shape: (K, w),  t shape: (w,)
    y_means = np.mean(blocks, axis=1)                    # (K,)
    ss_ty = blocks @ t - w * t_mean * y_means            # (K,)
    slopes = ss_ty / ss_tt                               # (K,)
    intercepts = y_means - slopes * t_mean               # (K,)

    # Compute trends: (K, w) = slopes[:, None] * t[None, :] + intercepts[:, None]
    trends = slopes[:, np.newaxis] * t[np.newaxis, :] + intercepts[:, np.newaxis]
    residuals = blocks - trends
    variances = np.mean(residuals * residuals, axis=1)   # (K,)

    return float(np.mean(variances))


def _ols_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Ordinary Least Squares slope for y = slope * x + intercept.

    Uses the direct formula:
        slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²

    Numerically stable via mean-centring.

    Parameters
    ----------
    x, y : NDArray[np.float64]
        Equal-length arrays.

    Returns
    -------
    float
        OLS slope.  Returns ``np.nan`` if degenerate.
    """
    n = len(x)
    if n < 2:
        return np.nan

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    dx = x - x_mean
    dy = y - y_mean
    ss_xx = np.dot(dx, dx)

    if ss_xx < _EPSILON:
        return np.nan

    ss_xy = np.dot(dx, dy)
    return float(ss_xy / ss_xx)


def _segment_detrended_variance(
    segment: NDArray[np.float64],
    order: int,
) -> float:
    """
    Fit a polynomial of given ``order`` to the segment and return the
    variance (mean squared residual) of the detrended segment.

    Parameters
    ----------
    segment : NDArray[np.float64]
        Profile segment of length *n*.
    order : int
        Polynomial degree (1 = linear, 2 = quadratic, ...).

    Returns
    -------
    float
        Mean squared residual (F² for this segment).
        Returns ``np.nan`` if the fit fails.
    """
    m = len(segment)
    if m < order + 2:
        return np.nan

    if order == 1:
        # Fast analytical linear detrend — avoids np.polyfit overhead.
        # For y = a*t + b over t = 0..m-1:
        #   a = (Σ t*y - m*t̄*ȳ) / (Σ t² - m*t̄²)
        #   b = ȳ - a*t̄
        # Pre-computed:  t̄ = (m-1)/2,  Σt² = m*(m-1)*(2m-1)/6
        y_mean = np.mean(segment)
        t_mean = (m - 1) * 0.5
        ss_tt = m * (m - 1) * (2 * m - 1) / 6.0 - m * t_mean * t_mean

        if abs(ss_tt) < _EPSILON:
            # Degenerate (m < 2 essentially) — just return variance
            return float(np.mean((segment - y_mean) ** 2))

        # Σ t*y  via dot product with arange
        t = np.arange(m, dtype=np.float64)
        ss_ty = np.dot(t, segment) - m * t_mean * y_mean
        slope = ss_ty / ss_tt
        intercept = y_mean - slope * t_mean

        residuals = segment - (slope * t + intercept)
        return float(np.mean(residuals * residuals))

    # General case: higher-order polynomial detrend
    t = np.arange(m, dtype=np.float64)
    try:
        coeffs = np.polyfit(t, segment, order)
        trend = np.polyval(coeffs, t)
        residuals = segment - trend
        return float(np.mean(residuals * residuals))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
