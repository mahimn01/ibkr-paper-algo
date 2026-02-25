"""
Information-Theoretic Signal Detection for Quantitative Finance.

Provides nonlinear dependence measures and complexity estimators that go far
beyond Pearson correlation.  In financial markets most exploitable
relationships are nonlinear and time-varying; information-theoretic quantities
capture *any* statistical dependence without assuming a functional form.

Key measures implemented:

    Transfer Entropy (TE)
        Directional information flow X -> Y.  Detects lead-lag relationships
        between instruments even when linear cross-correlation is zero.
        Uses the Kraskov-Stoegbauer-Grassberger (KSG) k-nearest-neighbor
        estimator for continuous data.

    Approximate Entropy / Sample Entropy
        Time-series regularity measures.  Low values indicate deterministic,
        exploitable structure; high values indicate noise.  Used for regime
        classification (trending vs. choppy markets).

    Permutation Entropy
        Ordinal-pattern complexity of Bandt & Pompe (2002).  Extremely fast,
        robust to noise, and monotone-invariant.  Ideal as a real-time
        regime filter.

    Mutual Information (MI)
        Symmetric nonlinear dependence between two variables.  Used for
        feature selection (MI between candidate feature and forward return)
        and for detecting breakdown of hedging relationships.

All entropy values are in **nats** (natural logarithm base) unless stated
otherwise.  Rolling variants return NaN-padded arrays aligned with the input
so they can be directly used as features in a FeatureSet.

References:
    - Kraskov, Stoegbauer & Grassberger (2004): "Estimating mutual
      information", Phys. Rev. E 69, 066138
    - Schreiber (2000): "Measuring information transfer", Phys. Rev. Lett.
      85, 461
    - Pincus (1991): "Approximate entropy as a measure of system complexity",
      Proc. Natl. Acad. Sci. 88, 2297
    - Richman & Moorman (2000): "Physiological time-series analysis using
      approximate entropy and sample entropy", Am. J. Physiol. 278, H2039
    - Bandt & Pompe (2002): "Permutation entropy: a natural complexity
      measure for time series", Phys. Rev. Lett. 88, 174102
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.special import digamma, factorial

__all__ = [
    "transfer_entropy",
    "approximate_entropy",
    "sample_entropy",
    "permutation_entropy",
    "mutual_information",
    "rolling_transfer_entropy",
    "rolling_sample_entropy",
    "rolling_permutation_entropy",
]

# ---------------------------------------------------------------------------
# Numerical-stability helpers
# ---------------------------------------------------------------------------

_EPS: float = 1e-15  # Avoids log(0) and zero-distance issues in KD-trees


def _validate_1d(x: NDArray, min_length: int = 3) -> Tuple[bool, NDArray[np.float64]]:
    """Validate and sanitise a 1-D input array.

    Returns (is_valid, cleaned_array).  If *is_valid* is ``False`` the caller
    should return a sentinel (NaN or 0.0) immediately.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    # Strip leading/trailing NaN
    mask = np.isfinite(x)
    if mask.sum() < min_length:
        return False, x
    # Keep only finite values (preserves order)
    x = x[mask]
    return True, x


def _is_constant(x: NDArray[np.float64]) -> bool:
    """Return True if all finite values are identical."""
    return np.ptp(x) == 0.0


# ---------------------------------------------------------------------------
# 1. Transfer Entropy  (KSG estimator)
# ---------------------------------------------------------------------------

def transfer_entropy(
    x: NDArray,
    y: NDArray,
    lag: int = 1,
    k: int = 5,
) -> float:
    """Estimate Transfer Entropy from *x* to *y*: TE(X -> Y).

    Uses the Kraskov-Stoegbauer-Grassberger (KSG) nearest-neighbour
    estimator to compute the conditional mutual information

        TE(X -> Y) = I(Y_future ; X_past | Y_past)
                    = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Parameters
    ----------
    x : array_like, shape (N,)
        Source time series.
    y : array_like, shape (N,)
        Target time series.
    lag : int, default 1
        Number of time steps for the "past" embedding.
    k : int, default 5
        Number of nearest neighbours for the KSG estimator.

    Returns
    -------
    float
        Transfer entropy in **nats**.  Returns ``0.0`` for degenerate inputs
        (constant arrays, insufficient data) and ``NaN`` when inputs contain
        only NaN.

    References
    ----------
    Kraskov, Stoegbauer & Grassberger (2004), Phys. Rev. E 69, 066138.
    Schreiber (2000), Phys. Rev. Lett. 85, 461.
    """
    # ---- input validation --------------------------------------------------
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    n = min(len(x), len(y))
    if n < lag + k + 2:
        return np.nan if n == 0 else 0.0

    x = x[:n]
    y = y[:n]

    # Aligned finite mask
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < lag + k + 2:
        return 0.0

    x = x[mask]
    y = y[mask]
    n = len(x)

    if _is_constant(x) or _is_constant(y):
        return 0.0

    # ---- construct state vectors -------------------------------------------
    # Y_future  = y[lag:]
    # Y_past    = y[:n-lag]
    # X_past    = x[:n-lag]
    m = n - lag
    if m < k + 2:
        return 0.0

    y_future = y[lag:lag + m].reshape(-1, 1)
    y_past = y[:m].reshape(-1, 1)
    x_past = x[:m].reshape(-1, 1)

    # Joint space: (Y_future, Y_past, X_past)
    joint = np.hstack([y_future, y_past, x_past])

    # ---- KSG estimation (Chebyshev / max norm) -----------------------------
    tree_joint = cKDTree(joint)
    # k-th neighbour distance in Chebyshev norm (index k because query
    # point itself is at distance 0).
    dists, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # distance to k-th neighbour
    eps = np.maximum(eps, _EPS)  # numerical floor

    # Count neighbours within eps in marginal subspaces
    tree_yf_yp = cKDTree(np.hstack([y_future, y_past]))
    tree_yp_xp = cKDTree(np.hstack([y_past, x_past]))
    tree_yp = cKDTree(y_past)

    n_yf_yp = np.array([
        tree_yf_yp.query_ball_point(pt, r=eps[i] - _EPS, p=np.inf).__len__() - 1
        for i, pt in enumerate(np.hstack([y_future, y_past]))
    ], dtype=np.float64)

    n_yp_xp = np.array([
        tree_yp_xp.query_ball_point(pt, r=eps[i] - _EPS, p=np.inf).__len__() - 1
        for i, pt in enumerate(np.hstack([y_past, x_past]))
    ], dtype=np.float64)

    n_yp = np.array([
        tree_yp.query_ball_point(pt, r=eps[i] - _EPS, p=np.inf).__len__() - 1
        for i, pt in enumerate(y_past)
    ], dtype=np.float64)

    # Clip to >= 1 for digamma stability
    n_yf_yp = np.maximum(n_yf_yp, 1)
    n_yp_xp = np.maximum(n_yp_xp, 1)
    n_yp = np.maximum(n_yp, 1)

    # TE = psi(k) - <psi(n_yf_yp + 1) + psi(n_yp_xp + 1) - psi(n_yp + 1)>
    te = (
        digamma(k)
        - np.mean(digamma(n_yf_yp + 1))
        - np.mean(digamma(n_yp_xp + 1))
        + np.mean(digamma(n_yp + 1))
    )

    return max(float(te), 0.0)  # TE is non-negative by construction


# ---------------------------------------------------------------------------
# 2. Approximate Entropy
# ---------------------------------------------------------------------------

def approximate_entropy(
    x: NDArray,
    m: int = 2,
    r: float = 0.2,
) -> float:
    """Compute Approximate Entropy (ApEn) of Pincus (1991).

    ApEn quantifies the regularity/predictability of a time series.
    Lower values indicate more regularity (deterministic structure);
    higher values indicate randomness.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input time series.
    m : int, default 2
        Embedding dimension (template length).
    r : float, default 0.2
        Tolerance as a fraction of the series standard deviation.

    Returns
    -------
    float
        Approximate entropy in nats.  Returns ``0.0`` for constant or
        degenerate series and ``NaN`` for all-NaN input.

    References
    ----------
    Pincus (1991), Proc. Natl. Acad. Sci. 88, 2297-2301.
    """
    ok, x = _validate_1d(x, min_length=m + 2)
    if not ok:
        return np.nan
    if _is_constant(x):
        return 0.0

    n = len(x)
    r_thr = r * np.std(x, ddof=0)
    if r_thr < _EPS:
        return 0.0

    def _phi(dim: int) -> float:
        """Compute phi(dim) = (1/N') * sum( log(C_i) ) where N' = n - dim + 1."""
        n_templates = n - dim + 1
        if n_templates <= 0:
            return 0.0
        # Build template matrix
        templates = np.array([x[i: i + dim] for i in range(n_templates)])
        counts = np.zeros(n_templates, dtype=np.float64)
        for i in range(n_templates):
            # Chebyshev distance between template i and all others
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.sum(dists <= r_thr)
        # Counts include self-match
        counts = counts / n_templates
        return float(np.mean(np.log(np.maximum(counts, _EPS))))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    apen = phi_m - phi_m1
    return max(float(apen), 0.0)


# ---------------------------------------------------------------------------
# 3. Sample Entropy
# ---------------------------------------------------------------------------

def sample_entropy(
    x: NDArray,
    m: int = 2,
    r: float = 0.2,
) -> float:
    """Compute Sample Entropy (SampEn) of Richman & Moorman (2000).

    An improved version of Approximate Entropy that:
      * does **not** count self-matches (reduces bias),
      * is independent of series length (more consistent for short series).

    SampEn = -ln(A / B)

    where B = number of template matches of length *m* (excluding self),
    and A = number of template matches of length *m + 1* (excluding self).

    Parameters
    ----------
    x : array_like, shape (N,)
        Input time series.
    m : int, default 2
        Embedding dimension.
    r : float, default 0.2
        Tolerance as a fraction of the standard deviation.

    Returns
    -------
    float
        Sample entropy in nats.  Returns ``inf`` when no matches at
        dimension *m + 1*, ``0.0`` for constant series, and ``NaN`` for
        degenerate input.

    References
    ----------
    Richman & Moorman (2000), Am. J. Physiol. 278, H2039-H2049.
    """
    ok, x = _validate_1d(x, min_length=m + 2)
    if not ok:
        return np.nan
    if _is_constant(x):
        return 0.0

    n = len(x)
    r_thr = r * np.std(x, ddof=0)
    if r_thr < _EPS:
        return 0.0

    def _count_matches(dim: int) -> int:
        """Count template matches of length *dim*, excluding self-matches."""
        n_templates = n - dim
        if n_templates <= 0:
            return 0
        templates = np.array([x[i: i + dim] for i in range(n_templates)])
        total = 0
        for i in range(n_templates):
            dists = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            total += int(np.sum(dists <= r_thr))
        return total

    b_count = _count_matches(m)      # matches at dimension m
    a_count = _count_matches(m + 1)  # matches at dimension m + 1

    if b_count == 0:
        return np.nan  # undefined -- series too short / irregular
    if a_count == 0:
        return float("inf")

    return -math.log(a_count / b_count)


# ---------------------------------------------------------------------------
# 4. Permutation Entropy
# ---------------------------------------------------------------------------

def permutation_entropy(
    x: NDArray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute Permutation Entropy of Bandt & Pompe (2002).

    Converts overlapping windows of length *order* (with time-delay *delay*)
    into ordinal patterns (rank orderings), then computes the Shannon entropy
    of the resulting pattern distribution.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input time series.
    order : int, default 3
        Order (length) of the ordinal patterns.  Must be >= 2.
    delay : int, default 1
        Time delay between successive elements in a pattern.
    normalize : bool, default True
        If ``True``, divide by ``log(order!)`` so the result lies in [0, 1].

    Returns
    -------
    float
        Permutation entropy.  Returns ``0.0`` for constant or degenerate
        series and ``NaN`` for all-NaN input.

    Notes
    -----
    Ties are broken by order of occurrence (stable argsort), following the
    original Bandt & Pompe prescription.

    References
    ----------
    Bandt & Pompe (2002), Phys. Rev. Lett. 88, 174102.
    """
    ok, x = _validate_1d(x, min_length=order * delay + 1)
    if not ok:
        return np.nan
    if _is_constant(x):
        return 0.0
    if order < 2:
        raise ValueError("order must be >= 2")

    n = len(x)
    n_patterns = n - (order - 1) * delay
    if n_patterns <= 0:
        return np.nan

    # Build ordinal patterns ---
    # For each window, extract elements and compute the rank ordering.
    # Using a dict of tuple(rank) -> count is faster than hashing large
    # arrays through numpy for typical order values (3-7).
    pattern_counts: dict[tuple[int, ...], int] = {}

    for i in range(n_patterns):
        window = x[i: i + order * delay: delay]
        # Stable argsort gives tie-breaking by occurrence order
        pattern = tuple(np.argsort(window, kind="mergesort").tolist())
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Shannon entropy of the distribution
    counts = np.array(list(pattern_counts.values()), dtype=np.float64)
    probs = counts / counts.sum()
    h = -float(np.sum(probs * np.log(np.maximum(probs, _EPS))))

    if normalize:
        max_entropy = math.log(math.factorial(order))
        if max_entropy > 0:
            h = h / max_entropy

    return float(h)


# ---------------------------------------------------------------------------
# 5. Mutual Information  (KSG estimator)
# ---------------------------------------------------------------------------

def mutual_information(
    x: NDArray,
    y: NDArray,
    k: int = 5,
) -> float:
    """Estimate Mutual Information I(X; Y) via the KSG estimator.

    MI(X, Y) = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(N)

    where psi is the digamma function, n_x and n_y are the number of
    neighbours within the k-th-neighbour distance in the marginal spaces,
    and N is the total number of points.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Two continuous variables.
    k : int, default 5
        Number of nearest neighbours.

    Returns
    -------
    float
        Mutual information in **nats**.  Returns ``0.0`` for degenerate
        inputs.

    References
    ----------
    Kraskov, Stoegbauer & Grassberger (2004), Phys. Rev. E 69, 066138.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(x), len(y))
    if n < k + 2:
        return 0.0

    x = x[:n]
    y = y[:n]

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < k + 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    n = len(x)

    if _is_constant(x) or _is_constant(y):
        return 0.0

    # Joint space
    joint = np.column_stack([x, y])
    tree_joint = cKDTree(joint)
    dists, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, -1]
    eps = np.maximum(eps, _EPS)

    # Marginal counts
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    n_x = np.array([
        tree_x.query_ball_point(x[i: i + 1], r=eps[i] - _EPS, p=np.inf).__len__() - 1
        for i in range(n)
    ], dtype=np.float64)

    n_y = np.array([
        tree_y.query_ball_point(y[i: i + 1], r=eps[i] - _EPS, p=np.inf).__len__() - 1
        for i in range(n)
    ], dtype=np.float64)

    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return max(float(mi), 0.0)


# ---------------------------------------------------------------------------
# 6. Rolling variants
# ---------------------------------------------------------------------------

def rolling_transfer_entropy(
    x: NDArray,
    y: NDArray,
    window: int = 390,
    lag: int = 1,
    k: int = 3,
) -> NDArray[np.float64]:
    """Rolling window Transfer Entropy TE(X -> Y).

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Source and target time series.
    window : int, default 390
        Rolling window size in bars (390 = one full trading day at 1-min).
    lag : int, default 1
        Lag for the TE estimator.
    k : int, default 3
        Number of neighbours (reduced from default 5 for speed).

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Rolling TE values; first ``window - 1`` entries are ``NaN``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out

    for i in range(window - 1, n):
        start = i - window + 1
        out[i] = transfer_entropy(x[start: i + 1], y[start: i + 1], lag=lag, k=k)

    return out


def rolling_sample_entropy(
    x: NDArray,
    window: int = 78,
    m: int = 2,
    r: float = 0.2,
) -> NDArray[np.float64]:
    """Rolling window Sample Entropy.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input time series.
    window : int, default 78
        Rolling window size in bars (78 ~ 2 hours at 1-min resolution,
        or ~1 quarter at daily resolution).
    m : int, default 2
        Embedding dimension.
    r : float, default 0.2
        Tolerance as fraction of window standard deviation.

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Rolling SampEn; first ``window - 1`` entries are ``NaN``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out

    for i in range(window - 1, n):
        start = i - window + 1
        val = sample_entropy(x[start: i + 1], m=m, r=r)
        out[i] = val

    return out


def rolling_permutation_entropy(
    x: NDArray,
    window: int = 78,
    order: int = 3,
    delay: int = 1,
) -> NDArray[np.float64]:
    """Rolling window Permutation Entropy (normalised to [0, 1]).

    Parameters
    ----------
    x : array_like, shape (N,)
        Input time series.
    window : int, default 78
        Rolling window size.
    order : int, default 3
        Ordinal pattern order.
    delay : int, default 1
        Time delay for pattern extraction.

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Rolling normalised permutation entropy; first ``window - 1``
        entries are ``NaN``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out

    for i in range(window - 1, n):
        start = i - window + 1
        out[i] = permutation_entropy(
            x[start: i + 1], order=order, delay=delay, normalize=True,
        )

    return out
