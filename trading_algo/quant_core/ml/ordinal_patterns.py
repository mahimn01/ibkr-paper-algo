"""
Ordinal Pattern Analysis for Financial Time Series.

Implements the Bandt & Pompe (2002) ordinal pattern framework for detecting
hidden deterministic structure in noisy financial time series. Ordinal patterns
convert continuous price series to discrete symbol sequences based on rank
ordering, revealing dynamics that are masked by noise.

Key insight: in a random walk all ordinal patterns are equally likely. In a
market with exploitable structure:
    - Certain patterns are "forbidden" (never occur) -- deterministic dynamics
    - Pattern distribution shifts -- regime changes
    - Pattern transitions -- predictable next moves
    - Low permutation entropy -- predictable, tradeable
    - High permutation entropy -- random, sit out

The ``rolling_permutation_entropy`` and ``pattern_direction_signal`` outputs
are designed for direct use as features in the NonlinearSignalCombiner and as
inputs to the entropy regime filter.

References:
    - Bandt & Pompe (2002): "Permutation Entropy: A Natural Complexity Measure
      for Time Series"
    - Fadlallah et al. (2013): "Weighted-Permutation Entropy: A Complexity
      Measure for Time Series Incorporating Amplitude Information"
    - Zanin et al. (2012): "Permutation Entropy and Its Main Biomedical and
      Econophysics Applications: A Review"
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Optional numba acceleration
# ---------------------------------------------------------------------------

_NUMBA_AVAILABLE = False
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "ordinal_pattern",
    "ordinal_pattern_sequence",
    "ordinal_distribution",
    "rolling_permutation_entropy",
    "forbidden_patterns",
    "pattern_transition_matrix",
    "weighted_permutation_entropy",
    "pattern_direction_signal",
]


# ===========================================================================
# 1. Ordinal Pattern Extraction
# ===========================================================================


def ordinal_pattern(window: NDArray[np.float64]) -> Tuple[int, ...]:
    """
    Convert a window of values to its ordinal pattern (rank ordering).

    The ordinal pattern is the permutation that would sort the window values
    in ascending order.  Ties are broken by order of occurrence (stable sort).

    Example::

        >>> ordinal_pattern(np.array([3.1, 1.5, 2.8]))
        (2, 0, 1)
        # 3.1 is the largest (rank 2), 1.5 the smallest (rank 0), 2.8 middle (rank 1)

    Args:
        window: 1-D array of float values.  Length determines the pattern
            order (embedding dimension).

    Returns:
        Tuple of integers representing the rank of each position.
        Values range from 0 (smallest) to ``len(window) - 1`` (largest).
    """
    window = np.asarray(window, dtype=np.float64)
    n = len(window)
    if n == 0:
        return ()

    # argsort with stable sort to break ties by order of occurrence
    sorted_indices = np.argsort(window, kind="mergesort")

    # Build the rank array: rank[i] = position of window[i] in sorted order
    ranks = np.empty(n, dtype=np.intp)
    ranks[sorted_indices] = np.arange(n)

    return tuple(int(r) for r in ranks)


# ===========================================================================
# 2. Ordinal Pattern Sequence
# ===========================================================================


def ordinal_pattern_sequence(
    x: NDArray[np.float64],
    order: int = 3,
    delay: int = 1,
) -> List[Tuple[int, ...]]:
    """
    Extract all overlapping ordinal patterns from a time series.

    For each valid starting position *t* the embedded window is::

        [x[t], x[t + delay], x[t + 2*delay], ..., x[t + (order-1)*delay]]

    and the ordinal pattern of this window is recorded.

    Args:
        x: 1-D time series array.
        order: Embedding dimension (pattern length).  Must be >= 2.
        delay: Embedding delay.  For delay > 1 every delay-th element in each
            window is taken.  Must be >= 1.

    Returns:
        List of ordinal pattern tuples, one per valid starting position.
        Length is ``len(x) - (order - 1) * delay``.

    Raises:
        ValueError: If *order* < 2, *delay* < 1, or *x* is too short.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    required_length = (order - 1) * delay + 1
    if n < required_length:
        raise ValueError(
            f"Time series length {n} is too short for order={order}, "
            f"delay={delay}.  Need at least {required_length} values."
        )

    n_patterns = n - (order - 1) * delay
    patterns: List[Tuple[int, ...]] = []

    for t in range(n_patterns):
        # Build the embedded window using the delay parameter
        indices = [t + k * delay for k in range(order)]
        window = x[indices]
        patterns.append(ordinal_pattern(window))

    return patterns


# ===========================================================================
# 3. Ordinal Distribution
# ===========================================================================


def ordinal_distribution(
    x: NDArray[np.float64],
    order: int = 3,
    delay: int = 1,
) -> Dict[Tuple[int, ...], float]:
    """
    Compute the probability distribution of ordinal patterns.

    Each unique ordinal pattern is mapped to its observed relative frequency
    (probability).  All probabilities sum to 1.0.

    For order=3 the maximum number of distinct patterns is 3! = 6, for
    order=4 it is 4! = 24, and for order=5 it is 5! = 120.

    Args:
        x: 1-D time series array.
        order: Embedding dimension.  Must be >= 2.
        delay: Embedding delay.  Must be >= 1.

    Returns:
        Dictionary mapping each observed ordinal pattern to its probability.
        Patterns that never occur are omitted (probability 0).

    Raises:
        ValueError: If *x* is too short for the given *order* and *delay*.
    """
    patterns = ordinal_pattern_sequence(x, order=order, delay=delay)
    n_total = len(patterns)

    if n_total == 0:
        return {}

    counts = Counter(patterns)
    return {pat: count / n_total for pat, count in counts.items()}


# ===========================================================================
# 4. Rolling Permutation Entropy
# ===========================================================================


def _permutation_entropy_from_distribution(
    dist: Dict[Tuple[int, ...], float],
    order: int,
    normalize: bool,
) -> float:
    """
    Compute permutation entropy from a pre-computed ordinal distribution.

    PE = -sum(p * ln(p)) for all patterns with p > 0.

    Args:
        dist: Ordinal pattern probability distribution.
        order: Pattern order (for normalization denominator).
        normalize: If True, divide by ln(order!) to get PE in [0, 1].

    Returns:
        Permutation entropy value.
    """
    if not dist:
        return np.nan

    entropy = 0.0
    for p in dist.values():
        if p > 0.0:
            entropy -= p * math.log(p)

    if normalize:
        max_entropy = math.log(math.factorial(order))
        if max_entropy > 0.0:
            entropy /= max_entropy

    return entropy


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _rolling_pe_numba(
        x: NDArray[np.float64],
        order: int,
        delay: int,
        window: int,
        normalize: bool,
    ) -> NDArray[np.float64]:
        """
        Numba-accelerated rolling permutation entropy computation.

        Uses integer encoding of ordinal patterns for efficient counting.
        """
        n = len(x)
        warmup = window + (order - 1) * delay - 1
        result = np.full(n, np.nan, dtype=np.float64)
        n_factorial = 1
        for k in range(2, order + 1):
            n_factorial *= k
        max_entropy = math.log(n_factorial) if n_factorial > 1 else 1.0

        for t in range(warmup, n):
            # Extract sub-series for this rolling window
            start = t - window + 1
            sub = x[start : t + 1]
            sub_len = len(sub)
            n_patterns = sub_len - (order - 1) * delay

            if n_patterns <= 0:
                continue

            # Count patterns using integer encoding
            counts = np.zeros(n_factorial, dtype=np.int64)

            for i in range(n_patterns):
                # Build embedded window
                window_vals = np.empty(order, dtype=np.float64)
                for k in range(order):
                    window_vals[k] = sub[i + k * delay]

                # Compute ordinal pattern as Lehmer code (integer)
                # This maps each permutation to a unique integer in [0, order!)
                code = 0
                for a in range(order):
                    rank = 0
                    for b in range(a + 1, order):
                        if window_vals[b] < window_vals[a]:
                            rank += 1
                        elif window_vals[b] == window_vals[a] and b < a:
                            rank += 1
                    # Multiply by (order - 1 - a)!
                    fac = 1
                    for f in range(1, order - a):
                        fac *= f
                    code += rank * fac

                if 0 <= code < n_factorial:
                    counts[code] += 1

            # Compute entropy from counts
            entropy = 0.0
            for c in range(n_factorial):
                if counts[c] > 0:
                    p = counts[c] / n_patterns
                    entropy -= p * math.log(p)

            if normalize and max_entropy > 0.0:
                entropy /= max_entropy

            result[t] = entropy

        return result


def _rolling_pe_python(
    x: NDArray[np.float64],
    order: int,
    delay: int,
    window: int,
    normalize: bool,
) -> NDArray[np.float64]:
    """
    Pure-Python rolling permutation entropy (fallback when numba unavailable).

    Uses pre-computed pattern-to-index mapping for efficient counting.
    """
    n = len(x)
    warmup = window + (order - 1) * delay - 1
    result = np.full(n, np.nan, dtype=np.float64)

    n_factorial = math.factorial(order)
    max_entropy = math.log(n_factorial) if n_factorial > 1 else 1.0

    # Build mapping from pattern tuple to integer index
    all_perms = list(permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    for t in range(warmup, n):
        start = t - window + 1
        sub = x[start : t + 1]
        sub_len = len(sub)
        n_patterns = sub_len - (order - 1) * delay

        if n_patterns <= 0:
            continue

        # Check for NaN in the sub-array
        if np.any(np.isnan(sub)):
            continue

        counts = np.zeros(n_factorial, dtype=np.int64)

        for i in range(n_patterns):
            indices = [i + k * delay for k in range(order)]
            window_vals = sub[indices]

            # Get ordinal pattern via argsort
            sorted_idx = np.argsort(window_vals, kind="mergesort")
            ranks = np.empty(order, dtype=np.intp)
            ranks[sorted_idx] = np.arange(order)
            pat = tuple(int(r) for r in ranks)

            idx = perm_to_idx.get(pat)
            if idx is not None:
                counts[idx] += 1

        # Compute entropy
        entropy = 0.0
        for c in range(n_factorial):
            if counts[c] > 0:
                p = counts[c] / n_patterns
                entropy -= p * math.log(p)

        if normalize and max_entropy > 0.0:
            entropy /= max_entropy

        result[t] = entropy

    return result


def rolling_permutation_entropy(
    x: NDArray[np.float64],
    order: int = 3,
    delay: int = 1,
    window: int = 78,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """
    Compute permutation entropy over rolling windows.

    Permutation entropy (PE) quantifies the complexity of a time series by
    measuring the Shannon entropy of its ordinal pattern distribution:

        PE = -sum_{pi} p(pi) * ln(p(pi))

    When ``normalize=True`` the output is scaled to [0, 1]:

        PE_norm = PE / ln(order!)

    Interpretation:
        - PE_norm ~ 0: highly predictable (deterministic dynamics)
        - PE_norm ~ 1: completely random (no exploitable structure)

    The rolling computation uses either a numba-accelerated kernel (when
    available) or a pure-Python fallback for arrays of 40,000+ bars.

    Args:
        x: 1-D time series array.
        order: Embedding dimension.  Must be >= 2.  Default 3.
        delay: Embedding delay.  Must be >= 1.  Default 1.
        window: Rolling window size in bars.  Default 78 (~1 quarter of
            daily data).
        normalize: If True, normalize PE to [0, 1].  Default True.

    Returns:
        Array of length ``len(x)`` with PE values.  The first
        ``window + (order - 1) * delay - 1`` positions are NaN (warm-up).

    Raises:
        ValueError: If *order* < 2 or *delay* < 1.
    """
    x = np.asarray(x, dtype=np.float64)

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if window < (order - 1) * delay + 1:
        raise ValueError(
            f"window ({window}) must be >= (order-1)*delay + 1 = "
            f"{(order - 1) * delay + 1}"
        )

    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)

    warmup = window + (order - 1) * delay - 1
    if warmup >= n:
        return np.full(n, np.nan, dtype=np.float64)

    if _NUMBA_AVAILABLE:
        return _rolling_pe_numba(x, order, delay, window, normalize)
    else:
        return _rolling_pe_python(x, order, delay, window, normalize)


# ===========================================================================
# 5. Forbidden Patterns Detection
# ===========================================================================


def forbidden_patterns(
    x: NDArray[np.float64],
    order: int = 4,
    delay: int = 1,
    min_length: int = 1000,
) -> Set[Tuple[int, ...]]:
    """
    Find ordinal patterns that never occur (or occur with negligible frequency).

    In a truly random process all order! patterns appear with equal probability.
    Patterns that are absent ("forbidden") indicate deterministic structure in
    the generating process -- structure that can potentially be exploited.

    A pattern is considered forbidden if its observed count is less than
    ``1 / N`` where ``N`` is the total number of observed patterns.

    Args:
        x: 1-D time series array.
        order: Embedding dimension.  Must be >= 2.  Default 4 (24 possible
            patterns).
        delay: Embedding delay.  Default 1.
        min_length: Minimum time series length for meaningful results.
            Short series may have missing patterns simply due to insufficient
            sampling, not deterministic dynamics.

    Returns:
        Set of ordinal pattern tuples that are absent or negligibly rare.
        Empty set if the series is too short or all patterns are observed.

    Raises:
        ValueError: If *order* < 2 or *delay* < 1.
    """
    x = np.asarray(x, dtype=np.float64)

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    # Filter out NaN values for a clean series
    x_clean = x[~np.isnan(x)]

    if len(x_clean) < min_length:
        return set()

    required_length = (order - 1) * delay + 1
    if len(x_clean) < required_length:
        return set()

    # Get observed distribution
    dist = ordinal_distribution(x_clean, order=order, delay=delay)
    n_patterns = len(x_clean) - (order - 1) * delay
    threshold = 1.0 / n_patterns if n_patterns > 0 else 0.0

    # Generate all possible patterns for this order
    all_possible = set(permutations(range(order)))

    # Find forbidden patterns: those absent or below threshold
    forbidden = set()
    for pat in all_possible:
        freq = dist.get(pat, 0.0)
        if freq < threshold:
            forbidden.add(pat)

    return forbidden


# ===========================================================================
# 6. Pattern Transition Matrix
# ===========================================================================


def pattern_transition_matrix(
    x: NDArray[np.float64],
    order: int = 3,
    delay: int = 1,
) -> Tuple[NDArray[np.float64], List[Tuple[int, ...]]]:
    """
    Compute transition probabilities between consecutive ordinal patterns.

    The transition matrix ``T[i, j]`` gives the probability of observing
    pattern ``j`` at time ``t + 1`` given that pattern ``i`` was observed at
    time ``t``.  This directly reveals which patterns predict which -- a
    non-uniform row indicates exploitable predictability.

    Rows sum to 1.0 (or 0.0 for patterns that were never observed as a
    predecessor).

    Args:
        x: 1-D time series array.
        order: Embedding dimension.  Must be >= 2.
        delay: Embedding delay.  Must be >= 1.

    Returns:
        Tuple of:
            - ``transition_matrix``: Square matrix of shape
              ``(n_patterns, n_patterns)`` where ``n_patterns = order!``.
            - ``pattern_list``: Ordered list of all possible pattern tuples,
              defining the row/column indices of the matrix.

    Raises:
        ValueError: If *x* is too short for the given parameters.
    """
    x = np.asarray(x, dtype=np.float64)

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    # Generate canonical ordering of all possible patterns
    pattern_list = sorted(permutations(range(order)))
    n_pats = len(pattern_list)  # = order!
    pat_to_idx = {p: i for i, p in enumerate(pattern_list)}

    # Extract the full pattern sequence
    patterns = ordinal_pattern_sequence(x, order=order, delay=delay)

    # Count transitions between consecutive patterns
    transition_counts = np.zeros((n_pats, n_pats), dtype=np.float64)

    for t in range(len(patterns) - 1):
        curr = patterns[t]
        nxt = patterns[t + 1]
        i = pat_to_idx.get(curr)
        j = pat_to_idx.get(nxt)
        if i is not None and j is not None:
            transition_counts[i, j] += 1.0

    # Normalize rows to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for patterns that never appeared as a predecessor
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    transition_matrix = transition_counts / row_sums

    return transition_matrix, [tuple(p) for p in pattern_list]


# ===========================================================================
# 7. Weighted Permutation Entropy
# ===========================================================================


def weighted_permutation_entropy(
    x: NDArray[np.float64],
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Compute Weighted Permutation Entropy (WPE).

    Fadlallah et al. (2013) extended standard PE to account for amplitude
    information.  Standard PE treats all patterns equally regardless of the
    magnitude of fluctuations within the window.  WPE weights each pattern
    by the variance of its embedding window, so large-amplitude patterns
    contribute more to the entropy than small-amplitude (noisy) ones.

    This is particularly valuable for financial data where price magnitude
    matters: a (0, 1, 2) pattern from a 5% swing carries more information
    than the same pattern from a 0.01% fluctuation.

    Weight for pattern at position t:
        w_t = Var([x[t], x[t+delay], ..., x[t+(order-1)*delay]])

    Weighted probability for pattern pi:
        p_w(pi) = sum(w_t for t where pattern(t) == pi) / sum(all w_t)

    WPE = -sum(p_w(pi) * ln(p_w(pi)))

    Args:
        x: 1-D time series array.
        order: Embedding dimension.  Must be >= 2.
        delay: Embedding delay.  Must be >= 1.
        normalize: If True, normalize by ln(order!) to [0, 1].

    Returns:
        Weighted permutation entropy as a float.  Returns NaN if the
        series is too short or all constant.

    Raises:
        ValueError: If *order* < 2 or *delay* < 1.
    """
    x = np.asarray(x, dtype=np.float64)

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    required_length = (order - 1) * delay + 1
    if len(x) < required_length:
        return np.nan

    # Filter NaN
    if np.any(np.isnan(x)):
        x = x[~np.isnan(x)]
        if len(x) < required_length:
            return np.nan

    n_patterns = len(x) - (order - 1) * delay

    # Compute weighted counts
    weighted_counts: Dict[Tuple[int, ...], float] = {}
    total_weight = 0.0

    for t in range(n_patterns):
        indices = [t + k * delay for k in range(order)]
        window_vals = x[indices]

        pat = ordinal_pattern(window_vals)

        # Weight = variance of the embedding window values
        weight = float(np.var(window_vals))

        # Accumulate weighted count
        weighted_counts[pat] = weighted_counts.get(pat, 0.0) + weight
        total_weight += weight

    if total_weight <= 0.0:
        # All windows are constant -- zero entropy
        return 0.0

    # Compute weighted entropy
    entropy = 0.0
    for w_count in weighted_counts.values():
        p_w = w_count / total_weight
        if p_w > 0.0:
            entropy -= p_w * math.log(p_w)

    if normalize:
        max_entropy = math.log(math.factorial(order))
        if max_entropy > 0.0:
            entropy /= max_entropy

    return entropy


# ===========================================================================
# 8. Directional Signal from Patterns
# ===========================================================================


def pattern_direction_signal(
    x: NDArray[np.float64],
    order: int = 4,
    delay: int = 1,
    window: int = 252,
) -> NDArray[np.float64]:
    """
    Generate a directional trading signal from ordinal pattern distribution shifts.

    The signal measures the imbalance between "ascending" and "descending"
    ordinal patterns over a rolling window:

        signal = frac(ascending) - frac(descending)

    Where:
        - **Ascending pattern**: ends with the highest rank (last element
          is the largest in the window).  These capture upward price
          momentum in the ordinal structure.
        - **Descending pattern**: ends with the lowest rank (last element
          is the smallest in the window).  These capture downward price
          momentum.

    Output interpretation:
        - signal > 0: more ascending patterns = bullish ordinal bias
        - signal < 0: more descending patterns = bearish ordinal bias
        - signal ~ 0: balanced / random

    The signal range is [-1, 1].

    Args:
        x: 1-D time series array (e.g., close prices or returns).
        order: Embedding dimension.  Default 4.
        delay: Embedding delay.  Default 1.
        window: Rolling window for computing pattern fractions.  Default 252
            (~1 year of daily data).

    Returns:
        Array of length ``len(x)`` with signal values in [-1, 1].
        The first ``window + (order - 1) * delay - 1`` positions are NaN
        (warm-up period).

    Raises:
        ValueError: If *order* < 2 or *delay* < 1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    if n == 0:
        return np.array([], dtype=np.float64)

    # Extract the full ordinal pattern sequence
    required_length = (order - 1) * delay + 1
    if n < required_length:
        return np.full(n, np.nan, dtype=np.float64)

    patterns = ordinal_pattern_sequence(x, order=order, delay=delay)
    n_patterns = len(patterns)

    # Classify each pattern as ascending (+1), descending (-1), or neutral (0)
    highest_rank = order - 1
    lowest_rank = 0

    classification = np.zeros(n_patterns, dtype=np.float64)
    for i, pat in enumerate(patterns):
        if pat[-1] == highest_rank:
            # Last element is the largest -- ascending
            classification[i] = 1.0
        elif pat[-1] == lowest_rank:
            # Last element is the smallest -- descending
            classification[i] = -1.0
        # else: neutral (0.0)

    # Compute rolling signal: fraction ascending - fraction descending
    # This is equivalent to rolling mean of the classification array
    result = np.full(n, np.nan, dtype=np.float64)

    # The ordinal pattern sequence starts at position 0 and corresponds to
    # original time series position 0.  Pattern at index i corresponds to
    # x[i] through x[i + (order-1)*delay].
    # We attribute each pattern's classification to its ending position:
    #   ending_pos = i + (order - 1) * delay
    offset = (order - 1) * delay

    if n_patterns < 1:
        return result

    # Compute rolling mean of classification over the window
    # We need at least `window` patterns for a valid signal
    if n_patterns < window:
        return result

    # Use cumulative sum for efficient rolling mean
    cum_class = np.cumsum(classification)

    for i in range(window - 1, n_patterns):
        if i < window:
            rolling_sum = cum_class[i]
        else:
            rolling_sum = cum_class[i] - cum_class[i - window]
        signal_val = rolling_sum / window

        # Map pattern index to original time series position
        ts_pos = i + offset
        if ts_pos < n:
            result[ts_pos] = signal_val

    return result
