"""
Time-Series Cross-Validation

Implements proper cross-validation for financial time series:
    - Purged K-Fold: Removes samples near test set to prevent leakage
    - Walk-Forward: Rolling train/test splits
    - Combinatorial Purged: For strategy validation (López de Prado)

Key Considerations:
    - No future information leakage
    - Serial correlation in returns
    - Non-stationarity of financial data
    - Overlapping labels (holding periods > 1)

References:
    - López de Prado (2018): "Advances in Financial Machine Learning", Chapter 7
    - Bailey et al. (2014): "The Probability of Backtest Overfitting"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterator, Generator
from itertools import combinations


@dataclass
class CVSplit:
    """Single cross-validation split."""
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class TimeSeriesCV:
    """
    Time-series cross-validation with purging and embargo.

    Standard K-fold CV is invalid for time series because:
    1. Future information leaks into training set
    2. Serial correlation violates independence assumption
    3. Non-stationarity makes early/late splits different

    This implementation provides:
    - Walk-forward validation
    - Purged K-Fold
    - Expanding window
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ):
        """
        Initialize time-series CV.

        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, computed from n_splits)
            gap: Number of samples to skip between train and test (embargo)
            max_train_size: Maximum training set size (for rolling window)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.float64]] = None,
    ) -> Generator[CVSplit, None, None]:
        """
        Generate train/test splits.

        Yields CVSplit objects with indices.
        """
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Minimum training size
        min_train_size = max(test_size, n_samples // (self.n_splits + 1))

        test_starts = []
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            if test_start >= min_train_size + self.gap:
                test_starts.append(test_start)

        for test_start in test_starts:
            test_end = min(test_start + test_size, n_samples)

            # Training set ends before gap
            train_end = test_start - self.gap

            # Apply max_train_size if set
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Implements the purged K-fold CV from López de Prado (2018).

    When labels are computed using overlapping data (e.g., returns
    over multiple days), we must purge training samples that overlap
    with test samples to prevent leakage.

    Also implements embargo: additional gap after test set to prevent
    leakage from features that use forward-looking data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_length: int = 1,
        embargo_length: int = 0,
    ):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_length: Number of samples to purge around test set
            embargo_length: Additional samples to skip after test set
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.float64]] = None,
        sample_times: Optional[NDArray] = None,
    ) -> Generator[CVSplit, None, None]:
        """
        Generate purged K-fold splits.

        Args:
            X: Features
            y: Labels
            sample_times: Sample timestamps (for time-based purging)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Create fold boundaries
        fold_size = n_samples // self.n_splits
        folds = []

        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            folds.append((start, end))

        # Generate splits
        for test_fold_idx in range(self.n_splits):
            test_start, test_end = folds[test_fold_idx]
            test_indices = indices[test_start:test_end]

            # Purge samples near test set
            purge_start = max(0, test_start - self.purge_length)
            purge_end = min(n_samples, test_end + self.purge_length + self.embargo_length)

            # Training indices = all except purged region
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:purge_end] = False
            train_indices = indices[train_mask]

            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=0,
                train_end=n_samples,
                test_start=test_start,
                test_end=test_end,
            )


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    From López de Prado (2018), Chapter 12.

    Used for:
    1. Testing trading strategies with realistic conditions
    2. Calculating Probability of Backtest Overfitting (PBO)
    3. Deflated Sharpe Ratio

    Creates all possible train/test combinations to estimate
    the probability that a strategy is overfit.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_length: int = 1,
        embargo_length: int = 0,
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of groups
            n_test_splits: Number of groups in test set
            purge_length: Samples to purge between train/test
            embargo_length: Additional embargo samples
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.float64]] = None,
    ) -> Generator[CVSplit, None, None]:
        """
        Generate all combinatorial splits.

        Number of splits = C(n_splits, n_test_splits)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Create groups
        group_size = n_samples // self.n_splits
        groups = []

        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups.append((start, end))

        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_splits):
            # Identify test regions
            test_regions = [groups[i] for i in test_group_indices]

            # Build test indices
            test_indices_list = []
            for start, end in test_regions:
                test_indices_list.extend(range(start, end))
            test_indices = np.array(test_indices_list)

            # Build train indices with purging
            train_mask = np.ones(n_samples, dtype=bool)

            for start, end in test_regions:
                # Purge around this test region
                purge_start = max(0, start - self.purge_length)
                purge_end = min(n_samples, end + self.purge_length + self.embargo_length)
                train_mask[purge_start:purge_end] = False

            train_indices = indices[train_mask]

            if len(train_indices) == 0:
                continue

            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=0,
                train_end=n_samples,
                test_start=min(test_indices),
                test_end=max(test_indices) + 1,
            )

    def get_n_splits(self) -> int:
        """Get number of splits."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


def calculate_pbo(
    performance_metrics: NDArray[np.float64],
    n_strategies: int,
) -> Tuple[float, float]:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    From Bailey et al. (2014).

    Args:
        performance_metrics: Matrix of (n_cv_splits, n_strategies)
            performance for each strategy in each CV split
        n_strategies: Number of strategies tested

    Returns:
        Tuple of (PBO, deflated_sharpe_ratio_haircut)

    PBO interpretation:
        - PBO < 0.05: Low probability of overfitting
        - 0.05 < PBO < 0.25: Moderate
        - PBO > 0.25: High probability of overfitting
    """
    if performance_metrics.shape[1] != n_strategies:
        raise ValueError("Mismatch between metrics shape and n_strategies")

    n_splits = performance_metrics.shape[0]

    # For each split, find if the best in-sample strategy is also best out-of-sample
    # This is a simplified approximation

    # Rank strategies in each split
    ranks = np.zeros_like(performance_metrics)
    for i in range(n_splits):
        ranks[i] = np.argsort(np.argsort(-performance_metrics[i]))

    # Calculate rank correlation between splits
    from scipy.stats import spearmanr

    # Split into "in-sample" (first half) and "out-of-sample" (second half)
    mid = n_splits // 2
    is_ranks = ranks[:mid].mean(axis=0)
    oos_ranks = ranks[mid:].mean(axis=0)

    correlation, _ = spearmanr(is_ranks, oos_ranks)

    # PBO approximation: probability that best IS strategy has poor OOS rank
    # Based on rank correlation
    pbo = max(0.0, (1 - correlation) / 2)

    # Deflated Sharpe ratio haircut
    # Accounts for multiple testing
    haircut = np.sqrt(1 + np.log(n_strategies))

    return float(pbo), float(haircut)


def walk_forward_cv(
    X: NDArray[np.float64],
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> Generator[CVSplit, None, None]:
    """
    Simple walk-forward cross-validation.

    Rolls a fixed-size window through the data.

    Args:
        X: Feature matrix
        train_size: Size of training window
        test_size: Size of test window
        step_size: Step size between windows (default: test_size)
    """
    n_samples = len(X)
    if step_size is None:
        step_size = test_size

    indices = np.arange(n_samples)

    start = 0
    while start + train_size + test_size <= n_samples:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size

        yield CVSplit(
            train_indices=indices[train_start:train_end],
            test_indices=indices[test_start:test_end],
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        start += step_size


def expanding_window_cv(
    X: NDArray[np.float64],
    min_train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> Generator[CVSplit, None, None]:
    """
    Expanding window cross-validation.

    Training set grows while test set slides forward.

    Args:
        X: Feature matrix
        min_train_size: Minimum training set size
        test_size: Size of test window
        step_size: Step size between windows
    """
    n_samples = len(X)
    if step_size is None:
        step_size = test_size

    indices = np.arange(n_samples)

    train_end = min_train_size
    while train_end + test_size <= n_samples:
        train_start = 0
        test_start = train_end
        test_end = test_start + test_size

        yield CVSplit(
            train_indices=indices[train_start:train_end],
            test_indices=indices[test_start:test_end],
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        train_end += step_size
