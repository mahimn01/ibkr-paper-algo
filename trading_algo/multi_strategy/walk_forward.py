"""
Walk-Forward Validation for Multi-Strategy System

Implements N-fold walk-forward analysis to detect overfitting.
Uses the existing PBO (Probability of Backtest Overfitting) framework
from quant_core for statistical validation.

Walk-forward procedure:
    1. Split data into N sequential folds.
    2. For each fold, train on all preceding folds and test on the
       current fold.
    3. Compare in-sample vs out-of-sample performance.
    4. Compute PBO and Deflated Sharpe to assess overfitting risk.

A well-designed strategy should show <30% degradation from
in-sample to out-of-sample performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float
    degradation: float  # (IS - OOS) / IS


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation results."""
    folds: List[WalkForwardFold] = field(default_factory=list)

    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_degradation: float = 0.0
    """Average (IS - OOS) / IS.  <0.30 is acceptable."""

    pbo: Optional[float] = None
    """Probability of Backtest Overfitting (0-1).  <0.50 is acceptable."""

    deflated_sharpe: Optional[float] = None
    """Deflated Sharpe accounting for multiple testing."""

    is_overfit: bool = False
    """True if PBO > 0.50 or degradation > 50%."""


class WalkForwardValidator:
    """
    N-fold walk-forward validation.

    Usage::

        validator = WalkForwardValidator(n_folds=5)
        result = validator.validate(daily_returns_matrix)
    """

    def __init__(self, n_folds: int = 5, min_fold_size: int = 60):
        self.n_folds = n_folds
        self.min_fold_size = min_fold_size

    def validate(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on a returns series.

        Args:
            returns: 1D array of portfolio daily returns.
            timestamps: Optional list of timestamps matching returns.

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        n = len(returns)
        if n < self.min_fold_size * 2:
            logger.warning("Not enough data for walk-forward: %d bars", n)
            return WalkForwardResult()

        fold_size = n // self.n_folds
        if fold_size < self.min_fold_size:
            # Reduce folds to meet minimum size
            self.n_folds = max(2, n // self.min_fold_size)
            fold_size = n // self.n_folds

        folds: List[WalkForwardFold] = []

        for i in range(1, self.n_folds):
            # Train on folds 0..i-1, test on fold i
            train_end = i * fold_size
            test_start = train_end
            test_end = min((i + 1) * fold_size, n)

            train_returns = returns[:train_end]
            test_returns = returns[test_start:test_end]

            if len(train_returns) < self.min_fold_size or len(test_returns) < 10:
                continue

            is_sharpe = self._compute_sharpe(train_returns)
            oos_sharpe = self._compute_sharpe(test_returns)
            is_return = float(np.sum(train_returns))
            oos_return = float(np.sum(test_returns))

            degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe) if abs(is_sharpe) > 0.01 else 0

            train_ts_start = timestamps[0] if timestamps else datetime(2020, 1, 1)
            train_ts_end = timestamps[train_end - 1] if timestamps and train_end <= len(timestamps) else datetime(2020, 6, 1)
            test_ts_start = timestamps[test_start] if timestamps and test_start < len(timestamps) else datetime(2020, 6, 1)
            test_ts_end = timestamps[test_end - 1] if timestamps and test_end <= len(timestamps) else datetime(2020, 12, 1)

            folds.append(WalkForwardFold(
                fold_index=i,
                train_start=train_ts_start,
                train_end=train_ts_end,
                test_start=test_ts_start,
                test_end=test_ts_end,
                in_sample_sharpe=is_sharpe,
                out_of_sample_sharpe=oos_sharpe,
                in_sample_return=is_return,
                out_of_sample_return=oos_return,
                degradation=degradation,
            ))

        if not folds:
            return WalkForwardResult()

        avg_is = float(np.mean([f.in_sample_sharpe for f in folds]))
        avg_oos = float(np.mean([f.out_of_sample_sharpe for f in folds]))
        avg_deg = float(np.mean([f.degradation for f in folds]))

        # Compute PBO if available
        pbo_value = None
        deflated = None
        try:
            from trading_algo.quant_core.validation.pbo import (
                PBOCalculator,
                DeflatedSharpe,
            )
            pbo_calc = PBOCalculator(n_groups=min(16, self.n_folds * 2))
            # Reshape returns for PBO (needs matrix of strategy variants)
            if len(returns) >= 100:
                pbo_result = pbo_calc.calculate(returns)
                pbo_value = pbo_result.pbo

            ds = DeflatedSharpe()
            sharpe_obs = self._compute_sharpe(returns)
            ds_result = ds.calculate(
                observed_sharpe=sharpe_obs,
                n_trials=self.n_folds,
                n_observations=len(returns),
            )
            deflated = ds_result.deflated_sharpe

        except Exception as e:
            logger.debug("PBO/DeflatedSharpe unavailable: %s", e)

        is_overfit = avg_deg > 0.50 or (pbo_value is not None and pbo_value > 0.50)

        return WalkForwardResult(
            folds=folds,
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            avg_degradation=avg_deg,
            pbo=pbo_value,
            deflated_sharpe=deflated,
            is_overfit=is_overfit,
        )

    @staticmethod
    def _compute_sharpe(returns: np.ndarray) -> float:
        """Annualized Sharpe ratio from daily returns."""
        if len(returns) < 2:
            return 0.0
        ann_ret = float(np.mean(returns) * 252)
        ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
        if ann_vol < 1e-8:
            return 0.0
        return (ann_ret - 0.02) / ann_vol
