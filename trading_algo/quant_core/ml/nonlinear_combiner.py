"""
Nonlinear Signal Combination via Gradient-Boosted Ensembles.

Replaces linear signal weighting with a gradient-boosted tree ensemble that
automatically discovers nonlinear interactions between trading signals. Most
alpha resides in INTERACTIONS: signal A alone = noise, signal B alone = noise,
A x B in a specific regime = 4-sigma signal. Tree-based models discover these
interaction effects through recursive partitioning without explicit feature
engineering.

Mathematical foundation:
    - Friedman (2001): "Greedy Function Approximation: A Gradient Boosting Machine"
    - Lopez de Prado (2018): "Advances in Financial Machine Learning" (purged CV)
    - Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"

Key advantages over linear combination:
    1. Automatic interaction detection (tree splits discover A x B effects)
    2. Regime conditioning (trees naturally partition by regime state)
    3. Nonlinear signal transformation (threshold effects, saturation)
    4. Feature importance ranking (which signals/interactions matter most)

Walk-forward training protocol prevents lookahead bias:
    - Train on expanding window up to T - purge_window
    - Predict at time T
    - Retrain every retrain_frequency bars
    - Embargo window after test set for overlapping returns
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.utils.constants import (
    EPSILON,
    SQRT_252,
    ML_N_ESTIMATORS,
    ML_MAX_DEPTH,
    ML_LEARNING_RATE,
    ML_VALIDATION_SPLIT,
    ML_PURGE_WINDOW,
)
from trading_algo.quant_core.utils.math_utils import rolling_mean, rolling_std

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability flags for optional ML libraries
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.metrics import mean_squared_error, explained_variance_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

_XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    pass

_LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

_JOBLIB_AVAILABLE = False
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    pass


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class NonlinearCombinerConfig:
    """
    Configuration for the nonlinear signal combiner.

    Attributes:
        model_type: Underlying model backend. One of ``"gradient_boosting"``,
            ``"random_forest"``, ``"xgboost"``, ``"lightgbm"``. Falls through
            the priority list if the requested backend is unavailable.
        n_estimators: Number of boosting rounds / trees.
        max_depth: Maximum tree depth. Shallow trees (3-5) regularize well for
            financial data where signal-to-noise is low.
        learning_rate: Shrinkage factor per boosting step.
        min_samples_leaf: Minimum samples in a leaf node. Higher values prevent
            overfitting to idiosyncratic noise.
        subsample: Fraction of samples used per boosting round (stochastic GB).
        train_window: Number of bars in the initial training window
            (504 ~ 2 years of daily data).
        retrain_frequency: Retrain every N bars (21 ~ monthly).
        purge_window: Gap between training and test sets to avoid information
            leakage from overlapping return labels.
        embargo_window: Additional gap after the test set to prevent leakage
            from autocorrelated features.
        include_regime_features: Append one-hot encoded regime labels.
        include_time_features: Append calendar features (day-of-week, month,
            quarter).
        include_interaction_features: Append pairwise products of top signals.
        top_n_interactions: Number of top signals (by variance) to use when
            constructing pairwise interaction features.
        max_features: Fraction of features considered per tree split. Values
            below 1.0 add column subsampling regularization.
        early_stopping_rounds: Stop boosting if validation loss does not
            improve for this many rounds. Set to 0 to disable.
        output_type: How to interpret model output.
            ``"direction"`` clips to [-1, 1].
            ``"probability"`` clips to [0, 1].
            ``"return_forecast"`` returns raw prediction.
        clip_predictions: Absolute bound for clipping output in ``"direction"``
            and ``"return_forecast"`` modes.
        fallback_to_linear: If True, fall back to Ridge regression when no
            tree-based library is available.
    """

    # Model selection
    model_type: str = "gradient_boosting"
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    subsample: float = 0.8

    # Walk-forward training
    train_window: int = 504
    retrain_frequency: int = 21
    purge_window: int = 5
    embargo_window: int = 5

    # Feature engineering
    include_regime_features: bool = True
    include_time_features: bool = True
    include_interaction_features: bool = True
    top_n_interactions: int = 10

    # Regularization
    max_features: float = 0.8
    early_stopping_rounds: int = 20

    # Output
    output_type: str = "direction"
    clip_predictions: float = 1.0

    # Fallback
    fallback_to_linear: bool = True


# ===========================================================================
# Ridge fallback (zero external dependencies)
# ===========================================================================

class _RidgeFallback:
    """
    Minimal Ridge regression for environments without scikit-learn.

    Solves the closed-form Ridge solution:

        beta = (X^T X + alpha * I)^{-1} X^T y

    Numerical stability is ensured by adding the ridge penalty and using
    ``numpy.linalg.solve`` rather than explicit matrix inversion.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: Optional[NDArray[np.float64]] = None
        self.intercept_: float = 0.0
        self.feature_importances_: Optional[NDArray[np.float64]] = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> "_RidgeFallback":
        """
        Fit Ridge regression via closed-form solution.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
            sample_weight: Per-sample weights. If provided the weighted
                normal equations are solved.

        Returns:
            Self for method chaining.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        # Centre targets to compute intercept
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            sw_sum = sample_weight.sum()
            if sw_sum < EPSILON:
                sw_sum = 1.0
            y_mean = np.dot(sample_weight, y) / sw_sum
            X_mean = (sample_weight[:, None] * X).sum(axis=0) / sw_sum
        else:
            y_mean = y.mean()
            X_mean = X.mean(axis=0)

        X_c = X - X_mean
        y_c = y - y_mean

        if sample_weight is not None:
            W = np.diag(sample_weight)
            XtWX = X_c.T @ W @ X_c
            XtWy = X_c.T @ W @ y_c
        else:
            XtWX = X_c.T @ X_c
            XtWy = X_c.T @ y_c

        A = XtWX + self.alpha * np.eye(n_features)

        try:
            self.coef_ = np.linalg.solve(A, XtWy)
        except np.linalg.LinAlgError:
            logger.warning(
                "Ridge fallback: singular matrix encountered, "
                "returning zero coefficients."
            )
            self.coef_ = np.zeros(n_features, dtype=np.float64)

        self.intercept_ = float(y_mean - X_mean @ self.coef_)

        # Pseudo-importance from absolute coefficient magnitude
        abs_coef = np.abs(self.coef_)
        total = abs_coef.sum()
        if total > EPSILON:
            self.feature_importances_ = abs_coef / total
        else:
            self.feature_importances_ = np.full(
                n_features, 1.0 / max(n_features, 1)
            )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict using the fitted Ridge model.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.coef_ is None:
            raise RuntimeError("_RidgeFallback.predict called before fit.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


# ===========================================================================
# Core combiner
# ===========================================================================

class NonlinearSignalCombiner:
    """
    Nonlinear signal combination using gradient-boosted trees.

    Key advantages over linear combination:

    1. **Automatic interaction detection** -- tree splits discover A x B
       effects without requiring explicit interaction features.
    2. **Regime conditioning** -- trees naturally partition feature space by
       regime state, learning different signal weights per regime.
    3. **Nonlinear signal transformation** -- captures threshold effects,
       saturation, and other non-monotonic relationships.
    4. **Feature importance ranking** -- quantifies which signals and
       interactions contribute most to prediction.

    Walk-forward training prevents lookahead bias:

    * Train on an expanding window up to ``T - purge_window``.
    * Predict at time ``T``.
    * Retrain every ``retrain_frequency`` bars.

    Usage::

        combiner = NonlinearSignalCombiner(config)

        # Walk-forward fit and predict (production method)
        predictions = combiner.walk_forward_predict(
            signals=signal_matrix,        # (T, N_signals)
            returns=forward_returns,       # (T,)
            regime_labels=regimes,         # (T,) optional
        )

        # Or fit once and predict (research / debugging)
        combiner.fit(X_train, y_train)
        pred = combiner.predict(X_test)
    """

    def __init__(self, config: Optional[NonlinearCombinerConfig] = None) -> None:
        """
        Initialize the nonlinear signal combiner.

        Args:
            config: Combiner configuration. Uses sensible defaults when
                ``None``.
        """
        self.config = config or NonlinearCombinerConfig()
        self._model: Any = None
        self._fitted: bool = False
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._model_backend: str = "none"
        self._train_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API: fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Fit the gradient-boosted model on labelled data.

        NaN rows are dropped prior to fitting. If fewer than
        ``config.min_samples_leaf * 5`` valid rows remain, the model
        defaults to a Ridge fallback (or zeros if no fallback is available).

        Args:
            X: Feature / signal matrix of shape ``(T, K)``.
            y: Forward return targets of shape ``(T,)``.
            sample_weight: Optional per-sample weights of shape ``(T,)``.
            feature_names: Human-readable feature names. Auto-generated as
                ``signal_0 .. signal_{K-1}`` when ``None``.

        Raises:
            ValueError: If ``X`` and ``y`` have incompatible first dimensions.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} elements."
            )

        # Handle feature names
        if feature_names is not None:
            self._feature_names = list(feature_names)
        elif not self._feature_names or len(self._feature_names) != X.shape[1]:
            self._feature_names = [
                f"signal_{i}" for i in range(X.shape[1])
            ]

        # Remove NaN rows
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        sw_clean: Optional[NDArray[np.float64]] = None
        if sample_weight is not None:
            sw_clean = np.asarray(sample_weight, dtype=np.float64)[valid_mask]

        min_samples = max(self.config.min_samples_leaf * 5, 50)
        if X_clean.shape[0] < min_samples:
            logger.warning(
                "NonlinearSignalCombiner.fit: only %d valid samples "
                "(need %d). Falling back to Ridge.",
                X_clean.shape[0],
                min_samples,
            )
            self._model = self._create_fallback_model()
            if self._model is not None and X_clean.shape[0] > 0:
                self._model.fit(X_clean, y_clean, sample_weight=sw_clean)
                self._fitted = True
                self._extract_feature_importance()
            return

        # Create and fit the model
        self._model = self._create_model()

        # Attempt early stopping when a validation set is available
        val_fraction = ML_VALIDATION_SPLIT
        n_val = max(1, int(X_clean.shape[0] * val_fraction))
        use_early_stopping = (
            self.config.early_stopping_rounds > 0
            and n_val >= self.config.min_samples_leaf
            and X_clean.shape[0] - n_val >= min_samples
        )

        if use_early_stopping and self._model_backend in (
            "xgboost", "lightgbm"
        ):
            X_trn = X_clean[:-n_val]
            y_trn = y_clean[:-n_val]
            X_val = X_clean[-n_val:]
            y_val = y_clean[-n_val:]
            sw_trn = sw_clean[:-n_val] if sw_clean is not None else None

            self._fit_with_early_stopping(
                X_trn, y_trn, X_val, y_val, sw_trn
            )
        else:
            self._safe_fit(self._model, X_clean, y_clean, sw_clean)

        self._fitted = True
        self._extract_feature_importance()

        logger.info(
            "NonlinearSignalCombiner fitted with backend=%s, "
            "n_samples=%d, n_features=%d.",
            self._model_backend,
            X_clean.shape[0],
            X_clean.shape[1],
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generate combined signal predictions from new data.

        Args:
            X: Feature matrix of shape ``(T, K)`` or ``(K,)`` for a single
                observation.

        Returns:
            Predictions of shape ``(T,)``, clipped according to
            ``config.output_type`` and ``config.clip_predictions``.
        """
        if not self._fitted or self._model is None:
            if X.ndim == 1:
                return np.zeros(1, dtype=np.float64)
            return np.zeros(X.shape[0], dtype=np.float64)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Replace NaN with 0 for prediction stability
        if np.any(np.isnan(X)):
            X = np.where(np.isnan(X), 0.0, X)

        raw = self._model.predict(X)

        return self._clip_output(raw)

    # ------------------------------------------------------------------
    # Walk-forward prediction
    # ------------------------------------------------------------------

    def walk_forward_predict(
        self,
        signals: NDArray[np.float64],
        returns: NDArray[np.float64],
        regime_labels: Optional[NDArray] = None,
        timestamps: Optional[NDArray] = None,
    ) -> NDArray[np.float64]:
        """
        Walk-forward out-of-sample prediction with expanding window.

        This is the primary production method. At each retrain point the model
        is re-fitted on all available history (respecting purge and embargo
        windows) and then used to predict the next ``retrain_frequency`` bars.

        The first ``train_window`` bars are used for the initial fit and
        therefore receive ``NaN`` predictions (warm-up period).

        Args:
            signals: Raw signal matrix of shape ``(T, N_signals)``.
            returns: Forward return targets of shape ``(T,)``.
            regime_labels: Optional regime labels of shape ``(T,)``.
                Integer or string labels; converted to one-hot internally.
            timestamps: Optional timestamp array of shape ``(T,)``. Used to
                derive calendar features. Elements should be
                ``datetime``-like or integer ordinals.

        Returns:
            Prediction array of shape ``(T,)``. The first
            ``train_window`` entries are ``NaN``.
        """
        signals = np.asarray(signals, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64).ravel()
        T = signals.shape[0]

        if T != returns.shape[0]:
            raise ValueError(
                f"signals has {T} rows but returns has {returns.shape[0]}."
            )

        # Prepare augmented feature matrix
        X_full = self._prepare_features(signals, regime_labels, timestamps)
        n_features = X_full.shape[1]

        # Generate feature names for the augmented matrix
        self._generate_augmented_feature_names(
            n_raw_signals=signals.shape[1],
            regime_labels=regime_labels,
            timestamps=timestamps,
            n_total_features=n_features,
        )

        predictions = np.full(T, np.nan, dtype=np.float64)

        tw = self.config.train_window
        rf = self.config.retrain_frequency
        purge = self.config.purge_window
        embargo = self.config.embargo_window

        if tw >= T:
            logger.warning(
                "walk_forward_predict: train_window (%d) >= T (%d). "
                "Returning all NaN.",
                tw,
                T,
            )
            return predictions

        # Walk forward
        last_train_end = -1
        for t in range(tw, T):
            need_retrain = (
                last_train_end < 0 or (t - last_train_end) >= rf
            )

            if need_retrain:
                train_end = t - purge
                if train_end <= 0:
                    continue

                X_train, y_train = self._purged_train_test_split(
                    X_full, returns, train_end=train_end,
                    test_start=t, purge=purge, embargo=embargo,
                )

                if X_train.shape[0] < max(self.config.min_samples_leaf * 5, 50):
                    continue

                try:
                    self.fit(
                        X_train, y_train,
                        feature_names=self._feature_names,
                    )
                    last_train_end = t
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "walk_forward_predict: fit failed at t=%d: %s", t, exc
                    )
                    continue

            if self._fitted:
                predictions[t] = float(self.predict(X_full[t])[0])

        return predictions

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _prepare_features(
        self,
        signals: NDArray[np.float64],
        regime_labels: Optional[NDArray] = None,
        timestamps: Optional[NDArray] = None,
    ) -> NDArray[np.float64]:
        """
        Augment raw signals with engineered features.

        The augmented feature matrix includes (in order):

        1. Raw signals (passed through).
        2. Regime one-hot encoding (if ``regime_labels`` provided and
           ``config.include_regime_features`` is True).
        3. Calendar features: day-of-week, month, quarter (if ``timestamps``
           provided and ``config.include_time_features`` is True).
        4. Pairwise interaction features for the top-N signals ranked by
           variance (if ``config.include_interaction_features`` is True).
        5. Rolling signal statistics: 5-bar and 20-bar rolling mean of each
           raw signal.

        Args:
            signals: Raw signal matrix ``(T, K)``.
            regime_labels: Optional regime labels ``(T,)``.
            timestamps: Optional timestamps ``(T,)``.

        Returns:
            Augmented feature matrix ``(T, K')``.
        """
        signals = np.asarray(signals, dtype=np.float64)
        T, K = signals.shape
        parts: List[NDArray[np.float64]] = [signals]

        # --- Regime one-hot ---
        if (
            self.config.include_regime_features
            and regime_labels is not None
        ):
            regime_arr = np.asarray(regime_labels)
            unique_regimes = np.unique(regime_arr[~_isnan_safe(regime_arr)])
            n_regimes = len(unique_regimes)
            if n_regimes > 0:
                regime_map = {
                    val: idx for idx, val in enumerate(unique_regimes)
                }
                one_hot = np.zeros((T, n_regimes), dtype=np.float64)
                for t in range(T):
                    if not _isnan_safe_scalar(regime_arr[t]):
                        idx = regime_map.get(regime_arr[t])
                        if idx is not None:
                            one_hot[t, idx] = 1.0
                parts.append(one_hot)

        # --- Calendar features ---
        if self.config.include_time_features and timestamps is not None:
            cal = self._extract_calendar_features(timestamps, T)
            if cal is not None:
                parts.append(cal)

        # --- Pairwise interactions ---
        if self.config.include_interaction_features and K >= 2:
            interactions = self._build_interaction_features(signals)
            if interactions.shape[1] > 0:
                parts.append(interactions)

        # --- Rolling signal statistics ---
        rolling_feats = self._build_rolling_features(signals)
        if rolling_feats.shape[1] > 0:
            parts.append(rolling_feats)

        return np.hstack(parts)

    def _build_interaction_features(
        self, signals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Construct pairwise product features for top-N signals by variance.

        Args:
            signals: Raw signal matrix ``(T, K)``.

        Returns:
            Interaction feature matrix ``(T, n_pairs)``.
        """
        T, K = signals.shape
        top_n = min(self.config.top_n_interactions, K)

        # Rank signals by variance (ignore NaN)
        variances = np.nanvar(signals, axis=0)
        top_indices = np.argsort(variances)[::-1][:top_n]

        pairs = list(combinations(top_indices, 2))
        if not pairs:
            return np.empty((T, 0), dtype=np.float64)

        interactions = np.empty((T, len(pairs)), dtype=np.float64)
        for col, (i, j) in enumerate(pairs):
            interactions[:, col] = signals[:, i] * signals[:, j]

        return interactions

    def _build_rolling_features(
        self, signals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute 5-bar and 20-bar rolling means of each signal.

        Args:
            signals: Raw signal matrix ``(T, K)``.

        Returns:
            Rolling feature matrix ``(T, 2*K)``.
        """
        T, K = signals.shape
        result = np.full((T, 2 * K), np.nan, dtype=np.float64)

        for k in range(K):
            col = signals[:, k].copy()
            # Replace NaN with 0 for rolling calculation
            nan_mask = np.isnan(col)
            col[nan_mask] = 0.0

            result[:, k] = rolling_mean(col, 5)
            result[:, K + k] = rolling_mean(col, 20)

        # Replace remaining NaN with 0 for model consumption
        result = np.where(np.isnan(result), 0.0, result)
        return result

    @staticmethod
    def _extract_calendar_features(
        timestamps: NDArray, T: int
    ) -> Optional[NDArray[np.float64]]:
        """
        Extract day-of-week, month, and quarter from timestamps.

        Handles ``datetime``, ``np.datetime64``, and integer ordinals. Returns
        ``None`` if extraction fails.

        Args:
            timestamps: Array of timestamp-like values.
            T: Expected number of rows.

        Returns:
            Calendar feature matrix ``(T, 3)`` or ``None``.
        """
        try:
            cal = np.zeros((T, 3), dtype=np.float64)
            for t in range(T):
                ts = timestamps[t]
                if isinstance(ts, (datetime, np.datetime64)):
                    if isinstance(ts, np.datetime64):
                        # Convert to Python datetime
                        ts = ts.astype("datetime64[ms]").astype(datetime)
                    cal[t, 0] = float(ts.weekday()) / 4.0  # 0-1 scale
                    cal[t, 1] = float(ts.month - 1) / 11.0
                    cal[t, 2] = float((ts.month - 1) // 3) / 3.0
                elif isinstance(ts, (int, float, np.integer, np.floating)):
                    # Treat as ordinal: use modular arithmetic for cyclic features
                    val = int(ts)
                    cal[t, 0] = float(val % 5) / 4.0
                    cal[t, 1] = float(val % 12) / 11.0
                    cal[t, 2] = float((val % 12) // 3) / 3.0
            return cal
        except Exception:  # noqa: BLE001
            logger.debug(
                "Calendar feature extraction failed; skipping."
            )
            return None

    def _generate_augmented_feature_names(
        self,
        n_raw_signals: int,
        regime_labels: Optional[NDArray],
        timestamps: Optional[NDArray],
        n_total_features: int,
    ) -> None:
        """
        Build human-readable names for all features in the augmented matrix.

        Populates ``self._feature_names``.
        """
        names: List[str] = [
            f"signal_{i}" for i in range(n_raw_signals)
        ]

        if (
            self.config.include_regime_features
            and regime_labels is not None
        ):
            unique_regimes = np.unique(
                np.asarray(regime_labels)[
                    ~_isnan_safe(np.asarray(regime_labels))
                ]
            )
            for r in unique_regimes:
                names.append(f"regime_{r}")

        if self.config.include_time_features and timestamps is not None:
            names.extend(["day_of_week", "month", "quarter"])

        if self.config.include_interaction_features and n_raw_signals >= 2:
            top_n = min(self.config.top_n_interactions, n_raw_signals)
            variances_placeholder = np.arange(n_raw_signals, dtype=np.float64)
            top_indices = list(range(top_n))
            for i, j in combinations(top_indices, 2):
                names.append(f"interact_{i}x{j}")

        for k in range(n_raw_signals):
            names.append(f"roll5_signal_{k}")
        for k in range(n_raw_signals):
            names.append(f"roll20_signal_{k}")

        # Ensure length matches (trim or pad)
        if len(names) < n_total_features:
            for i in range(n_total_features - len(names)):
                names.append(f"extra_feat_{i}")
        elif len(names) > n_total_features:
            names = names[:n_total_features]

        self._feature_names = names

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_model(self) -> Any:
        """
        Create the underlying ML model.

        Priority order based on ``config.model_type``:

        1. XGBoost (if requested and available)
        2. LightGBM (if requested and available)
        3. scikit-learn ``GradientBoostingRegressor`` or
           ``RandomForestRegressor``
        4. ``_RidgeFallback`` (if ``config.fallback_to_linear``)

        Returns:
            A fitted-API compatible model object.

        Raises:
            RuntimeError: If no backend is available and
                ``config.fallback_to_linear`` is False.
        """
        cfg = self.config

        # --- XGBoost ---
        if cfg.model_type == "xgboost" and _XGBOOST_AVAILABLE:
            self._model_backend = "xgboost"
            return xgb.XGBRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                min_child_weight=cfg.min_samples_leaf,
                subsample=cfg.subsample,
                colsample_bytree=cfg.max_features,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

        # --- LightGBM ---
        if cfg.model_type == "lightgbm" and _LIGHTGBM_AVAILABLE:
            self._model_backend = "lightgbm"
            return lgb.LGBMRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                min_child_samples=cfg.min_samples_leaf,
                subsample=cfg.subsample,
                colsample_bytree=cfg.max_features,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        # --- XGBoost fallback if gradient_boosting requested ---
        if cfg.model_type == "gradient_boosting" and _XGBOOST_AVAILABLE:
            self._model_backend = "xgboost"
            return xgb.XGBRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                min_child_weight=cfg.min_samples_leaf,
                subsample=cfg.subsample,
                colsample_bytree=cfg.max_features,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

        # --- LightGBM fallback ---
        if cfg.model_type in ("gradient_boosting", "xgboost") and _LIGHTGBM_AVAILABLE:
            self._model_backend = "lightgbm"
            return lgb.LGBMRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                min_child_samples=cfg.min_samples_leaf,
                subsample=cfg.subsample,
                colsample_bytree=cfg.max_features,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        # --- sklearn GBR ---
        if _SKLEARN_AVAILABLE:
            if cfg.model_type in (
                "gradient_boosting", "xgboost", "lightgbm"
            ):
                self._model_backend = "sklearn_gbr"
                return GradientBoostingRegressor(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    learning_rate=cfg.learning_rate,
                    min_samples_leaf=cfg.min_samples_leaf,
                    subsample=cfg.subsample,
                    max_features=cfg.max_features,
                    random_state=42,
                )
            elif cfg.model_type == "random_forest":
                self._model_backend = "sklearn_rf"
                return RandomForestRegressor(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    min_samples_leaf=cfg.min_samples_leaf,
                    max_features=cfg.max_features,
                    random_state=42,
                    n_jobs=-1,
                )

        # --- Fallback ---
        return self._create_fallback_model()

    def _create_fallback_model(self) -> Any:
        """
        Create a fallback linear model.

        Returns sklearn ``Ridge`` if available, otherwise ``_RidgeFallback``.
        Returns ``None`` if ``config.fallback_to_linear`` is False.
        """
        if not self.config.fallback_to_linear:
            raise RuntimeError(
                "No ML backend available and fallback_to_linear is False. "
                "Install scikit-learn, xgboost, or lightgbm."
            )

        if _SKLEARN_AVAILABLE:
            self._model_backend = "sklearn_ridge"
            logger.info(
                "Falling back to sklearn Ridge regression."
            )
            return SklearnRidge(alpha=1.0)

        self._model_backend = "ridge_fallback"
        logger.info(
            "Falling back to built-in Ridge regression (no sklearn)."
        )
        return _RidgeFallback(alpha=1.0)

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_fit(
        model: Any,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]],
    ) -> None:
        """
        Fit a model, routing ``sample_weight`` only when supported.
        """
        try:
            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
        except TypeError:
            # Model does not accept sample_weight
            model.fit(X, y)

    def _fit_with_early_stopping(
        self,
        X_trn: NDArray[np.float64],
        y_trn: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]],
    ) -> None:
        """
        Fit XGBoost or LightGBM with early stopping on a held-out
        validation set.
        """
        es_rounds = self.config.early_stopping_rounds

        if self._model_backend == "xgboost":
            fit_kwargs: Dict[str, Any] = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
            # XGBoost >= 1.6 uses callbacks for early stopping
            try:
                callback = xgb.callback.EarlyStopping(
                    rounds=es_rounds,
                    metric_name="rmse",
                    save_best=True,
                )
                fit_kwargs["callbacks"] = [callback]
            except AttributeError:
                fit_kwargs["early_stopping_rounds"] = es_rounds

            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self._model.fit(X_trn, y_trn, **fit_kwargs)

        elif self._model_backend == "lightgbm":
            callbacks = [
                lgb.early_stopping(es_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "callbacks": callbacks,
            }
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self._model.fit(X_trn, y_trn, **fit_kwargs)

        else:
            # Fallback: no early stopping
            self._safe_fit(self._model, X_trn, y_trn, sample_weight)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _extract_feature_importance(self) -> None:
        """
        Extract feature importance from the fitted model and store in
        ``self._feature_importance`` as a sorted dict.
        """
        if self._model is None:
            return

        importances: Optional[NDArray[np.float64]] = None

        if hasattr(self._model, "feature_importances_"):
            importances = np.asarray(self._model.feature_importances_)
        elif hasattr(self._model, "coef_"):
            coef = np.asarray(self._model.coef_).ravel()
            abs_coef = np.abs(coef)
            total = abs_coef.sum()
            importances = abs_coef / total if total > EPSILON else abs_coef

        if importances is None:
            self._feature_importance = {}
            return

        names = self._feature_names
        if len(names) != len(importances):
            names = [f"feature_{i}" for i in range(len(importances))]

        imp_dict = {
            name: float(val) for name, val in zip(names, importances)
        }
        self._feature_importance = dict(
            sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True)
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance from the fitted model.

        Returns:
            Dictionary mapping feature name to importance score, sorted
            descending by importance.
        """
        return dict(self._feature_importance)

    def get_interaction_effects(self) -> List[Tuple[str, str, float]]:
        """
        Analyse tree structure to identify top interaction effects.

        For tree-based models this inspects feature importances of the
        explicit interaction features (those named ``interact_*``). For
        non-tree models a heuristic based on coefficient products is used.

        Returns:
            List of ``(feature_a, feature_b, interaction_strength)`` tuples,
            sorted descending by strength.
        """
        interactions: List[Tuple[str, str, float]] = []

        for name, importance in self._feature_importance.items():
            if name.startswith("interact_"):
                # Parse "interact_AxB" -> (signal_A, signal_B)
                suffix = name[len("interact_"):]
                parts = suffix.split("x")
                if len(parts) == 2:
                    feat_a = f"signal_{parts[0]}"
                    feat_b = f"signal_{parts[1]}"
                    interactions.append((feat_a, feat_b, importance))

        # If no explicit interaction features, approximate from raw importances
        if not interactions and len(self._feature_importance) >= 2:
            raw_feats = [
                (n, v) for n, v in self._feature_importance.items()
                if n.startswith("signal_") and not n.startswith("signal_roll")
            ]
            for (n_a, v_a), (n_b, v_b) in combinations(raw_feats, 2):
                interactions.append((n_a, n_b, v_a * v_b))

        interactions.sort(key=lambda x: x[2], reverse=True)
        return interactions

    # ------------------------------------------------------------------
    # Purged train/test split
    # ------------------------------------------------------------------

    @staticmethod
    def _purged_train_test_split(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        train_end: int,
        test_start: int,
        purge: int,
        embargo: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Split data with purge and embargo windows to prevent information
        leakage.

        The training set runs from index 0 to ``train_end - purge``
        (exclusive). The purge window removes rows that might contain
        forward-looking information due to overlapping return labels.

        The embargo window after the test point is excluded from training
        in future calls (not enforced here since we use an expanding window).

        Args:
            X: Feature matrix ``(T, K)``.
            y: Target vector ``(T,)``.
            train_end: Last index (exclusive) before the purge gap.
            test_start: First index of the test observation.
            purge: Number of rows to drop between train and test.
            embargo: Extra rows to drop after the test point.

        Returns:
            Tuple of ``(X_train, y_train)``.
        """
        effective_end = max(0, train_end - purge)
        if effective_end <= 0:
            return (
                np.empty((0, X.shape[1]), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        # Embargo: also remove rows right after the test point from training
        # (relevant for expanding window where future retrains include past
        # test regions). For the current call we simply take [0, effective_end).
        X_train = X[:effective_end]
        y_train = y[:effective_end]

        # Remove NaN rows
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        return X_train[valid], y_train[valid]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> Dict[str, float]:
        """
        Compute prediction quality metrics.

        Metrics returned:

        * **IC**: Information Coefficient (Pearson correlation between
          predictions and realised returns).
        * **IC_IR**: IC Information Ratio (mean IC / std IC across rolling
          windows -- approximated here as IC / assumed IC vol of 0.15).
        * **hit_rate**: Fraction of observations where the sign of the
          prediction matches the sign of the realised return.
        * **MSE**: Mean squared error.
        * **explained_variance**: Fraction of target variance explained.

        Args:
            X: Feature matrix ``(T, K)``.
            y: Realised return vector ``(T,)``.

        Returns:
            Dictionary of metric names to values.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)

        # Remove NaN
        valid = ~(np.isnan(pred) | np.isnan(y))
        pred_v = pred[valid]
        y_v = y[valid]

        metrics: Dict[str, float] = {}

        if len(pred_v) < 2:
            return {
                "IC": 0.0, "IC_IR": 0.0, "hit_rate": 0.5,
                "MSE": float("inf"), "explained_variance": 0.0,
            }

        # IC (Pearson)
        corr_matrix = np.corrcoef(pred_v, y_v)
        ic = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        metrics["IC"] = ic

        # IC_IR (approximate)
        assumed_ic_vol = 0.15
        metrics["IC_IR"] = ic / assumed_ic_vol if assumed_ic_vol > EPSILON else 0.0

        # Hit rate
        correct_sign = np.sign(pred_v) == np.sign(y_v)
        metrics["hit_rate"] = float(correct_sign.mean())

        # MSE
        metrics["MSE"] = float(np.mean((pred_v - y_v) ** 2))

        # Explained variance
        y_var = np.var(y_v)
        if y_var > EPSILON:
            residual_var = np.var(y_v - pred_v)
            metrics["explained_variance"] = float(
                1.0 - residual_var / y_var
            )
        else:
            metrics["explained_variance"] = 0.0

        self._train_metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the fitted combiner to disk.

        Uses ``joblib`` if available (better for large numpy arrays inside
        sklearn models), otherwise falls back to ``pickle``.

        Args:
            path: File path for the serialized model.
        """
        state = {
            "config": self.config,
            "model": self._model,
            "fitted": self._fitted,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "model_backend": self._model_backend,
            "train_metrics": self._train_metrics,
        }

        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if _JOBLIB_AVAILABLE:
            joblib.dump(state, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("NonlinearSignalCombiner saved to %s.", path)

    def load(self, path: str) -> None:
        """
        Deserialize a previously saved combiner from disk.

        Args:
            path: File path to the serialized model.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        if _JOBLIB_AVAILABLE:
            state = joblib.load(path)
        else:
            with open(path, "rb") as f:
                state = pickle.load(f)  # noqa: S301

        self.config = state["config"]
        self._model = state["model"]
        self._fitted = state["fitted"]
        self._feature_names = state["feature_names"]
        self._feature_importance = state["feature_importance"]
        self._model_backend = state["model_backend"]
        self._train_metrics = state.get("train_metrics", {})

        logger.info(
            "NonlinearSignalCombiner loaded from %s (backend=%s).",
            path,
            self._model_backend,
        )

    # ------------------------------------------------------------------
    # Output clipping
    # ------------------------------------------------------------------

    def _clip_output(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Clip model output according to ``config.output_type``.
        """
        out_type = self.config.output_type
        clip_val = self.config.clip_predictions

        if out_type == "direction":
            return np.clip(raw, -clip_val, clip_val)
        elif out_type == "probability":
            return np.clip(raw, 0.0, 1.0)
        elif out_type == "return_forecast":
            return np.clip(raw, -clip_val, clip_val)
        else:
            return np.clip(raw, -clip_val, clip_val)


# ===========================================================================
# Adapter for the RAT combiner interface
# ===========================================================================

class NonlinearCombinerAdapter:
    """
    Adapter that makes :class:`NonlinearSignalCombiner` compatible with the
    existing RAT combiner's :class:`CombinedDecision` interface.

    This bridges the gap between the matrix-oriented ML combiner and the
    event-driven RAT signal pipeline. Incoming ``Signal`` objects are
    converted to a feature vector, passed through the nonlinear combiner,
    and the output is wrapped in a ``CombinedDecision``.

    Usage::

        adapter = NonlinearCombinerAdapter(combiner)

        # Identical call signature to RAT SignalCombiner.combine()
        decision = adapter.combine(signals, current_regime="bull")
    """

    def __init__(
        self,
        combiner: NonlinearSignalCombiner,
        max_position_pct: float = 0.25,
        min_confidence: float = 0.1,
    ) -> None:
        """
        Args:
            combiner: A fitted :class:`NonlinearSignalCombiner` instance.
            max_position_pct: Maximum suggested position size.
            min_confidence: Minimum absolute prediction for non-hold action.
        """
        self._combiner = combiner
        self._max_position_pct = max_position_pct
        self._min_confidence = min_confidence

    def combine(
        self,
        signals: List,
        current_regime: Optional[str] = None,
    ) -> "CombinedDecision":
        """
        Combine a list of RAT Signal objects into a CombinedDecision.

        Each signal's ``direction * confidence`` is used as a feature value.
        The nonlinear combiner predicts a combined direction from the feature
        vector.

        Args:
            signals: List of ``Signal`` objects from RAT modules.
            current_regime: Optional current market regime label.

        Returns:
            A ``CombinedDecision`` with action, direction, confidence, and
            suggested position size.
        """
        # Lazy import to avoid circular dependencies
        from trading_algo.rat.combiner.combiner import CombinedDecision
        from trading_algo.rat.signals import SignalType, SignalSource

        timestamp = datetime.now()

        if not signals:
            return self._hold_decision(
                timestamp=timestamp,
                symbol="UNKNOWN",
                signals=signals,
            )

        symbol = signals[0].symbol

        # Build feature vector: one feature per source
        all_sources = list(SignalSource)
        source_to_idx = {src: i for i, src in enumerate(all_sources)}
        feature_vec = np.zeros(len(all_sources), dtype=np.float64)

        raw_signals: Dict[SignalSource, float] = {}
        contributing_sources: List[SignalSource] = []
        weights_used: Dict[SignalSource, float] = {}

        for sig in signals:
            idx = source_to_idx.get(sig.source)
            if idx is not None:
                feature_vec[idx] = sig.direction * sig.confidence
                raw_signals[sig.source] = sig.direction
                contributing_sources.append(sig.source)

        # Predict
        if self._combiner._fitted:
            pred = float(self._combiner.predict(feature_vec.reshape(1, -1))[0])
        else:
            # If not fitted, use simple average as fallback
            if contributing_sources:
                pred = float(np.mean(
                    [feature_vec[source_to_idx[s]] for s in contributing_sources]
                ))
            else:
                pred = 0.0

        direction = np.clip(pred, -1.0, 1.0)
        confidence = min(1.0, abs(pred) / max(self._min_confidence, EPSILON))

        # Determine action
        if abs(direction) < self._min_confidence:
            action = "hold"
            signal_type = SignalType.HOLD
        elif direction > 0:
            action = "buy"
            signal_type = SignalType.LONG
        else:
            action = "sell"
            signal_type = SignalType.SHORT

        # Position sizing: proportional to |direction| * confidence
        position_size = min(
            abs(direction) * confidence,
            self._max_position_pct,
        )

        # Populate weights from feature importance
        importance = self._combiner.get_feature_importance()
        for src in contributing_sources:
            feat_name = f"signal_{source_to_idx[src]}"
            weights_used[src] = importance.get(feat_name, 1.0 / max(len(contributing_sources), 1))

        # Compute urgency as max urgency across contributing signals
        urgency = max((s.urgency for s in signals), default=0.0)

        return CombinedDecision(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            signal_type=signal_type,
            direction=float(direction),
            confidence=float(confidence),
            urgency=float(urgency),
            position_size_pct=float(position_size),
            contributing_sources=contributing_sources,
            weights_used=weights_used,
            raw_signals=raw_signals,
        )

    def _hold_decision(
        self,
        timestamp: datetime,
        symbol: str,
        signals: List,
    ) -> "CombinedDecision":
        """Produce a neutral hold decision."""
        from trading_algo.rat.combiner.combiner import CombinedDecision
        from trading_algo.rat.signals import SignalType

        return CombinedDecision(
            timestamp=timestamp,
            symbol=symbol,
            action="hold",
            signal_type=SignalType.HOLD,
            direction=0.0,
            confidence=0.0,
            urgency=0.0,
            position_size_pct=0.0,
            contributing_sources=[],
            weights_used={},
            raw_signals={s.source: s.direction for s in signals} if signals else {},
        )


# ===========================================================================
# Utility helpers
# ===========================================================================

def _isnan_safe(arr: NDArray) -> NDArray[np.bool_]:
    """
    Element-wise NaN check that works for both numeric and object arrays.

    For object/string dtypes (e.g. regime labels), returns an all-False mask
    since string values cannot be NaN.
    """
    try:
        return np.isnan(arr)
    except (TypeError, ValueError):
        return np.zeros(len(arr), dtype=np.bool_)


def _isnan_safe_scalar(val: Any) -> bool:
    """
    Scalar NaN check that handles non-numeric types gracefully.
    """
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False
