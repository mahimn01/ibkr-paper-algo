"""
ML-Based Signal Combination

Implements signal combination using machine learning methods from:
    - Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"
    - López de Prado (2018): "Advances in Financial Machine Learning"

Methods:
    1. Linear combination with regularization (Ridge, Lasso, Elastic Net)
    2. Tree-based ensembles (Random Forest, Gradient Boosting)
    3. Neural networks (simple feed-forward)
    4. Model stacking (ensemble of models)

Key Considerations:
    - Time-series nature of data (no lookahead)
    - Non-stationarity of financial data
    - Class imbalance (few extreme returns)
    - Feature importance and model interpretability

References:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import EPSILON


class CombinerMethod(Enum):
    """Signal combination methods."""
    RIDGE = auto()              # L2 regularized linear
    LASSO = auto()              # L1 regularized linear
    ELASTIC_NET = auto()        # L1 + L2 regularized
    RANDOM_FOREST = auto()      # Random forest regression
    GRADIENT_BOOST = auto()     # Gradient boosting
    NEURAL_NET = auto()         # Simple neural network
    STACKING = auto()           # Stacked ensemble
    EQUAL_WEIGHT = auto()       # Simple average


@dataclass
class CombinerConfig:
    """Configuration for signal combiner."""
    method: CombinerMethod = CombinerMethod.RIDGE
    regularization: float = 1.0
    n_estimators: int = 100
    max_depth: int = 5
    hidden_layers: List[int] = field(default_factory=lambda: [32, 16])
    learning_rate: float = 0.01
    min_samples_leaf: int = 20


@dataclass
class CombinedSignal:
    """Result of signal combination."""
    signal: float                       # Combined signal (-1 to 1)
    confidence: float                   # Prediction confidence
    feature_importance: Dict[str, float]  # Feature contributions
    model_predictions: Dict[str, float]   # Individual model predictions


class SignalCombiner:
    """
    ML-based signal combination.

    Combines multiple trading signals using machine learning
    to improve predictive accuracy.

    Usage:
        combiner = SignalCombiner(config)

        # Train on historical data
        combiner.fit(features, returns)

        # Generate combined signal
        result = combiner.predict(current_features)
        signal = result.signal
    """

    def __init__(self, config: Optional[CombinerConfig] = None):
        """
        Initialize signal combiner.

        Args:
            config: Combiner configuration
        """
        self.config = config or CombinerConfig()
        self.model = None
        self.feature_names: List[str] = []
        self._fitted = False
        self._feature_importance: Dict[str, float] = {}

    def fit(
        self,
        features: NDArray[np.float64],
        returns: NDArray[np.float64],
        feature_names: Optional[List[str]] = None,
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> "SignalCombiner":
        """
        Fit the signal combination model.

        Args:
            features: Feature matrix (T x K)
            returns: Target returns (T,)
            feature_names: Names of features
            sample_weights: Sample weights (optional)

        Returns:
            Self for chaining
        """
        # Remove NaN samples
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(returns))
        X = features[valid_mask]
        y = returns[valid_mask]

        if len(X) < 100:
            # Not enough data, use simple averaging
            self.config.method = CombinerMethod.EQUAL_WEIGHT
            self._fitted = True
            return self

        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if sample_weights is not None:
            sample_weights = sample_weights[valid_mask]

        # Fit appropriate model
        if self.config.method == CombinerMethod.RIDGE:
            self._fit_ridge(X, y, sample_weights)
        elif self.config.method == CombinerMethod.LASSO:
            self._fit_lasso(X, y, sample_weights)
        elif self.config.method == CombinerMethod.ELASTIC_NET:
            self._fit_elastic_net(X, y, sample_weights)
        elif self.config.method == CombinerMethod.RANDOM_FOREST:
            self._fit_random_forest(X, y, sample_weights)
        elif self.config.method == CombinerMethod.GRADIENT_BOOST:
            self._fit_gradient_boost(X, y, sample_weights)
        elif self.config.method == CombinerMethod.NEURAL_NET:
            self._fit_neural_net(X, y, sample_weights)
        elif self.config.method == CombinerMethod.STACKING:
            self._fit_stacking(X, y, sample_weights)
        else:
            pass  # Equal weight doesn't need fitting

        self._fitted = True
        return self

    def predict(
        self,
        features: NDArray[np.float64],
    ) -> CombinedSignal:
        """
        Generate combined signal from features.

        Args:
            features: Feature vector or matrix

        Returns:
            CombinedSignal with prediction and metadata
        """
        if not self._fitted:
            return CombinedSignal(
                signal=0.0,
                confidence=0.0,
                feature_importance={},
                model_predictions={},
            )

        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Handle NaN
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)

        # Get prediction
        if self.config.method == CombinerMethod.EQUAL_WEIGHT:
            prediction = np.mean(features, axis=1)
        else:
            prediction = self.model.predict(features)

        # Convert to signal (-1 to 1)
        signal = float(np.clip(np.tanh(prediction[0]), -1, 1))

        # Confidence based on prediction magnitude
        confidence = min(1.0, abs(prediction[0]) / 0.01)  # 1% return = full confidence

        return CombinedSignal(
            signal=signal,
            confidence=confidence,
            feature_importance=self._feature_importance,
            model_predictions={},
        )

    def _fit_ridge(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit Ridge regression."""
        try:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=self.config.regularization)
            self.model.fit(X, y, sample_weight=sample_weights)

            # Feature importance from coefficients
            self._feature_importance = {
                name: float(abs(coef))
                for name, coef in zip(self.feature_names, self.model.coef_)
            }
        except ImportError:
            self._fit_simple_ridge(X, y)

    def _fit_simple_ridge(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        """Simple Ridge without sklearn."""
        n_features = X.shape[1]
        XtX = X.T @ X + self.config.regularization * np.eye(n_features)
        Xty = X.T @ y

        try:
            self._coefficients = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self._coefficients = np.zeros(n_features)

        self._feature_importance = {
            name: float(abs(coef))
            for name, coef in zip(self.feature_names, self._coefficients)
        }

        # Create simple model object
        class SimpleRidge:
            def __init__(self, coef):
                self.coef_ = coef

            def predict(self, X):
                return X @ self.coef_

        self.model = SimpleRidge(self._coefficients)

    def _fit_lasso(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit Lasso regression."""
        try:
            from sklearn.linear_model import Lasso
            self.model = Lasso(alpha=self.config.regularization)
            self.model.fit(X, y, sample_weight=sample_weights)

            self._feature_importance = {
                name: float(abs(coef))
                for name, coef in zip(self.feature_names, self.model.coef_)
            }
        except ImportError:
            # Fallback to ridge
            self._fit_simple_ridge(X, y)

    def _fit_elastic_net(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit Elastic Net regression."""
        try:
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(alpha=self.config.regularization, l1_ratio=0.5)
            self.model.fit(X, y)

            self._feature_importance = {
                name: float(abs(coef))
                for name, coef in zip(self.feature_names, self.model.coef_)
            }
        except ImportError:
            self._fit_simple_ridge(X, y)

    def _fit_random_forest(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit Random Forest."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                n_jobs=-1,
                random_state=42,
            )
            self.model.fit(X, y, sample_weight=sample_weights)

            self._feature_importance = {
                name: float(imp)
                for name, imp in zip(self.feature_names, self.model.feature_importances_)
            }
        except ImportError:
            self._fit_simple_ridge(X, y)

    def _fit_gradient_boost(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit Gradient Boosting."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                random_state=42,
            )
            self.model.fit(X, y, sample_weight=sample_weights)

            self._feature_importance = {
                name: float(imp)
                for name, imp in zip(self.feature_names, self.model.feature_importances_)
            }
        except ImportError:
            self._fit_simple_ridge(X, y)

    def _fit_neural_net(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit simple neural network."""
        try:
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.config.hidden_layers),
                learning_rate_init=self.config.learning_rate,
                max_iter=1000,
                random_state=42,
            )
            self.model.fit(X, y)

            # Approximate feature importance from first layer weights
            first_layer = np.abs(self.model.coefs_[0]).sum(axis=1)
            self._feature_importance = {
                name: float(imp)
                for name, imp in zip(self.feature_names, first_layer)
            }
        except ImportError:
            self._fit_simple_ridge(X, y)

    def _fit_stacking(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weights: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Fit stacked ensemble."""
        try:
            from sklearn.ensemble import StackingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

            estimators = [
                ('ridge', Ridge(alpha=1.0)),
                ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)),
            ]

            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=0.1),
                cv=5,
            )
            self.model.fit(X, y)

            # Use RF feature importance as proxy
            rf_model = self.model.estimators_[1]
            self._feature_importance = {
                name: float(imp)
                for name, imp in zip(self.feature_names, rf_model.feature_importances_)
            }
        except ImportError:
            self._fit_simple_ridge(X, y)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from fitted model."""
        return self._feature_importance.copy()

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]


class EnsembleSignalCombiner:
    """
    Ensemble of multiple signal combiners.

    Combines predictions from multiple models using weighted averaging.
    """

    def __init__(
        self,
        methods: Optional[List[CombinerMethod]] = None,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble combiner.

        Args:
            methods: List of combination methods to use
            weights: Weights for each method (equal if None)
        """
        if methods is None:
            methods = [
                CombinerMethod.RIDGE,
                CombinerMethod.RANDOM_FOREST,
                CombinerMethod.GRADIENT_BOOST,
            ]

        self.combiners = [
            SignalCombiner(CombinerConfig(method=method))
            for method in methods
        ]

        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()

    def fit(
        self,
        features: NDArray[np.float64],
        returns: NDArray[np.float64],
        feature_names: Optional[List[str]] = None,
    ) -> "EnsembleSignalCombiner":
        """Fit all component combiners."""
        for combiner in self.combiners:
            combiner.fit(features, returns, feature_names)
        return self

    def predict(
        self,
        features: NDArray[np.float64],
    ) -> CombinedSignal:
        """Generate weighted ensemble prediction."""
        predictions = []
        model_preds = {}

        for i, combiner in enumerate(self.combiners):
            result = combiner.predict(features)
            predictions.append(result.signal)
            model_preds[f"model_{i}"] = result.signal

        # Weighted average
        ensemble_signal = float(np.sum(np.array(predictions) * self.weights))

        # Average confidence
        avg_confidence = np.mean([c.predict(features).confidence for c in self.combiners])

        # Combine feature importance (weighted average)
        combined_importance: Dict[str, float] = {}
        for i, combiner in enumerate(self.combiners):
            for name, imp in combiner.get_feature_importance().items():
                if name not in combined_importance:
                    combined_importance[name] = 0.0
                combined_importance[name] += imp * self.weights[i]

        return CombinedSignal(
            signal=ensemble_signal,
            confidence=avg_confidence,
            feature_importance=combined_importance,
            model_predictions=model_preds,
        )
