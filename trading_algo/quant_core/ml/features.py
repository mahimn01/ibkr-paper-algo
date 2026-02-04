"""
Feature Engineering for Trading Signals

Implements feature extraction based on:
    - Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"
    - Freyberger et al. (2020): "Dissecting Characteristics Nonparametrically"

Feature Categories:
    1. Price-based: Returns, momentum, reversal
    2. Volume-based: Volume ratios, illiquidity
    3. Volatility-based: Realized vol, GARCH-type
    4. Technical: Moving averages, oscillators
    5. Microstructure: Spread, order flow

The features are designed to capture:
    - Cross-sectional return predictability
    - Time-series momentum and reversal
    - Volatility clustering
    - Liquidity effects

References:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.utils.math_utils import (
    log_returns,
    rolling_mean,
    rolling_std,
    ewma_volatility,
)


class FeatureCategory(Enum):
    """Categories of features."""
    PRICE = auto()
    VOLUME = auto()
    VOLATILITY = auto()
    TECHNICAL = auto()
    MICROSTRUCTURE = auto()
    CROSS_SECTIONAL = auto()


@dataclass
class FeatureSet:
    """Container for computed features."""
    features: NDArray[np.float64]      # Feature matrix (T x K)
    feature_names: List[str]           # Names of features
    timestamps: Optional[NDArray] = None
    categories: Dict[str, FeatureCategory] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    def get_feature(self, name: str) -> Optional[NDArray[np.float64]]:
        """Get a specific feature by name."""
        if name in self.feature_names:
            idx = self.feature_names.index(name)
            return self.features[:, idx]
        return None

    def to_dict(self) -> Dict[str, NDArray[np.float64]]:
        """Convert to dictionary of feature arrays."""
        return {name: self.features[:, i] for i, name in enumerate(self.feature_names)}


class FeatureEngine:
    """
    Feature engineering for trading signals.

    Computes a comprehensive set of features from price and volume data,
    following best practices from empirical asset pricing literature.

    Usage:
        engine = FeatureEngine()

        # Compute features for single asset
        features = engine.compute_features(prices, volumes)

        # For multiple assets (cross-sectional)
        features = engine.compute_cross_sectional_features(price_dict, volume_dict)
    """

    def __init__(
        self,
        momentum_windows: List[int] = [5, 10, 20, 60, 120, 252],
        volatility_windows: List[int] = [5, 10, 20, 60],
        volume_windows: List[int] = [5, 10, 20],
        include_technical: bool = True,
        include_microstructure: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize feature engine.

        Args:
            momentum_windows: Lookback periods for momentum
            volatility_windows: Lookback periods for volatility
            volume_windows: Lookback periods for volume features
            include_technical: Include technical indicators
            include_microstructure: Include microstructure features
            normalize: Normalize features to z-scores
        """
        self.momentum_windows = momentum_windows
        self.volatility_windows = volatility_windows
        self.volume_windows = volume_windows
        self.include_technical = include_technical
        self.include_microstructure = include_microstructure
        self.normalize = normalize

    def compute_features(
        self,
        prices: NDArray[np.float64],
        volumes: Optional[NDArray[np.float64]] = None,
        high: Optional[NDArray[np.float64]] = None,
        low: Optional[NDArray[np.float64]] = None,
        bid: Optional[NDArray[np.float64]] = None,
        ask: Optional[NDArray[np.float64]] = None,
    ) -> FeatureSet:
        """
        Compute all features for a single asset.

        Args:
            prices: Close prices
            volumes: Trading volumes
            high: High prices (optional)
            low: Low prices (optional)
            bid: Bid prices (optional)
            ask: Ask prices (optional)

        Returns:
            FeatureSet with computed features
        """
        features = {}
        categories = {}

        # 1. Price-based features
        price_features = self._price_features(prices)
        features.update(price_features)
        for name in price_features:
            categories[name] = FeatureCategory.PRICE

        # 2. Volatility features
        vol_features = self._volatility_features(prices, high, low)
        features.update(vol_features)
        for name in vol_features:
            categories[name] = FeatureCategory.VOLATILITY

        # 3. Volume features
        if volumes is not None:
            volume_features = self._volume_features(prices, volumes)
            features.update(volume_features)
            for name in volume_features:
                categories[name] = FeatureCategory.VOLUME

        # 4. Technical indicators
        if self.include_technical:
            tech_features = self._technical_features(prices, high, low, volumes)
            features.update(tech_features)
            for name in tech_features:
                categories[name] = FeatureCategory.TECHNICAL

        # 5. Microstructure features
        if self.include_microstructure and bid is not None and ask is not None:
            micro_features = self._microstructure_features(bid, ask, volumes)
            features.update(micro_features)
            for name in micro_features:
                categories[name] = FeatureCategory.MICROSTRUCTURE

        # Convert to matrix
        feature_names = list(features.keys())
        n = len(prices)
        feature_matrix = np.column_stack([
            self._pad_feature(features[name], n) for name in feature_names
        ])

        # Normalize if requested
        if self.normalize:
            feature_matrix = self._normalize_features(feature_matrix)

        return FeatureSet(
            features=feature_matrix,
            feature_names=feature_names,
            categories=categories,
        )

    def _price_features(self, prices: NDArray[np.float64]) -> Dict[str, NDArray]:
        """
        Compute price-based features.

        Includes:
            - Returns at various horizons
            - Momentum (cumulative returns)
            - Short-term reversal
        """
        features = {}
        returns = log_returns(prices)

        # Simple returns
        features["ret_1d"] = returns

        # Momentum at various horizons
        for window in self.momentum_windows:
            if len(prices) > window:
                mom = prices[window:] / prices[:-window] - 1
                features[f"mom_{window}d"] = mom

        # Short-term reversal (Jegadeesh 1990)
        if len(returns) > 5:
            features["reversal_5d"] = -rolling_mean(returns, 5)

        # Price relative to moving averages
        for window in [20, 50, 200]:
            if len(prices) > window:
                ma = rolling_mean(prices, window)
                features[f"price_to_ma{window}"] = prices[window-1:] / ma - 1

        # Maximum drawdown (lookback)
        if len(prices) > 20:
            features["max_dd_20d"] = self._rolling_max_drawdown(prices, 20)

        return features

    def _volatility_features(
        self,
        prices: NDArray[np.float64],
        high: Optional[NDArray[np.float64]] = None,
        low: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, NDArray]:
        """
        Compute volatility-based features.

        Includes:
            - Realized volatility
            - EWMA volatility
            - Parkinson estimator (if H/L available)
            - Volatility of volatility
        """
        features = {}
        returns = log_returns(prices)

        # Realized volatility at various windows
        for window in self.volatility_windows:
            if len(returns) > window:
                vol = rolling_std(returns, window) * SQRT_252
                features[f"rvol_{window}d"] = vol

        # EWMA volatility
        if len(returns) > 10:
            ewma_vol = ewma_volatility(returns)
            features["ewma_vol"] = ewma_vol

        # Parkinson volatility (uses high/low)
        if high is not None and low is not None:
            pk_vol = self._parkinson_volatility(high, low)
            features["parkinson_vol"] = pk_vol

        # Volatility ratio (short/long)
        if len(returns) > 60:
            vol_short = rolling_std(returns, 10)
            vol_long = rolling_std(returns, 60)
            vol_ratio = vol_short[50:] / (vol_long + EPSILON)
            features["vol_ratio_10_60"] = vol_ratio

        # Volatility of volatility
        if len(returns) > 40:
            vol_20 = rolling_std(returns, 20)
            vol_of_vol = rolling_std(vol_20, 20)
            features["vol_of_vol"] = vol_of_vol

        return features

    def _volume_features(
        self,
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
    ) -> Dict[str, NDArray]:
        """
        Compute volume-based features.

        Includes:
            - Volume ratios
            - Amihud illiquidity
            - Volume-price correlation
        """
        features = {}
        returns = log_returns(prices)

        # Volume relative to average
        for window in self.volume_windows:
            if len(volumes) > window:
                avg_vol = rolling_mean(volumes, window)
                features[f"vol_ratio_{window}d"] = volumes[window-1:] / (avg_vol + EPSILON)

        # Amihud illiquidity (|return| / volume)
        if len(returns) > 0 and len(volumes) > 1:
            min_len = min(len(returns), len(volumes) - 1)
            illiq = np.abs(returns[:min_len]) / (volumes[1:min_len+1] + EPSILON)
            features["amihud_illiq"] = illiq

            # Rolling average illiquidity
            if len(illiq) > 20:
                features["amihud_illiq_20d"] = rolling_mean(illiq, 20)

        # On-balance volume momentum
        if len(returns) > 0:
            min_len = min(len(returns), len(volumes) - 1)
            obv_change = np.sign(returns[:min_len]) * volumes[1:min_len+1]
            if len(obv_change) > 20:
                features["obv_mom_20d"] = rolling_mean(obv_change, 20)

        return features

    def _technical_features(
        self,
        prices: NDArray[np.float64],
        high: Optional[NDArray[np.float64]] = None,
        low: Optional[NDArray[np.float64]] = None,
        volumes: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, NDArray]:
        """
        Compute technical indicator features.

        Includes:
            - RSI (Relative Strength Index)
            - MACD
            - Bollinger Band position
            - ATR (Average True Range)
        """
        features = {}
        returns = log_returns(prices)

        # RSI
        if len(returns) > 14:
            rsi = self._compute_rsi(returns, 14)
            features["rsi_14"] = rsi

        # MACD
        if len(prices) > 26:
            macd, signal = self._compute_macd(prices)
            features["macd"] = macd
            features["macd_signal"] = macd - signal

        # Bollinger Band position
        if len(prices) > 20:
            ma = rolling_mean(prices, 20)
            std = rolling_std(prices, 20)
            upper = ma + 2 * std
            lower = ma - 2 * std
            bb_pos = (prices[19:] - lower) / (upper - lower + EPSILON)
            features["bb_position"] = bb_pos

        # ATR (requires high/low)
        if high is not None and low is not None and len(high) > 14:
            atr = self._compute_atr(high, low, prices, 14)
            features["atr_14"] = atr

        return features

    def _microstructure_features(
        self,
        bid: NDArray[np.float64],
        ask: NDArray[np.float64],
        volumes: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, NDArray]:
        """
        Compute microstructure features.

        Includes:
            - Bid-ask spread
            - Effective spread estimates
            - Kyle's lambda proxy
        """
        features = {}

        # Quoted spread
        mid = (bid + ask) / 2
        spread = (ask - bid) / (mid + EPSILON)
        features["quoted_spread"] = spread

        # Rolling average spread
        if len(spread) > 20:
            features["avg_spread_20d"] = rolling_mean(spread, 20)

        # Spread volatility
        if len(spread) > 20:
            features["spread_vol_20d"] = rolling_std(spread, 20)

        return features

    def _parkinson_volatility(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        window: int = 20,
    ) -> NDArray[np.float64]:
        """
        Parkinson volatility estimator using high/low prices.

        More efficient than close-to-close volatility.
        """
        log_hl = np.log(high / low)
        pk_var = log_hl ** 2 / (4 * np.log(2))

        if len(pk_var) > window:
            return np.sqrt(rolling_mean(pk_var, window)) * SQRT_252
        return np.sqrt(pk_var) * SQRT_252

    def _rolling_max_drawdown(
        self,
        prices: NDArray[np.float64],
        window: int,
    ) -> NDArray[np.float64]:
        """Calculate rolling maximum drawdown."""
        n = len(prices)
        if n <= window:
            return np.zeros(n)

        max_dd = np.zeros(n - window + 1)
        for i in range(n - window + 1):
            window_prices = prices[i:i + window]
            cummax = np.maximum.accumulate(window_prices)
            dd = (cummax - window_prices) / cummax
            max_dd[i] = np.max(dd)

        return max_dd

    def _compute_rsi(
        self,
        returns: NDArray[np.float64],
        window: int = 14,
    ) -> NDArray[np.float64]:
        """Compute Relative Strength Index."""
        gains = np.maximum(returns, 0)
        losses = np.abs(np.minimum(returns, 0))

        avg_gain = rolling_mean(gains, window)
        avg_loss = rolling_mean(losses, window)

        rs = avg_gain / (avg_loss + EPSILON)
        rsi = 100 - 100 / (1 + rs)

        return rsi

    def _compute_macd(
        self,
        prices: NDArray[np.float64],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute MACD and signal line."""
        # EMA calculation
        def ema(arr, span):
            alpha = 2 / (span + 1)
            result = np.zeros_like(arr)
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)

        # Trim to valid length
        start = slow - 1
        return macd_line[start:], signal_line[start:]

    def _compute_atr(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        window: int = 14,
    ) -> NDArray[np.float64]:
        """Compute Average True Range."""
        n = len(high)
        if n < 2:
            return np.zeros(n)

        tr = np.zeros(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        return rolling_mean(tr, window)

    def _pad_feature(
        self,
        feature: NDArray[np.float64],
        target_length: int,
    ) -> NDArray[np.float64]:
        """Pad feature to target length with NaN at beginning."""
        if len(feature) >= target_length:
            return feature[-target_length:]

        padding = np.full(target_length - len(feature), np.nan)
        return np.concatenate([padding, feature])

    def _normalize_features(
        self,
        features: NDArray[np.float64],
        window: int = 252,
    ) -> NDArray[np.float64]:
        """
        Normalize features using rolling z-score.

        Uses expanding window initially, then rolling.
        """
        n, k = features.shape
        normalized = np.zeros_like(features)

        for j in range(k):
            col = features[:, j]
            for i in range(n):
                if np.isnan(col[i]):
                    normalized[i, j] = np.nan
                    continue

                # Use available history (up to window)
                start = max(0, i - window + 1)
                history = col[start:i+1]
                valid = history[~np.isnan(history)]

                if len(valid) < 2:
                    normalized[i, j] = 0.0
                else:
                    mean = np.mean(valid)
                    std = np.std(valid, ddof=1)
                    normalized[i, j] = (col[i] - mean) / (std + EPSILON)

        return normalized


def compute_cross_sectional_features(
    prices_dict: Dict[str, NDArray[np.float64]],
    reference_feature: str = "mom_20d",
) -> Dict[str, float]:
    """
    Compute cross-sectional features (relative to universe).

    Returns cross-sectional rank/z-score for each asset.
    """
    engine = FeatureEngine(normalize=False)

    # Compute features for each asset
    feature_values = {}
    for symbol, prices in prices_dict.items():
        feature_set = engine.compute_features(prices)
        if reference_feature in feature_set.feature_names:
            values = feature_set.get_feature(reference_feature)
            if values is not None and len(values) > 0:
                feature_values[symbol] = values[-1]

    if not feature_values:
        return {}

    # Cross-sectional z-score
    values = np.array(list(feature_values.values()))
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    return {
        symbol: (val - mean) / (std + EPSILON)
        for symbol, val in feature_values.items()
    }
