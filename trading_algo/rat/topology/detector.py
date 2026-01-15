"""
Topology Detector: Market regime detection using persistent homology.

Uses Topological Data Analysis to classify market regimes:
- β₀ (connected components) - Market fragmentation
- β₁ (loops/cycles) - Cyclical patterns
- β₂ (voids) - Complex structural holes

Regimes detected:
- TRENDING: Low β₀, low β₁ (connected, no cycles)
- CONSOLIDATION: Low β₀, high β₁ (connected with cycles)
- ROTATION: High β₀, varying β₁ (fragmented sectors)
- BUBBLE: Specific β signature with expanding structure
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


class TopologyRegime(Enum):
    """Market regimes detected via topology."""

    UNKNOWN = auto()        # Insufficient data
    TRENDING = auto()       # Strong directional move
    CONSOLIDATION = auto()  # Range-bound with mean reversion
    ROTATION = auto()       # Sector rotation
    FRAGMENTED = auto()     # Disconnected market structure
    BUBBLE = auto()         # Bubble-like expansion


@dataclass
class TopologyState:
    """Current topological state."""

    timestamp: datetime
    symbol: str
    regime: TopologyRegime
    betti_0: float          # Connected components
    betti_1: float          # Loops/cycles
    betti_2: float          # Voids
    persistence: float      # How stable the structure is
    regime_confidence: float

    @property
    def is_stable(self) -> bool:
        """Check if regime is stable."""
        return self.persistence > 0.5


class TopologyDetector:
    """
    Detect market regimes using persistent homology.

    Uses Takens embedding to construct point cloud from time series,
    then computes Betti numbers to classify regime.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        time_delay: int = 1,
        max_dimension: int = 2,
        max_edge_length: float = 2.0,
        min_persistence: float = 0.1,
        window_size: int = 100,
    ):
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.min_persistence = min_persistence
        self.window_size = window_size

        # Price history per symbol
        self._price_history: Dict[str, Deque[float]] = {}

        # State cache
        self._last_state: Dict[str, TopologyState] = {}
        self._last_regime: Dict[str, TopologyRegime] = {}

        # Check for ripser
        self._has_ripser = self._check_ripser()

    def _check_ripser(self) -> bool:
        """Check if ripser is available."""
        try:
            import ripser
            return True
        except ImportError:
            return False

    def update(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update with new price data."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.window_size)

        self._price_history[symbol].append(price)

    def detect(self, symbol: str) -> TopologyState:
        """Detect current market regime."""
        if symbol not in self._price_history:
            return self._create_default_state(symbol)

        prices = list(self._price_history[symbol])

        if len(prices) < self.embedding_dim * self.time_delay + 10:
            return self._create_default_state(symbol)

        # Build point cloud via Takens embedding
        point_cloud = self._build_point_cloud(prices)

        # Compute Betti numbers
        if self._has_ripser:
            betti = self._compute_betti_numbers_ripser(point_cloud)
        else:
            betti = self._compute_betti_numbers_simple(point_cloud)

        # Classify regime
        regime, confidence = self._classify_regime(betti)

        # Compute persistence (stability)
        persistence = self._compute_persistence(symbol, regime)

        state = TopologyState(
            timestamp=datetime.now(),
            symbol=symbol,
            regime=regime,
            betti_0=betti[0],
            betti_1=betti[1],
            betti_2=betti[2] if len(betti) > 2 else 0.0,
            persistence=persistence,
            regime_confidence=confidence,
        )

        self._last_state[symbol] = state
        self._last_regime[symbol] = regime

        return state

    def _build_point_cloud(self, prices: List[float]) -> List[List[float]]:
        """
        Build point cloud using Takens embedding.

        Each point is (x_t, x_{t-τ}, x_{t-2τ}, ...) for embedding dimension d.
        """
        # Normalize prices
        mean_p = sum(prices) / len(prices)
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in prices) / len(prices))
        if std_p == 0:
            std_p = 1.0

        normalized = [(p - mean_p) / std_p for p in prices]

        # Build embedded points
        points = []
        start_idx = (self.embedding_dim - 1) * self.time_delay

        for i in range(start_idx, len(normalized)):
            point = []
            for d in range(self.embedding_dim):
                idx = i - d * self.time_delay
                point.append(normalized[idx])
            points.append(point)

        return points

    def _compute_betti_numbers_simple(
        self, point_cloud: List[List[float]]
    ) -> Tuple[float, float, float]:
        """
        Simple Betti number approximation without ripser.

        Uses density-based approach as proxy for topological features.
        """
        if len(point_cloud) < 10:
            return (1.0, 0.0, 0.0)

        n = len(point_cloud)

        # Compute pairwise distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = self._euclidean_distance(point_cloud[i], point_cloud[j])
                distances.append(d)

        if not distances:
            return (1.0, 0.0, 0.0)

        # Statistics of distance distribution
        mean_d = sum(distances) / len(distances)
        std_d = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / len(distances))

        # Estimate β₀ (components) from clustering
        # Points within 1 std are likely connected
        threshold = mean_d - 0.5 * std_d
        connected = sum(1 for d in distances if d < threshold)
        connectivity_ratio = connected / len(distances)

        # β₀ approximation (lower connectivity = more components)
        beta_0 = 1.0 / (connectivity_ratio + 0.1)

        # Estimate β₁ (loops) from local density variation
        # High variance in local density suggests cycles
        local_densities = []
        for i in range(n):
            local_dists = [
                self._euclidean_distance(point_cloud[i], point_cloud[j])
                for j in range(n) if i != j
            ]
            local_dists.sort()
            # k-nearest neighbor density
            k = min(5, len(local_dists))
            local_density = k / (sum(local_dists[:k]) + 0.001)
            local_densities.append(local_density)

        mean_density = sum(local_densities) / len(local_densities)
        density_var = sum((d - mean_density) ** 2 for d in local_densities) / len(local_densities)

        # β₁ approximation (higher density variance = more cycles)
        beta_1 = min(5.0, density_var / (mean_density ** 2 + 0.001))

        # β₂ approximation (simplified - based on outlier ratio)
        outlier_threshold = mean_d + 2 * std_d
        outliers = sum(1 for d in distances if d > outlier_threshold)
        beta_2 = outliers / len(distances) * 5

        return (beta_0, beta_1, beta_2)

    def _compute_betti_numbers_ripser(
        self, point_cloud: List[List[float]]
    ) -> Tuple[float, float, float]:
        """Compute Betti numbers using ripser library."""
        try:
            import ripser
            import numpy as np

            points = np.array(point_cloud)

            result = ripser.ripser(
                points,
                maxdim=self.max_dimension,
                thresh=self.max_edge_length,
            )

            dgms = result['dgms']

            # Count persistent features
            betti = []
            for dim in range(self.max_dimension + 1):
                if dim < len(dgms):
                    dgm = dgms[dim]
                    # Count features with persistence > threshold
                    persistent = sum(
                        1 for birth, death in dgm
                        if death - birth > self.min_persistence
                    )
                    betti.append(float(persistent))
                else:
                    betti.append(0.0)

            return tuple(betti) + (0.0,) * (3 - len(betti))

        except Exception:
            return self._compute_betti_numbers_simple(point_cloud)

    def _euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Compute Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _classify_regime(
        self, betti: Tuple[float, float, float]
    ) -> Tuple[TopologyRegime, float]:
        """
        Classify market regime from Betti numbers.

        Classification rules:
        - TRENDING: β₀ ≈ 1, β₁ low (single connected component, no cycles)
        - CONSOLIDATION: β₀ ≈ 1, β₁ high (connected with cycles = range bound)
        - ROTATION: β₀ > 2, varying β₁ (multiple components)
        - BUBBLE: β₁ increasing rapidly with β₂ > 0
        """
        beta_0, beta_1, beta_2 = betti

        # Normalize for comparison
        total = beta_0 + beta_1 + beta_2 + 1

        # Classification logic
        if beta_0 <= 1.5 and beta_1 <= 1.0:
            # Connected, no cycles = trending
            confidence = 0.7 + 0.3 * (1.0 - beta_1)
            return (TopologyRegime.TRENDING, confidence)

        elif beta_0 <= 1.5 and beta_1 > 1.0:
            # Connected with cycles = consolidation
            confidence = 0.6 + 0.2 * min(1.0, beta_1 / 3.0)
            return (TopologyRegime.CONSOLIDATION, confidence)

        elif beta_0 > 2.5:
            # Multiple components = fragmented or rotation
            if beta_1 > 0.5:
                confidence = 0.5 + 0.2 * min(1.0, beta_0 / 5.0)
                return (TopologyRegime.ROTATION, confidence)
            else:
                confidence = 0.5 + 0.2 * min(1.0, beta_0 / 5.0)
                return (TopologyRegime.FRAGMENTED, confidence)

        elif beta_2 > 0.5 and beta_1 > 2.0:
            # Complex structure with voids = potential bubble
            confidence = 0.4 + 0.3 * min(1.0, beta_2)
            return (TopologyRegime.BUBBLE, confidence)

        # Default
        return (TopologyRegime.UNKNOWN, 0.3)

    def _compute_persistence(self, symbol: str, current_regime: TopologyRegime) -> float:
        """Compute regime persistence (how long it's been stable)."""
        if symbol not in self._last_regime:
            return 0.0

        if self._last_regime[symbol] == current_regime:
            # Same regime - high persistence
            return 0.8
        else:
            # Regime change - low persistence
            return 0.2

    def _create_default_state(self, symbol: str) -> TopologyState:
        """Create default state."""
        return TopologyState(
            timestamp=datetime.now(),
            symbol=symbol,
            regime=TopologyRegime.UNKNOWN,
            betti_0=1.0,
            betti_1=0.0,
            betti_2=0.0,
            persistence=0.0,
            regime_confidence=0.0,
        )

    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """Generate trading signal from topological state."""
        state = self.detect(symbol)

        if state.regime == TopologyRegime.UNKNOWN:
            return None

        # Generate signal based on regime
        if state.regime == TopologyRegime.TRENDING:
            # Follow the trend
            prices = list(self._price_history.get(symbol, []))
            if len(prices) < 10:
                return None

            trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
            direction = 1.0 if trend > 0 else -1.0
            signal_type = SignalType.LONG if direction > 0 else SignalType.SHORT
            confidence = state.regime_confidence * 0.8

        elif state.regime == TopologyRegime.CONSOLIDATION:
            # Mean reversion
            prices = list(self._price_history.get(symbol, []))
            if len(prices) < 20:
                return None

            mean_price = sum(prices[-20:]) / 20
            current = prices[-1]
            deviation = (current - mean_price) / mean_price if mean_price != 0 else 0

            # Fade the deviation
            direction = -1.0 if deviation > 0 else 1.0
            signal_type = SignalType.SHORT if deviation > 0 else SignalType.LONG
            confidence = state.regime_confidence * min(1.0, abs(deviation) * 20)

        elif state.regime == TopologyRegime.BUBBLE:
            # Careful - potential reversal
            direction = -0.5  # Slight short bias
            signal_type = SignalType.SHORT
            confidence = state.regime_confidence * 0.5

        elif state.regime == TopologyRegime.ROTATION:
            # No clear signal in rotation
            return None

        else:
            return None

        return Signal(
            source=SignalSource.TOPOLOGY,
            signal_type=signal_type,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            urgency=0.5 if state.is_stable else 0.3,
            metadata={
                "regime": state.regime.name,
                "betti_0": state.betti_0,
                "betti_1": state.betti_1,
                "betti_2": state.betti_2,
                "persistence": state.persistence,
            },
        )

    def inject_backtest_data(
        self,
        symbol: str,
        prices: List[float],
    ) -> None:
        """Inject historical prices for backtesting."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.window_size)

        for price in prices:
            self._price_history[symbol].append(price)
