"""
Tail Risk Management

Monitors and responds to extreme market conditions.
Implements tail hedging triggers and correlation breakdown detection.

Key Components:
    - Tail risk monitoring via ES and EVT
    - Correlation breakdown detection
    - Dynamic hedging triggers
    - Position de-risking rules
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto

from trading_algo.quant_core.utils.constants import (
    MAX_DRAWDOWN_THRESHOLD,
    EPSILON,
    SQRT_252,
)


class TailRiskLevel(Enum):
    """Tail risk severity levels."""
    NORMAL = auto()
    ELEVATED = auto()
    HIGH = auto()
    EXTREME = auto()
    CRISIS = auto()


@dataclass
class TailRiskAssessment:
    """Assessment of current tail risk."""
    level: TailRiskLevel
    vol_percentile: float          # Current vol vs history (0-100)
    correlation_spike: bool        # Correlation breakdown detected
    drawdown_pct: float           # Current drawdown
    days_in_drawdown: int
    recommended_exposure: float    # Suggested exposure reduction
    triggers_active: List[str]     # Which risk triggers are active


class TailRiskManager:
    """
    Tail risk detection and management.

    Monitors multiple risk indicators and provides
    recommended actions for tail risk scenarios.

    Monitors:
    - Volatility regime (VIX-like)
    - Correlation changes
    - Drawdown severity
    - Consecutive losses

    Responds with:
    - Exposure reduction recommendations
    - Hedging trigger signals
    - Position de-risking alerts
    """

    def __init__(
        self,
        vol_lookback: int = 252,
        correlation_lookback: int = 60,
        drawdown_threshold: float = MAX_DRAWDOWN_THRESHOLD,
        vol_crisis_percentile: float = 95,
        correlation_spike_threshold: float = 0.3,
    ):
        """
        Initialize tail risk manager.

        Args:
            vol_lookback: Days for volatility history
            correlation_lookback: Days for correlation calculation
            drawdown_threshold: Drawdown level triggering alert
            vol_crisis_percentile: Vol percentile for crisis
            correlation_spike_threshold: Change in correlation for spike
        """
        self.vol_lookback = vol_lookback
        self.correlation_lookback = correlation_lookback
        self.drawdown_threshold = drawdown_threshold
        self.vol_crisis_percentile = vol_crisis_percentile
        self.correlation_spike_threshold = correlation_spike_threshold

        # State
        self._vol_history: List[float] = []
        self._correlation_history: List[float] = []
        self._peak_equity: float = 0.0
        self._drawdown_start: int = 0
        self._current_bar: int = 0

    def assess_risk(
        self,
        returns: NDArray[np.float64],
        current_equity: float,
        asset_returns: Optional[Dict[str, NDArray]] = None,
    ) -> TailRiskAssessment:
        """
        Assess current tail risk level.

        Args:
            returns: Portfolio return series
            current_equity: Current equity value
            asset_returns: Optional dict of asset returns for correlation

        Returns:
            TailRiskAssessment with risk level and recommendations
        """
        triggers: List[str] = []

        # 1. Volatility assessment
        current_vol = np.std(returns[-20:], ddof=1) * SQRT_252 if len(returns) >= 20 else 0.15
        self._vol_history.append(current_vol)
        if len(self._vol_history) > self.vol_lookback:
            self._vol_history = self._vol_history[-self.vol_lookback:]

        vol_percentile = self._calculate_percentile(current_vol, self._vol_history)

        if vol_percentile > self.vol_crisis_percentile:
            triggers.append("volatility_crisis")

        # 2. Correlation assessment
        correlation_spike = False
        if asset_returns and len(asset_returns) >= 2:
            correlation_spike = self._detect_correlation_spike(asset_returns)
            if correlation_spike:
                triggers.append("correlation_breakdown")

        # 3. Drawdown assessment
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._drawdown_start = self._current_bar

        drawdown_pct = 1 - current_equity / self._peak_equity if self._peak_equity > 0 else 0.0
        days_in_drawdown = self._current_bar - self._drawdown_start

        if drawdown_pct > self.drawdown_threshold:
            triggers.append("drawdown_breach")

        if drawdown_pct > self.drawdown_threshold * 1.5:
            triggers.append("severe_drawdown")

        # 4. Consecutive losses
        if len(returns) >= 5:
            recent = returns[-5:]
            if all(r < 0 for r in recent):
                triggers.append("consecutive_losses")

        # Determine risk level
        level = self._determine_risk_level(
            vol_percentile, correlation_spike, drawdown_pct, len(triggers)
        )

        # Calculate recommended exposure
        recommended_exposure = self._calculate_exposure_reduction(level)

        self._current_bar += 1

        return TailRiskAssessment(
            level=level,
            vol_percentile=vol_percentile,
            correlation_spike=correlation_spike,
            drawdown_pct=drawdown_pct,
            days_in_drawdown=days_in_drawdown,
            recommended_exposure=recommended_exposure,
            triggers_active=triggers,
        )

    def _detect_correlation_spike(
        self,
        asset_returns: Dict[str, NDArray],
    ) -> bool:
        """
        Detect correlation breakdown (spike in correlations).

        During crises, correlations tend to spike toward 1.
        """
        symbols = list(asset_returns.keys())
        if len(symbols) < 2:
            return False

        # Calculate recent correlation
        min_len = min(len(r) for r in asset_returns.values())
        if min_len < 30:
            return False

        # Get recent 20-day correlation
        recent_len = min(20, min_len)
        returns_recent = np.column_stack([
            asset_returns[s][-recent_len:] for s in symbols
        ])
        recent_corr = np.corrcoef(returns_recent, rowvar=False)
        avg_recent_corr = np.mean(recent_corr[np.triu_indices_from(recent_corr, k=1)])

        # Get historical average correlation
        hist_len = min(60, min_len)
        returns_hist = np.column_stack([
            asset_returns[s][-hist_len:-20] for s in symbols
        ]) if min_len > 30 else returns_recent
        hist_corr = np.corrcoef(returns_hist, rowvar=False)
        avg_hist_corr = np.mean(hist_corr[np.triu_indices_from(hist_corr, k=1)])

        # Track correlation history
        self._correlation_history.append(avg_recent_corr)
        if len(self._correlation_history) > self.correlation_lookback:
            self._correlation_history = self._correlation_history[-self.correlation_lookback:]

        # Spike if recent >> historical
        return (avg_recent_corr - avg_hist_corr) > self.correlation_spike_threshold

    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile of value in history."""
        if not history:
            return 50.0
        return float(np.sum(np.array(history) < value) / len(history) * 100)

    def _determine_risk_level(
        self,
        vol_percentile: float,
        correlation_spike: bool,
        drawdown_pct: float,
        n_triggers: int,
    ) -> TailRiskLevel:
        """Determine overall risk level."""
        if n_triggers >= 3 or drawdown_pct > 0.30:
            return TailRiskLevel.CRISIS
        elif n_triggers >= 2 or vol_percentile > 95:
            return TailRiskLevel.EXTREME
        elif n_triggers >= 1 or vol_percentile > 85:
            return TailRiskLevel.HIGH
        elif vol_percentile > 70 or drawdown_pct > 0.10:
            return TailRiskLevel.ELEVATED
        else:
            return TailRiskLevel.NORMAL

    def _calculate_exposure_reduction(self, level: TailRiskLevel) -> float:
        """Calculate recommended exposure multiplier."""
        exposure_map = {
            TailRiskLevel.NORMAL: 1.0,
            TailRiskLevel.ELEVATED: 0.8,
            TailRiskLevel.HIGH: 0.5,
            TailRiskLevel.EXTREME: 0.25,
            TailRiskLevel.CRISIS: 0.1,
        }
        return exposure_map.get(level, 1.0)

    def reset(self) -> None:
        """Reset manager state."""
        self._vol_history.clear()
        self._correlation_history.clear()
        self._peak_equity = 0.0
        self._drawdown_start = 0
        self._current_bar = 0
