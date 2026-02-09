"""
Regime-Conditional Performance Analysis

Breaks down each strategy's performance by HMM market regime.
The output feeds into the Phase 5 dynamic allocation weights so the
controller can overweight strategies that perform well in the
current regime.

Regime-conditional Sharpe ratios tell us:
  - Bull: Momentum and Orchestrator tend to outperform.
  - Bear: Reversal and Pairs Trading tend to outperform.
  - High Vol: ORB and Pairs tend to outperform.
  - Low Vol: Momentum tends to outperform.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Performance metrics within a single regime."""
    regime: str
    n_bars: int = 0
    n_signals: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_signal_weight: float = 0.0


@dataclass
class StrategyRegimeProfile:
    """Regime-conditional profile for one strategy."""
    strategy_name: str
    regimes: Dict[str, RegimePerformance] = field(default_factory=dict)

    @property
    def best_regime(self) -> Optional[str]:
        """Regime where this strategy has the highest Sharpe."""
        if not self.regimes:
            return None
        return max(self.regimes, key=lambda r: self.regimes[r].sharpe_ratio)

    @property
    def worst_regime(self) -> Optional[str]:
        """Regime where this strategy has the lowest Sharpe."""
        if not self.regimes:
            return None
        return min(self.regimes, key=lambda r: self.regimes[r].sharpe_ratio)


@dataclass
class RegimeAnalysisResult:
    """Complete regime analysis across all strategies."""
    strategy_profiles: Dict[str, StrategyRegimeProfile] = field(default_factory=dict)
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """Recommended allocation weights per regime.
    E.g. {"BULL": {"Orchestrator": 0.40, "Momentum": 0.30, ...}}"""


class RegimeAnalyzer:
    """
    Analyze per-strategy performance across market regimes.

    Usage::

        analyzer = RegimeAnalyzer()
        result = analyzer.analyze(
            strategy_returns={"Orchestrator": [...], "Momentum": [...]},
            regime_labels=["BULL", "BULL", "BEAR", ...]
        )
        # result.regime_weights["BULL"] → optimal weights for bull market
    """

    # Default priors for regime tilts (used when data is insufficient)
    DEFAULT_REGIME_TILTS = {
        "BULL": {
            "Orchestrator": 0.35,
            "PureMomentum": 0.30,
            "IntradayMomentum": 0.10,
            "ORB": 0.10,
            "PairsTrading": 0.05,
            "ShortTermReversal": 0.05,
            "OvernightReturns": 0.05,
        },
        "BEAR": {
            "Orchestrator": 0.20,
            "PureMomentum": 0.05,
            "IntradayMomentum": 0.10,
            "ORB": 0.15,
            "PairsTrading": 0.20,
            "ShortTermReversal": 0.20,
            "OvernightReturns": 0.10,
        },
        "HIGH_VOL": {
            "Orchestrator": 0.15,
            "PureMomentum": 0.10,
            "IntradayMomentum": 0.10,
            "ORB": 0.20,
            "PairsTrading": 0.25,
            "ShortTermReversal": 0.10,
            "OvernightReturns": 0.10,
        },
        "NEUTRAL": {
            "Orchestrator": 0.25,
            "PureMomentum": 0.20,
            "IntradayMomentum": 0.10,
            "ORB": 0.10,
            "PairsTrading": 0.15,
            "ShortTermReversal": 0.10,
            "OvernightReturns": 0.10,
        },
    }

    def __init__(self, min_regime_bars: int = 30):
        self.min_regime_bars = min_regime_bars

    def analyze(
        self,
        strategy_returns: Dict[str, List[float]],
        regime_labels: List[str],
    ) -> RegimeAnalysisResult:
        """
        Analyze strategy performance per regime.

        Args:
            strategy_returns: Dict of strategy_name -> daily return series.
            regime_labels: Regime label per day (same length as return series).

        Returns:
            RegimeAnalysisResult with profiles and recommended weights.
        """
        profiles: Dict[str, StrategyRegimeProfile] = {}
        unique_regimes = sorted(set(regime_labels))

        for strat_name, returns in strategy_returns.items():
            if len(returns) != len(regime_labels):
                logger.warning(
                    "Length mismatch for %s: %d returns vs %d labels",
                    strat_name, len(returns), len(regime_labels),
                )
                continue

            profile = StrategyRegimeProfile(strategy_name=strat_name)

            for regime in unique_regimes:
                mask = [i for i, r in enumerate(regime_labels) if r == regime]
                if len(mask) < self.min_regime_bars:
                    continue

                regime_rets = np.array([returns[i] for i in mask])

                sharpe = self._compute_sharpe(regime_rets)
                total_ret = float(np.sum(regime_rets))
                wins = np.sum(regime_rets > 0)
                wr = float(wins / len(regime_rets)) if len(regime_rets) > 0 else 0

                profile.regimes[regime] = RegimePerformance(
                    regime=regime,
                    n_bars=len(mask),
                    total_return=total_ret,
                    sharpe_ratio=sharpe,
                    win_rate=wr,
                )

            profiles[strat_name] = profile

        # Compute regime-optimal weights
        regime_weights = self._compute_regime_weights(profiles, unique_regimes)

        return RegimeAnalysisResult(
            strategy_profiles=profiles,
            regime_weights=regime_weights,
        )

    def _compute_regime_weights(
        self,
        profiles: Dict[str, StrategyRegimeProfile],
        regimes: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute optimal strategy weights per regime.

        Uses Sharpe-weighted allocation: higher Sharpe in a regime
        → higher allocation.  Falls back to default priors when data
        is insufficient.
        """
        weights: Dict[str, Dict[str, float]] = {}

        for regime in regimes:
            sharpes: Dict[str, float] = {}
            for strat_name, profile in profiles.items():
                perf = profile.regimes.get(regime)
                if perf and perf.n_bars >= self.min_regime_bars:
                    sharpes[strat_name] = max(0.01, perf.sharpe_ratio + 1.0)
                    # +1.0 shift so even negative Sharpe gets some weight

            if not sharpes:
                # Use default priors
                weights[regime] = self.DEFAULT_REGIME_TILTS.get(
                    regime,
                    self.DEFAULT_REGIME_TILTS.get("NEUTRAL", {}),
                )
                continue

            # Sharpe-weighted allocation
            total = sum(sharpes.values())
            if total > 0:
                weights[regime] = {s: v / total for s, v in sharpes.items()}
            else:
                n = len(sharpes)
                weights[regime] = {s: 1.0 / n for s in sharpes}

        return weights

    @staticmethod
    def _compute_sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        ann_ret = float(np.mean(returns) * 252)
        ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
        if ann_vol < 1e-8:
            return 0.0
        return (ann_ret - 0.02) / ann_vol
