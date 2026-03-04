"""
Alpha Decay Detector — per-edge rolling performance tracker.

Monitors each edge's rolling Sharpe ratio and adjusts weight multipliers
dynamically.  When an edge's alpha is decaying (rolling Sharpe declining),
its allocation is reduced.  When an edge is running hot, allocation is
increased.  This prevents the portfolio from staying anchored to stale
edge weights that no longer reflect reality.

Standalone — no dependencies on other meta-innovations.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional


class AlphaDecayDetector:
    """Track per-edge rolling performance and dynamically adjust weights.

    Parameters
    ----------
    edge_names : List[str]
        Names of all edges being tracked.
    rolling_window : int
        Number of daily PnL observations used for rolling Sharpe.
    min_sharpe : float
        Rolling Sharpe below this triggers the decay penalty.
    decay_penalty : float
        Weight multiplier applied to decaying edges (< 1.0 to reduce).
    hot_bonus : float
        Weight multiplier applied to hot edges (> 1.0 to increase).
    """

    def __init__(
        self,
        edge_names: List[str],
        rolling_window: int = 60,
        min_sharpe: float = 0.3,
        decay_penalty: float = 0.5,
        hot_bonus: float = 1.3,
    ):
        self._edge_daily_pnl: Dict[str, deque] = {
            name: deque(maxlen=rolling_window) for name in edge_names
        }
        self._rolling_window = rolling_window
        self._min_sharpe = min_sharpe
        self._decay_penalty = decay_penalty
        self._hot_bonus = hot_bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_edge_pnl(self, edge_name: str, daily_pnl: float) -> None:
        """Record a day's PnL for an edge.

        Parameters
        ----------
        edge_name : str
            Must match one of the names passed at construction.
        daily_pnl : float
            Net PnL for the day (in portfolio-currency units or % return).
        """
        if edge_name not in self._edge_daily_pnl:
            self._edge_daily_pnl[edge_name] = deque(maxlen=self._rolling_window)
        self._edge_daily_pnl[edge_name].append(daily_pnl)

    def get_weight_multipliers(self) -> Dict[str, float]:
        """Return a weight multiplier for every tracked edge.

        Returns
        -------
        Dict[str, float]
            Mapping of edge_name -> multiplier in the range
            [decay_penalty * 0.8, hot_bonus].

        Logic
        -----
        1. Compute rolling Sharpe for each edge over the full window.
        2. ``rolling_sharpe < min_sharpe``  ->  ``multiplier = decay_penalty``
        3. ``rolling_sharpe > 2 * min_sharpe``  ->  ``multiplier = hot_bonus``
        4. Otherwise  ->  ``multiplier = 1.0``
        5. Sharpe *trend* check — compare the last-20-day rolling Sharpe to
           the last-40-day rolling Sharpe.  If the short window is lower
           (alpha declining), apply an additional 0.8x penalty.
        """
        multipliers: Dict[str, float] = {}

        for name, pnl_deque in self._edge_daily_pnl.items():
            pnl_list = list(pnl_deque)

            # Need a minimum number of observations to be meaningful
            if len(pnl_list) < 10:
                multipliers[name] = 1.0
                continue

            full_sharpe = self._annualized_sharpe(pnl_list)

            # Base multiplier from absolute Sharpe level
            if full_sharpe < self._min_sharpe:
                mult = self._decay_penalty
            elif full_sharpe > 2.0 * self._min_sharpe:
                mult = self._hot_bonus
            else:
                mult = 1.0

            # Trend adjustment: compare recent vs longer-term Sharpe
            if len(pnl_list) >= 40:
                short_sharpe = self._annualized_sharpe(pnl_list[-20:])
                long_sharpe = self._annualized_sharpe(pnl_list[-40:])
                if short_sharpe < long_sharpe:
                    mult *= 0.8

            multipliers[name] = mult

        return multipliers

    def get_decay_report(self) -> Dict[str, Dict]:
        """Return a detailed per-edge performance report.

        Returns
        -------
        Dict[str, Dict]
            Keyed by edge name, each value contains:
            - ``observations``: number of daily PnL records
            - ``rolling_sharpe``: annualized Sharpe over full window
            - ``sharpe_20d``: annualized Sharpe over last 20 days
            - ``sharpe_40d``: annualized Sharpe over last 40 days
            - ``trend``: ``"declining"`` | ``"improving"`` | ``"stable"``
            - ``multiplier``: the current weight multiplier
        """
        multipliers = self.get_weight_multipliers()
        report: Dict[str, Dict] = {}

        for name, pnl_deque in self._edge_daily_pnl.items():
            pnl_list = list(pnl_deque)
            obs = len(pnl_list)
            full_sharpe = self._annualized_sharpe(pnl_list) if obs >= 10 else float("nan")

            sharpe_20 = float("nan")
            sharpe_40 = float("nan")
            trend = "stable"

            if obs >= 20:
                sharpe_20 = self._annualized_sharpe(pnl_list[-20:])
            if obs >= 40:
                sharpe_40 = self._annualized_sharpe(pnl_list[-40:])
                if sharpe_20 < sharpe_40:
                    trend = "declining"
                elif sharpe_20 > sharpe_40:
                    trend = "improving"

            report[name] = {
                "observations": obs,
                "rolling_sharpe": full_sharpe,
                "sharpe_20d": sharpe_20,
                "sharpe_40d": sharpe_40,
                "trend": trend,
                "multiplier": multipliers.get(name, 1.0),
            }

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _annualized_sharpe(pnl: List[float], trading_days: int = 365) -> float:
        """Compute annualized Sharpe from a list of daily PnL values.

        Crypto markets trade 365 days/year so we annualize accordingly.
        Returns 0.0 if standard deviation is negligible to avoid division
        by zero.
        """
        if len(pnl) < 2:
            return 0.0
        mean = sum(pnl) / len(pnl)
        var = sum((x - mean) ** 2 for x in pnl) / (len(pnl) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-12:
            return 0.0
        return (mean / std) * math.sqrt(trading_days)
