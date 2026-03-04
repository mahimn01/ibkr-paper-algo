"""
Correlation Regime Switch — cross-asset correlation monitor.

When pairwise correlations spike (risk-off / single-factor dominance),
the tradeable universe is contracted to the most liquid assets
(BTC/USDT, ETH/USDT) to avoid taking correlated bets that all move
together.  When correlations are low (diverse alpha opportunities),
the full universe is available.

Uses both average pairwise correlation and eigenvalue concentration
(top eigenvalue share) as complementary risk indicators.

Standalone — no dependencies on other meta-innovations.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class CorrelationRegimeSwitch:
    """Dynamically contract/expand the tradeable symbol universe based on
    cross-asset correlation regime.

    Parameters
    ----------
    symbols : List[str]
        Full universe of tradeable symbols (e.g. ``["BTC/USDT", ...]``).
    lookback : int
        Number of daily returns used to compute the correlation matrix.
    high_corr_threshold : float
        Average pairwise correlation above this triggers risk-off
        (contract to BTC/ETH only).
    low_corr_threshold : float
        Average pairwise correlation below this triggers full-universe mode.
    safe_symbols : List[str] or None
        Symbols to trade during risk-off.  Defaults to BTC/USDT, ETH/USDT.
    eigenvalue_concentration_limit : float
        If the top eigenvalue accounts for more than this fraction of total
        variance, treat the market as single-factor dominated and contract.
    """

    def __init__(
        self,
        symbols: List[str],
        lookback: int = 30,
        high_corr_threshold: float = 0.70,
        low_corr_threshold: float = 0.35,
        safe_symbols: Optional[List[str]] = None,
        eigenvalue_concentration_limit: float = 0.80,
    ):
        self._symbols = list(symbols)
        self._lookback = lookback
        self._high_corr_threshold = high_corr_threshold
        self._low_corr_threshold = low_corr_threshold
        self._safe_symbols = safe_symbols or ["BTC/USDT", "ETH/USDT"]
        self._eigenvalue_concentration_limit = eigenvalue_concentration_limit

        # Buffer length slightly larger than lookback to tolerate gaps
        buf_len = lookback + 10
        self._daily_returns: Dict[str, deque] = {
            s: deque(maxlen=buf_len) for s in symbols
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_daily_return(self, symbol: str, ret: float) -> None:
        """Record a daily return observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Must be one of the symbols the instance was constructed with.
        ret : float
            Daily simple return (e.g. 0.02 for +2 %).
        """
        if symbol not in self._daily_returns:
            self._daily_returns[symbol] = deque(maxlen=self._lookback + 10)
        self._daily_returns[symbol].append(ret)

    def get_tradeable_symbols(self) -> List[str]:
        """Return the current list of symbols allowed for trading.

        Decision logic
        --------------
        1. Build a correlation matrix from the most recent ``lookback``
           daily returns (only symbols with enough data participate).
        2. Compute average pairwise correlation (mean of off-diagonal).
        3. Compute eigenvalue concentration (top eigenvalue / total).
        4. If ``avg_corr > high_corr_threshold`` **or** eigenvalue
           concentration exceeds its limit, return only the *safe* symbols.
        5. If ``avg_corr < low_corr_threshold``, return the full universe.
        6. In the intermediate zone, return symbols sorted by *lowest*
           correlation to BTC — i.e. prioritise diversifying assets.
        """
        corr_matrix, participating = self._compute_correlation_matrix()

        if corr_matrix is None or len(participating) < 3:
            # Not enough data — allow full universe
            return list(self._symbols)

        avg_corr = self._avg_off_diagonal(corr_matrix)
        eigen_concentration = self._eigenvalue_concentration(corr_matrix)

        # Risk-off: high correlation or single-factor dominance
        if avg_corr > self._high_corr_threshold:
            return self._filter_safe(participating)
        if eigen_concentration > self._eigenvalue_concentration_limit:
            return self._filter_safe(participating)

        # Full diversification available
        if avg_corr < self._low_corr_threshold:
            return list(self._symbols)

        # Intermediate: rank by lowest correlation to BTC
        return self._rank_by_btc_correlation(corr_matrix, participating)

    def get_correlation_info(self) -> Dict:
        """Return a snapshot of the current correlation regime.

        Returns
        -------
        Dict with keys:
            - ``regime``: ``"risk_off"`` | ``"diverse"`` | ``"intermediate"``
            - ``avg_correlation``: float
            - ``eigenvalue_concentration``: float
            - ``tradeable_symbols``: List[str]
            - ``participating_symbols``: List[str] (symbols with enough data)
        """
        corr_matrix, participating = self._compute_correlation_matrix()

        if corr_matrix is None or len(participating) < 3:
            return {
                "regime": "unknown",
                "avg_correlation": float("nan"),
                "eigenvalue_concentration": float("nan"),
                "tradeable_symbols": list(self._symbols),
                "participating_symbols": participating,
            }

        avg_corr = self._avg_off_diagonal(corr_matrix)
        eigen_conc = self._eigenvalue_concentration(corr_matrix)
        tradeable = self.get_tradeable_symbols()

        if avg_corr > self._high_corr_threshold or eigen_conc > self._eigenvalue_concentration_limit:
            regime = "risk_off"
        elif avg_corr < self._low_corr_threshold:
            regime = "diverse"
        else:
            regime = "intermediate"

        return {
            "regime": regime,
            "avg_correlation": round(avg_corr, 4),
            "eigenvalue_concentration": round(eigen_conc, 4),
            "tradeable_symbols": tradeable,
            "participating_symbols": participating,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_correlation_matrix(
        self,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Build a correlation matrix from available daily returns.

        Only symbols with at least ``lookback`` observations are included.
        Returns ``(None, [])`` if fewer than 2 symbols qualify.
        """
        eligible: List[str] = []
        return_vecs: List[np.ndarray] = []

        for sym in self._symbols:
            rets = self._daily_returns.get(sym)
            if rets is None or len(rets) < self._lookback:
                continue
            arr = np.array(list(rets)[-self._lookback :])
            eligible.append(sym)
            return_vecs.append(arr)

        if len(eligible) < 2:
            return None, eligible

        matrix = np.stack(return_vecs)  # shape (n_symbols, lookback)
        # np.corrcoef returns (n, n) correlation matrix
        corr = np.corrcoef(matrix)
        # Guard against NaN from constant series
        corr = np.nan_to_num(corr, nan=0.0)
        return corr, eligible

    @staticmethod
    def _avg_off_diagonal(corr: np.ndarray) -> float:
        """Mean of off-diagonal elements of a square correlation matrix."""
        n = corr.shape[0]
        if n < 2:
            return 0.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(corr[mask]))

    @staticmethod
    def _eigenvalue_concentration(corr: np.ndarray) -> float:
        """Fraction of total variance explained by the top eigenvalue."""
        eigenvalues = np.linalg.eigvalsh(corr)
        total = np.sum(np.abs(eigenvalues))
        if total < 1e-12:
            return 0.0
        return float(np.max(np.abs(eigenvalues)) / total)

    def _filter_safe(self, participating: List[str]) -> List[str]:
        """Return safe symbols that are also in the participating set."""
        safe = [s for s in self._safe_symbols if s in participating]
        # If none of the safe symbols have data, fall back to whatever we have
        return safe if safe else participating[:2]

    def _rank_by_btc_correlation(
        self, corr: np.ndarray, participating: List[str]
    ) -> List[str]:
        """Return all symbols sorted by ascending correlation to BTC.

        If BTC is not in the participating set, return symbols as-is.
        """
        btc_candidates = [s for s in participating if "BTC" in s.upper()]
        if not btc_candidates:
            return list(self._symbols)

        btc_idx = participating.index(btc_candidates[0])
        btc_row = corr[btc_idx]

        # Pair each symbol with its correlation to BTC, sort ascending
        ranked = sorted(
            zip(participating, btc_row),
            key=lambda pair: pair[1],
        )
        return [sym for sym, _ in ranked]
