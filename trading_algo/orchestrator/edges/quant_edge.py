"""
Quant Edge — Bridges quant_core model signals into the Orchestrator.

This edge wraps the quant_core SignalAggregator (Ornstein-Uhlenbeck,
Time-Series Momentum, Volatility-Managed Momentum, HMM regime) and
translates its continuous signal into the Orchestrator's discrete
EdgeVote system.

By adding this as a 7th edge, the production Orchestrator benefits from
the quantitative models without replacing the existing 6-edge consensus.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from trading_algo.orchestrator.types import AssetState, EdgeSignal, EdgeVote

logger = logging.getLogger(__name__)

# Lazy-imported to avoid hard dependency on scipy/numpy at module level.
_aggregator_cls = None
_aggregator_cfg_cls = None


def _ensure_imports():
    global _aggregator_cls, _aggregator_cfg_cls
    if _aggregator_cls is not None:
        return
    from trading_algo.quant_core.engine.signal_aggregator import (
        AggregatorConfig,
        SignalAggregator,
    )
    _aggregator_cls = SignalAggregator
    _aggregator_cfg_cls = AggregatorConfig


class QuantEdge:
    """
    7th edge that wraps quant_core's SignalAggregator.

    Converts a continuous signal in [-1, 1] with a confidence in [0, 1]
    into an EdgeVote consumable by the Orchestrator consensus system.

    Vote mapping:
        signal >=  0.5 and confidence >= 0.6  → STRONG_LONG
        signal >=  0.15                        → LONG
        signal <= -0.5 and confidence >= 0.6  → STRONG_SHORT
        signal <= -0.15                        → SHORT
        otherwise                              → NEUTRAL
    """

    # Thresholds for translating continuous signal → discrete vote.
    STRONG_SIGNAL: float = 0.5
    WEAK_SIGNAL: float = 0.15
    STRONG_CONF: float = 0.6

    def __init__(self, min_bars: int = 60) -> None:
        _ensure_imports()
        self._aggregator = _aggregator_cls(_aggregator_cfg_cls())
        self._initialized_symbols: set[str] = set()
        self._min_bars = min_bars
        self._benchmark_fitted = False

    # ── public API (matches edge interface pattern) ───────────────────

    def update(self, symbol: str, state: AssetState) -> None:
        """Register a symbol with the aggregator on first sight."""
        if symbol not in self._initialized_symbols:
            self._aggregator.initialize([symbol])
            self._initialized_symbols.add(symbol)

    def update_regime(self, benchmark_prices: np.ndarray) -> None:
        """Feed SPY (or benchmark) prices into the HMM regime model."""
        if len(benchmark_prices) >= self._min_bars:
            try:
                self._aggregator.update_regime(benchmark_prices)
                self._benchmark_fitted = True
            except Exception:
                logger.debug("QuantEdge: HMM regime update failed", exc_info=True)

    def get_vote(
        self,
        symbol: str,
        state: AssetState,
        intended_direction: Optional[str] = None,
    ) -> EdgeSignal:
        """
        Generate an EdgeSignal from the quant_core blended signal.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        state : AssetState
            Current asset state (provides price/volume history).
        intended_direction : str | None
            'long' or 'short' — used to boost agreement when the quant
            signal aligns with the Orchestrator's planned direction.

        Returns
        -------
        EdgeSignal
        """
        prices = np.array(list(state.prices), dtype=np.float64)
        if len(prices) < self._min_bars:
            return EdgeSignal(
                "Quant", EdgeVote.NEUTRAL, 0.3,
                f"Insufficient data ({len(prices)}/{self._min_bars} bars)",
            )

        if symbol not in self._initialized_symbols:
            self.update(symbol, state)

        try:
            agg_signal = self._aggregator.generate_signal(symbol, prices)
        except Exception:
            logger.debug("QuantEdge: signal generation failed for %s", symbol, exc_info=True)
            return EdgeSignal("Quant", EdgeVote.NEUTRAL, 0.3, "Model error")

        sig = float(agg_signal.signal)
        conf = float(agg_signal.confidence)
        regime_name = str(agg_signal.market_regime.name) if agg_signal.market_regime else "UNKNOWN"

        vote = self._map_vote(sig, conf)
        reason = (
            f"signal={sig:+.3f} conf={conf:.2f} regime={regime_name}"
        )
        return EdgeSignal("Quant", vote, conf, reason, data={
            "signal": sig,
            "confidence": conf,
            "regime": regime_name,
        })

    # ── internal helpers ──────────────────────────────────────────────

    @classmethod
    def _map_vote(cls, signal: float, confidence: float) -> EdgeVote:
        if signal >= cls.STRONG_SIGNAL and confidence >= cls.STRONG_CONF:
            return EdgeVote.STRONG_LONG
        if signal <= -cls.STRONG_SIGNAL and confidence >= cls.STRONG_CONF:
            return EdgeVote.STRONG_SHORT
        if signal >= cls.WEAK_SIGNAL:
            return EdgeVote.LONG
        if signal <= -cls.WEAK_SIGNAL:
            return EdgeVote.SHORT
        return EdgeVote.NEUTRAL
