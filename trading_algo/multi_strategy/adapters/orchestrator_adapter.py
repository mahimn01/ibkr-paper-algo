"""
Adapter wrapping the production Orchestrator for multi-strategy use.

The Orchestrator already has a well-defined interface (update_asset /
generate_signal) so this adapter is thin — it just translates the
OrchestratorSignal into a StrategySignal.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.orchestrator.config import OrchestratorConfig
from trading_algo.orchestrator.strategy import Orchestrator, create_orchestrator

logger = logging.getLogger(__name__)


class OrchestratorStrategyAdapter(TradingStrategy):
    """
    Wraps the 6-edge Orchestrator as a TradingStrategy.

    The Orchestrator manages its own positions internally, so we
    translate its buy/sell/hold signals into StrategySignal objects.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self._orchestrator = create_orchestrator(config)
        self._state = StrategyState.WARMING_UP
        self._bar_count = 0
        self._warmup_bars = (config or OrchestratorConfig()).warmup_bars

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Orchestrator"

    @property
    def state(self) -> StrategyState:
        return self._state

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self._orchestrator.update_asset(
            symbol, timestamp, open_price, high, low, close, volume,
        )
        self._bar_count += 1
        if self._state == StrategyState.WARMING_UP and self._bar_count >= self._warmup_bars:
            self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        signals: List[StrategySignal] = []
        ref = self._orchestrator.reference_assets

        for symbol in symbols:
            if symbol in ref:
                continue  # Don't trade reference assets

            sig = self._orchestrator.generate_signal(symbol, timestamp)
            if sig.action in ("buy", "short"):
                direction = 1 if sig.action == "buy" else -1
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=direction,
                    target_weight=sig.size,
                    confidence=sig.confidence,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    entry_price=sig.entry_price,
                    trade_type=sig.trade_type.name if sig.trade_type else "ensemble",
                    metadata={
                        "consensus_score": sig.consensus_score,
                        "regime": sig.market_regime.name if sig.market_regime else "UNKNOWN",
                        "reason": sig.reason,
                    },
                ))
            elif sig.action in ("sell", "cover"):
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=0,  # exit signal
                    target_weight=0.0,
                    confidence=sig.confidence,
                    entry_price=sig.entry_price,
                    trade_type="exit",
                    metadata={"reason": sig.reason},
                ))

        return signals

    def get_current_exposure(self) -> float:
        n_positions = len(self._orchestrator.positions)
        max_pos = self._orchestrator.max_position_pct
        return n_positions * max_pos

    def get_performance_stats(self) -> Dict[str, float]:
        return self._orchestrator.trade_stats

    def reset(self) -> None:
        self._orchestrator.clear_positions()
        self._state = StrategyState.WARMING_UP
        self._bar_count = 0
