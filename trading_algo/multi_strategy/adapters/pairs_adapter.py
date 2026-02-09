"""
Adapter wrapping the Pairs Trading (Statistical Arbitrage) strategy.

Pairs trading is market-neutral: it simultaneously longs one stock and
shorts its cointegrated pair.  The adapter maps the pair-level signals
into per-symbol StrategySignals so the controller can manage them
alongside directional strategies.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.intraday.pairs_trading import (
    PairsTradingStrategy,
    PairConfig,
)

logger = logging.getLogger(__name__)


class PairsStrategyAdapter(TradingStrategy):
    """
    Wraps PairsTradingStrategy as a TradingStrategy.

    The pairs strategy emits *two* StrategySignals per trade (one long,
    one short) to ensure the controller can track each leg separately.
    """

    # Minimum price history bars before the strategy activates
    MIN_WARMUP_BARS = 60

    def __init__(self, config: Optional[PairConfig] = None):
        self._pairs = PairsTradingStrategy(config)
        self._config = config or PairConfig()
        self._state = StrategyState.WARMING_UP

        # Price history for cointegration / z-score calculation
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._current_prices: Dict[str, float] = {}
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "PairsTrading"

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
        self._price_history[symbol].append(close)
        self._current_prices[symbol] = close
        self._bars_per_symbol[symbol] += 1

        # Keep bounded history
        max_len = self._config.lookback_period + 20
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

        # Activate once we have enough history for at least 2 symbols
        if self._state == StrategyState.WARMING_UP:
            ready = sum(
                1 for n in self._bars_per_symbol.values()
                if n >= self.MIN_WARMUP_BARS
            )
            if ready >= 2:
                self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        # Build price_data dict expected by PairsTradingStrategy
        price_data = {
            s: np.array(h, dtype=np.float64)
            for s, h in self._price_history.items()
            if len(h) >= self._config.lookback_period
        }
        if len(price_data) < 2:
            return []

        # First update existing positions (check exits)
        self._pairs.update_positions(
            self._current_prices, price_data, timestamp,
        )

        # Generate new pair signals
        pair_signals = self._pairs.generate_signals(
            symbols=list(price_data.keys()),
            price_data=price_data,
            current_prices=self._current_prices,
            current_date=timestamp,
        )

        signals: List[StrategySignal] = []
        for ps in pair_signals:
            action = ps.get("action", "")
            stock_a = ps.get("stock_a", "")
            stock_b = ps.get("stock_b", "")
            zscore = abs(ps.get("zscore", 0.0))
            half_weight = self._config.position_size / 2

            if action == "long_spread":
                # Long stock_a, short stock_b
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=stock_a,
                    direction=1,
                    target_weight=half_weight,
                    confidence=min(1.0, zscore / 3.0),
                    entry_price=ps.get("price_a"),
                    trade_type="pairs_long_leg",
                    metadata={"pair": ps.get("pair_name"), "zscore": ps.get("zscore")},
                ))
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=stock_b,
                    direction=-1,
                    target_weight=half_weight,
                    confidence=min(1.0, zscore / 3.0),
                    entry_price=ps.get("price_b"),
                    trade_type="pairs_short_leg",
                    metadata={"pair": ps.get("pair_name"), "zscore": ps.get("zscore")},
                ))
            elif action == "short_spread":
                # Short stock_a, long stock_b
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=stock_a,
                    direction=-1,
                    target_weight=half_weight,
                    confidence=min(1.0, zscore / 3.0),
                    entry_price=ps.get("price_a"),
                    trade_type="pairs_short_leg",
                    metadata={"pair": ps.get("pair_name"), "zscore": ps.get("zscore")},
                ))
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=stock_b,
                    direction=1,
                    target_weight=half_weight,
                    confidence=min(1.0, zscore / 3.0),
                    entry_price=ps.get("price_b"),
                    trade_type="pairs_long_leg",
                    metadata={"pair": ps.get("pair_name"), "zscore": ps.get("zscore")},
                ))

        return signals

    def get_current_exposure(self) -> float:
        n_pairs = len(self._pairs.positions) if hasattr(self._pairs, "positions") else 0
        return n_pairs * self._config.position_size

    def get_performance_stats(self) -> Dict[str, float]:
        try:
            return self._pairs.get_performance_stats()
        except Exception:
            return {}

    def reset(self) -> None:
        self._pairs = PairsTradingStrategy(self._config)
        self._price_history.clear()
        self._current_prices.clear()
        self._bars_per_symbol.clear()
        self._state = StrategyState.WARMING_UP
