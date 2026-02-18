"""
Adapter wrapping the Cross-Asset Divergence strategy.

Cross-Asset Divergence exploits the empirical lead of credit and bond
markets over equities.  When equity momentum and credit/bond momentum
diverge, the credit market is almost always right.  The strategy
monitors multiple asset pairs across multiple timeframes, detects
statistically significant divergences, and generates sized signals.

The adapter feeds per-symbol close prices into the underlying
CrossAssetDivergenceStrategy, builds numpy price arrays, and maps
DivergenceSignal objects into per-symbol StrategySignals for the
controller.  Symbols not present in the configured asset pairs are
silently accepted (their prices are stored and may serve as reference
instruments for divergence calculations).
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
from trading_algo.quant_core.strategies.cross_asset_divergence import (
    CrossAssetDivergenceStrategy,
    DivergenceConfig,
)

logger = logging.getLogger(__name__)

# Minimum bars before the strategy can activate.
_WARMUP_BARS = 60


class CrossAssetDivergenceAdapter(TradingStrategy):
    """
    Wraps CrossAssetDivergenceStrategy as a TradingStrategy.

    Needs at least ~60 bars of price data for the configured asset pairs
    before divergence z-scores can be computed reliably.  After warmup
    it emits one signal per target symbol in each divergent pair.

    The underlying strategy's ``update()`` method accepts a symbol and
    close price; ``generate_signals()`` accepts a dict of symbol to
    numpy price arrays.  This adapter accumulates close-price histories
    and bridges the two APIs.
    """

    def __init__(self, config: Optional[DivergenceConfig] = None):
        self._strategy = CrossAssetDivergenceStrategy(config)
        self._config = config or DivergenceConfig()
        self._state = StrategyState.WARMING_UP

        # Per-symbol close-price history (mirrored to strategy.update)
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

        # Track last target weights for exposure calculation
        self._last_weights: Dict[str, float] = {}

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "CrossAssetDivergence"

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
        # Feed the underlying strategy (it only needs symbol + close)
        self._strategy.update(symbol, close)

        # Keep our own history for generate_signals
        self._price_history[symbol].append(close)
        self._bars_per_symbol[symbol] += 1

        # Trim to bounded length
        max_len = (
            max(
                max(self._config.lookback_windows),
                max(self._config.confirmation_timeframes),
                self._config.zscore_lookback,
                self._config.correlation_lookback,
                self._config.vol_lookback,
            )
            + self._config.zscore_lookback
            + 30
        )
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

        # Activate once we have enough history for at least one asset pair
        if self._state == StrategyState.WARMING_UP:
            pair_symbols = set()
            for target, reference in self._config.asset_pairs:
                pair_symbols.add(target)
                pair_symbols.add(reference)

            ready_pairs = 0
            for target, reference in self._config.asset_pairs:
                t_bars = self._bars_per_symbol.get(target, 0)
                r_bars = self._bars_per_symbol.get(reference, 0)
                if t_bars >= _WARMUP_BARS and r_bars >= _WARMUP_BARS:
                    ready_pairs += 1

            if ready_pairs >= 1:
                self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        # Build numpy price arrays from our history
        prices: Dict[str, np.ndarray] = {}
        for s in self._price_history:
            hist = self._price_history[s]
            if len(hist) >= self._config.min_history:
                prices[s] = np.array(hist, dtype=np.float64)

        if not prices:
            return []

        # Generate divergence signals
        try:
            div_signals = self._strategy.generate_signals(prices)
        except Exception:
            logger.debug(
                "Cross-asset divergence signal generation failed",
                exc_info=True,
            )
            return []

        # Also compute target weights for exposure tracking
        try:
            self._last_weights = self._strategy.get_target_weights(prices)
        except Exception:
            self._last_weights = {}

        signals: List[StrategySignal] = []
        for ds in div_signals:
            # Map continuous direction [-1, +1] to discrete
            if ds.direction > 0:
                direction = 1
            elif ds.direction < 0:
                direction = -1
            else:
                continue

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    symbol=ds.target_symbol,
                    direction=direction,
                    target_weight=ds.position_size,
                    confidence=ds.confidence,
                    entry_price=float(prices[ds.target_symbol][-1])
                    if ds.target_symbol in prices
                    else None,
                    trade_type="cross_asset_divergence",
                    metadata={
                        "reference_symbol": ds.reference_symbol,
                        "divergence_score": ds.divergence_score,
                        "timeframe_agreement": ds.timeframe_agreement,
                        "holding_period_est": ds.holding_period_est,
                    },
                )
            )

        return signals

    def get_current_exposure(self) -> float:
        return sum(abs(w) for w in self._last_weights.values())

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self._strategy.reset()
        self._price_history.clear()
        self._bars_per_symbol.clear()
        self._last_weights.clear()
        self._state = StrategyState.WARMING_UP
