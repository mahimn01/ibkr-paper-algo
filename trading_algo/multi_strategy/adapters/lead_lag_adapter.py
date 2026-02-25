"""
Adapter wrapping the Lead-Lag Arbitrage strategy.

Lead-Lag Arbitrage exploits cross-asset information flow at 5-minute
frequency.  It monitors known leader-follower pairs with empirically
discovered lag structures.  When a leader makes a statistically
significant move, it enters the follower in the direction predicted
by the sign of the lagged cross-correlation.

The adapter feeds per-symbol OHLCV data into the underlying
LeadLagArbitrage strategy, manages the warmup lifecycle, and maps
raw signal dicts into per-symbol StrategySignals for the multi-strategy
controller.

Symbols not part of any configured pair are silently accepted (their
prices are stored in case they serve as leaders or followers in
future pair definitions).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.lead_lag_arbitrage import (
    LeadLagArbitrage,
    LeadLagConfig,
)

logger = logging.getLogger(__name__)


class LeadLagAdapter(TradingStrategy):
    """
    Wraps LeadLagArbitrage as a TradingStrategy for the multi-strategy
    controller.

    Needs at least ``config.warmup`` bars (default 100) per symbol
    before producing signals.  After warmup, it emits one signal per
    active lead-lag pair that triggers.

    The underlying strategy manages its own position tracking and exit
    logic.  This adapter is responsible only for data feeding and signal
    format translation.
    """

    def __init__(self, config: Optional[LeadLagConfig] = None) -> None:
        self._config = config or LeadLagConfig()
        self._strategy = LeadLagArbitrage(self._config)
        self._state = StrategyState.WARMING_UP

        # Track bars per symbol for warmup detection
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

        # Cache the latest close prices for entry_price metadata
        self._current_prices: Dict[str, float] = {}

    # -- Protocol implementation ----------------------------------------

    @property
    def name(self) -> str:
        return "LeadLagArbitrage"

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
        """Feed a single OHLCV bar to the inner strategy.

        The adapter tracks bar counts for warmup management and
        delegates data storage to the underlying LeadLagArbitrage.

        Args:
            symbol:     Ticker/instrument identifier.
            timestamp:  Bar timestamp.
            open_price: Open price for this bar.
            high:       High price for this bar.
            low:        Low price for this bar.
            close:      Close price for this bar.
            volume:     Trade volume for this bar.
        """
        # Feed the underlying strategy
        self._strategy.update(symbol, close, high, low, volume)

        # Track state for warmup
        self._bars_per_symbol[symbol] += 1
        self._current_prices[symbol] = close

        # Activate once all symbols in at least one pair have enough history
        if self._state == StrategyState.WARMING_UP:
            warmup_threshold = self._config.warmup
            for leader, follower, _lag, _sign in self._config.pairs:
                leader_bars = self._bars_per_symbol.get(leader, 0)
                follower_bars = self._bars_per_symbol.get(follower, 0)
                if (leader_bars >= warmup_threshold
                        and follower_bars >= warmup_threshold):
                    self._state = StrategyState.ACTIVE
                    logger.info(
                        "LeadLagAdapter activated: pair (%s -> %s) "
                        "has %d / %d bars",
                        leader, follower, leader_bars, follower_bars,
                    )
                    break

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        """Generate signals by delegating to the inner strategy.

        Translates raw signal dicts from LeadLagArbitrage into
        StrategySignal objects that the controller can consume.

        Args:
            symbols:   List of symbols in the current universe.
            timestamp: Timestamp of the current bar.

        Returns:
            List of StrategySignal objects.  Empty list if still
            warming up or if no pairs triggered.
        """
        if self._state != StrategyState.ACTIVE:
            return []

        # Generate raw signals from the inner strategy
        try:
            raw_signals = self._strategy.generate_signals(symbols, timestamp)
        except Exception:
            logger.debug(
                "Lead-lag signal generation failed", exc_info=True,
            )
            return []

        # Map raw dicts to StrategySignal objects
        signals: List[StrategySignal] = []
        for raw in raw_signals:
            direction = raw.get("direction", 0)
            symbol = raw.get("symbol", "")
            confidence = raw.get("confidence", 0.0)
            weight = raw.get("weight", self._config.per_pair_weight)
            metadata = raw.get("metadata", {})

            # Determine trade type from the metadata action field
            action = metadata.get("action", "entry")
            if action == "exit":
                trade_type = "lead_lag_exit"
            else:
                # Classify by correlation sign
                corr = metadata.get("corr", 0.0)
                expected_sign = metadata.get("expected_sign", 1)
                if expected_sign < 0 or corr < 0:
                    trade_type = "lead_lag_mean_reversion"
                else:
                    trade_type = "lead_lag_continuation"

            # Get entry price from current prices
            entry_price = self._current_prices.get(symbol)

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=direction,
                    target_weight=weight,
                    confidence=confidence,
                    entry_price=entry_price,
                    trade_type=trade_type,
                    metadata=metadata,
                )
            )

        return signals

    def get_current_exposure(self) -> float:
        """Return gross exposure based on active trade count.

        Each active trade contributes one ``per_pair_weight`` unit
        of exposure.
        """
        return self._strategy.active_trade_count * self._config.per_pair_weight

    def get_performance_stats(self) -> Dict[str, float]:
        """Return strategy-level performance statistics."""
        return {
            "active_trades": float(self._strategy.active_trade_count),
            "pairs_configured": float(len(self._config.pairs)),
        }

    def reset(self) -> None:
        """Reset all internal state for a new session or backtest run."""
        self._strategy.reset()
        self._bars_per_symbol.clear()
        self._current_prices.clear()
        self._state = StrategyState.WARMING_UP
