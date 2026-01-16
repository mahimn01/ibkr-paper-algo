"""
RAT Engine: Main orchestrator for Reflexive Attention Topology framework.

Coordinates all five modules:
1. Attention Flow - Market attention tracking
2. Reflexivity Meter - Soros-style feedback detection
3. Topology Detector - Regime classification via persistent homology
4. Adversarial Detector - Algorithm pattern recognition
5. Alpha Tracker/Mutator - Self-cannibalizing alpha management

Plus signal combination and filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from trading_algo.rat.config import RATConfig
from trading_algo.rat.signals import Signal, SignalType, SignalSource

# Module imports
from trading_algo.rat.attention.tracker import AttentionTracker
from trading_algo.rat.reflexivity.meter import ReflexivityMeter, ReflexivityStage
from trading_algo.rat.topology.detector import TopologyDetector, TopologyRegime
from trading_algo.rat.adversarial.detector import AdversarialDetector
from trading_algo.rat.alpha.tracker import AlphaTracker, DecayStage
from trading_algo.rat.alpha.mutator import AlphaMutator
from trading_algo.rat.combiner.combiner import SignalCombiner, CombinedDecision
from trading_algo.rat.combiner.filters import SignalFilter


logger = logging.getLogger(__name__)


@dataclass
class RATState:
    """Complete state of the RAT engine."""

    timestamp: datetime
    symbol: str

    # Module states
    attention_score: float
    reflexivity_stage: ReflexivityStage
    topology_regime: TopologyRegime
    adversarial_archetype: Optional[str]
    alpha_health: float

    # Signals from each module
    signals: Dict[SignalSource, Optional[Signal]]

    # Combined decision
    decision: Optional[CombinedDecision]

    # Position info
    current_position: float
    suggested_action: str
    suggested_size: float


@dataclass
class Position:
    """Track current position."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class RATEngine:
    """
    Main RAT trading engine.

    Orchestrates all modules and produces trading decisions.
    Designed for both live trading and backtesting.
    """

    def __init__(
        self,
        config: RATConfig,
        broker: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config
        self.broker = broker
        self.llm_client = llm_client

        # Initialize all modules
        self._attention = AttentionTracker(
            config=config.attention,
            broker=broker,
        )

        self._reflexivity = ReflexivityMeter(
            lookback=config.reflexivity.lookback,
            lag_order=config.reflexivity.lag_order,
            significance_level=config.reflexivity.significance_level,
        )

        self._topology = TopologyDetector(
            embedding_dim=config.topology.embedding_dim,
            time_delay=config.topology.time_delay,
            max_dimension=config.topology.max_dimension,
        )

        self._adversarial = AdversarialDetector(
            flow_window=config.adversarial.flow_window,
            detection_threshold=config.adversarial.detection_threshold,
        )

        self._alpha_tracker = AlphaTracker(
            sharpe_window=config.alpha.sharpe_window,
            ic_window=config.alpha.ic_window,
            decay_threshold=config.alpha.decay_threshold,
        )

        self._alpha_mutator = AlphaMutator(
            tracker=self._alpha_tracker,
            llm_client=llm_client,
            enable_llm=config.alpha.enable_llm_mutation,
            llm_cooldown_hours=config.alpha.llm_cooldown_hours,
        )

        self._combiner = SignalCombiner(
            weighting_method=config.signal.weighting_method,
            min_signals_required=config.signal.min_signals_required,
            agreement_threshold=config.signal.agreement_threshold,
            max_position_pct=config.signal.max_position_pct,
        )

        self._filter = SignalFilter(
            confidence_threshold=config.signal.confidence_threshold,
            max_signals_per_hour=config.signal.max_signals_per_hour,
        )

        # Engine state
        self._positions: Dict[str, Position] = {}
        self._last_state: Dict[str, RATState] = {}
        self._is_running = False

        # Performance tracking
        self._total_pnl = 0.0
        self._trade_count = 0

        # Initialize default alpha factors
        self._initialize_default_factors()

    def _initialize_default_factors(self) -> None:
        """Initialize default alpha factors for trading."""
        # Generate initial factors from templates
        self._alpha_mutator.generate_new_factors(
            count=self.config.alpha.initial_factors,
            template_name=None,  # Use all templates
        )

    def start(self) -> None:
        """Start the engine."""
        self._is_running = True
        logger.info("RAT Engine started")

    def stop(self) -> None:
        """Stop the engine."""
        self._is_running = False
        logger.info("RAT Engine stopped")

    def process_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[RATState]:
        """
        Process a single market tick.

        Updates all modules and produces trading decision.
        """
        ts = timestamp or datetime.now()

        # Update all modules with new data
        self._update_modules(symbol, price, volume, bid, ask, ts)

        # Generate signals from each module
        signals = self._generate_signals(symbol)

        # Get current regime for context
        topo_state = self._topology.detect(symbol)
        current_regime = topo_state.regime.name if topo_state else None

        # Filter signals
        filtered_signals = self._filter.filter_batch(
            [s for s in signals.values() if s is not None],
            timestamp=ts,
        )

        # Combine signals into decision
        decision = None
        if filtered_signals:
            decision = self._combiner.combine(
                filtered_signals,
                current_regime=current_regime,
            )

        # Update filter with current regime
        if current_regime:
            self._filter.set_regime(current_regime)

        # Build state
        state = self._build_state(symbol, ts, signals, decision)
        self._last_state[symbol] = state

        # Execute if appropriate
        if decision and decision.should_trade(self.config.signal.confidence_threshold):
            self._execute_decision(symbol, decision, price, ts)

        return state

    def _update_modules(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float,
        ask: float,
        timestamp: datetime,
    ) -> None:
        """Update all modules with new market data."""
        # Estimate buy/sell split (simplified)
        mid = (bid + ask) / 2
        if price > mid:
            aggressor = "buy"
            buy_vol, sell_vol = volume, 0.0
        else:
            aggressor = "sell"
            buy_vol, sell_vol = 0.0, volume

        # Update attention tracker
        self._attention.process_snapshot({
            "symbol": symbol,
            "last": price,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": timestamp,
        })

        # Update reflexivity meter
        self._reflexivity.update(
            symbol=symbol,
            price=price,
            fundamental=price,  # Simplified - would use actual fundamental
            timestamp=timestamp,
        )

        # Update topology detector
        self._topology.update(symbol, price, timestamp)

        # Update adversarial detector
        self._adversarial.update(
            symbol=symbol,
            price=price,
            volume=volume,
            aggressor=aggressor,
            bid=bid,
            ask=ask,
            timestamp=timestamp,
        )

    def _generate_signals(self, symbol: str) -> Dict[SignalSource, Optional[Signal]]:
        """Generate signals from all modules."""
        signals: Dict[SignalSource, Optional[Signal]] = {}

        # Attention signal
        try:
            signals[SignalSource.ATTENTION] = self._attention.generate_signal(symbol)
        except Exception as e:
            logger.debug(f"Attention signal error: {e}")
            signals[SignalSource.ATTENTION] = None

        # Reflexivity signal
        try:
            signals[SignalSource.REFLEXIVITY] = self._reflexivity.generate_signal(symbol)
        except Exception as e:
            logger.debug(f"Reflexivity signal error: {e}")
            signals[SignalSource.REFLEXIVITY] = None

        # Topology signal
        try:
            signals[SignalSource.TOPOLOGY] = self._topology.generate_signal(symbol)
        except Exception as e:
            logger.debug(f"Topology signal error: {e}")
            signals[SignalSource.TOPOLOGY] = None

        # Adversarial signal
        try:
            signals[SignalSource.ADVERSARIAL] = self._adversarial.generate_signal(symbol)
        except Exception as e:
            logger.debug(f"Adversarial signal error: {e}")
            signals[SignalSource.ADVERSARIAL] = None

        # Alpha signal
        try:
            # Build data dict for alpha factors
            data = self._build_alpha_data(symbol)
            signals[SignalSource.ALPHA] = self._alpha_tracker.generate_signal(symbol, data)
        except Exception as e:
            logger.debug(f"Alpha signal error: {e}")
            signals[SignalSource.ALPHA] = None

        return signals

    def _build_alpha_data(self, symbol: str) -> Dict[str, Any]:
        """Build data dictionary for alpha factor computation."""
        # This would pull from historical data store
        # Simplified for now
        return {
            "prices": list(self._topology._price_history.get(symbol, [])),
            "volumes": [],
            "buy_volume": [],
            "sell_volume": [],
        }

    def _build_state(
        self,
        symbol: str,
        timestamp: datetime,
        signals: Dict[SignalSource, Optional[Signal]],
        decision: Optional[CombinedDecision],
    ) -> RATState:
        """Build complete engine state."""
        # Get module states
        attention_state = self._attention._last_state.get(symbol)
        reflexivity_state = self._reflexivity._last_state.get(symbol)
        topology_state = self._topology._last_state.get(symbol)
        adversarial_state = self._adversarial._last_state.get(symbol)
        alpha_state = self._alpha_tracker.analyze()

        # Current position
        position = self._positions.get(symbol)
        current_qty = position.quantity if position else 0.0

        # Suggested action
        if decision:
            action = decision.action
            size = decision.position_size_pct
        else:
            action = "hold"
            size = 0.0

        return RATState(
            timestamp=timestamp,
            symbol=symbol,
            attention_score=attention_state.attention_score if attention_state else 0.0,
            reflexivity_stage=(
                reflexivity_state.stage if reflexivity_state
                else ReflexivityStage.EFFICIENT
            ),
            topology_regime=(
                topology_state.regime if topology_state
                else TopologyRegime.UNKNOWN
            ),
            adversarial_archetype=(
                adversarial_state.dominant_archetype.name if adversarial_state
                else None
            ),
            alpha_health=alpha_state.overall_alpha_health,
            signals=signals,
            decision=decision,
            current_position=current_qty,
            suggested_action=action,
            suggested_size=size,
        )

    def _execute_decision(
        self,
        symbol: str,
        decision: CombinedDecision,
        current_price: float,
        timestamp: datetime,
    ) -> None:
        """Execute a trading decision."""
        if not self.broker:
            logger.info(f"Would execute {decision.action} for {symbol}")
            return

        position = self._positions.get(symbol)

        if decision.action == "buy":
            # Calculate quantity
            # This would use account value from broker
            target_qty = decision.position_size_pct * 1000  # Simplified

            if position:
                # Add to position
                delta = target_qty - position.quantity
                if delta > 0:
                    self._place_order(symbol, "BUY", delta, current_price, timestamp)
            else:
                # New position
                self._place_order(symbol, "BUY", target_qty, current_price, timestamp)

        elif decision.action == "sell":
            if position and position.quantity > 0:
                # Close or reduce position
                self._place_order(
                    symbol, "SELL", position.quantity, current_price, timestamp
                )

    def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Place an order through the broker."""
        logger.info(f"Placing {side} order: {quantity} {symbol} @ {price}")

        if self.broker and hasattr(self.broker, "place_order"):
            try:
                self.broker.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="MARKET",
                )
            except Exception as e:
                logger.error(f"Order placement failed: {e}")

        # Update position tracking (simplified)
        if side == "BUY":
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.quantity += quantity
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=timestamp,
                )
        elif side == "SELL":
            if symbol in self._positions:
                self._positions[symbol].quantity -= quantity
                if self._positions[symbol].quantity <= 0:
                    del self._positions[symbol]

        self._trade_count += 1

    def update_pnl(self, symbol: str, pnl: float) -> None:
        """Update P&L tracking and module performance."""
        self._total_pnl += pnl
        self._filter.update_equity(pnl)

        # Update source performance based on last signals
        if symbol in self._last_state:
            state = self._last_state[symbol]
            for source, signal in state.signals.items():
                if signal:
                    self._combiner.update_performance(
                        source=source,
                        prediction=signal.direction,
                        actual=1.0 if pnl > 0 else -1.0,
                        pnl=pnl,
                    )

    def check_alpha_health(self) -> bool:
        """Check if alpha factors need regeneration."""
        state = self._alpha_tracker.analyze()

        if state.needs_mutation:
            logger.warning("Alpha health critical - generating new factors")
            self._alpha_mutator.generate_new_factors(count=3)
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        mutation_stats = self._alpha_mutator.get_mutation_stats()
        source_stats = self._combiner.get_source_stats()
        alpha_state = self._alpha_tracker.analyze()

        return {
            "total_pnl": self._total_pnl,
            "trade_count": self._trade_count,
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for sym, pos in self._positions.items()
            },
            "alpha_health": alpha_state.overall_alpha_health,
            "active_factors": len(alpha_state.active_factors),
            "decaying_factors": len(alpha_state.decaying_factors),
            "mutation_stats": mutation_stats,
            "source_performance": source_stats,
        }

    def get_last_state(self, symbol: str) -> Optional[RATState]:
        """Get last state for a symbol."""
        return self._last_state.get(symbol)

    # ==================== Backtesting Interface ====================

    def inject_backtest_tick(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        buy_volume: Optional[float] = None,
        sell_volume: Optional[float] = None,
    ) -> Optional[RATState]:
        """
        Inject a single bar of historical data for backtesting.

        Returns the state after processing.
        """
        # Use close as current price, estimate bid/ask
        spread = (high - low) * 0.001  # Simplified spread estimate
        bid = close - spread / 2
        ask = close + spread / 2

        return self.process_tick(
            symbol=symbol,
            price=close,
            volume=volume,
            bid=bid,
            ask=ask,
            timestamp=timestamp,
        )

    def inject_news_event(
        self,
        symbol: str,
        timestamp: datetime,
        headline: str,
        sentiment: float,
    ) -> None:
        """Inject a news event for backtesting attention module."""
        self._attention.process_news({
            "symbol": symbol,
            "headline": headline,
            "timestamp": timestamp,
            "sentiment": sentiment,
        })

    def reset_for_backtest(self) -> None:
        """Reset engine state for new backtest run."""
        self._positions.clear()
        self._last_state.clear()
        self._total_pnl = 0.0
        self._trade_count = 0

        # Reset modules
        self._attention = AttentionTracker(
            config=self.config.attention,
            broker=None,
        )
        self._reflexivity = ReflexivityMeter(
            lookback=self.config.reflexivity.lookback,
            lag_order=self.config.reflexivity.lag_order,
        )
        self._topology = TopologyDetector(
            embedding_dim=self.config.topology.embedding_dim,
            time_delay=self.config.topology.time_delay,
        )
        self._adversarial = AdversarialDetector(
            flow_window=self.config.adversarial.flow_window,
        )
        self._alpha_tracker = AlphaTracker(
            sharpe_window=self.config.alpha.sharpe_window,
        )
        self._alpha_mutator = AlphaMutator(
            tracker=self._alpha_tracker,
            llm_client=None,  # No LLM in backtest
            enable_llm=False,
        )

        # Regenerate initial factors
        self._initialize_default_factors()

        # Reset combiner and filter
        self._combiner = SignalCombiner(
            weighting_method=self.config.signal.weighting_method,
        )
        self._filter = SignalFilter(
            confidence_threshold=self.config.signal.confidence_threshold,
        )

        logger.info("RAT Engine reset for backtesting")
