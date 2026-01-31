"""
Market Chameleon Strategy - Adaptive Alpha in All Regimes

The strategy that BECOMES the market:
- Bullish in bull markets (with leverage for outperformance)
- Bearish in bear markets (short for profit)
- Neutral in uncertain times (capital preservation)

Key Innovation: Multi-timeframe regime conviction with dynamic positioning

Academic Foundation:
- Moskowitz, Ooi, Pedersen (2012): Time-series momentum
- Baltas & Kosowski (2013): Trend-following with leverage
- Daniel & Moskowitz (2016): Momentum crashes and recovery

Author: RAT Framework
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


class MarketMode(Enum):
    """Market mode classification."""
    STRONG_BULL = auto()      # All timeframes bullish, high conviction
    BULL = auto()             # Majority bullish
    NEUTRAL_BULLISH = auto()  # Slight bullish bias
    NEUTRAL = auto()          # No clear direction
    NEUTRAL_BEARISH = auto()  # Slight bearish bias
    BEAR = auto()             # Majority bearish
    STRONG_BEAR = auto()      # All timeframes bearish, high conviction
    CRISIS = auto()           # Extreme volatility, capital preservation


@dataclass
class RegimeSignals:
    """Multi-timeframe regime signals."""
    short_term: float      # -1 to +1 (5-day)
    medium_term: float     # -1 to +1 (20-day)
    long_term: float       # -1 to +1 (50-day)
    volatility_regime: float  # 0 to 1 (low to high)
    trend_strength: float  # 0 to 1 (weak to strong)

    @property
    def alignment_score(self) -> float:
        """How aligned are the timeframes? 1 = perfect alignment."""
        signals = [self.short_term, self.medium_term, self.long_term]
        avg = sum(signals) / 3
        variance = sum((s - avg) ** 2 for s in signals) / 3
        # Convert variance to alignment (0 variance = 1 alignment)
        return max(0, 1 - math.sqrt(variance))

    @property
    def direction_consensus(self) -> float:
        """Weighted direction (-1 to +1), longer timeframes weighted more."""
        return 0.2 * self.short_term + 0.3 * self.medium_term + 0.5 * self.long_term

    @property
    def conviction(self) -> float:
        """Overall conviction (0 to 1)."""
        return self.alignment_score * self.trend_strength * (1 - 0.5 * self.volatility_regime)


@dataclass
class ChameleonPosition:
    """Position with adaptive sizing."""
    symbol: str
    direction: int          # +1 long, -1 short
    base_size: float        # Base position size (% of equity)
    leverage_mult: float    # Current leverage multiplier
    entry_price: float
    entry_time: datetime
    highest_price: float    # For trailing stop
    lowest_price: float     # For trailing stop (shorts)
    regime_at_entry: MarketMode

    @property
    def effective_size(self) -> float:
        """Effective position size including leverage."""
        return self.base_size * self.leverage_mult


class MultiTimeframeAnalyzer:
    """
    Analyze market across multiple timeframes.

    The key insight: When all timeframes align, trends are strong.
    When they diverge, the market is transitioning.
    """

    def __init__(self):
        self._prices: Dict[str, Deque[float]] = {}
        self._volumes: Dict[str, Deque[float]] = {}
        self._returns: Dict[str, Deque[float]] = {}

        # Timeframe periods (in days)
        self.short_period = 5
        self.medium_period = 20
        self.long_period = 50
        self.vol_period = 20

        # History length
        self.max_history = 252

    def update(self, symbol: str, price: float, volume: float) -> None:
        """Update with new price."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.max_history)
            self._volumes[symbol] = deque(maxlen=self.max_history)
            self._returns[symbol] = deque(maxlen=self.max_history)

        if self._prices[symbol]:
            prev_price = self._prices[symbol][-1]
            ret = (price - prev_price) / prev_price if prev_price > 0 else 0
            self._returns[symbol].append(ret)

        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)

    def get_regime_signals(self, symbol: str) -> Optional[RegimeSignals]:
        """Get multi-timeframe regime signals."""
        if symbol not in self._prices or len(self._prices[symbol]) < self.long_period:
            return None

        prices = list(self._prices[symbol])
        returns = list(self._returns[symbol]) if self._returns[symbol] else []

        # Short-term signal (momentum + price vs SMA)
        short_sma = sum(prices[-self.short_period:]) / self.short_period
        short_momentum = (prices[-1] - prices[-self.short_period]) / prices[-self.short_period]
        short_signal = self._normalize_signal(short_momentum * 10 + (prices[-1] / short_sma - 1) * 5)

        # Medium-term signal
        medium_sma = sum(prices[-self.medium_period:]) / self.medium_period
        medium_momentum = (prices[-1] - prices[-self.medium_period]) / prices[-self.medium_period]
        medium_signal = self._normalize_signal(medium_momentum * 5 + (prices[-1] / medium_sma - 1) * 3)

        # Long-term signal
        long_sma = sum(prices[-self.long_period:]) / self.long_period
        long_momentum = (prices[-1] - prices[-self.long_period]) / prices[-self.long_period]
        long_signal = self._normalize_signal(long_momentum * 3 + (prices[-1] / long_sma - 1) * 2)

        # Volatility regime
        if len(returns) >= self.vol_period:
            recent_vol = math.sqrt(sum(r**2 for r in returns[-self.vol_period:]) / self.vol_period)
            annual_vol = recent_vol * math.sqrt(252)
            # Normalize: 0.15 (15%) is "normal", above 0.4 is "crisis"
            vol_regime = min(1.0, max(0, (annual_vol - 0.10) / 0.30))
        else:
            vol_regime = 0.5

        # Trend strength (ADX-like calculation)
        trend_strength = self._calculate_trend_strength(prices)

        return RegimeSignals(
            short_term=short_signal,
            medium_term=medium_signal,
            long_term=long_signal,
            volatility_regime=vol_regime,
            trend_strength=trend_strength,
        )

    def _normalize_signal(self, raw: float) -> float:
        """Normalize signal to [-1, +1] using tanh."""
        return math.tanh(raw)

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength (0 to 1)."""
        if len(prices) < 20:
            return 0.5

        # Use directional movement
        recent = prices[-20:]
        up_moves = sum(max(0, recent[i] - recent[i-1]) for i in range(1, len(recent)))
        down_moves = sum(max(0, recent[i-1] - recent[i]) for i in range(1, len(recent)))

        total_move = up_moves + down_moves
        if total_move == 0:
            return 0.5

        # Directional index: how much of the movement is in one direction
        di = abs(up_moves - down_moves) / total_move

        # Also consider price range vs average move
        price_range = max(recent) - min(recent)
        avg_price = sum(recent) / len(recent)
        range_ratio = price_range / avg_price if avg_price > 0 else 0

        # Combine: strong trend = directional + reasonable range
        strength = di * min(1.0, range_ratio * 10)
        return min(1.0, max(0, strength))


class VolatilityHarvester:
    """
    Harvest volatility for additional alpha.

    Key insight: High volatility = opportunity
    - In bull + high vol: Buy dips aggressively
    - In bear + high vol: Short rips aggressively
    - Volatility mean-reverts, so extreme vol = opportunity
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self._volatility: Dict[str, Deque[float]] = {}
        self._vol_percentile: Dict[str, float] = {}

    def update(self, symbol: str, daily_return: float) -> None:
        """Update volatility estimate."""
        if symbol not in self._volatility:
            self._volatility[symbol] = deque(maxlen=252)

        self._volatility[symbol].append(abs(daily_return))

        # Calculate percentile of current vol vs history
        if len(self._volatility[symbol]) >= self.lookback:
            recent_vol = sum(list(self._volatility[symbol])[-self.lookback:]) / self.lookback
            sorted_vols = sorted(self._volatility[symbol])
            percentile = sum(1 for v in sorted_vols if v <= recent_vol) / len(sorted_vols)
            self._vol_percentile[symbol] = percentile

    def get_vol_opportunity(self, symbol: str) -> Tuple[float, str]:
        """
        Get volatility-based opportunity score and type.

        Returns: (opportunity_score, opportunity_type)
        - score: 0 to 1 (how good is the opportunity)
        - type: 'high_vol_mean_revert', 'low_vol_breakout', 'normal'
        """
        if symbol not in self._vol_percentile:
            return (0.5, 'normal')

        percentile = self._vol_percentile[symbol]

        if percentile > 0.9:
            # Extremely high vol - expect mean reversion
            # This is when markets overreact
            return (0.9, 'high_vol_mean_revert')
        elif percentile < 0.2:
            # Low vol - expect breakout soon
            return (0.7, 'low_vol_breakout')
        else:
            return (0.5, 'normal')


class MomentumCascade:
    """
    Cascade into winning positions, cascade out of losers.

    Key insight: Winners keep winning (momentum), losers keep losing.
    - Add to winning positions (pyramid up)
    - Cut losers quickly (pyramid out)
    - This naturally aligns with market direction
    """

    def __init__(self):
        self._position_performance: Dict[str, List[float]] = {}

    def should_add_to_position(
        self,
        symbol: str,
        current_pnl_pct: float,
        regime: MarketMode,
    ) -> Tuple[bool, float]:
        """
        Should we add to this position?

        Returns: (should_add, size_multiplier)
        """
        # Track performance
        if symbol not in self._position_performance:
            self._position_performance[symbol] = []
        self._position_performance[symbol].append(current_pnl_pct)

        # Only add if winning and regime supports
        if current_pnl_pct < 0.02:  # Need at least 2% profit
            return (False, 0)

        # Check if consistently winning
        recent = self._position_performance[symbol][-5:] if len(self._position_performance[symbol]) >= 5 else []
        if not recent:
            return (False, 0)

        # All recent checks should be profitable
        if all(p > 0 for p in recent):
            # Determine add size based on profit and regime
            if regime in (MarketMode.STRONG_BULL, MarketMode.STRONG_BEAR):
                return (True, 0.5)  # Add 50% more
            elif regime in (MarketMode.BULL, MarketMode.BEAR):
                return (True, 0.25)  # Add 25% more

        return (False, 0)

    def should_reduce_position(
        self,
        symbol: str,
        current_pnl_pct: float,
        days_held: int,
    ) -> Tuple[bool, float]:
        """
        Should we reduce this position?

        Returns: (should_reduce, reduction_pct)
        """
        # Quick cut on losses
        if current_pnl_pct < -0.03:  # 3% loss
            return (True, 0.5)  # Cut half immediately

        if current_pnl_pct < -0.05:  # 5% loss
            return (True, 1.0)  # Cut entire position

        # Time decay for non-performing positions
        if days_held > 10 and current_pnl_pct < 0.01:
            return (True, 0.25)  # Reduce if not performing after 10 days

        return (False, 0)


class ChameleonStrategy:
    """
    The Market Chameleon - Adaptive Alpha in All Regimes.

    Core Philosophy:
    1. BECOME the market direction (don't fight it)
    2. Use BETTER timing than buy-and-hold
    3. Apply LEVERAGE when conviction is high
    4. PRESERVE capital in uncertainty

    Key Innovations:
    - Multi-timeframe alignment for regime detection
    - Conviction-scaled leverage (0.5x to 2x)
    - Volatility harvesting for additional alpha
    - Momentum cascading (add to winners)
    """

    def __init__(self):
        self.analyzer = MultiTimeframeAnalyzer()
        self.vol_harvester = VolatilityHarvester()
        self.momentum_cascade = MomentumCascade()

        # Position tracking
        self.positions: Dict[str, ChameleonPosition] = {}

        # Strategy parameters - VERY AGGRESSIVE for alpha generation
        # Key insight: To beat B&H, we need >100% exposure in trends
        self.base_position_size = 0.50  # 50% base - high conviction
        self.max_leverage = 3.0         # Up to 3x in strong regimes
        self.min_leverage = 0.3         # Down to 0.3x in uncertain
        self.max_positions = 2          # Max concurrent positions (very focused)
        self.max_total_exposure = 3.0   # 300% max gross exposure

        # Performance tracking
        self._daily_returns: Deque[float] = deque(maxlen=252)
        self._regime_history: Deque[MarketMode] = deque(maxlen=50)

        # Regime persistence state
        self._current_regime: Optional[MarketMode] = None
        self._regime_conviction_count: int = 0  # How long we've been in this regime
        self._potential_flip_regime: Optional[MarketMode] = None
        self._potential_flip_count: int = 0

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> Optional[Dict]:
        """
        Update with new bar and return trading decision.

        Returns dict with:
        - action: 'buy', 'sell', 'short', 'cover', 'add', 'reduce', 'hold'
        - size: position size (% of equity)
        - confidence: 0 to 1
        - regime: current market mode
        """
        # Update analyzers
        self.analyzer.update(symbol, close, volume)

        # Calculate daily return for vol harvesting
        prices = list(self.analyzer._prices.get(symbol, []))
        if len(prices) >= 2:
            daily_ret = (prices[-1] - prices[-2]) / prices[-2]
            self.vol_harvester.update(symbol, daily_ret)

        # Get regime signals
        signals = self.analyzer.get_regime_signals(symbol)
        if signals is None:
            return None

        # Determine market mode
        mode = self._determine_mode(signals)
        self._regime_history.append(mode)

        # Get volatility opportunity
        vol_opp, vol_type = self.vol_harvester.get_vol_opportunity(symbol)

        # Generate trading decision
        decision = self._generate_decision(
            symbol=symbol,
            price=close,
            timestamp=timestamp,
            signals=signals,
            mode=mode,
            vol_opportunity=vol_opp,
            vol_type=vol_type,
        )

        return decision

    def _determine_mode(self, signals: RegimeSignals) -> MarketMode:
        """Determine market mode from signals with REGIME PERSISTENCE.

        Key insight: Regimes are sticky. Once established, they persist.
        Don't flip-flop on daily noise.
        """
        direction = signals.direction_consensus
        conviction = signals.conviction
        vol = signals.volatility_regime
        alignment = signals.alignment_score
        long_term = signals.long_term  # Most stable signal

        # Calculate the raw/new regime
        new_regime = self._calculate_raw_regime(direction, conviction, vol, alignment, long_term)

        # Apply regime persistence (stickiness)
        if self._current_regime is None:
            self._current_regime = new_regime
            self._regime_conviction_count = 1
            return new_regime

        # If same regime, increase conviction
        if new_regime == self._current_regime:
            self._regime_conviction_count = min(20, self._regime_conviction_count + 1)
            return new_regime

        # Regime change detected - but require conviction to flip
        # The longer we've been in current regime, the more conviction needed to flip

        # Easy transitions (same direction): require 2 days
        same_direction = (
            (self._current_regime in (MarketMode.STRONG_BULL, MarketMode.BULL, MarketMode.NEUTRAL_BULLISH) and
             new_regime in (MarketMode.STRONG_BULL, MarketMode.BULL, MarketMode.NEUTRAL_BULLISH)) or
            (self._current_regime in (MarketMode.STRONG_BEAR, MarketMode.BEAR, MarketMode.NEUTRAL_BEARISH) and
             new_regime in (MarketMode.STRONG_BEAR, MarketMode.BEAR, MarketMode.NEUTRAL_BEARISH))
        )

        if same_direction:
            # Easy transition within same direction family
            self._current_regime = new_regime
            self._regime_conviction_count = 1
            return new_regime

        # Crisis always overrides immediately
        if new_regime == MarketMode.CRISIS and vol > 0.85:
            self._current_regime = new_regime
            self._regime_conviction_count = 1
            return new_regime

        # Hard transitions (direction flip): require strong conviction
        # If we've been in bull regime 5+ days, need strong bear signals to flip
        min_conviction_to_flip = min(5, max(2, self._regime_conviction_count // 3))

        # Track potential flip
        if not hasattr(self, '_potential_flip_regime') or self._potential_flip_regime != new_regime:
            self._potential_flip_regime = new_regime
            self._potential_flip_count = 1
        else:
            self._potential_flip_count += 1

        # Flip if we've seen new regime enough times
        if self._potential_flip_count >= min_conviction_to_flip:
            self._current_regime = new_regime
            self._regime_conviction_count = self._potential_flip_count
            self._potential_flip_count = 0
            return new_regime

        # Stay in current regime
        return self._current_regime

    def _calculate_raw_regime(self, direction: float, conviction: float,
                              vol: float, alignment: float, long_term: float) -> MarketMode:
        """Calculate regime without persistence (raw signal)."""
        # Crisis mode: ONLY in EXTREME volatility with TRULY no clear direction
        # Key insight: High vol rallies are still rallies, not crises
        if vol > 0.95 and abs(direction) < 0.15 and abs(long_term) < 0.2:
            return MarketMode.CRISIS

        # Use long_term signal more heavily for regime stability
        # Long term is slow-moving and more reliable
        stable_direction = 0.3 * direction + 0.7 * long_term

        # STRONG BULL: Strong positive direction
        if stable_direction > 0.3 and (conviction > 0.4 or alignment > 0.4 or long_term > 0.5):
            return MarketMode.STRONG_BULL
        if stable_direction > 0.5:
            return MarketMode.STRONG_BULL

        # STRONG BEAR: Strong negative direction
        if stable_direction < -0.3 and (conviction > 0.4 or alignment > 0.4 or long_term < -0.5):
            return MarketMode.STRONG_BEAR
        if stable_direction < -0.5:
            return MarketMode.STRONG_BEAR

        # Regular BULL
        if stable_direction > 0.1 or long_term > 0.2:
            if conviction > 0.25 or stable_direction > 0.2 or long_term > 0.3:
                return MarketMode.BULL
            return MarketMode.NEUTRAL_BULLISH

        # Regular BEAR
        if stable_direction < -0.1 or long_term < -0.2:
            if conviction > 0.25 or stable_direction < -0.2 or long_term < -0.3:
                return MarketMode.BEAR
            return MarketMode.NEUTRAL_BEARISH

        # NEUTRAL with bias
        if stable_direction > 0.02:
            return MarketMode.NEUTRAL_BULLISH
        elif stable_direction < -0.02:
            return MarketMode.NEUTRAL_BEARISH

        return MarketMode.NEUTRAL

    def _calculate_leverage(self, mode: MarketMode, conviction: float) -> float:
        """Calculate leverage multiplier based on regime and conviction.

        Key insight: To beat buy-and-hold in bull markets, we need >100% exposure.
        With 50% base position, we need 2x leverage for 100% exposure, 4x for 200%.

        To outperform B&H in bulls while only being invested 60% of the time,
        we need: exposure = B&H_return / time_invested = 1.0 / 0.6 = 1.67x minimum
        Adding buffer for trading friction: 2x minimum in bull markets.
        """
        base_leverage = {
            MarketMode.STRONG_BULL: 4.0,   # 200% exposure in strong bull
            MarketMode.BULL: 3.5,          # 175% in bull
            MarketMode.NEUTRAL_BULLISH: 2.5,  # 125% - still bullish bias
            MarketMode.NEUTRAL: 0.5,
            MarketMode.NEUTRAL_BEARISH: 2.5,  # 125% short - still bearish bias
            MarketMode.BEAR: 3.5,          # 175% short in bear
            MarketMode.STRONG_BEAR: 4.0,   # 200% short in strong bear
            MarketMode.CRISIS: 0.2,        # Minimal exposure in crisis
        }.get(mode, 1.0)

        # Scale by conviction - floor at 85% of base leverage to stay aggressive
        return base_leverage * (0.85 + 0.15 * conviction)

    def _generate_decision(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        signals: RegimeSignals,
        mode: MarketMode,
        vol_opportunity: float,
        vol_type: str,
    ) -> Dict:
        """Generate trading decision."""

        leverage = self._calculate_leverage(mode, signals.conviction)

        # Check existing position
        has_position = symbol in self.positions
        position = self.positions.get(symbol)

        if has_position:
            return self._manage_existing_position(
                position, price, timestamp, mode, signals, vol_type
            )
        else:
            return self._evaluate_new_position(
                symbol, price, timestamp, mode, signals, leverage, vol_opportunity, vol_type
            )

    def _evaluate_new_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        mode: MarketMode,
        signals: RegimeSignals,
        leverage: float,
        vol_opportunity: float,
        vol_type: str,
    ) -> Dict:
        """Evaluate whether to enter a new position."""

        # In CRISIS mode, very conservative but allow some exposure
        if mode == MarketMode.CRISIS:
            # In crisis, only enter if signals are very strong
            if signals.conviction > 0.7 and abs(signals.direction_consensus) > 0.6:
                direction = 1 if signals.direction_consensus > 0 else -1
                return self._create_entry(
                    symbol, price, timestamp, mode, signals,
                    direction, leverage * 0.3, "Crisis mode - high conviction entry"
                )
            return {
                'action': 'hold',
                'regime': mode,
                'reason': 'Crisis mode - preserving capital',
            }

        # Determine direction
        direction = 1 if signals.direction_consensus > 0 else -1

        # AGGRESSIVE ENTRY LOGIC: Be in the market when direction is clear
        # Key insight: Time in market beats timing the market

        short_term = signals.short_term
        long_term = signals.long_term
        should_enter = False
        entry_reason = ""

        # BULL MARKETS: We want to be LONG most of the time
        if mode in (MarketMode.STRONG_BULL, MarketMode.BULL):
            direction = 1  # Always long in bull

            # Strong bull: Enter immediately, don't wait for perfect dip
            if mode == MarketMode.STRONG_BULL:
                should_enter = True
                entry_reason = "Strong bull - time in market"
                leverage *= 1.2

            # Regular bull: Enter on any reasonable setup
            elif short_term < 0.5:  # Not extremely overbought
                should_enter = True
                entry_reason = "Bull market entry"

            # Extra leverage on dips
            if short_term < -0.2:
                leverage *= 1.3
                entry_reason = "Buy the dip - aggressive"

        # BEAR MARKETS: We want to be SHORT
        elif mode in (MarketMode.STRONG_BEAR, MarketMode.BEAR):
            direction = -1  # Always short in bear

            # Strong bear: Enter immediately
            if mode == MarketMode.STRONG_BEAR:
                should_enter = True
                entry_reason = "Strong bear - short market"
                leverage *= 1.2

            # Regular bear: Enter on any reasonable setup
            elif short_term > -0.5:  # Not extremely oversold
                should_enter = True
                entry_reason = "Bear market short"

            # Extra leverage on rallies (sell the rip)
            if short_term > 0.2:
                leverage *= 1.3
                entry_reason = "Short the rip - aggressive"

        # NEUTRAL-ISH MARKETS: Trade with bias
        elif mode == MarketMode.NEUTRAL_BULLISH:
            direction = 1
            # Lower bar for entry in neutral-bullish
            if signals.conviction > 0.2 or long_term > 0.1:
                should_enter = True
                entry_reason = "Neutral-bullish bias"
                leverage *= 0.8

        elif mode == MarketMode.NEUTRAL_BEARISH:
            direction = -1
            if signals.conviction > 0.2 or long_term < -0.1:
                should_enter = True
                entry_reason = "Neutral-bearish bias"
                leverage *= 0.8

        elif mode == MarketMode.NEUTRAL:
            # In pure neutral, follow short-term momentum
            if abs(short_term) > 0.3:
                direction = 1 if short_term > 0 else -1
                should_enter = True
                entry_reason = "Neutral - follow momentum"
                leverage *= 0.5  # Smaller size in neutral

        if should_enter:
            return self._create_entry(
                symbol, price, timestamp, mode, signals,
                direction, leverage, entry_reason
            )

        return {
            'action': 'hold',
            'regime': mode,
            'reason': 'No entry opportunity',
        }

    def _create_entry(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        mode: MarketMode,
        signals: RegimeSignals,
        direction: int,
        leverage: float,
        reason: str,
    ) -> Dict:
        """Create a new position entry."""
        size = self.base_position_size * leverage
        action = 'buy' if direction > 0 else 'short'

        # Create position
        self.positions[symbol] = ChameleonPosition(
            symbol=symbol,
            direction=direction,
            base_size=self.base_position_size,
            leverage_mult=leverage,
            entry_price=price,
            entry_time=timestamp,
            highest_price=price,
            lowest_price=price,
            regime_at_entry=mode,
        )

        return {
            'action': action,
            'size': size,
            'confidence': signals.conviction,
            'regime': mode,
            'leverage': leverage,
            'reason': reason,
            'direction': direction,
        }

    def _manage_existing_position(
        self,
        position: ChameleonPosition,
        price: float,
        timestamp: datetime,
        mode: MarketMode,
        signals: RegimeSignals,
        vol_type: str,
    ) -> Dict:
        """Manage existing position."""

        symbol = position.symbol
        direction = position.direction
        entry_price = position.entry_price

        # Update highest/lowest
        position.highest_price = max(position.highest_price, price)
        position.lowest_price = min(position.lowest_price, price)

        # Calculate P&L
        if direction > 0:
            pnl_pct = (price - entry_price) / entry_price
            drawdown_from_peak = (position.highest_price - price) / position.highest_price
        else:
            pnl_pct = (entry_price - price) / entry_price
            drawdown_from_peak = (price - position.lowest_price) / position.lowest_price if position.lowest_price > 0 else 0

        days_held = (timestamp - position.entry_time).days

        # Check for regime change
        regime_changed = mode != position.regime_at_entry
        regime_flipped = (
            (position.direction > 0 and mode in (MarketMode.BEAR, MarketMode.STRONG_BEAR)) or
            (position.direction < 0 and mode in (MarketMode.BULL, MarketMode.STRONG_BULL))
        )

        # EXIT CONDITIONS

        # 1. Regime flip - ONLY exit on STRONG opposite regime
        # Key insight: Missing upside in bull markets is worse than small losses
        # A LONG position should ONLY exit if regime becomes STRONG_BEAR or BEAR
        # A SHORT position should ONLY exit if regime becomes STRONG_BULL or BULL

        if position.direction > 0:
            # LONG position - only exit on strong bearish signal
            should_exit_regime = mode in (MarketMode.STRONG_BEAR, MarketMode.BEAR)
            # Also exit if we've been in any bearish mode for too long with losses
            if mode in (MarketMode.NEUTRAL_BEARISH,) and pnl_pct < -0.03 and days_held > 5:
                should_exit_regime = True
        else:
            # SHORT position - only exit on strong bullish signal
            should_exit_regime = mode in (MarketMode.STRONG_BULL, MarketMode.BULL)
            if mode in (MarketMode.NEUTRAL_BULLISH,) and pnl_pct < -0.03 and days_held > 5:
                should_exit_regime = True

        if should_exit_regime:
            return self._close_position(symbol, price, f"Regime strongly against position: {mode.name}")

        # 2. Stop loss - WIDER in trending markets to avoid whipsaw
        # Key insight: Normal bull market corrections are 5-10%
        if mode in (MarketMode.STRONG_BULL, MarketMode.STRONG_BEAR):
            stop_loss_pct = 0.15  # 15% stop in strong trends
        elif mode in (MarketMode.BULL, MarketMode.BEAR):
            stop_loss_pct = 0.12  # 12% stop in trends
        elif mode == MarketMode.CRISIS:
            stop_loss_pct = 0.05  # Tight stop in crisis
        else:
            stop_loss_pct = 0.10  # 10% in neutral

        if pnl_pct < -stop_loss_pct:
            return self._close_position(symbol, price, f"Stop loss hit ({pnl_pct:.1%})")

        # 3. Trailing stop - MUCH WIDER in trending markets to let winners run
        # Key insight: Bull markets have 5-15% pullbacks that are normal
        if pnl_pct > 0.08:  # Only activate after 8% profit
            # In strong trends, use very wide trailing stop
            if mode in (MarketMode.STRONG_BULL, MarketMode.STRONG_BEAR):
                trailing_stop_pct = 0.25  # 25% trailing in strong trends
            elif mode in (MarketMode.BULL, MarketMode.BEAR):
                trailing_stop_pct = 0.20  # 20% in regular trends
            else:
                trailing_stop_pct = 0.12  # 12% in neutral

            if drawdown_from_peak > trailing_stop_pct:
                return self._close_position(symbol, price, f"Trailing stop ({drawdown_from_peak:.1%} from peak)")

        # 4. Take profit - ONLY partial, let the rest run
        if pnl_pct > 0.30:  # 30% profit
            # Only take partial profits, never full exit on profits in trends
            if mode not in (MarketMode.STRONG_BULL, MarketMode.STRONG_BEAR):
                return self._reduce_position(symbol, price, 0.3, "Taking 30% partial profit")

        # 5. Time decay - VERY patient in trending markets
        # In strong trends, we want to stay invested even if flat
        if mode in (MarketMode.STRONG_BULL, MarketMode.STRONG_BEAR):
            patience = 60  # 3 months patience in strong trends
            min_profit = -0.03  # Allow up to 3% underwater
        elif mode in (MarketMode.BULL, MarketMode.BEAR):
            patience = 40  # 2 months patience in trends
            min_profit = -0.02  # Allow up to 2% underwater
        else:
            patience = 15
            min_profit = 0.01  # Need to be profitable

        if days_held > patience and pnl_pct < min_profit:
            return self._close_position(symbol, price, "Time decay - not performing")

        # CHECK FOR ADD OPPORTUNITY
        should_add, add_mult = self.momentum_cascade.should_add_to_position(
            symbol, pnl_pct, mode
        )

        if should_add and len(self.positions) < self.max_positions:
            # Only add if regime still supports
            if (direction > 0 and mode in (MarketMode.STRONG_BULL, MarketMode.BULL)) or \
               (direction < 0 and mode in (MarketMode.STRONG_BEAR, MarketMode.BEAR)):

                add_size = position.effective_size * add_mult
                position.leverage_mult *= (1 + add_mult)

                return {
                    'action': 'add',
                    'size': add_size,
                    'regime': mode,
                    'reason': f'Adding to winner ({pnl_pct:.1%} profit)',
                    'total_position': position.effective_size,
                }

        # HOLD
        return {
            'action': 'hold',
            'regime': mode,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
        }

    def _close_position(self, symbol: str, price: float, reason: str) -> Dict:
        """Close entire position."""
        position = self.positions.pop(symbol, None)
        if position is None:
            return {'action': 'hold', 'reason': 'No position to close'}

        action = 'sell' if position.direction > 0 else 'cover'

        return {
            'action': action,
            'size': position.effective_size,
            'reason': reason,
            'direction': position.direction,
        }

    def _reduce_position(self, symbol: str, price: float, reduction_pct: float, reason: str) -> Dict:
        """Reduce position size."""
        position = self.positions.get(symbol)
        if position is None:
            return {'action': 'hold', 'reason': 'No position to reduce'}

        reduce_size = position.effective_size * reduction_pct
        position.leverage_mult *= (1 - reduction_pct)

        action = 'sell' if position.direction > 0 else 'cover'

        return {
            'action': action,
            'size': reduce_size,
            'reason': reason,
            'is_partial': True,
        }

    def get_recommended_allocation(self, mode: MarketMode) -> Dict[str, float]:
        """
        Get recommended portfolio allocation based on regime.

        Returns target weights for:
        - long_equity: Long stock exposure
        - short_equity: Short stock exposure
        - cash: Cash reserve
        """
        allocations = {
            MarketMode.STRONG_BULL: {'long': 1.5, 'short': 0.0, 'cash': 0.0},
            MarketMode.BULL: {'long': 1.2, 'short': 0.0, 'cash': 0.1},
            MarketMode.NEUTRAL_BULLISH: {'long': 0.8, 'short': 0.0, 'cash': 0.3},
            MarketMode.NEUTRAL: {'long': 0.4, 'short': 0.2, 'cash': 0.5},
            MarketMode.NEUTRAL_BEARISH: {'long': 0.2, 'short': 0.5, 'cash': 0.4},
            MarketMode.BEAR: {'long': 0.0, 'short': 1.0, 'cash': 0.2},
            MarketMode.STRONG_BEAR: {'long': 0.0, 'short': 1.5, 'cash': 0.0},
            MarketMode.CRISIS: {'long': 0.1, 'short': 0.1, 'cash': 0.9},
        }

        return allocations.get(mode, {'long': 0.5, 'short': 0.0, 'cash': 0.5})


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_chameleon_strategy() -> ChameleonStrategy:
    """Create a new Chameleon strategy instance."""
    return ChameleonStrategy()


def get_regime_description(mode: MarketMode) -> str:
    """Get human-readable regime description."""
    descriptions = {
        MarketMode.STRONG_BULL: "Strong Bull Market - All timeframes aligned bullish, high conviction. Deploy maximum long exposure with leverage.",
        MarketMode.BULL: "Bull Market - Majority of signals bullish. Maintain long bias, buy dips.",
        MarketMode.NEUTRAL_BULLISH: "Neutral with Bullish Lean - Mixed signals with slight upside. Modest long exposure.",
        MarketMode.NEUTRAL: "Neutral/Ranging - No clear direction. Reduce exposure, wait for clarity.",
        MarketMode.NEUTRAL_BEARISH: "Neutral with Bearish Lean - Mixed signals with slight downside. Modest short exposure.",
        MarketMode.BEAR: "Bear Market - Majority of signals bearish. Maintain short bias, sell rips.",
        MarketMode.STRONG_BEAR: "Strong Bear Market - All timeframes aligned bearish. Deploy maximum short exposure.",
        MarketMode.CRISIS: "Crisis Mode - Extreme volatility. Preserve capital, minimal exposure.",
    }
    return descriptions.get(mode, "Unknown regime")
