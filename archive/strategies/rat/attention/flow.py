"""
Attention Flow: Core attention tracking logic.

Combines three attention metrics:
1. News velocity - Normalized rate of news about symbol
2. Flow imbalance - Buy/sell order ratio
3. Price acceleration - Rate of price momentum change

All metrics normalized to [-1, 1] range.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


@dataclass
class AttentionState:
    """Current attention state for a symbol."""

    timestamp: datetime
    symbol: str
    news_velocity: float        # -1 to 1 (negative = declining, positive = increasing)
    flow_imbalance: float       # -1 to 1 (negative = sell pressure, positive = buy pressure)
    price_acceleration: float   # -1 to 1 (negative = decelerating, positive = accelerating)
    attention_score: float      # Combined weighted score

    @property
    def is_high_attention(self) -> bool:
        """Check if attention is elevated."""
        return abs(self.attention_score) > 0.5


class AttentionFlow:
    """
    Track attention flow for a symbol.

    Uses exponentially weighted moving averages for smoothing.
    """

    def __init__(
        self,
        news_weight: float = 0.4,
        flow_weight: float = 0.35,
        price_weight: float = 0.25,
        decay_half_life: float = 300.0,  # seconds
        window_size: int = 100,
    ):
        self.news_weight = news_weight
        self.flow_weight = flow_weight
        self.price_weight = price_weight
        self.decay_half_life = decay_half_life
        self.window_size = window_size

        # Data storage
        self._news_times: Deque[datetime] = deque(maxlen=1000)
        self._flow_data: Deque[Tuple[datetime, float, float]] = deque(maxlen=window_size)  # (time, buy_vol, sell_vol)
        self._price_data: Deque[Tuple[datetime, float]] = deque(maxlen=window_size)

        # Cached computations
        self._last_state: Optional[AttentionState] = None
        self._symbol: str = ""

    def update_news(self, timestamp: datetime) -> None:
        """Record a news event."""
        self._news_times.append(timestamp)

    def update_flow(
        self,
        timestamp: datetime,
        buy_volume: float,
        sell_volume: float,
    ) -> None:
        """Update order flow data."""
        self._flow_data.append((timestamp, buy_volume, sell_volume))

    def update_price(self, timestamp: datetime, price: float) -> None:
        """Update price data."""
        self._price_data.append((timestamp, price))

    def compute_attention_state(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> AttentionState:
        """Compute current attention state."""
        ts = timestamp or datetime.now()
        self._symbol = symbol

        news_velocity = self._compute_news_velocity(ts)
        flow_imbalance = self._compute_flow_imbalance(ts)
        price_acceleration = self._compute_price_acceleration(ts)

        # Weighted combination
        attention_score = (
            self.news_weight * news_velocity +
            self.flow_weight * flow_imbalance +
            self.price_weight * price_acceleration
        )

        state = AttentionState(
            timestamp=ts,
            symbol=symbol,
            news_velocity=news_velocity,
            flow_imbalance=flow_imbalance,
            price_acceleration=price_acceleration,
            attention_score=attention_score,
        )

        self._last_state = state
        return state

    def _compute_news_velocity(self, timestamp: datetime) -> float:
        """
        Compute news velocity as change in news rate.

        Uses exponential decay weighting.
        """
        if len(self._news_times) < 2:
            return 0.0

        # Count news in recent vs older window
        cutoff_recent = timestamp - timedelta(seconds=self.decay_half_life)
        cutoff_older = cutoff_recent - timedelta(seconds=self.decay_half_life)

        recent_count = sum(1 for t in self._news_times if t > cutoff_recent)
        older_count = sum(1 for t in self._news_times if cutoff_older < t <= cutoff_recent)

        if older_count == 0:
            return min(1.0, recent_count * 0.2) if recent_count > 0 else 0.0

        # Velocity is change in rate
        velocity = (recent_count - older_count) / max(older_count, 1)

        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, velocity))

    def _compute_flow_imbalance(self, timestamp: datetime) -> float:
        """
        Compute order flow imbalance.

        Positive = more buying, Negative = more selling.
        """
        if len(self._flow_data) < 5:
            return 0.0

        # Use recent data with time decay
        total_buy = 0.0
        total_sell = 0.0
        total_weight = 0.0

        for t, buy_vol, sell_vol in self._flow_data:
            age = (timestamp - t).total_seconds()
            weight = math.exp(-age / self.decay_half_life)

            total_buy += buy_vol * weight
            total_sell += sell_vol * weight
            total_weight += weight

        if total_weight == 0 or (total_buy + total_sell) == 0:
            return 0.0

        # Imbalance normalized to [-1, 1]
        imbalance = (total_buy - total_sell) / (total_buy + total_sell)
        return imbalance

    def _compute_price_acceleration(self, timestamp: datetime) -> float:
        """
        Compute price acceleration (second derivative).

        Positive = momentum increasing, Negative = momentum decreasing.
        """
        if len(self._price_data) < 10:
            return 0.0

        prices = [p for t, p in self._price_data]

        # Compute first derivative (momentum)
        n = len(prices)
        half = n // 2

        momentum_early = (prices[half] - prices[0]) / prices[0] if prices[0] != 0 else 0
        momentum_late = (prices[-1] - prices[half]) / prices[half] if prices[half] != 0 else 0

        # Second derivative is change in momentum
        acceleration = momentum_late - momentum_early

        # Normalize (typical daily moves are ~1-2%)
        normalized = acceleration * 50  # Scale factor

        return max(-1.0, min(1.0, normalized))

    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """Generate trading signal from attention state."""
        state = self.compute_attention_state(symbol)

        if abs(state.attention_score) < 0.3:
            return None

        if state.attention_score > 0:
            signal_type = SignalType.LONG
            direction = state.attention_score
        else:
            signal_type = SignalType.SHORT
            direction = state.attention_score

        # Confidence based on component agreement
        components = [state.news_velocity, state.flow_imbalance, state.price_acceleration]
        same_sign = sum(1 for c in components if c * state.attention_score > 0)
        confidence = same_sign / 3.0

        # Urgency based on absolute score
        urgency = abs(state.attention_score)

        return Signal(
            source=SignalSource.ATTENTION,
            signal_type=signal_type,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            urgency=urgency,
            metadata={
                "news_velocity": state.news_velocity,
                "flow_imbalance": state.flow_imbalance,
                "price_acceleration": state.price_acceleration,
            },
        )
