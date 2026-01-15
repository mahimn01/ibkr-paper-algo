"""
Adversarial Meta-Trader: Detect and exploit predictable algorithm behavior.

Mathematical approach to identifying algorithm archetypes:
1. Momentum algos: Autocorrelation signatures in order flow
2. Mean reversion algos: Predictable entry at Bollinger/zscore levels
3. Index rebalancers: Calendar-based volume spikes
4. Stop hunters: Thin book exploitation near round numbers

No AI required - pure statistical pattern recognition.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from trading_algo.rat.signals import Signal, SignalType, SignalSource


class AlgoArchetype(Enum):
    """Known algorithm behavior patterns."""

    UNKNOWN = auto()
    MOMENTUM = auto()           # Trend followers, CTA-style
    MEAN_REVERSION = auto()     # Statistical arbitrage, pairs
    INDEX_REBALANCE = auto()    # ETF creation/redemption, rebalancing
    STOP_HUNT = auto()          # Exploits thin books at round numbers
    VWAP_TWAP = auto()          # Execution algorithms
    MARKET_MAKER = auto()       # Liquidity provision with inventory mgmt


@dataclass(frozen=True)
class AlgoSignature:
    """Detected algorithm signature."""

    archetype: AlgoArchetype
    confidence: float           # 0-1 detection confidence
    predicted_action: str       # "buy", "sell", "fade"
    predicted_size: float       # Relative size estimate
    predicted_timing: float     # Seconds until action
    exploitation_edge: float    # Expected edge from counter-trade

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            object.__setattr__(self, 'confidence', max(0, min(1, self.confidence)))


@dataclass
class OrderFlowTick:
    """Single order flow observation."""

    timestamp: datetime
    price: float
    volume: float
    aggressor: str      # "buy" or "sell"
    bid: float
    ask: float


@dataclass
class AdversarialState:
    """Current adversarial analysis state."""

    timestamp: datetime
    detected_algos: List[AlgoSignature]
    dominant_archetype: AlgoArchetype
    exploitation_signal: Optional[str]  # "front_run", "fade", "avoid"
    total_confidence: float

    def has_opportunity(self) -> bool:
        """Check if there's a tradeable opportunity."""
        return (
            self.total_confidence > 0.6 and
            self.exploitation_signal in ("front_run", "fade")
        )


class AdversarialDetector:
    """
    Detect and exploit predictable algorithm behavior.

    Uses statistical signatures to identify:
    - MOMENTUM: High autocorrelation in order flow direction
    - MEAN_REVERSION: Increased activity at statistical extremes
    - INDEX_REBALANCE: Calendar-correlated volume patterns
    - STOP_HUNT: Probing behavior near round numbers
    - VWAP_TWAP: Even-paced execution throughout day
    - MARKET_MAKER: Quote refreshing patterns
    """

    def __init__(
        self,
        flow_window: int = 500,
        detection_threshold: float = 0.65,
        round_number_tolerance: float = 0.001,
    ):
        self.flow_window = flow_window
        self.detection_threshold = detection_threshold
        self.round_number_tolerance = round_number_tolerance

        # Order flow history per symbol
        self._flow_history: Dict[str, Deque[OrderFlowTick]] = {}

        # Price history for pattern detection
        self._price_history: Dict[str, Deque[Tuple[datetime, float]]] = {}

        # Volume profile by time of day
        self._volume_profile: Dict[str, Dict[int, float]] = {}  # minute -> avg volume

        # Detection state
        self._last_state: Dict[str, AdversarialState] = {}

        # Known rebalancing dates (simplified - would be calendar-driven)
        self._rebalance_times = [
            dt_time(15, 45),  # Near close for index rebalance
            dt_time(16, 0),   # MOC orders
        ]

    def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        aggressor: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update with new order flow data."""
        ts = timestamp or datetime.now()

        # Initialize if needed
        if symbol not in self._flow_history:
            self._flow_history[symbol] = deque(maxlen=self.flow_window)
            self._price_history[symbol] = deque(maxlen=self.flow_window)
            self._volume_profile[symbol] = {}

        tick = OrderFlowTick(
            timestamp=ts,
            price=price,
            volume=volume,
            aggressor=aggressor,
            bid=bid,
            ask=ask,
        )

        self._flow_history[symbol].append(tick)
        self._price_history[symbol].append((ts, price))

        # Update volume profile
        minute_key = ts.hour * 60 + ts.minute
        if minute_key not in self._volume_profile[symbol]:
            self._volume_profile[symbol][minute_key] = volume
        else:
            # Exponential moving average
            alpha = 0.1
            self._volume_profile[symbol][minute_key] = (
                alpha * volume +
                (1 - alpha) * self._volume_profile[symbol][minute_key]
            )

    def detect(self, symbol: str) -> AdversarialState:
        """Detect algorithm signatures in current flow."""
        if symbol not in self._flow_history or len(self._flow_history[symbol]) < 50:
            return AdversarialState(
                timestamp=datetime.now(),
                detected_algos=[],
                dominant_archetype=AlgoArchetype.UNKNOWN,
                exploitation_signal=None,
                total_confidence=0.0,
            )

        flow = list(self._flow_history[symbol])

        # Run all detectors
        signatures = []

        momentum_sig = self._detect_momentum(flow)
        if momentum_sig:
            signatures.append(momentum_sig)

        mean_rev_sig = self._detect_mean_reversion(flow)
        if mean_rev_sig:
            signatures.append(mean_rev_sig)

        rebalance_sig = self._detect_index_rebalance(symbol, flow)
        if rebalance_sig:
            signatures.append(rebalance_sig)

        stop_hunt_sig = self._detect_stop_hunt(flow)
        if stop_hunt_sig:
            signatures.append(stop_hunt_sig)

        vwap_sig = self._detect_vwap_twap(flow)
        if vwap_sig:
            signatures.append(vwap_sig)

        mm_sig = self._detect_market_maker(flow)
        if mm_sig:
            signatures.append(mm_sig)

        # Determine dominant and exploitation strategy
        dominant = AlgoArchetype.UNKNOWN
        exploitation = None
        total_conf = 0.0

        if signatures:
            # Sort by confidence
            signatures.sort(key=lambda s: s.confidence, reverse=True)
            dominant = signatures[0].archetype
            total_conf = signatures[0].confidence

            # Determine exploitation strategy
            exploitation = self._determine_exploitation(signatures[0])

        state = AdversarialState(
            timestamp=datetime.now(),
            detected_algos=signatures,
            dominant_archetype=dominant,
            exploitation_signal=exploitation,
            total_confidence=total_conf,
        )

        self._last_state[symbol] = state
        return state

    def _detect_momentum(self, flow: List[OrderFlowTick]) -> Optional[AlgoSignature]:
        """
        Detect momentum algorithms via autocorrelation.

        Momentum algos show positive autocorrelation in trade direction:
        if recent trades were buys, next trades likely buys too.
        """
        if len(flow) < 100:
            return None

        # Convert to direction series: +1 for buy, -1 for sell
        directions = [1 if t.aggressor == "buy" else -1 for t in flow[-100:]]

        # Compute lag-1 autocorrelation
        n = len(directions)
        mean_d = sum(directions) / n

        numerator = sum(
            (directions[i] - mean_d) * (directions[i-1] - mean_d)
            for i in range(1, n)
        )
        denominator = sum((d - mean_d) ** 2 for d in directions)

        if denominator == 0:
            return None

        autocorr = numerator / denominator

        # Also check for trend persistence
        recent_direction = sum(directions[-20:]) / 20

        # High positive autocorr + directional bias = momentum algo
        if autocorr > 0.3 and abs(recent_direction) > 0.5:
            predicted_action = "buy" if recent_direction > 0 else "sell"

            return AlgoSignature(
                archetype=AlgoArchetype.MOMENTUM,
                confidence=min(0.95, 0.5 + autocorr),
                predicted_action=predicted_action,
                predicted_size=self._estimate_momentum_size(flow),
                predicted_timing=2.0,  # Momentum algos are fast
                exploitation_edge=0.02 * autocorr,  # Edge proportional to predictability
            )

        return None

    def _detect_mean_reversion(self, flow: List[OrderFlowTick]) -> Optional[AlgoSignature]:
        """
        Detect mean reversion algorithms.

        Mean reversion algos increase activity at statistical extremes
        (2+ standard deviations from mean).
        """
        if len(flow) < 100:
            return None

        prices = [t.price for t in flow]

        # Compute z-score of current price
        mean_p = sum(prices) / len(prices)
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in prices) / len(prices))

        if std_p == 0:
            return None

        current_z = (prices[-1] - mean_p) / std_p

        # Check if we're at extremes and seeing counter-trend flow
        recent_flow = flow[-20:]
        buy_count = sum(1 for t in recent_flow if t.aggressor == "buy")
        sell_count = len(recent_flow) - buy_count

        # At positive extreme, mean rev algos sell
        # At negative extreme, mean rev algos buy
        is_mean_rev = False
        predicted_action = None

        if current_z > 1.5 and sell_count > buy_count * 1.5:
            is_mean_rev = True
            predicted_action = "sell"
        elif current_z < -1.5 and buy_count > sell_count * 1.5:
            is_mean_rev = True
            predicted_action = "buy"

        if is_mean_rev and predicted_action:
            confidence = min(0.9, 0.4 + abs(current_z) * 0.15)

            return AlgoSignature(
                archetype=AlgoArchetype.MEAN_REVERSION,
                confidence=confidence,
                predicted_action=predicted_action,
                predicted_size=self._estimate_mean_rev_size(flow, current_z),
                predicted_timing=5.0,  # Mean rev is more patient
                exploitation_edge=0.015 * abs(current_z),
            )

        return None

    def _detect_index_rebalance(
        self, symbol: str, flow: List[OrderFlowTick]
    ) -> Optional[AlgoSignature]:
        """
        Detect index rebalancing activity.

        Index rebalancers have extremely predictable timing:
        - End of quarter
        - Near market close
        - Large, steady order flow
        """
        if len(flow) < 50:
            return None

        current_time = flow[-1].timestamp.time()

        # Check if we're in rebalancing window
        in_rebalance_window = any(
            abs(
                (current_time.hour * 60 + current_time.minute) -
                (rt.hour * 60 + rt.minute)
            ) < 30
            for rt in self._rebalance_times
        )

        if not in_rebalance_window:
            return None

        # Check for steady, one-sided flow
        recent = flow[-50:]
        buy_vol = sum(t.volume for t in recent if t.aggressor == "buy")
        sell_vol = sum(t.volume for t in recent if t.aggressor == "sell")

        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return None

        imbalance = abs(buy_vol - sell_vol) / total_vol

        # Check for unusually high volume vs profile
        minute_key = current_time.hour * 60 + current_time.minute
        expected_vol = self._volume_profile.get(symbol, {}).get(minute_key, total_vol)
        vol_ratio = total_vol / max(expected_vol, 1)

        if imbalance > 0.6 and vol_ratio > 1.5:
            predicted_action = "buy" if buy_vol > sell_vol else "sell"

            return AlgoSignature(
                archetype=AlgoArchetype.INDEX_REBALANCE,
                confidence=min(0.95, 0.6 + imbalance * 0.3),
                predicted_action=predicted_action,
                predicted_size=total_vol * 0.5,  # Estimate remaining
                predicted_timing=15.0 * 60,  # Through close
                exploitation_edge=0.01 * imbalance,
            )

        return None

    def _detect_stop_hunt(self, flow: List[OrderFlowTick]) -> Optional[AlgoSignature]:
        """
        Detect stop hunting behavior.

        Stop hunters probe near round numbers where stops cluster:
        - Quick moves to trigger stops
        - Immediate reversal after triggering
        """
        if len(flow) < 30:
            return None

        recent = flow[-30:]
        current_price = recent[-1].price

        # Find nearest round number
        round_price = self._nearest_round_number(current_price)
        distance_to_round = abs(current_price - round_price) / current_price

        if distance_to_round > self.round_number_tolerance * 5:
            return None  # Too far from round number

        # Check for probing behavior: quick move toward round, then reversal
        prices = [t.price for t in recent]

        # Did we approach and bounce?
        approached_round = any(
            abs(p - round_price) / round_price < self.round_number_tolerance
            for p in prices[:-5]
        )

        if approached_round:
            # Check for reversal
            pre_approach = prices[:15]
            post_approach = prices[-10:]

            pre_direction = 1 if pre_approach[-1] > pre_approach[0] else -1
            post_direction = 1 if post_approach[-1] > post_approach[0] else -1

            if pre_direction != post_direction:
                # Reversal detected - likely stop hunt
                confidence = 0.7

                # Predict they'll try again
                predicted_action = "buy" if current_price < round_price else "sell"

                return AlgoSignature(
                    archetype=AlgoArchetype.STOP_HUNT,
                    confidence=confidence,
                    predicted_action=predicted_action,
                    predicted_size=sum(t.volume for t in recent) / len(recent),
                    predicted_timing=30.0,  # Quick probes
                    exploitation_edge=0.02,  # Can fade the hunt
                )

        return None

    def _detect_vwap_twap(self, flow: List[OrderFlowTick]) -> Optional[AlgoSignature]:
        """
        Detect VWAP/TWAP execution algorithms.

        Characteristics:
        - Steady, even-paced execution
        - Volume proportional to market volume
        - One-sided but patient
        """
        if len(flow) < 100:
            return None

        recent = flow[-100:]

        # Check for steady one-sided flow
        directions = [1 if t.aggressor == "buy" else -1 for t in recent]
        net_direction = sum(directions) / len(directions)

        if abs(net_direction) < 0.3:
            return None  # Not one-sided enough

        # Check for even pacing (low variance in inter-trade times)
        time_gaps = []
        for i in range(1, len(recent)):
            gap = (recent[i].timestamp - recent[i-1].timestamp).total_seconds()
            if gap > 0:
                time_gaps.append(gap)

        if not time_gaps:
            return None

        mean_gap = sum(time_gaps) / len(time_gaps)
        var_gap = sum((g - mean_gap) ** 2 for g in time_gaps) / len(time_gaps)
        cv_gap = math.sqrt(var_gap) / mean_gap if mean_gap > 0 else float('inf')

        # VWAP/TWAP has low coefficient of variation in timing
        if cv_gap < 0.5:
            predicted_action = "buy" if net_direction > 0 else "sell"

            return AlgoSignature(
                archetype=AlgoArchetype.VWAP_TWAP,
                confidence=min(0.85, 0.5 + (1 - cv_gap) * 0.5),
                predicted_action=predicted_action,
                predicted_size=sum(t.volume for t in recent) / len(recent),
                predicted_timing=mean_gap,
                exploitation_edge=0.005,  # Small edge, but consistent
            )

        return None

    def _detect_market_maker(self, flow: List[OrderFlowTick]) -> Optional[AlgoSignature]:
        """
        Detect market maker behavior.

        Characteristics:
        - Symmetric quoting around mid
        - Quick quote updates after trades
        - Inventory management: lean quotes after imbalance
        """
        if len(flow) < 50:
            return None

        recent = flow[-50:]

        # Check spread consistency
        spreads = [(t.ask - t.bid) for t in recent]
        mean_spread = sum(spreads) / len(spreads)
        var_spread = sum((s - mean_spread) ** 2 for s in spreads) / len(spreads)

        # Tight, consistent spreads indicate market maker
        if var_spread / (mean_spread ** 2 + 1e-10) > 0.1:
            return None  # Too much spread variation

        # Check for quote leaning after trades
        buy_trades = [t for t in recent if t.aggressor == "buy"]
        sell_trades = [t for t in recent if t.aggressor == "sell"]

        if not buy_trades or not sell_trades:
            return None

        # After buys, MM should lean offer (inventory)
        # After sells, MM should lean bid
        buy_vol = sum(t.volume for t in buy_trades)
        sell_vol = sum(t.volume for t in sell_trades)

        if buy_vol > sell_vol * 1.5:
            # MM likely has long inventory, will want to sell
            return AlgoSignature(
                archetype=AlgoArchetype.MARKET_MAKER,
                confidence=0.6,
                predicted_action="sell",
                predicted_size=buy_vol - sell_vol,
                predicted_timing=1.0,
                exploitation_edge=mean_spread * 0.3,
            )
        elif sell_vol > buy_vol * 1.5:
            return AlgoSignature(
                archetype=AlgoArchetype.MARKET_MAKER,
                confidence=0.6,
                predicted_action="buy",
                predicted_size=sell_vol - buy_vol,
                predicted_timing=1.0,
                exploitation_edge=mean_spread * 0.3,
            )

        return None

    def _nearest_round_number(self, price: float) -> float:
        """Find nearest psychologically significant round number."""
        # Determine appropriate rounding based on price magnitude
        if price >= 1000:
            base = 100
        elif price >= 100:
            base = 10
        elif price >= 10:
            base = 1
        else:
            base = 0.5

        # Use math.floor with +0.5 for standard rounding (not banker's rounding)
        return math.floor(price / base + 0.5) * base

    def _estimate_momentum_size(self, flow: List[OrderFlowTick]) -> float:
        """Estimate likely size of momentum continuation."""
        recent = flow[-20:]
        return sum(t.volume for t in recent) / len(recent) * 1.2

    def _estimate_mean_rev_size(self, flow: List[OrderFlowTick], z_score: float) -> float:
        """Estimate mean reversion position size."""
        recent = flow[-20:]
        base_size = sum(t.volume for t in recent) / len(recent)
        # Larger positions at more extreme z-scores
        return base_size * min(3.0, abs(z_score))

    def _determine_exploitation(self, sig: AlgoSignature) -> Optional[str]:
        """Determine how to exploit detected algorithm."""
        if sig.confidence < self.detection_threshold:
            return None

        if sig.archetype == AlgoArchetype.MOMENTUM:
            # Front-run momentum continuation
            return "front_run"

        elif sig.archetype == AlgoArchetype.MEAN_REVERSION:
            # Can either front-run their entry or fade their position
            return "front_run" if sig.confidence > 0.8 else "avoid"

        elif sig.archetype == AlgoArchetype.INDEX_REBALANCE:
            # Front-run the predictable flow
            return "front_run"

        elif sig.archetype == AlgoArchetype.STOP_HUNT:
            # Fade the hunt - don't get stopped out
            return "fade"

        elif sig.archetype == AlgoArchetype.VWAP_TWAP:
            # Can front-run if confident
            return "front_run" if sig.confidence > 0.75 else "avoid"

        elif sig.archetype == AlgoArchetype.MARKET_MAKER:
            # Trade in direction of their inventory unwind
            return "front_run"

        return None

    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """Generate trading signal from adversarial analysis."""
        state = self.detect(symbol)

        if not state.has_opportunity():
            return None

        top_sig = state.detected_algos[0]

        # Determine signal direction based on exploitation strategy
        if state.exploitation_signal == "front_run":
            # Trade in same direction as predicted algo action
            if top_sig.predicted_action == "buy":
                signal_type = SignalType.LONG
                direction = 1.0
            else:
                signal_type = SignalType.SHORT
                direction = -1.0

        elif state.exploitation_signal == "fade":
            # Trade opposite to predicted algo action
            if top_sig.predicted_action == "buy":
                signal_type = SignalType.SHORT
                direction = -1.0
            else:
                signal_type = SignalType.LONG
                direction = 1.0

        else:
            return None

        # Urgency based on predicted timing
        urgency = 1.0 / (1.0 + top_sig.predicted_timing / 60.0)

        return Signal(
            source=SignalSource.ADVERSARIAL,
            signal_type=signal_type,
            symbol=symbol,
            direction=direction,
            confidence=top_sig.confidence,
            urgency=urgency,
            metadata={
                "archetype": top_sig.archetype.name,
                "exploitation": state.exploitation_signal,
                "predicted_action": top_sig.predicted_action,
                "predicted_timing_sec": top_sig.predicted_timing,
                "exploitation_edge": top_sig.exploitation_edge,
            },
        )

    def inject_backtest_data(
        self,
        symbol: str,
        trades: List[Dict],
    ) -> None:
        """
        Inject historical trade data for backtesting.

        Args:
            symbol: Ticker symbol
            trades: List of dicts with keys:
                - timestamp: datetime
                - price: float
                - volume: float
                - aggressor: "buy" or "sell"
                - bid: float
                - ask: float
        """
        for trade in trades:
            self.update(
                symbol=symbol,
                price=trade["price"],
                volume=trade["volume"],
                aggressor=trade["aggressor"],
                bid=trade["bid"],
                ask=trade["ask"],
                timestamp=trade["timestamp"],
            )
