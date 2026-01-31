"""
The Orchestrator - Multi-Edge Ensemble Day Trading System

The main strategy that orchestrates all edge sources to make trading decisions.

Decision process:
1. Update all edge engines with latest data
2. Detect market regime
3. For each potential trade:
   a. Get votes from all 6 edges
   b. Check for vetoes (any veto blocks the trade)
   c. Calculate consensus score
   d. Only trade if >= 4 edges agree
4. Size position based on consensus strength
"""

from datetime import datetime, time
from typing import Dict, Set

from .types import (
    AssetState,
    EdgeSignal,
    EdgeVote,
    MarketRegime,
    OrchestratorSignal,
    TradeType,
)
from .edges import (
    MarketRegimeEngine,
    RelativeStrengthEngine,
    StatisticalExtremeDetector,
    VolumeProfileEngine,
    CrossAssetEngine,
    TimeOfDayEngine,
)


class Orchestrator:
    """
    The main strategy that orchestrates all edge sources.

    Decision process:
    1. Update all edge engines with latest data
    2. Detect market regime
    3. For each potential trade:
       a. Get votes from all 6 edges
       b. Check for vetoes (any veto blocks the trade)
       c. Calculate consensus score
       d. Only trade if >= 4 edges agree
    4. Size position based on consensus strength
    """

    def __init__(self):
        # Initialize all edge engines
        self.regime_engine = MarketRegimeEngine()
        self.relative_strength = RelativeStrengthEngine()
        self.statistics = StatisticalExtremeDetector()
        self.volume_profile = VolumeProfileEngine()
        self.cross_asset = CrossAssetEngine()
        self.time_of_day = TimeOfDayEngine()

        # Asset states
        self.asset_states: Dict[str, AssetState] = {}

        # Current positions
        self.positions: Dict[str, dict] = {}

        # Settings
        self.min_consensus_edges = 4  # At least 4 edges must agree
        self.min_consensus_score = 0.5  # Minimum weighted score
        self.max_position_pct = 0.03  # 3% of account max
        self.atr_stop_mult = 2.5
        self.atr_target_mult = 4.0

        # Reference assets we need to track
        self.reference_assets: Set[str] = {"SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE", "XLY", "XLV"}

    def update_asset(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Update state for a single asset."""

        # Initialize state if needed
        if symbol not in self.asset_states:
            self.asset_states[symbol] = AssetState(symbol=symbol)

        state = self.asset_states[symbol]

        # Update price/volume history
        state.prices.append(close)
        state.highs.append(high)
        state.lows.append(low)
        state.volumes.append(volume)
        state.timestamps.append(timestamp)

        # Update day's metrics
        if len(state.prices) == 1 or timestamp.time() <= time(9, 35):
            state.day_open = open_price
            state.day_high = high
            state.day_low = low
            state.day_volume = volume
        else:
            state.day_high = max(state.day_high, high)
            state.day_low = min(state.day_low, low)
            state.day_volume += volume

        # Calculate VWAP
        if len(state.prices) > 0:
            typical_prices = [(h + l + c) / 3 for h, l, c in
                            zip(list(state.highs), list(state.lows), list(state.prices))]
            volumes_list = list(state.volumes)
            cum_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes_list))
            cum_vol = sum(volumes_list)
            state.vwap = cum_tp_vol / cum_vol if cum_vol > 0 else close

        # Calculate ATR
        if len(state.prices) > 1:
            trs = []
            prices = list(state.prices)
            highs = list(state.highs)
            lows = list(state.lows)
            for i in range(-min(14, len(prices)-1), 0):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i-1]),
                    abs(lows[i] - prices[i-1])
                )
                trs.append(tr)
            state.atr = sum(trs) / len(trs) if trs else 0
            state.atr_pct = state.atr / close if close > 0 else 0

        # Calculate RSI
        if len(state.prices) > 14:
            prices = list(state.prices)
            gains = []
            losses = []
            for i in range(-14, 0):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            avg_gain = sum(gains) / 14
            avg_loss = max(sum(losses) / 14, 0.0001)
            rs = avg_gain / avg_loss
            state.rsi = 100 - (100 / (1 + rs))

        # Update all edge engines
        self.regime_engine.update(symbol, state)
        self.relative_strength.update(symbol, state)
        self.cross_asset.update(symbol, state)

    def generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> OrchestratorSignal:
        """
        Generate trading signal using ensemble of all edges.
        """
        if symbol not in self.asset_states:
            return self._hold_signal(symbol, timestamp, 0, "No data")

        state = self.asset_states[symbol]
        if len(state.prices) < 30:
            return self._hold_signal(symbol, timestamp, state.prices[-1] if state.prices else 0,
                                    "Warming up")

        price = state.prices[-1]

        # Check existing position
        if symbol in self.positions:
            return self._check_exit(symbol, timestamp, state)

        # Step 1: Get market regime
        regime, regime_conf, regime_reason = self.regime_engine.detect_regime(timestamp)

        # Step 2: Skip if market is too uncertain or volatile
        if regime == MarketRegime.UNKNOWN:
            return self._hold_signal(symbol, timestamp, price, "Market regime unknown")
        if regime == MarketRegime.HIGH_VOLATILITY:
            return self._hold_signal(symbol, timestamp, price, "High volatility - sitting out")

        # Step 3: Determine potential trade direction based on regime
        if regime in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP, MarketRegime.REVERSAL_UP]:
            potential_direction = "long"
            trade_type = TradeType.MOMENTUM_CONTINUATION
        elif regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN, MarketRegime.REVERSAL_DOWN]:
            potential_direction = "short"
            trade_type = TradeType.MOMENTUM_CONTINUATION
        else:  # Range bound
            # Look for mean reversion
            stats = self.statistics.analyze(state)
            if stats["price_zscore"] > 1.5:
                potential_direction = "short"
                trade_type = TradeType.MEAN_REVERSION
            elif stats["price_zscore"] < -1.5:
                potential_direction = "long"
                trade_type = TradeType.MEAN_REVERSION
            else:
                return self._hold_signal(symbol, timestamp, price,
                                        f"Range-bound, no extreme. z={stats['price_zscore']:.1f}")

        # Step 4: Collect votes from all edges
        votes: Dict[str, EdgeSignal] = {}

        # Edge 1: Relative Strength
        votes["RelativeStrength"] = self.relative_strength.get_vote(symbol)

        # Edge 2: Statistics
        votes["Statistics"] = self.statistics.get_vote(state, regime)

        # Edge 3: Volume Profile
        votes["VolumeProfile"] = self.volume_profile.get_vote(state, regime)

        # Edge 4: Cross-Asset
        votes["CrossAsset"] = self.cross_asset.get_vote(symbol, potential_direction)

        # Edge 5: Time of Day
        votes["TimeOfDay"] = self.time_of_day.get_vote(timestamp, trade_type)

        # Edge 6: Regime (implicit vote based on regime strength)
        if regime_conf > 0.7:
            if potential_direction == "long":
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.STRONG_LONG, regime_conf, regime_reason)
            else:
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.STRONG_SHORT, regime_conf, regime_reason)
        elif regime_conf > 0.5:
            if potential_direction == "long":
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.LONG, regime_conf, regime_reason)
            else:
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.SHORT, regime_conf, regime_reason)
        else:
            votes["Regime"] = EdgeSignal("Regime", EdgeVote.NEUTRAL, regime_conf, regime_reason)

        # Step 5: Check for vetoes
        for edge_name, signal in votes.items():
            if potential_direction == "long" and signal.vote == EdgeVote.VETO_LONG:
                return self._hold_signal(symbol, timestamp, price,
                                        f"VETO from {edge_name}: {signal.reason}")
            if potential_direction == "short" and signal.vote == EdgeVote.VETO_SHORT:
                return self._hold_signal(symbol, timestamp, price,
                                        f"VETO from {edge_name}: {signal.reason}")

        # Step 6: Calculate consensus
        agreeing_edges = 0
        consensus_score = 0.0

        for edge_name, signal in votes.items():
            vote_value = signal.vote.value

            if potential_direction == "long":
                if vote_value > 0:
                    agreeing_edges += 1
                    consensus_score += vote_value * signal.confidence
            else:  # short
                if vote_value < 0:
                    agreeing_edges += 1
                    consensus_score += abs(vote_value) * signal.confidence

        # Normalize score
        consensus_score = consensus_score / len(votes) if votes else 0

        # Step 7: Check if we have enough agreement
        if agreeing_edges < self.min_consensus_edges:
            return self._hold_signal(symbol, timestamp, price,
                                    f"Insufficient consensus: {agreeing_edges}/6 edges agree")

        if consensus_score < self.min_consensus_score:
            return self._hold_signal(symbol, timestamp, price,
                                    f"Weak consensus score: {consensus_score:.2f}")

        # Step 8: Calculate position size based on consensus strength
        base_size = 0.01  # 1% base
        size_multiplier = min(2.0, 1.0 + consensus_score)
        position_size = min(self.max_position_pct, base_size * size_multiplier)

        # Step 9: Calculate stops based on ATR
        atr = state.atr
        if potential_direction == "long":
            stop_loss = price - (atr * self.atr_stop_mult)
            take_profit = price + (atr * self.atr_target_mult)
        else:
            stop_loss = price + (atr * self.atr_stop_mult)
            take_profit = price - (atr * self.atr_target_mult)

        # Step 10: Create position
        self.positions[symbol] = {
            "direction": 1 if potential_direction == "long" else -1,
            "entry_price": price,
            "entry_time": timestamp,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "best_price": price,
            "trailing_active": False,
            "regime_at_entry": regime,
        }

        # Build reason string
        vote_summary = ", ".join([f"{k}:{v.vote.name}" for k, v in votes.items()])
        reason = f"{agreeing_edges}/6 edges agree, score={consensus_score:.2f}. {vote_summary}"

        return OrchestratorSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="buy" if potential_direction == "long" else "short",
            trade_type=trade_type,
            size=position_size,
            confidence=consensus_score,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            edge_votes={k: v.vote for k, v in votes.items()},
            edge_reasons={k: v.reason for k, v in votes.items()},
            consensus_score=consensus_score,
            market_regime=regime,
            reason=reason,
        )

    def _check_exit(self, symbol: str, timestamp: datetime, state: AssetState) -> OrchestratorSignal:
        """Check for exit signals on existing position."""
        position = self.positions[symbol]
        direction = position["direction"]
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        atr = position["atr"]
        best_price = position["best_price"]

        price = state.prices[-1]
        high = state.highs[-1]
        low = state.lows[-1]

        action = "hold"
        reason = ""

        # Update best price for trailing
        if direction > 0 and price > best_price:
            position["best_price"] = price
            best_price = price
        elif direction < 0 and price < best_price:
            position["best_price"] = price
            best_price = price

        # Check trailing stop activation (at 2x ATR profit)
        if not position["trailing_active"]:
            profit_distance = (best_price - entry_price) * direction
            if profit_distance >= atr * 2.0:
                position["trailing_active"] = True
                # Move stop to breakeven + 1 ATR
                if direction > 0:
                    new_stop = entry_price + atr
                    if new_stop > stop_loss:
                        position["stop_loss"] = new_stop
                        stop_loss = new_stop
                else:
                    new_stop = entry_price - atr
                    if new_stop < stop_loss:
                        position["stop_loss"] = new_stop
                        stop_loss = new_stop

        # Update trailing stop
        if position["trailing_active"]:
            if direction > 0:
                trailing_stop = best_price - atr * 1.5
                if trailing_stop > stop_loss:
                    position["stop_loss"] = trailing_stop
                    stop_loss = trailing_stop
            else:
                trailing_stop = best_price + atr * 1.5
                if trailing_stop < stop_loss:
                    position["stop_loss"] = trailing_stop
                    stop_loss = trailing_stop

        # Check exit conditions
        if direction > 0:  # Long
            if low <= stop_loss:
                action = "sell"
                pnl_pct = (stop_loss - entry_price) / entry_price * 100
                reason = f"Stop hit at ${stop_loss:.2f} | P&L: {pnl_pct:+.2f}%"
            elif high >= take_profit:
                action = "sell"
                pnl_pct = (take_profit - entry_price) / entry_price * 100
                reason = f"Target hit at ${take_profit:.2f} | P&L: {pnl_pct:+.2f}%"
        else:  # Short
            if high >= stop_loss:
                action = "cover"
                pnl_pct = (entry_price - stop_loss) / entry_price * 100
                reason = f"Stop hit at ${stop_loss:.2f} | P&L: {pnl_pct:+.2f}%"
            elif low <= take_profit:
                action = "cover"
                pnl_pct = (entry_price - take_profit) / entry_price * 100
                reason = f"Target hit at ${take_profit:.2f} | P&L: {pnl_pct:+.2f}%"

        # End of day close
        if timestamp.time() >= time(15, 55):
            action = "sell" if direction > 0 else "cover"
            pnl_pct = (price - entry_price) * direction / entry_price * 100
            reason = f"End of day close | P&L: {pnl_pct:+.2f}%"

        if action in ("sell", "cover"):
            del self.positions[symbol]

        return OrchestratorSignal(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            trade_type=TradeType.MOMENTUM_CONTINUATION,
            size=0,
            confidence=0.8 if action != "hold" else 0,
            entry_price=price,
            reason=reason,
        )

    def _hold_signal(self, symbol: str, timestamp: datetime, price: float, reason: str) -> OrchestratorSignal:
        """Generate a hold signal."""
        return OrchestratorSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="hold",
            trade_type=TradeType.MOMENTUM_CONTINUATION,
            size=0,
            confidence=0,
            entry_price=price,
            reason=reason,
        )

    def clear_positions(self):
        """Clear all positions (for warmup)."""
        self.positions.clear()


def create_orchestrator() -> Orchestrator:
    """Create an Orchestrator instance with default settings."""
    return Orchestrator()
