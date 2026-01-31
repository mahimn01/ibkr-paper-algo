"""
The Orchestrator - Multi-Edge Ensemble Day Trading System

A fundamentally different approach to day trading that combines multiple
independent edge sources into a unified decision framework.

=============================================================================
PHILOSOPHY: WHY THIS IS DIFFERENT
=============================================================================

Traditional retail algo: "If RSI < 30 and price > SMA, BUY"
  - Uses lagging indicators
  - Trades in isolation
  - No market context
  - Same strategy for all conditions

The Orchestrator: "What is the market telling us, and where do multiple
                   independent signals align?"
  - Context-first (understand regime before any trade)
  - Multi-asset (never trade a stock in isolation)
  - Statistical (only act at measurable extremes)
  - Ensemble (multiple edges must agree)
  - Anti-fragile (actively detects and avoids traps)

=============================================================================
THE 6 INDEPENDENT EDGE SOURCES
=============================================================================

1. MARKET REGIME ENGINE
   - Detects: Trend day vs Range day vs Reversal day
   - Uses: Market internals, breadth, sector rotation
   - Output: What TYPE of day is it? (changes strategy selection)

2. RELATIVE STRENGTH ENGINE
   - Compares: Stock vs Sector vs Market
   - Detects: Leaders (outperforming) and Laggards (underperforming)
   - Output: Is this stock showing unusual strength/weakness?

3. STATISTICAL EXTREME DETECTOR
   - Measures: Z-scores, percentile ranks, standard deviations
   - Detects: When price/volume/momentum are at extremes
   - Output: Is this a statistically significant setup?

4. VOLUME PROFILE / AUCTION THEORY
   - Analyzes: Where volume occurs at each price level
   - Detects: Value Area, Point of Control, Poor Structure
   - Output: Is price at an auction extreme?

5. CROSS-ASSET CONFIRMATION
   - Monitors: Related stocks, sector ETF, index futures, VIX
   - Detects: Divergences and confirmations
   - Output: Do related assets agree with this trade?

6. TIME-OF-DAY PATTERNS
   - Knows: Opening drive, mid-morning reversal, lunch chop,
            power hour, MOC imbalances
   - Detects: Which time pattern is active
   - Output: Is this the right time for this type of trade?

=============================================================================
ENSEMBLE VOTING
=============================================================================

Each edge source casts a vote: LONG, SHORT, or NEUTRAL
Confidence comes from AGREEMENT, not from any single indicator.

Trade only when:
- At least 4 of 6 edges agree on direction
- No edge is strongly opposed (veto power)
- Position size scales with consensus strength

This means we take FEWER trades, but each trade has multiple reasons to work.

=============================================================================
THE KEY INSIGHT
=============================================================================

Most retail algos ask: "Should I buy or sell based on this indicator?"

The Orchestrator asks: "What kind of day is it, what is this stock doing
relative to everything else, is this statistically extreme, do related
assets confirm, and is this the right time?"

Only when ALL these questions have favorable answers do we trade.

=============================================================================
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple, Set
import statistics


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class MarketRegime(Enum):
    """What type of day is the market having?"""
    STRONG_TREND_UP = auto()     # Clear uptrend, buy dips
    TREND_UP = auto()            # Upward bias, be long
    RANGE_BOUND = auto()         # Chop, fade extremes
    TREND_DOWN = auto()          # Downward bias, be short
    STRONG_TREND_DOWN = auto()   # Clear downtrend, sell rallies
    REVERSAL_UP = auto()         # Was down, now reversing up
    REVERSAL_DOWN = auto()       # Was up, now reversing down
    HIGH_VOLATILITY = auto()     # Extreme moves, reduce size
    UNKNOWN = auto()             # Not enough data


class EdgeVote(Enum):
    """Vote from each edge source."""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2
    VETO_LONG = -99    # Blocks long trades
    VETO_SHORT = 99    # Blocks short trades


class TradeType(Enum):
    """What kind of trade setup is this?"""
    MOMENTUM_CONTINUATION = auto()  # Ride existing trend
    MEAN_REVERSION = auto()          # Fade to the mean
    BREAKOUT = auto()                # New range expansion
    RELATIVE_VALUE = auto()          # Pairs/spread trade
    OPENING_DRIVE = auto()           # First 30 min momentum
    REVERSAL = auto()                # Trend change


@dataclass
class EdgeSignal:
    """Signal from a single edge source."""
    edge_name: str
    vote: EdgeVote
    confidence: float  # 0 to 1
    reason: str
    data: Dict = field(default_factory=dict)


@dataclass
class OrchestratorSignal:
    """Final signal from the ensemble."""
    symbol: str
    timestamp: datetime
    action: str  # 'buy', 'short', 'sell', 'cover', 'hold'
    trade_type: TradeType
    size: float  # Position size as fraction
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Edge breakdown
    edge_votes: Dict[str, EdgeVote] = field(default_factory=dict)
    edge_reasons: Dict[str, str] = field(default_factory=dict)
    consensus_score: float = 0.0  # -2 to +2

    # Context
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    relative_strength_rank: float = 0.0  # 0-100 percentile
    statistical_zscore: float = 0.0

    reason: str = ""


@dataclass
class AssetState:
    """Current state of a single asset."""
    symbol: str
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    highs: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    lows: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=500))

    # Derived metrics (updated on each bar)
    vwap: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    rsi: float = 50.0

    # Volume profile
    volume_by_price: Dict[float, float] = field(default_factory=dict)
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    point_of_control: float = 0.0

    # Statistical metrics
    price_zscore: float = 0.0
    volume_zscore: float = 0.0
    momentum_zscore: float = 0.0

    # Today's metrics
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    day_volume: float = 0.0


# =============================================================================
# EDGE 1: MARKET REGIME ENGINE
# =============================================================================

class MarketRegimeEngine:
    """
    Determines what type of day the market is having.

    This is CRITICAL - different regimes require different strategies:
    - Trend days: Trade with trend, don't fade
    - Range days: Fade extremes, mean revert
    - Reversal days: Be patient, wait for confirmation
    - High volatility: Reduce size, widen stops

    Uses SPY as the market proxy, with sector ETF confirmation.
    """

    def __init__(self):
        self.spy_state: Optional[AssetState] = None
        self.qqq_state: Optional[AssetState] = None
        self.iwm_state: Optional[AssetState] = None  # Small caps
        self.vix_proxy: float = 0.0  # We'll estimate from SPY volatility

        # Intraday tracking
        self.morning_direction: Optional[str] = None  # 'up', 'down', 'flat'
        self.midday_direction: Optional[str] = None
        self.trend_bars: int = 0  # Consecutive bars in same direction

    def update(self, symbol: str, state: AssetState):
        """Update with latest asset state."""
        if symbol == "SPY":
            self.spy_state = state
        elif symbol == "QQQ":
            self.qqq_state = state
        elif symbol == "IWM":
            self.iwm_state = state

    def detect_regime(self, current_time: datetime) -> Tuple[MarketRegime, float, str]:
        """
        Detect current market regime.

        Returns: (regime, confidence, reason)
        """
        if self.spy_state is None or len(self.spy_state.prices) < 20:
            return MarketRegime.UNKNOWN, 0.0, "Insufficient data"

        prices = list(self.spy_state.prices)
        highs = list(self.spy_state.highs)
        lows = list(self.spy_state.lows)

        current = prices[-1]

        # Calculate day's metrics
        if self.spy_state.day_open > 0:
            day_return = (current - self.spy_state.day_open) / self.spy_state.day_open
        else:
            day_return = 0

        # Short-term momentum (last 5 bars = 25 min)
        short_return = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # Medium-term momentum (last 20 bars = ~1.5 hours)
        medium_return = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0

        # Trend consistency: How many of last 20 bars closed in same direction?
        if len(prices) >= 20:
            up_bars = sum(1 for i in range(-20, 0) if prices[i] > prices[i-1])
            trend_consistency = abs(up_bars - 10) / 10  # 0 = balanced, 1 = all same direction
            trend_is_up = up_bars > 10
        else:
            trend_consistency = 0
            trend_is_up = day_return > 0

        # Range analysis: Is price making new highs/lows or stuck?
        if len(highs) >= 20:
            recent_high = max(highs[-20:])
            recent_low = min(lows[-20:])
            range_position = (current - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        else:
            range_position = 0.5

        # Volatility (ATR-based)
        volatility = self.spy_state.atr_pct if self.spy_state.atr_pct > 0 else 0.01

        # Breadth proxy: Compare SPY to QQQ and IWM
        breadth_score = 0
        if self.qqq_state and len(self.qqq_state.prices) >= 20:
            qqq_return = (self.qqq_state.prices[-1] - self.qqq_state.prices[-20]) / self.qqq_state.prices[-20]
            if (qqq_return > 0) == (medium_return > 0):
                breadth_score += 1
        if self.iwm_state and len(self.iwm_state.prices) >= 20:
            iwm_return = (self.iwm_state.prices[-1] - self.iwm_state.prices[-20]) / self.iwm_state.prices[-20]
            if (iwm_return > 0) == (medium_return > 0):
                breadth_score += 1

        # Now classify the regime

        # High volatility supersedes other regimes
        if volatility > 0.02:  # 2% ATR is huge
            return MarketRegime.HIGH_VOLATILITY, 0.8, f"Extreme volatility: ATR={volatility*100:.1f}%"

        # Strong trends
        if day_return > 0.01 and trend_consistency > 0.6 and trend_is_up:
            conf = min(0.9, 0.5 + trend_consistency * 0.4)
            return MarketRegime.STRONG_TREND_UP, conf, f"Strong uptrend: +{day_return*100:.1f}%, consistency={trend_consistency:.0%}"

        if day_return < -0.01 and trend_consistency > 0.6 and not trend_is_up:
            conf = min(0.9, 0.5 + trend_consistency * 0.4)
            return MarketRegime.STRONG_TREND_DOWN, conf, f"Strong downtrend: {day_return*100:.1f}%, consistency={trend_consistency:.0%}"

        # Moderate trends
        if day_return > 0.003 and short_return > 0 and medium_return > 0:
            return MarketRegime.TREND_UP, 0.6, f"Upward bias: +{day_return*100:.2f}%"

        if day_return < -0.003 and short_return < 0 and medium_return < 0:
            return MarketRegime.TREND_DOWN, 0.6, f"Downward bias: {day_return*100:.2f}%"

        # Reversals
        if day_return < -0.005 and short_return > 0.002:
            return MarketRegime.REVERSAL_UP, 0.5, f"Potential reversal up from {day_return*100:.2f}%"

        if day_return > 0.005 and short_return < -0.002:
            return MarketRegime.REVERSAL_DOWN, 0.5, f"Potential reversal down from +{day_return*100:.2f}%"

        # Range bound (default)
        return MarketRegime.RANGE_BOUND, 0.7, f"Range-bound: {day_return*100:+.2f}%, low trend consistency"


# =============================================================================
# EDGE 2: RELATIVE STRENGTH ENGINE
# =============================================================================

class RelativeStrengthEngine:
    """
    Compares a stock's performance to its sector and the market.

    Key insight: Stocks that outperform their sector tend to continue outperforming.
    Stocks that underperform their sector tend to continue underperforming.

    This is NOT about absolute momentum - it's about RELATIVE performance.

    Example:
    - INTC up 11%, SMH up 6%, SPY up 1% → INTC is a leader, trade WITH it
    - INTC up 2%, SMH up 6%, SPY up 2% → INTC is a laggard, avoid or fade
    """

    # Sector ETF mappings
    SECTOR_MAP = {
        # Tech / Semiconductors
        "INTC": "SMH", "AMD": "SMH", "NVDA": "SMH", "MU": "SMH",
        "AVGO": "SMH", "QCOM": "SMH", "TSM": "SMH", "MRVL": "SMH",
        "SMCI": "SMH", "ARM": "SMH",
        # Tech general
        "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK",
        "CRM": "XLK", "ORCL": "XLK", "ADBE": "XLK",
        # Consumer tech
        "AMZN": "XLY", "TSLA": "XLY", "NFLX": "XLY",
        # Financials
        "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
        "C": "XLF", "WFC": "XLF", "SCHW": "XLF",
        # Energy
        "XOM": "XLE", "CVX": "XLE", "OXY": "XLE", "SLB": "XLE",
        # EV / Clean energy
        "RIVN": "QCLN", "LCID": "QCLN", "NIO": "QCLN", "PLUG": "QCLN",
        # Consumer
        "SBUX": "XLY", "MCD": "XLY", "NKE": "XLY", "DIS": "XLY",
        # Healthcare
        "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "ABBV": "XLV",
        # Leveraged ETFs - compare to underlying
        "SOXL": "SMH", "TQQQ": "QQQ",
    }

    def __init__(self):
        self.asset_states: Dict[str, AssetState] = {}
        self.spy_state: Optional[AssetState] = None

    def update(self, symbol: str, state: AssetState):
        """Update with latest asset state."""
        self.asset_states[symbol] = state
        if symbol == "SPY":
            self.spy_state = state

    def calculate_relative_strength(
        self,
        symbol: str,
        lookback_bars: int = 20
    ) -> Tuple[float, float, float, str]:
        """
        Calculate relative strength vs sector and market.

        Returns: (rs_vs_sector, rs_vs_market, percentile_rank, reason)

        rs > 0 means outperforming
        rs < 0 means underperforming
        percentile_rank is 0-100 (100 = strongest)
        """
        if symbol not in self.asset_states:
            return 0.0, 0.0, 50.0, "No data"

        state = self.asset_states[symbol]
        if len(state.prices) < lookback_bars:
            return 0.0, 0.0, 50.0, "Insufficient history"

        prices = list(state.prices)
        stock_return = (prices[-1] - prices[-lookback_bars]) / prices[-lookback_bars]

        # Get sector ETF return
        sector_etf = self.SECTOR_MAP.get(symbol, "SPY")
        sector_return = 0.0
        if sector_etf in self.asset_states:
            sector_prices = list(self.asset_states[sector_etf].prices)
            if len(sector_prices) >= lookback_bars:
                sector_return = (sector_prices[-1] - sector_prices[-lookback_bars]) / sector_prices[-lookback_bars]

        # Get SPY return
        market_return = 0.0
        if self.spy_state and len(self.spy_state.prices) >= lookback_bars:
            spy_prices = list(self.spy_state.prices)
            market_return = (spy_prices[-1] - spy_prices[-lookback_bars]) / spy_prices[-lookback_bars]

        # Relative strength
        rs_vs_sector = stock_return - sector_return
        rs_vs_market = stock_return - market_return

        # Calculate percentile rank among all tracked stocks
        all_returns = []
        for sym, st in self.asset_states.items():
            if sym not in ["SPY", "QQQ", "IWM"] and len(st.prices) >= lookback_bars:
                ret = (st.prices[-1] - st.prices[-lookback_bars]) / st.prices[-lookback_bars]
                all_returns.append((sym, ret))

        if all_returns:
            sorted_returns = sorted(all_returns, key=lambda x: x[1])
            rank = next((i for i, (s, r) in enumerate(sorted_returns) if s == symbol), len(sorted_returns)//2)
            percentile = (rank / len(sorted_returns)) * 100
        else:
            percentile = 50.0

        # Build reason
        reason_parts = []
        if rs_vs_sector > 0.005:
            reason_parts.append(f"Outperforming {sector_etf} by {rs_vs_sector*100:.1f}%")
        elif rs_vs_sector < -0.005:
            reason_parts.append(f"Underperforming {sector_etf} by {abs(rs_vs_sector)*100:.1f}%")

        if rs_vs_market > 0.01:
            reason_parts.append(f"Leading SPY by {rs_vs_market*100:.1f}%")
        elif rs_vs_market < -0.01:
            reason_parts.append(f"Lagging SPY by {abs(rs_vs_market)*100:.1f}%")

        reason_parts.append(f"Rank: {percentile:.0f}th percentile")

        return rs_vs_sector, rs_vs_market, percentile, " | ".join(reason_parts)

    def get_vote(self, symbol: str) -> EdgeSignal:
        """Get voting signal based on relative strength."""
        rs_sector, rs_market, percentile, reason = self.calculate_relative_strength(symbol)

        # Strong signals
        if percentile > 90 and rs_sector > 0.01:
            return EdgeSignal("RelativeStrength", EdgeVote.STRONG_LONG, 0.9, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        if percentile < 10 and rs_sector < -0.01:
            return EdgeSignal("RelativeStrength", EdgeVote.STRONG_SHORT, 0.9, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        # Moderate signals
        if percentile > 70 and rs_sector > 0.005:
            return EdgeSignal("RelativeStrength", EdgeVote.LONG, 0.6, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        if percentile < 30 and rs_sector < -0.005:
            return EdgeSignal("RelativeStrength", EdgeVote.SHORT, 0.6, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        # Conflict detection (stock vs sector divergence can be a trap)
        if rs_sector > 0.02 and rs_market < -0.01:
            # Stock up, sector down, market down - potential trap
            return EdgeSignal("RelativeStrength", EdgeVote.VETO_LONG, 0.7,
                            f"Divergence warning: stock up but sector/market down",
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        return EdgeSignal("RelativeStrength", EdgeVote.NEUTRAL, 0.5, reason,
                        {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})


# =============================================================================
# EDGE 3: STATISTICAL EXTREME DETECTOR
# =============================================================================

class StatisticalExtremeDetector:
    """
    Identifies when price, volume, or momentum are at statistical extremes.

    Key insight: Trade only at extremes where mean reversion OR breakout
    is statistically likely. Avoid the middle where outcomes are random.

    Uses z-scores to measure distance from mean in standard deviations.
    |z| > 2.0 is significant (95% confidence)
    |z| > 2.5 is very significant (99% confidence)
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def calculate_zscore(self, values: List[float]) -> float:
        """Calculate z-score of most recent value."""
        if len(values) < 10:
            return 0.0
        try:
            mean = statistics.mean(values[:-1])
            stdev = statistics.stdev(values[:-1])
            if stdev < 0.0001:
                return 0.0
            return (values[-1] - mean) / stdev
        except:
            return 0.0

    def calculate_percentile(self, values: List[float]) -> float:
        """Calculate percentile rank of most recent value."""
        if len(values) < 10:
            return 50.0
        sorted_vals = sorted(values[:-1])
        current = values[-1]
        rank = sum(1 for v in sorted_vals if v <= current)
        return (rank / len(sorted_vals)) * 100

    def analyze(self, state: AssetState) -> Dict[str, float]:
        """
        Calculate statistical metrics for the asset.

        Returns dict with:
        - price_zscore: How far from recent mean
        - volume_zscore: Is volume unusually high/low
        - momentum_zscore: Is momentum extreme
        - range_percentile: Where in today's range (0=low, 100=high)
        - extension_from_vwap: Distance from VWAP in ATR units
        """
        result = {
            "price_zscore": 0.0,
            "volume_zscore": 0.0,
            "momentum_zscore": 0.0,
            "range_percentile": 50.0,
            "extension_from_vwap": 0.0,
            "is_extreme": False,
            "extreme_type": "none",
        }

        prices = list(state.prices)
        volumes = list(state.volumes)

        if len(prices) < 20:
            return result

        # Price z-score (vs recent prices)
        result["price_zscore"] = self.calculate_zscore(prices[-self.lookback:])

        # Volume z-score
        if volumes:
            result["volume_zscore"] = self.calculate_zscore(volumes[-self.lookback:])

        # Momentum z-score (rate of change)
        returns = [(prices[i] - prices[i-1]) / prices[i-1]
                   for i in range(-min(self.lookback, len(prices)-1), 0)]
        if returns:
            result["momentum_zscore"] = self.calculate_zscore(returns)

        # Range percentile (where in today's high-low range)
        if state.day_high > state.day_low:
            result["range_percentile"] = ((prices[-1] - state.day_low) /
                                          (state.day_high - state.day_low)) * 100

        # Extension from VWAP
        if state.vwap > 0 and state.atr > 0:
            result["extension_from_vwap"] = (prices[-1] - state.vwap) / state.atr

        # Determine if we're at an extreme
        if abs(result["price_zscore"]) > 2.0 or abs(result["momentum_zscore"]) > 2.0:
            result["is_extreme"] = True
            if result["price_zscore"] > 2.0:
                result["extreme_type"] = "extended_high"
            elif result["price_zscore"] < -2.0:
                result["extreme_type"] = "extended_low"
            elif result["momentum_zscore"] > 2.0:
                result["extreme_type"] = "momentum_surge"
            elif result["momentum_zscore"] < -2.0:
                result["extreme_type"] = "momentum_crash"

        return result

    def get_vote(self, state: AssetState, regime: MarketRegime) -> EdgeSignal:
        """Get voting signal based on statistical analysis."""
        stats = self.analyze(state)

        price_z = stats["price_zscore"]
        mom_z = stats["momentum_zscore"]
        vol_z = stats["volume_zscore"]
        vwap_ext = stats["extension_from_vwap"]

        # In range-bound markets, fade extremes
        if regime == MarketRegime.RANGE_BOUND:
            if price_z > 2.0 and mom_z > 1.5:
                return EdgeSignal("Statistics", EdgeVote.SHORT, 0.7,
                                f"Mean reversion: price z={price_z:.1f}, extended from range",
                                stats)
            if price_z < -2.0 and mom_z < -1.5:
                return EdgeSignal("Statistics", EdgeVote.LONG, 0.7,
                                f"Mean reversion: price z={price_z:.1f}, oversold in range",
                                stats)

        # In trend markets, trade with momentum at moderate extremes
        if regime in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP]:
            if mom_z > 1.5 and price_z < 2.0:  # Momentum strong but not overextended
                return EdgeSignal("Statistics", EdgeVote.LONG, 0.6,
                                f"Momentum confirmation: mom z={mom_z:.1f} in uptrend",
                                stats)
            if price_z > 2.5:  # Too extended even in uptrend
                return EdgeSignal("Statistics", EdgeVote.VETO_LONG, 0.6,
                                f"Overextended: price z={price_z:.1f}, avoid longs",
                                stats)

        if regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN]:
            if mom_z < -1.5 and price_z > -2.0:
                return EdgeSignal("Statistics", EdgeVote.SHORT, 0.6,
                                f"Momentum confirmation: mom z={mom_z:.1f} in downtrend",
                                stats)
            if price_z < -2.5:
                return EdgeSignal("Statistics", EdgeVote.VETO_SHORT, 0.6,
                                f"Overextended: price z={price_z:.1f}, avoid shorts",
                                stats)

        # Volume confirmation
        if vol_z > 2.0 and abs(mom_z) > 1.0:
            # High volume confirms the move
            if mom_z > 1.0:
                return EdgeSignal("Statistics", EdgeVote.LONG, 0.5,
                                f"Volume surge ({vol_z:.1f}σ) confirms momentum",
                                stats)
            else:
                return EdgeSignal("Statistics", EdgeVote.SHORT, 0.5,
                                f"Volume surge ({vol_z:.1f}σ) confirms selling",
                                stats)

        return EdgeSignal("Statistics", EdgeVote.NEUTRAL, 0.4,
                        f"No statistical extreme: z={price_z:.1f}", stats)


# =============================================================================
# EDGE 4: VOLUME PROFILE / AUCTION THEORY
# =============================================================================

class VolumeProfileEngine:
    """
    Analyzes where volume occurs at each price level.

    Key concepts from auction market theory:
    - Value Area: Where 70% of volume occurs (fair value zone)
    - Point of Control (POC): Price with most volume (strongest level)
    - Poor Structure: Low volume areas (price moves fast through these)

    Trading implications:
    - Price above VA: Bullish, look for pullbacks to VA high
    - Price below VA: Bearish, look for rallies to VA low
    - At POC: Consolidation, wait for breakout
    - In poor structure: Expect rapid moves
    """

    def __init__(self, price_increment: float = 0.10):
        self.price_increment = price_increment

    def build_profile(self, state: AssetState) -> Dict[str, float]:
        """
        Build volume profile from intraday data.

        Returns dict with VA high, VA low, POC, and structure quality.
        """
        result = {
            "value_area_high": 0.0,
            "value_area_low": 0.0,
            "poc": 0.0,
            "current_vs_va": "inside",  # 'above', 'below', 'inside'
            "current_vs_poc": 0.0,  # Distance from POC in ATR
            "structure_quality": 0.5,  # 0-1, higher = better structure
        }

        prices = list(state.prices)
        volumes = list(state.volumes)

        if len(prices) < 20 or len(volumes) < 20:
            return result

        # Build volume at price histogram
        vol_by_price: Dict[float, float] = {}
        for price, volume in zip(prices, volumes):
            # Round price to increment
            rounded = round(price / self.price_increment) * self.price_increment
            vol_by_price[rounded] = vol_by_price.get(rounded, 0) + volume

        if not vol_by_price:
            return result

        # Find Point of Control (highest volume price)
        poc_price = max(vol_by_price.keys(), key=lambda p: vol_by_price[p])
        result["poc"] = poc_price

        # Calculate Value Area (70% of volume)
        total_volume = sum(vol_by_price.values())
        target_volume = total_volume * 0.70

        # Start from POC and expand outward
        sorted_prices = sorted(vol_by_price.keys())
        poc_idx = sorted_prices.index(poc_price)

        va_volume = vol_by_price[poc_price]
        low_idx = poc_idx
        high_idx = poc_idx

        while va_volume < target_volume and (low_idx > 0 or high_idx < len(sorted_prices) - 1):
            # Expand to whichever side has more volume
            low_vol = vol_by_price[sorted_prices[low_idx - 1]] if low_idx > 0 else 0
            high_vol = vol_by_price[sorted_prices[high_idx + 1]] if high_idx < len(sorted_prices) - 1 else 0

            if low_vol >= high_vol and low_idx > 0:
                low_idx -= 1
                va_volume += low_vol
            elif high_idx < len(sorted_prices) - 1:
                high_idx += 1
                va_volume += high_vol
            else:
                break

        result["value_area_low"] = sorted_prices[low_idx]
        result["value_area_high"] = sorted_prices[high_idx]

        # Current price vs value area
        current_price = prices[-1]
        if current_price > result["value_area_high"]:
            result["current_vs_va"] = "above"
        elif current_price < result["value_area_low"]:
            result["current_vs_va"] = "below"
        else:
            result["current_vs_va"] = "inside"

        # Distance from POC in ATR units
        if state.atr > 0:
            result["current_vs_poc"] = (current_price - poc_price) / state.atr

        # Structure quality: How evenly distributed is volume?
        volumes_list = list(vol_by_price.values())
        if len(volumes_list) > 2:
            mean_vol = statistics.mean(volumes_list)
            stdev_vol = statistics.stdev(volumes_list)
            # Lower coefficient of variation = more even distribution = better structure
            cv = stdev_vol / mean_vol if mean_vol > 0 else 1
            result["structure_quality"] = max(0, min(1, 1 - (cv / 2)))

        return result

    def get_vote(self, state: AssetState, regime: MarketRegime) -> EdgeSignal:
        """Get voting signal based on volume profile."""
        profile = self.build_profile(state)

        va_position = profile["current_vs_va"]
        poc_distance = profile["current_vs_poc"]
        structure = profile["structure_quality"]

        # Above value area = bullish, below = bearish
        if va_position == "above":
            if regime in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP]:
                return EdgeSignal("VolumeProfile", EdgeVote.LONG, 0.6,
                                f"Price above value area, in uptrend. POC dist: {poc_distance:.1f} ATR",
                                profile)
            else:
                # Above VA but not trending up - potential reversal
                return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.4,
                                f"Above VA but no trend support", profile)

        elif va_position == "below":
            if regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN]:
                return EdgeSignal("VolumeProfile", EdgeVote.SHORT, 0.6,
                                f"Price below value area, in downtrend. POC dist: {poc_distance:.1f} ATR",
                                profile)
            else:
                return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.4,
                                f"Below VA but no trend support", profile)

        else:  # Inside value area
            if structure < 0.3:
                # Poor structure = expect breakout
                return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.3,
                                f"Inside VA with poor structure - breakout likely", profile)
            else:
                # Good structure = range trade
                if poc_distance > 1.0:
                    return EdgeSignal("VolumeProfile", EdgeVote.SHORT, 0.5,
                                    f"At VA high, fade to POC", profile)
                elif poc_distance < -1.0:
                    return EdgeSignal("VolumeProfile", EdgeVote.LONG, 0.5,
                                    f"At VA low, fade to POC", profile)

        return EdgeSignal("VolumeProfile", EdgeVote.NEUTRAL, 0.3,
                        f"No clear volume profile signal", profile)


# =============================================================================
# EDGE 5: CROSS-ASSET CONFIRMATION
# =============================================================================

class CrossAssetEngine:
    """
    Checks if related assets confirm or contradict the trade thesis.

    Key insight: Moves confirmed by related assets are more likely to continue.
    Divergences often signal reversals or false moves.

    Examples:
    - INTC up but AMD down → Divergence, be cautious
    - INTC up, AMD up, SMH up → Confirmed sector strength
    - Stock up but VIX up → Fear in market, potential reversal
    """

    CORRELATIONS = {
        # Direct competitors
        "INTC": ["AMD", "NVDA", "SMH"],
        "AMD": ["INTC", "NVDA", "SMH"],
        "NVDA": ["AMD", "SMH", "TSM"],
        "AAPL": ["MSFT", "QQQ", "XLK"],
        "MSFT": ["AAPL", "GOOGL", "XLK"],
        "JPM": ["BAC", "GS", "XLF"],
        "XOM": ["CVX", "XLE", "OXY"],
        "TSLA": ["RIVN", "QQQ", "XLY"],
        # ETFs
        "SOXL": ["SMH", "AMD", "NVDA"],
        "TQQQ": ["QQQ", "AAPL", "MSFT"],
    }

    def __init__(self):
        self.asset_states: Dict[str, AssetState] = {}

    def update(self, symbol: str, state: AssetState):
        """Update with latest asset state."""
        self.asset_states[symbol] = state

    def check_confirmation(self, symbol: str, direction: str, lookback: int = 10) -> Tuple[float, List[str], str]:
        """
        Check if related assets confirm the direction.

        Args:
            symbol: The stock we want to trade
            direction: 'long' or 'short'
            lookback: Number of bars to analyze

        Returns:
            (confirmation_score, confirming_symbols, reason)
            Score: -1 to +1 (positive = confirms, negative = contradicts)
        """
        related = self.CORRELATIONS.get(symbol, [])
        if not related:
            return 0.0, [], "No related assets defined"

        # Get our symbol's return
        if symbol not in self.asset_states:
            return 0.0, [], "No data for symbol"
        our_prices = list(self.asset_states[symbol].prices)
        if len(our_prices) < lookback:
            return 0.0, [], "Insufficient data"
        our_return = (our_prices[-1] - our_prices[-lookback]) / our_prices[-lookback]

        # Check related assets
        confirming = []
        contradicting = []

        for related_sym in related:
            if related_sym in self.asset_states:
                rel_prices = list(self.asset_states[related_sym].prices)
                if len(rel_prices) >= lookback:
                    rel_return = (rel_prices[-1] - rel_prices[-lookback]) / rel_prices[-lookback]

                    # Same direction = confirming
                    if (direction == "long" and rel_return > 0.001) or \
                       (direction == "short" and rel_return < -0.001):
                        confirming.append(f"{related_sym} {rel_return*100:+.1f}%")
                    elif (direction == "long" and rel_return < -0.001) or \
                         (direction == "short" and rel_return > 0.001):
                        contradicting.append(f"{related_sym} {rel_return*100:+.1f}%")

        total = len(confirming) + len(contradicting)
        if total == 0:
            return 0.0, [], "No related asset data"

        score = (len(confirming) - len(contradicting)) / total

        reason_parts = []
        if confirming:
            reason_parts.append(f"Confirming: {', '.join(confirming)}")
        if contradicting:
            reason_parts.append(f"Contradicting: {', '.join(contradicting)}")

        return score, confirming, " | ".join(reason_parts)

    def get_vote(self, symbol: str, intended_direction: str) -> EdgeSignal:
        """Get voting signal based on cross-asset analysis."""
        score, confirming, reason = self.check_confirmation(symbol, intended_direction)

        if score > 0.6:
            vote = EdgeVote.STRONG_LONG if intended_direction == "long" else EdgeVote.STRONG_SHORT
            return EdgeSignal("CrossAsset", vote, 0.8, f"Strong confirmation: {reason}",
                            {"score": score, "confirming": confirming})

        elif score > 0.2:
            vote = EdgeVote.LONG if intended_direction == "long" else EdgeVote.SHORT
            return EdgeSignal("CrossAsset", vote, 0.6, f"Moderate confirmation: {reason}",
                            {"score": score, "confirming": confirming})

        elif score < -0.4:
            # Strong contradiction - veto the trade
            vote = EdgeVote.VETO_LONG if intended_direction == "long" else EdgeVote.VETO_SHORT
            return EdgeSignal("CrossAsset", vote, 0.7, f"Divergence warning: {reason}",
                            {"score": score, "confirming": confirming})

        elif score < -0.1:
            vote = EdgeVote.NEUTRAL
            return EdgeSignal("CrossAsset", vote, 0.5, f"Mixed signals: {reason}",
                            {"score": score, "confirming": confirming})

        return EdgeSignal("CrossAsset", EdgeVote.NEUTRAL, 0.4, reason,
                        {"score": score, "confirming": confirming})


# =============================================================================
# EDGE 6: TIME-OF-DAY PATTERNS
# =============================================================================

class TimeOfDayEngine:
    """
    Uses well-documented intraday patterns to time entries and exits.

    Key patterns:
    - Opening Drive (9:30-10:00): Initial momentum often continues
    - Morning Reversal (10:00-10:30): First pullback opportunity
    - Mid-Morning Trend (10:30-11:30): Best trending period
    - Lunch Chop (11:30-13:30): Avoid new entries
    - Afternoon Trend (13:30-14:30): Second trending period
    - Power Hour Setup (14:30-15:00): Institutions position
    - MOC Imbalance (15:30-16:00): Volume spike, be careful
    """

    def __init__(self):
        self.session_data = {
            "opening_range_high": 0.0,
            "opening_range_low": 0.0,
            "morning_trend": None,  # 'up', 'down', None
            "lunch_direction": None,
        }

    def get_time_window(self, timestamp: datetime) -> str:
        """Determine which time window we're in."""
        t = timestamp.time()

        if time(9, 30) <= t < time(10, 0):
            return "opening_drive"
        elif time(10, 0) <= t < time(10, 30):
            return "morning_reversal"
        elif time(10, 30) <= t < time(11, 30):
            return "mid_morning_trend"
        elif time(11, 30) <= t < time(13, 30):
            return "lunch_chop"
        elif time(13, 30) <= t < time(14, 30):
            return "afternoon_trend"
        elif time(14, 30) <= t < time(15, 30):
            return "power_hour"
        elif time(15, 30) <= t < time(16, 0):
            return "moc_period"
        else:
            return "outside_hours"

    def get_vote(self, timestamp: datetime, trade_type: TradeType) -> EdgeSignal:
        """Get voting signal based on time of day."""
        window = self.get_time_window(timestamp)

        # Opening drive - momentum trades good
        if window == "opening_drive":
            if trade_type == TradeType.MOMENTUM_CONTINUATION:
                return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.7,
                                "Opening drive favors momentum", {"window": window})
            else:
                return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.4,
                                "Opening drive - prefer momentum trades", {"window": window})

        # Morning reversal - mean reversion good
        elif window == "morning_reversal":
            if trade_type == TradeType.MEAN_REVERSION:
                return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.6,
                                "10 AM reversal window - good for mean reversion", {"window": window})
            else:
                return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.5,
                                "10 AM reversal possible", {"window": window})

        # Mid-morning trend - best period
        elif window == "mid_morning_trend":
            return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.7,
                            "Best trending period of day", {"window": window})

        # Lunch period - reduced confidence but still tradeable
        elif window == "lunch_chop":
            return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.4,
                            "Lunch period - lower volume but tradeable", {"window": window})

        # Afternoon trend - second best
        elif window == "afternoon_trend":
            return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.6,
                            "Afternoon trending period", {"window": window})

        # Power hour - institutions active
        elif window == "power_hour":
            if trade_type == TradeType.MOMENTUM_CONTINUATION:
                return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.6,
                                "Power hour - institutional flow", {"window": window})
            else:
                return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.5,
                                "Power hour - watch for reversals", {"window": window})

        # MOC period - high volatility, reduced confidence
        elif window == "moc_period":
            return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.3,
                            "MOC period - high volatility, use caution", {"window": window})

        return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.3,
                        f"Outside trading hours: {window}", {"window": window})


# =============================================================================
# THE ORCHESTRATOR - ENSEMBLE DECISION MAKER
# =============================================================================

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
        self.reference_assets = {"SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE", "XLY", "XLV"}

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


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_orchestrator() -> Orchestrator:
    """Create an Orchestrator instance with default settings."""
    return Orchestrator()
