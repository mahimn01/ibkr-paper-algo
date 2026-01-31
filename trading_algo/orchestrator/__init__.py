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
"""

from .types import (
    MarketRegime,
    EdgeVote,
    TradeType,
    EdgeSignal,
    OrchestratorSignal,
    AssetState,
)

from .strategy import Orchestrator, create_orchestrator

from .edges import (
    MarketRegimeEngine,
    RelativeStrengthEngine,
    StatisticalExtremeDetector,
    VolumeProfileEngine,
    CrossAssetEngine,
    TimeOfDayEngine,
)

__all__ = [
    # Core types
    "MarketRegime",
    "EdgeVote",
    "TradeType",
    "EdgeSignal",
    "OrchestratorSignal",
    "AssetState",
    # Main strategy
    "Orchestrator",
    "create_orchestrator",
    # Edge engines
    "MarketRegimeEngine",
    "RelativeStrengthEngine",
    "StatisticalExtremeDetector",
    "VolumeProfileEngine",
    "CrossAssetEngine",
    "TimeOfDayEngine",
]
