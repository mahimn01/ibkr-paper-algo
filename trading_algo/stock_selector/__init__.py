"""
AI-Driven Day Trading Stock Selector.

A sophisticated multi-factor system for identifying optimal day trading candidates.

Architecture:
    1. Universe Scanner - Filters liquid, tradeable stocks
    2. Multi-Factor Analyzers - Score stocks on various dimensions
    3. Composite Scorer - Combines factors with adaptive weighting
    4. Risk Filter - Eliminates unsuitable candidates
    5. Final Ranker - Produces ranked list with confidence scores

Key Innovations:
    - Regime-adaptive factor weighting (adjusts for bull/bear/crisis)
    - News velocity and sentiment momentum tracking
    - Volume anomaly detection (institutional activity)
    - Options flow signals (smart money detection)
    - Social media trend detection (Reddit, Twitter buzz)
    - Cross-factor interaction modeling (synergy bonuses)
    - Short squeeze potential scoring

Analyzers:
    - VolatilityAnalyzer: ATR, historical vol, gap analysis
    - VolumeAnalyzer: Liquidity, RVOL, accumulation/distribution
    - MomentumAnalyzer: Multi-timeframe momentum, RSI, relative strength
    - TechnicalAnalyzer: Support/resistance, breakout potential, patterns
    - NewsSentimentAnalyzer: News flow, sentiment scoring, catalyst detection
    - OptionsFlowAnalyzer: Unusual activity, sweep detection, gamma squeeze
    - ShortInterestAnalyzer: SI ratio, days to cover, squeeze setup

Usage:
    from trading_algo.stock_selector import DayTradeStockSelector

    selector = DayTradeStockSelector()
    result = selector.select(top_n=5)

    for candidate in result.top_candidates:
        print(f"{candidate.symbol}: {candidate.composite_score:.0f}")
"""

from trading_algo.stock_selector.models import (
    MarketRegime,
    CatalystType,
    SentimentLevel,
    StockCandidate,
    SelectionResult,
    VolatilityScore,
    VolumeScore,
    MomentumScore,
    TechnicalScore,
    NewsScore,
    SocialScore,
    OptionsScore,
    ShortInterestScore,
    RiskMetrics,
    Catalyst,
)
from trading_algo.stock_selector.selector import (
    DayTradeStockSelector,
    UniverseConfig,
    print_selection_results,
)
from trading_algo.stock_selector.scoring import (
    CompositeScorer,
    FactorWeights,
    RegimeAdaptiveWeights,
    rank_candidates,
)
from trading_algo.stock_selector.ibkr_scanner import (
    IBKRStockScanner,
    ScanResult,
    print_scan_results,
    DAY_TRADE_UNIVERSE,
)

__all__ = [
    # Main selector
    "DayTradeStockSelector",
    "UniverseConfig",
    "print_selection_results",
    # IBKR Scanner
    "IBKRStockScanner",
    "ScanResult",
    "print_scan_results",
    "DAY_TRADE_UNIVERSE",
    # Models
    "MarketRegime",
    "CatalystType",
    "SentimentLevel",
    "StockCandidate",
    "SelectionResult",
    "VolatilityScore",
    "VolumeScore",
    "MomentumScore",
    "TechnicalScore",
    "NewsScore",
    "SocialScore",
    "OptionsScore",
    "ShortInterestScore",
    "RiskMetrics",
    "Catalyst",
    # Scoring
    "CompositeScorer",
    "FactorWeights",
    "RegimeAdaptiveWeights",
    "rank_candidates",
]
