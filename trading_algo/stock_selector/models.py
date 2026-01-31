"""
Data models for the stock selector system.

These dataclasses represent the various scores, signals, and metadata
used throughout the selection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional


class MarketRegime(Enum):
    """Overall market regime affects factor weightings."""
    STRONG_BULL = auto()      # VIX < 15, SPY trending up strongly
    BULL = auto()             # VIX 15-20, SPY above 20-day MA
    NEUTRAL = auto()          # VIX 20-25, SPY consolidating
    BEAR = auto()             # VIX 25-35, SPY below 20-day MA
    CRISIS = auto()           # VIX > 35, high correlation, fear


class CatalystType(Enum):
    """Types of catalysts that can drive stock moves."""
    EARNINGS_UPCOMING = auto()       # Earnings within 5 days
    EARNINGS_JUST_REPORTED = auto()  # Reported within 2 days
    FDA_DECISION = auto()            # Biotech FDA date
    PRODUCT_LAUNCH = auto()          # New product announcement
    ANALYST_UPGRADE = auto()         # Analyst upgrade
    ANALYST_DOWNGRADE = auto()       # Analyst downgrade
    INSIDER_BUYING = auto()          # Insider purchases
    INSIDER_SELLING = auto()         # Insider sales
    SHORT_SQUEEZE_SETUP = auto()     # High short interest + momentum
    MERGER_ACQUISITION = auto()      # M&A news
    SECTOR_ROTATION = auto()         # Money flowing into sector
    SOCIAL_VIRAL = auto()            # Viral on social media
    BREAKING_NEWS = auto()           # Major breaking news
    TECHNICAL_BREAKOUT = auto()      # Breaking key level
    UNUSUAL_OPTIONS = auto()         # Large options activity


class SentimentLevel(Enum):
    """Sentiment classification."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class VolatilityScore:
    """Volatility analysis results."""
    # Raw metrics
    historical_volatility_20d: float  # Annualized 20-day vol
    historical_volatility_5d: float   # Annualized 5-day vol (recent)
    atr_percent: float                # ATR as % of price
    average_intraday_range: float     # Avg (high-low)/close
    gap_frequency: float              # How often it gaps >1%
    gap_average_size: float           # Average gap size

    # Derived scores (0-100)
    volatility_score: float           # Overall vol score
    volatility_trend: float           # Is vol increasing? (-1 to 1)
    tradeable_range_score: float      # Good range for day trading

    # Vol regime
    vol_percentile: float             # Current vol vs historical


@dataclass
class VolumeScore:
    """Volume analysis results."""
    # Raw metrics
    avg_volume_20d: float             # 20-day average volume
    avg_dollar_volume: float          # Price × volume
    relative_volume: float            # Today's vol vs average
    premarket_volume_ratio: float     # Premarket vs regular

    # Anomaly detection
    volume_spike_detected: bool       # Unusual volume today
    volume_spike_magnitude: float     # How unusual (z-score)
    accumulation_detected: bool       # Price up on high volume
    distribution_detected: bool       # Price down on high volume

    # Derived scores (0-100)
    liquidity_score: float            # Tradeable liquidity
    volume_momentum_score: float      # Volume trend
    smart_money_score: float          # Institutional signals


@dataclass
class MomentumScore:
    """Momentum and trend analysis."""
    # Price momentum
    return_1d: float                  # 1-day return
    return_5d: float                  # 5-day return
    return_20d: float                 # 20-day return
    return_intraday: float            # Today's return so far

    # Trend indicators
    price_vs_sma_20: float            # % above/below 20-day MA
    price_vs_sma_50: float            # % above/below 50-day MA
    sma_20_vs_50: float               # 20 vs 50 MA (trend)

    # Oscillators
    rsi_14: float                     # 14-period RSI
    rsi_trend: float                  # RSI direction

    # Relative strength
    relative_strength_sector: float   # vs sector
    relative_strength_market: float   # vs SPY

    # Derived scores (0-100)
    momentum_score: float             # Overall momentum
    trend_strength: float             # How strong is trend
    mean_reversion_score: float       # Oversold/overbought opportunity


@dataclass
class TechnicalScore:
    """Technical analysis signals."""
    # Support/Resistance
    distance_to_resistance: float     # % to nearest resistance
    distance_to_support: float        # % to nearest support
    at_key_level: bool                # Near important level

    # Patterns
    consolidation_days: int           # Days in tight range
    breakout_potential: float         # 0-1 breakout likelihood
    breakdown_potential: float        # 0-1 breakdown likelihood

    # Price action
    higher_highs: bool                # Making higher highs
    lower_lows: bool                  # Making lower lows
    inside_day: bool                  # Today's range inside yesterday

    # Derived scores (0-100)
    technical_setup_score: float      # Overall technical setup
    breakout_score: float             # Breakout probability
    reversal_score: float             # Reversal probability


@dataclass
class NewsScore:
    """News and sentiment analysis."""
    # News metrics
    news_count_24h: int               # Articles in last 24h
    news_count_7d: int                # Articles in last 7d
    news_velocity: float              # Rate of news increase
    news_recency_hours: float         # Hours since last major news

    # Sentiment
    sentiment_score: float            # -1 to 1 aggregate sentiment
    sentiment_magnitude: float        # Strength of sentiment
    sentiment_trend: float            # Sentiment direction

    # Source breakdown
    mainstream_sentiment: float       # Major news outlets
    financial_sentiment: float        # Financial news
    social_sentiment: float           # Social media

    # Key flags
    has_breaking_news: bool           # Breaking news detected
    has_analyst_action: bool          # Upgrade/downgrade
    has_insider_activity: bool        # Insider trading reported

    # Derived scores (0-100)
    news_score: float                 # Overall news score
    catalyst_score: float             # Catalyst strength


@dataclass
class SocialScore:
    """Social media and retail sentiment."""
    # Reddit metrics
    reddit_mentions_24h: int          # Mentions on WSB, stocks, etc
    reddit_mentions_trend: float      # Trend direction
    reddit_sentiment: float           # -1 to 1

    # Twitter metrics
    twitter_mentions_24h: int         # Twitter/X mentions
    twitter_sentiment: float          # -1 to 1
    influencer_mentions: int          # Mentions by large accounts

    # StockTwits
    stocktwits_sentiment: float       # Bull/bear ratio
    stocktwits_volume: int            # Message volume

    # Derived
    social_momentum: float            # Is buzz increasing?
    viral_potential: float            # Could go viral?
    retail_interest_score: float      # Overall retail attention (0-100)


@dataclass
class OptionsScore:
    """Options flow analysis."""
    # Volume metrics
    options_volume: int               # Total options volume
    put_call_ratio: float             # Put/call ratio
    unusual_activity: bool            # Unusual options detected

    # Flow analysis
    call_flow_score: float            # Bullish call flow
    put_flow_score: float             # Bearish put flow
    net_premium: float                # Net $ premium (calls - puts)

    # Implied volatility
    iv_rank: float                    # IV percentile (0-100)
    iv_skew: float                    # Put vs call IV

    # Smart money signals
    large_trades_detected: int        # Block trades
    sweep_orders_detected: int        # Aggressive sweeps
    smart_money_direction: float      # -1 to 1

    # Derived scores (0-100)
    options_signal_score: float       # Overall options signal
    squeeze_potential: float          # Gamma squeeze potential


@dataclass
class ShortInterestScore:
    """Short interest and squeeze analysis."""
    # Raw metrics
    short_interest_ratio: float       # Short interest as % of float
    days_to_cover: float              # Days to cover at avg volume
    short_interest_change: float      # Change in SI (%)

    # Squeeze indicators
    cost_to_borrow: float             # Borrow rate (%)
    shares_available: int             # Shares available to short
    utilization: float                # % of lendable shares shorted

    # Derived scores (0-100)
    squeeze_setup_score: float        # Short squeeze potential
    short_pressure_score: float       # Pressure on shorts


@dataclass
class RiskMetrics:
    """Risk assessment for the stock."""
    # Event risk
    earnings_days_away: Optional[int]  # Days until earnings
    has_binary_event: bool             # FDA, legal, etc
    halt_risk: float                   # Risk of trading halt

    # Liquidity risk
    bid_ask_spread_pct: float          # Typical spread
    slippage_estimate: float           # Expected slippage

    # Volatility risk
    gap_risk: float                    # Risk of adverse gap
    tail_risk: float                   # Fat tail probability

    # Overall
    risk_score: float                  # 0-100, higher = riskier
    tradeable: bool                    # Passes minimum criteria


@dataclass
class Catalyst:
    """Detected catalyst for a stock."""
    type: CatalystType
    description: str
    timestamp: datetime
    impact_score: float               # Expected impact (0-1)
    confidence: float                 # Detection confidence (0-1)
    source: str                       # Where detected


@dataclass
class StockCandidate:
    """Complete analysis of a stock as day trading candidate."""
    # Basic info
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    price: float
    timestamp: datetime

    # Component scores
    volatility: VolatilityScore
    volume: VolumeScore
    momentum: MomentumScore
    technical: TechnicalScore
    news: NewsScore
    social: SocialScore
    options: OptionsScore
    short_interest: ShortInterestScore
    risk: RiskMetrics

    # Catalysts
    catalysts: List[Catalyst] = field(default_factory=list)

    # Final composite scores
    composite_score: float = 0.0              # Overall score (0-100)
    confidence: float = 0.0                   # Confidence in score (0-1)
    expected_move: float = 0.0                # Expected % move
    recommended_direction: int = 0            # 1=long, -1=short, 0=either

    # Ranking
    rank: int = 0
    percentile: float = 0.0

    # Explanation
    top_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """Result of the stock selection process."""
    timestamp: datetime
    market_regime: MarketRegime

    # Selected stocks
    top_candidates: List[StockCandidate]

    # Universe stats
    total_scanned: int
    passed_filters: int

    # Factor weights used
    factor_weights: Dict[str, float]

    # Market context
    market_volatility: float
    market_trend: float
    sector_leaders: List[str]

    # Metadata
    processing_time_seconds: float
