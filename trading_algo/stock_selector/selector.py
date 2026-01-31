"""
Main Stock Selector - The Complete Day Trading Stock Selection System.

This is the top-level interface that orchestrates:
1. Universe definition (what stocks to scan)
2. Data collection (prices, news, options, etc.)
3. Multi-factor analysis
4. Composite scoring
5. Risk filtering
6. Final ranking and selection

Usage:
    selector = DayTradeStockSelector()
    result = selector.select(top_n=5)
    for candidate in result.top_candidates:
        print(f"{candidate.symbol}: {candidate.composite_score:.0f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from trading_algo.stock_selector.models import (
    MarketRegime,
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
from trading_algo.stock_selector.analyzers import (
    VolatilityAnalyzer,
    VolumeAnalyzer,
    MomentumAnalyzer,
    TechnicalAnalyzer,
    NewsSentimentAnalyzer,
    OptionsFlowAnalyzer,
    ShortInterestAnalyzer,
    PriceBar,
    NewsItem,
)
from trading_algo.stock_selector.scoring import (
    CompositeScorer,
    RegimeAdaptiveWeights,
    rank_candidates,
)


# Default universe of liquid, popular stocks suitable for day trading
DEFAULT_UNIVERSE = [
    # Mega caps (very liquid)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Large tech
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL",
    # Financials
    "JPM", "BAC", "GS", "MS", "C", "WFC",
    # Consumer
    "DIS", "NKE", "SBUX", "MCD", "HD", "WMT", "TGT", "COST",
    # Healthcare
    "JNJ", "PFE", "MRNA", "UNH", "ABBV",
    # Energy
    "XOM", "CVX", "OXY", "SLB",
    # EV/Clean energy
    "RIVN", "LCID", "NIO", "PLUG", "ENPH",
    # Popular retail/meme stocks
    "GME", "AMC", "PLTR", "SOFI", "HOOD", "COIN",
    # Chinese ADRs
    "BABA", "JD", "PDD", "BIDU", "NIO",
    # Biotech (volatile)
    "MRNA", "BNTX", "REGN",
    # Semiconductors
    "MU", "MRVL", "QCOM", "AVGO", "TSM",
    # Airlines/Travel
    "UAL", "DAL", "AAL", "CCL", "RCL",
    # ETFs for reference
    "SPY", "QQQ", "IWM", "XLF", "XLE", "ARKK",
]


@dataclass
class UniverseConfig:
    """Configuration for stock universe."""
    symbols: List[str] = field(default_factory=lambda: DEFAULT_UNIVERSE.copy())
    min_price: float = 5.0           # Minimum stock price
    max_price: float = 5000.0        # Maximum stock price
    min_avg_volume: float = 500_000  # Minimum average volume
    min_market_cap: float = 1e9      # $1B minimum market cap


@dataclass
class DataSources:
    """
    Data source configuration.

    In production, these would be actual API clients/connections.
    """
    price_source: str = "ibkr"       # IBKR, polygon, alpaca, etc.
    news_source: str = "finnhub"     # finnhub, alphavantage, benzinga
    options_source: str = "ibkr"     # IBKR, polygon, tradier
    social_source: str = "reddit"    # reddit API, stocktwits API


class DayTradeStockSelector:
    """
    Main stock selection engine.

    Scans a universe of stocks, analyzes each on multiple factors,
    and returns the best day trading candidates.
    """

    def __init__(
        self,
        universe_config: Optional[UniverseConfig] = None,
        data_sources: Optional[DataSources] = None,
    ):
        self.universe_config = universe_config or UniverseConfig()
        self.data_sources = data_sources or DataSources()

        # Initialize analyzers
        self.volatility_analyzer = VolatilityAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.options_analyzer = OptionsFlowAnalyzer()
        self.short_analyzer = ShortInterestAnalyzer()

        # Cache for efficiency
        self._market_data_cache: Dict[str, List[PriceBar]] = {}
        self._news_cache: Dict[str, List[NewsItem]] = {}

    def select(
        self,
        top_n: int = 10,
        market_regime: Optional[MarketRegime] = None,
        custom_universe: Optional[List[str]] = None,
    ) -> SelectionResult:
        """
        Run the complete stock selection process.

        Args:
            top_n: Number of top candidates to return
            market_regime: Override detected market regime
            custom_universe: Override default universe

        Returns:
            SelectionResult with top candidates and metadata
        """
        start_time = time.time()

        # Determine universe
        universe = custom_universe or self.universe_config.symbols

        # Detect market regime if not provided
        if market_regime is None:
            market_regime = self._detect_market_regime()

        # Initialize scorer with regime
        scorer = CompositeScorer(regime=market_regime)

        # Analyze each stock
        candidates = []
        passed_filters = 0

        for symbol in universe:
            try:
                candidate = self._analyze_stock(symbol)

                if candidate is None:
                    continue

                # Apply risk filters
                if not self._passes_filters(candidate):
                    continue

                passed_filters += 1

                # Score the candidate
                candidate = scorer.score(candidate)
                candidates.append(candidate)

            except Exception as e:
                # Log error but continue with other stocks
                print(f"Error analyzing {symbol}: {e}")
                continue

        # Rank and select top candidates
        top_candidates = rank_candidates(candidates, top_n)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get sector leaders
        sector_leaders = self._identify_sector_leaders(candidates)

        return SelectionResult(
            timestamp=datetime.now(),
            market_regime=market_regime,
            top_candidates=top_candidates,
            total_scanned=len(universe),
            passed_filters=passed_filters,
            factor_weights=scorer.weights.as_dict(),
            market_volatility=self._get_market_volatility(),
            market_trend=self._get_market_trend(),
            sector_leaders=sector_leaders,
            processing_time_seconds=processing_time,
        )

    def quick_scan(
        self,
        symbols: List[str],
        min_score: float = 60,
    ) -> List[StockCandidate]:
        """
        Quick scan of specific symbols without full analysis.

        Useful for monitoring a watchlist.
        """
        regime = self._detect_market_regime()
        scorer = CompositeScorer(regime=regime)

        candidates = []
        for symbol in symbols:
            try:
                candidate = self._analyze_stock(symbol, quick=True)
                if candidate and self._passes_filters(candidate):
                    candidate = scorer.score(candidate)
                    if candidate.composite_score >= min_score:
                        candidates.append(candidate)
            except Exception:
                continue

        return rank_candidates(candidates, len(symbols))

    def _analyze_stock(
        self,
        symbol: str,
        quick: bool = False,
    ) -> Optional[StockCandidate]:
        """
        Perform full multi-factor analysis on a stock.

        Args:
            symbol: Stock symbol
            quick: If True, skip some expensive analyses

        Returns:
            StockCandidate with all scores, or None if data unavailable
        """
        # Get price data
        bars = self._get_price_data(symbol)
        if bars is None or len(bars) < 60:
            return None

        current_price = bars[-1].close

        # Basic filters
        if current_price < self.universe_config.min_price:
            return None
        if current_price > self.universe_config.max_price:
            return None

        # Get market data for relative strength
        market_bars = self._get_price_data("SPY")

        # Run analyzers
        volatility_score = self.volatility_analyzer.analyze(bars)
        if volatility_score is None:
            return None

        volume_score = self.volume_analyzer.analyze(bars)
        if volume_score is None:
            return None

        # Quick filter: must have minimum liquidity
        if volume_score.liquidity_score < 30:
            return None

        momentum_score = self.momentum_analyzer.analyze(bars, market_bars)
        if momentum_score is None:
            return None

        technical_score = self.technical_analyzer.analyze(bars)
        if technical_score is None:
            return None

        # News analysis (can be expensive)
        if quick:
            news_score = self._empty_news_score()
            social_score = self._empty_social_score()
        else:
            news_items = self._get_news_data(symbol)
            news_score = self.news_analyzer.analyze(symbol, news_items)
            social_score = self._analyze_social(symbol)

        # Options analysis
        options_score = self._analyze_options(symbol, quick)

        # Short interest
        short_score = self._analyze_short_interest(symbol)

        # Risk assessment
        risk_metrics = self._assess_risk(symbol, bars, volume_score)

        # Detect catalysts
        catalysts = self._detect_catalysts(symbol, news_score)

        # Create candidate
        return StockCandidate(
            symbol=symbol,
            company_name=self._get_company_name(symbol),
            sector=self._get_sector(symbol),
            industry=self._get_industry(symbol),
            market_cap=self._get_market_cap(symbol),
            price=current_price,
            timestamp=datetime.now(),
            volatility=volatility_score,
            volume=volume_score,
            momentum=momentum_score,
            technical=technical_score,
            news=news_score,
            social=social_score,
            options=options_score,
            short_interest=short_score,
            risk=risk_metrics,
            catalysts=catalysts,
        )

    def _passes_filters(self, candidate: StockCandidate) -> bool:
        """Check if candidate passes basic filters."""
        # Minimum liquidity
        if candidate.volume.liquidity_score < 40:
            return False

        # Not too volatile (hard to trade)
        if candidate.volatility.atr_percent > 0.10:  # >10% daily ATR
            return False

        # Must be tradeable
        if not candidate.risk.tradeable:
            return False

        return True

    def _detect_market_regime(self) -> MarketRegime:
        """
        Detect current market regime from SPY and VIX.

        Uses:
        - VIX level (fear gauge)
        - SPY trend (above/below 20-day MA)
        - Recent volatility
        """
        spy_bars = self._get_price_data("SPY")
        if spy_bars is None or len(spy_bars) < 30:
            return MarketRegime.NEUTRAL

        # Calculate SPY metrics
        current_price = spy_bars[-1].close
        sma_20 = sum(b.close for b in spy_bars[-20:]) / 20
        price_vs_sma = (current_price - sma_20) / sma_20

        # Calculate recent volatility
        returns = [(spy_bars[i].close - spy_bars[i-1].close) / spy_bars[i-1].close
                   for i in range(-20, 0)]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * (252**0.5)

        # Determine regime
        if volatility > 0.35:  # Very high vol
            return MarketRegime.CRISIS
        elif volatility > 0.25:
            if price_vs_sma < -0.03:
                return MarketRegime.BEAR
            return MarketRegime.NEUTRAL
        elif price_vs_sma > 0.03:
            if price_vs_sma > 0.06:
                return MarketRegime.STRONG_BULL
            return MarketRegime.BULL
        elif price_vs_sma < -0.03:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL

    def _get_price_data(self, symbol: str) -> Optional[List[PriceBar]]:
        """
        Get historical price data for a symbol.

        In production, this would call IBKR or another data source.
        """
        # Check cache first
        if symbol in self._market_data_cache:
            return self._market_data_cache[symbol]

        # TODO: Implement actual data fetching from IBKR
        # For now, return None (will be implemented when connected to broker)
        return None

    def _get_news_data(self, symbol: str) -> List[NewsItem]:
        """
        Get recent news for a symbol.

        In production, this would call a news API.
        """
        if symbol in self._news_cache:
            return self._news_cache[symbol]

        # TODO: Implement actual news fetching
        return []

    def _analyze_social(self, symbol: str) -> SocialScore:
        """Analyze social media sentiment."""
        # TODO: Implement Reddit/Twitter/StockTwits analysis
        return self._empty_social_score()

    def _analyze_options(self, symbol: str, quick: bool = False) -> OptionsScore:
        """Analyze options flow."""
        # TODO: Implement options flow analysis from IBKR
        return self.options_analyzer._empty_score()

    def _analyze_short_interest(self, symbol: str) -> ShortInterestScore:
        """Get short interest data."""
        # TODO: Implement short interest data fetching
        return self.short_analyzer._empty_score()

    def _assess_risk(
        self,
        symbol: str,
        bars: List[PriceBar],
        volume: VolumeScore,
    ) -> RiskMetrics:
        """Assess trading risk for the stock."""
        # Calculate bid-ask spread estimate from daily ranges
        avg_range = sum((b.high - b.low) / b.close for b in bars[-20:]) / 20
        estimated_spread = avg_range * 0.1  # Rough estimate

        # Gap risk
        gaps = []
        for i in range(1, len(bars)):
            gap = abs(bars[i].open - bars[i-1].close) / bars[i-1].close
            gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        gap_risk = min(1.0, avg_gap * 20)

        # Overall risk score
        risk_score = 30  # Base
        if estimated_spread > 0.005:  # >0.5% spread
            risk_score += 20
        if gap_risk > 0.3:
            risk_score += 15
        if volume.liquidity_score < 50:
            risk_score += 15

        # Is it tradeable?
        tradeable = (
            volume.liquidity_score >= 30 and
            estimated_spread < 0.02 and
            risk_score < 80
        )

        return RiskMetrics(
            earnings_days_away=None,  # TODO: Implement earnings calendar
            has_binary_event=False,
            halt_risk=0.1 if symbol in ["GME", "AMC"] else 0.01,
            bid_ask_spread_pct=estimated_spread,
            slippage_estimate=estimated_spread * 0.5,
            gap_risk=gap_risk,
            tail_risk=gap_risk * 0.5,
            risk_score=risk_score,
            tradeable=tradeable,
        )

    def _detect_catalysts(
        self,
        symbol: str,
        news_score: NewsScore,
    ) -> List[Catalyst]:
        """Detect catalysts from news and other sources."""
        catalysts = []

        if news_score.has_breaking_news:
            catalysts.append(Catalyst(
                type=CatalystType.BREAKING_NEWS,
                description="Breaking news detected",
                timestamp=datetime.now(),
                impact_score=0.7,
                confidence=0.8,
                source="news",
            ))

        if news_score.has_analyst_action:
            catalysts.append(Catalyst(
                type=CatalystType.ANALYST_UPGRADE,  # or downgrade
                description="Analyst action detected",
                timestamp=datetime.now(),
                impact_score=0.6,
                confidence=0.9,
                source="news",
            ))

        return catalysts

    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol."""
        # TODO: Implement company info lookup
        return symbol

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        # Simple mapping for common stocks
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "AMZN": "Consumer", "TSLA": "Automotive", "NVDA": "Technology",
            "JPM": "Financials", "BAC": "Financials", "XOM": "Energy",
            "JNJ": "Healthcare", "PFE": "Healthcare",
        }
        return sector_map.get(symbol, "Unknown")

    def _get_industry(self, symbol: str) -> str:
        """Get industry for symbol."""
        return "Unknown"

    def _get_market_cap(self, symbol: str) -> float:
        """Get market cap for symbol."""
        # TODO: Implement market cap lookup
        return 0.0

    def _get_market_volatility(self) -> float:
        """Get current market volatility (VIX proxy)."""
        spy_bars = self._get_price_data("SPY")
        if spy_bars is None or len(spy_bars) < 20:
            return 0.20  # Default 20%

        returns = [(spy_bars[i].close - spy_bars[i-1].close) / spy_bars[i-1].close
                   for i in range(-20, 0)]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * (252**0.5)
        return volatility

    def _get_market_trend(self) -> float:
        """Get market trend (-1 to 1)."""
        spy_bars = self._get_price_data("SPY")
        if spy_bars is None or len(spy_bars) < 20:
            return 0.0

        current = spy_bars[-1].close
        sma_20 = sum(b.close for b in spy_bars[-20:]) / 20
        trend = (current - sma_20) / sma_20 * 10  # Scale to roughly -1 to 1
        return max(-1, min(1, trend))

    def _identify_sector_leaders(
        self,
        candidates: List[StockCandidate],
    ) -> List[str]:
        """Identify top performing sectors."""
        sector_scores: Dict[str, List[float]] = {}

        for c in candidates:
            if c.sector not in sector_scores:
                sector_scores[c.sector] = []
            sector_scores[c.sector].append(c.momentum.return_5d)

        # Calculate sector averages
        sector_avgs = {
            sector: sum(scores) / len(scores)
            for sector, scores in sector_scores.items()
            if scores
        }

        # Sort by performance
        sorted_sectors = sorted(sector_avgs.items(), key=lambda x: -x[1])
        return [s[0] for s in sorted_sectors[:3]]

    def _empty_news_score(self) -> NewsScore:
        """Return empty/neutral news score."""
        return NewsScore(
            news_count_24h=0, news_count_7d=0, news_velocity=0,
            news_recency_hours=168, sentiment_score=0, sentiment_magnitude=0,
            sentiment_trend=0, mainstream_sentiment=0, financial_sentiment=0,
            social_sentiment=0, has_breaking_news=False, has_analyst_action=False,
            has_insider_activity=False, news_score=30, catalyst_score=30,
        )

    def _empty_social_score(self) -> SocialScore:
        """Return empty/neutral social score."""
        return SocialScore(
            reddit_mentions_24h=0, reddit_mentions_trend=0, reddit_sentiment=0,
            twitter_mentions_24h=0, twitter_sentiment=0, influencer_mentions=0,
            stocktwits_sentiment=0, stocktwits_volume=0, social_momentum=0,
            viral_potential=0, retail_interest_score=30,
        )


def print_selection_results(result: SelectionResult):
    """Pretty print selection results."""
    print("\n" + "=" * 80)
    print("DAY TRADING STOCK SELECTOR RESULTS")
    print("=" * 80)
    print(f"Timestamp: {result.timestamp}")
    print(f"Market Regime: {result.market_regime.name}")
    print(f"Market Volatility: {result.market_volatility*100:.1f}%")
    print(f"Market Trend: {result.market_trend:+.2f}")
    print(f"Processing Time: {result.processing_time_seconds:.2f}s")
    print()
    print(f"Scanned: {result.total_scanned} | Passed Filters: {result.passed_filters}")
    print(f"Sector Leaders: {', '.join(result.sector_leaders)}")
    print()
    print("TOP CANDIDATES")
    print("-" * 80)
    print(f"{'Rank':<5} {'Symbol':<8} {'Score':>6} {'Conf':>5} {'Move':>6} {'Dir':<6} {'Top Factors'}")
    print("-" * 80)

    for c in result.top_candidates:
        direction = "LONG" if c.recommended_direction > 0 else ("SHORT" if c.recommended_direction < 0 else "EITHER")
        factors = ", ".join(c.top_factors[:2])
        print(f"{c.rank:<5} {c.symbol:<8} {c.composite_score:>6.1f} {c.confidence:>4.0%} "
              f"{c.expected_move*100:>5.1f}% {direction:<6} {factors[:40]}")

    print("=" * 80)

    # Detailed view of top 3
    print("\nDETAILED TOP 3:")
    for c in result.top_candidates[:3]:
        print(f"\n{'='*40}")
        print(f"{c.symbol} - Score: {c.composite_score:.1f}")
        print(f"{'='*40}")
        print(f"Price: ${c.price:.2f}")
        print(f"Direction: {'LONG' if c.recommended_direction > 0 else 'SHORT' if c.recommended_direction < 0 else 'EITHER'}")
        print(f"Expected Move: {c.expected_move*100:.1f}%")
        print(f"Confidence: {c.confidence:.0%}")
        print()
        print("Factor Scores:")
        print(f"  Volatility:    {c.volatility.volatility_score:.0f}")
        print(f"  Volume:        {c.volume.liquidity_score:.0f} (liquidity)")
        print(f"  Momentum:      {c.momentum.momentum_score:.0f}")
        print(f"  Technical:     {c.technical.technical_setup_score:.0f}")
        print(f"  News:          {c.news.news_score:.0f}")
        print(f"  Options:       {c.options.options_signal_score:.0f}")
        print(f"  Short Int:     {c.short_interest.squeeze_setup_score:.0f}")
        print()
        print("Top Factors:")
        for factor in c.top_factors:
            print(f"  + {factor}")
        if c.risk_factors:
            print("Risk Factors:")
            for risk in c.risk_factors:
                print(f"  ! {risk}")
