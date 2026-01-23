"""
News and Sentiment Analyzer for Day Trading Stock Selection.

Analyzes news flow and sentiment to identify:
- Breaking news catalysts
- Sentiment momentum (improving/declining)
- News velocity (increasing coverage = increasing interest)
- Analyst actions
- Insider activity

Multi-Source Approach:
1. Financial news APIs (Alpha Vantage, FMP, etc.)
2. Web scraping (Yahoo Finance, MarketWatch, etc.)
3. SEC filings (8-K for material events)
4. Social media sentiment

Sentiment Analysis:
- Keyword-based scoring
- Title vs body weighting
- Source credibility weighting
- Recency decay
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from trading_algo.stock_selector.models import NewsScore, Catalyst, CatalystType


@dataclass
class NewsItem:
    """Single news article/item."""
    title: str
    summary: str
    source: str
    url: str
    published: datetime
    symbols: List[str]
    sentiment: float = 0.0  # -1 to 1
    relevance: float = 1.0  # 0 to 1
    credibility: float = 0.5  # 0 to 1


class SentimentLexicon:
    """
    Keyword-based sentiment analysis.

    Contains domain-specific financial sentiment words
    with associated scores and context modifiers.
    """

    # Strongly positive words (score: 0.8 - 1.0)
    STRONG_POSITIVE = {
        'soars': 1.0, 'surges': 1.0, 'skyrockets': 1.0, 'explodes': 0.9,
        'crushes': 0.9, 'beats': 0.8, 'exceeds': 0.8, 'outperforms': 0.8,
        'breakthrough': 0.9, 'revolutionary': 0.8, 'blockbuster': 0.9,
        'record': 0.7, 'all-time high': 0.9, 'upgrade': 0.8, 'upgraded': 0.8,
        'bullish': 0.7, 'optimistic': 0.6, 'strong buy': 0.9,
        'accelerating': 0.7, 'expansion': 0.6, 'growth': 0.5,
        'approved': 0.8, 'fda approval': 0.9, 'partnership': 0.6,
        'acquisition': 0.5, 'buyback': 0.6, 'dividend increase': 0.7,
    }

    # Mildly positive words (score: 0.3 - 0.6)
    MILD_POSITIVE = {
        'gains': 0.5, 'rises': 0.4, 'climbs': 0.4, 'advances': 0.4,
        'positive': 0.4, 'improves': 0.4, 'solid': 0.4, 'steady': 0.3,
        'confident': 0.4, 'upbeat': 0.4, 'promising': 0.5, 'potential': 0.3,
        'opportunity': 0.4, 'momentum': 0.4, 'recovery': 0.5,
        'stabilizes': 0.3, 'resilient': 0.4,
    }

    # Strongly negative words (score: -0.8 to -1.0)
    STRONG_NEGATIVE = {
        'plunges': -1.0, 'crashes': -1.0, 'plummets': -1.0, 'collapses': -1.0,
        'tanks': -0.9, 'tumbles': -0.8, 'misses': -0.7, 'disappoints': -0.7,
        'downgrade': -0.8, 'downgraded': -0.8, 'sell': -0.6, 'bearish': -0.7,
        'warning': -0.7, 'warns': -0.6, 'lawsuit': -0.7, 'investigation': -0.7,
        'fraud': -0.9, 'scandal': -0.9, 'bankruptcy': -1.0, 'default': -0.9,
        'layoffs': -0.6, 'cuts': -0.5, 'recession': -0.7, 'crisis': -0.8,
        'halt': -0.7, 'halted': -0.7, 'suspended': -0.7, 'delisted': -0.9,
        'sec investigation': -0.8, 'subpoena': -0.7,
    }

    # Mildly negative words (score: -0.3 to -0.6)
    MILD_NEGATIVE = {
        'falls': -0.4, 'drops': -0.4, 'declines': -0.4, 'slips': -0.3,
        'weakness': -0.4, 'struggles': -0.5, 'concerns': -0.4, 'worried': -0.4,
        'uncertain': -0.3, 'volatile': -0.3, 'risk': -0.3, 'caution': -0.3,
        'headwinds': -0.4, 'challenges': -0.3, 'pressure': -0.4,
        'slowdown': -0.5, 'contraction': -0.5,
    }

    # Negation words that flip sentiment
    NEGATIONS = {
        'not', 'no', "n't", 'never', 'neither', 'nobody', 'nothing',
        'nowhere', 'hardly', 'barely', 'rarely', 'seldom',
    }

    # Intensifiers that amplify sentiment
    INTENSIFIERS = {
        'very': 1.3, 'extremely': 1.5, 'highly': 1.3, 'significantly': 1.3,
        'substantially': 1.3, 'dramatically': 1.4, 'sharply': 1.3,
        'massively': 1.4, 'huge': 1.3, 'major': 1.2,
    }

    # Source credibility scores
    SOURCE_CREDIBILITY = {
        'reuters': 0.95, 'bloomberg': 0.95, 'wsj': 0.95, 'ft': 0.95,
        'cnbc': 0.85, 'marketwatch': 0.80, 'yahoo': 0.75, 'seekingalpha': 0.70,
        'benzinga': 0.75, 'investopedia': 0.80, 'fool': 0.65,
        'zacks': 0.70, 'tipranks': 0.75, 'finviz': 0.70,
    }

    @classmethod
    def get_all_positive(cls) -> Dict[str, float]:
        return {**cls.STRONG_POSITIVE, **cls.MILD_POSITIVE}

    @classmethod
    def get_all_negative(cls) -> Dict[str, float]:
        return {**cls.STRONG_NEGATIVE, **cls.MILD_NEGATIVE}


class NewsSentimentAnalyzer:
    """
    Comprehensive news and sentiment analysis.

    Combines multiple signals:
    - Raw sentiment from text
    - News velocity (rate of coverage)
    - Source credibility weighting
    - Recency weighting
    - Catalyst detection
    """

    def __init__(
        self,
        recency_half_life_hours: float = 12.0,  # Older news weighted less
        min_relevance: float = 0.3,              # Filter low-relevance news
    ):
        self.recency_half_life = recency_half_life_hours
        self.min_relevance = min_relevance
        self.lexicon = SentimentLexicon()

    def analyze(
        self,
        symbol: str,
        news_items: List[NewsItem],
        current_time: Optional[datetime] = None,
    ) -> NewsScore:
        """
        Analyze news flow for a symbol.

        Args:
            symbol: Stock symbol
            news_items: List of news items (ideally last 7 days)
            current_time: Current time for recency calculations

        Returns:
            NewsScore with all metrics
        """
        if current_time is None:
            current_time = datetime.now()

        # Filter to relevant news
        relevant_news = [
            n for n in news_items
            if symbol in n.symbols or self._is_relevant(n, symbol)
        ]

        # Separate by timeframe
        news_24h = [n for n in relevant_news
                    if (current_time - n.published).total_seconds() < 86400]
        news_7d = relevant_news

        # News counts
        news_count_24h = len(news_24h)
        news_count_7d = len(news_7d)

        # News velocity (is coverage increasing?)
        news_velocity = self._calculate_news_velocity(relevant_news, current_time)

        # Most recent news
        if relevant_news:
            most_recent = max(relevant_news, key=lambda n: n.published)
            news_recency_hours = (current_time - most_recent.published).total_seconds() / 3600
        else:
            news_recency_hours = 168.0  # 7 days (no recent news)

        # Sentiment analysis
        sentiment_results = self._analyze_sentiment(relevant_news, current_time)

        # Catalyst detection
        catalysts = self._detect_catalysts(relevant_news, symbol)

        # Breaking news detection
        has_breaking = self._detect_breaking_news(news_24h)

        # Analyst action detection
        has_analyst = any(
            self._is_analyst_news(n) for n in news_24h
        )

        # Insider activity
        has_insider = any(
            self._is_insider_news(n) for n in news_7d
        )

        # Calculate derived scores
        news_score = self._score_news_flow(
            news_count_24h, news_velocity, sentiment_results['sentiment']
        )
        catalyst_score = self._score_catalysts(catalysts, has_breaking, has_analyst)

        return NewsScore(
            news_count_24h=news_count_24h,
            news_count_7d=news_count_7d,
            news_velocity=news_velocity,
            news_recency_hours=news_recency_hours,
            sentiment_score=sentiment_results['sentiment'],
            sentiment_magnitude=sentiment_results['magnitude'],
            sentiment_trend=sentiment_results['trend'],
            mainstream_sentiment=sentiment_results['mainstream'],
            financial_sentiment=sentiment_results['financial'],
            social_sentiment=sentiment_results['social'],
            has_breaking_news=has_breaking,
            has_analyst_action=has_analyst,
            has_insider_activity=has_insider,
            news_score=news_score,
            catalyst_score=catalyst_score,
        )

    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a text string.

        Returns:
            (sentiment, confidence) where sentiment is -1 to 1
        """
        if not text:
            return 0.0, 0.0

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_score = 0.0
        negative_score = 0.0
        word_count = 0
        negation_active = False

        all_positive = self.lexicon.get_all_positive()
        all_negative = self.lexicon.get_all_negative()

        for i, word in enumerate(words):
            # Check for negation
            if word in self.lexicon.NEGATIONS:
                negation_active = True
                continue

            # Check for intensifiers
            intensifier = 1.0
            if i > 0 and words[i-1] in self.lexicon.INTENSIFIERS:
                intensifier = self.lexicon.INTENSIFIERS[words[i-1]]

            # Check sentiment
            if word in all_positive:
                score = all_positive[word] * intensifier
                if negation_active:
                    negative_score += score * 0.5  # Negated positive = weaker negative
                else:
                    positive_score += score
                word_count += 1
                negation_active = False

            elif word in all_negative:
                score = abs(all_negative[word]) * intensifier
                if negation_active:
                    positive_score += score * 0.5  # Negated negative = weaker positive
                else:
                    negative_score += score
                word_count += 1
                negation_active = False

            # Also check for multi-word phrases
            if i < len(words) - 1:
                phrase = f"{word} {words[i+1]}"
                if phrase in all_positive:
                    positive_score += all_positive[phrase]
                    word_count += 1
                elif phrase in all_negative:
                    negative_score += abs(all_negative[phrase])
                    word_count += 1

        if word_count == 0:
            return 0.0, 0.0

        # Calculate net sentiment
        total = positive_score + negative_score
        if total == 0:
            return 0.0, 0.0

        sentiment = (positive_score - negative_score) / total
        confidence = min(1.0, word_count / 5)  # More words = more confident

        return sentiment, confidence

    def _is_relevant(self, news: NewsItem, symbol: str) -> bool:
        """Check if news is relevant to symbol."""
        # Check title and summary for symbol mention
        text = f"{news.title} {news.summary}".upper()
        return symbol.upper() in text

    def _calculate_news_velocity(
        self,
        news_items: List[NewsItem],
        current_time: datetime,
    ) -> float:
        """
        Calculate rate of change in news coverage.

        Returns -1 to 1 where:
        - Positive = increasing coverage
        - Negative = decreasing coverage
        """
        if len(news_items) < 3:
            return 0.0

        # Count news in recent vs older periods
        recent_cutoff = current_time - timedelta(hours=24)
        older_cutoff = current_time - timedelta(hours=72)

        recent_count = sum(1 for n in news_items if n.published > recent_cutoff)
        older_count = sum(1 for n in news_items
                         if older_cutoff < n.published <= recent_cutoff)

        # Normalize by time period (24h vs 48h)
        older_rate = older_count / 2  # Per 24h equivalent

        if older_rate == 0:
            if recent_count > 0:
                return 1.0  # New coverage starting
            return 0.0

        velocity = (recent_count - older_rate) / older_rate
        velocity = max(-1, min(1, velocity))

        return velocity

    def _analyze_sentiment(
        self,
        news_items: List[NewsItem],
        current_time: datetime,
    ) -> Dict[str, float]:
        """
        Aggregate sentiment across all news with weighting.
        """
        if not news_items:
            return {
                'sentiment': 0.0, 'magnitude': 0.0, 'trend': 0.0,
                'mainstream': 0.0, 'financial': 0.0, 'social': 0.0,
            }

        weighted_sentiments = []
        weights = []
        mainstream_sents = []
        financial_sents = []
        social_sents = []

        for news in news_items:
            # Calculate sentiment if not already done
            if news.sentiment == 0:
                sent, conf = self.analyze_text_sentiment(
                    f"{news.title}. {news.summary}"
                )
                # Title is more important
                title_sent, _ = self.analyze_text_sentiment(news.title)
                news.sentiment = title_sent * 0.6 + sent * 0.4

            # Recency weight
            hours_old = (current_time - news.published).total_seconds() / 3600
            recency_weight = 0.5 ** (hours_old / self.recency_half_life)

            # Credibility weight
            credibility = self._get_source_credibility(news.source)

            # Combined weight
            weight = recency_weight * credibility * news.relevance
            weights.append(weight)
            weighted_sentiments.append(news.sentiment * weight)

            # Categorize by source type
            source_type = self._categorize_source(news.source)
            if source_type == 'mainstream':
                mainstream_sents.append(news.sentiment)
            elif source_type == 'financial':
                financial_sents.append(news.sentiment)
            elif source_type == 'social':
                social_sents.append(news.sentiment)

        # Aggregate sentiment
        total_weight = sum(weights)
        if total_weight > 0:
            sentiment = sum(weighted_sentiments) / total_weight
        else:
            sentiment = 0.0

        # Magnitude (how strong are opinions?)
        magnitude = sum(abs(s) * w for s, w in zip(
            [n.sentiment for n in news_items], weights
        )) / total_weight if total_weight > 0 else 0.0

        # Trend (is sentiment improving?)
        trend = self._calculate_sentiment_trend(news_items, current_time)

        # Source-specific sentiment
        mainstream = sum(mainstream_sents) / len(mainstream_sents) if mainstream_sents else 0
        financial = sum(financial_sents) / len(financial_sents) if financial_sents else 0
        social = sum(social_sents) / len(social_sents) if social_sents else 0

        return {
            'sentiment': sentiment,
            'magnitude': magnitude,
            'trend': trend,
            'mainstream': mainstream,
            'financial': financial,
            'social': social,
        }

    def _calculate_sentiment_trend(
        self,
        news_items: List[NewsItem],
        current_time: datetime,
    ) -> float:
        """Calculate if sentiment is improving or declining."""
        if len(news_items) < 3:
            return 0.0

        # Split into recent and older
        cutoff = current_time - timedelta(hours=48)
        recent = [n for n in news_items if n.published > cutoff]
        older = [n for n in news_items if n.published <= cutoff]

        if not recent or not older:
            return 0.0

        recent_sent = sum(n.sentiment for n in recent) / len(recent)
        older_sent = sum(n.sentiment for n in older) / len(older)

        trend = recent_sent - older_sent
        return max(-1, min(1, trend))

    def _detect_catalysts(
        self,
        news_items: List[NewsItem],
        symbol: str,
    ) -> List[Catalyst]:
        """Detect specific catalysts from news."""
        catalysts = []

        catalyst_patterns = {
            CatalystType.EARNINGS_JUST_REPORTED: [
                r'reports?\s+(?:q[1-4]|quarterly|annual)\s+(?:earnings|results)',
                r'(?:beats?|misses?)\s+(?:earnings|estimates|expectations)',
                r'eps\s+(?:of\s+)?\$[\d.]+',
            ],
            CatalystType.ANALYST_UPGRADE: [
                r'upgrade[sd]?\s+(?:to|from)',
                r'raises?\s+(?:price\s+)?target',
                r'initiates?\s+(?:with\s+)?buy',
            ],
            CatalystType.ANALYST_DOWNGRADE: [
                r'downgrade[sd]?\s+(?:to|from)',
                r'lowers?\s+(?:price\s+)?target',
                r'cuts?\s+(?:rating|price\s+target)',
            ],
            CatalystType.FDA_DECISION: [
                r'fda\s+(?:approves?|approval|rejects?|rejection)',
                r'(?:drug|treatment)\s+(?:approved|rejected)',
            ],
            CatalystType.MERGER_ACQUISITION: [
                r'(?:acquires?|acquisition|merger|buyout)',
                r'(?:to\s+)?(?:buy|purchase|acquire)\s+',
                r'deal\s+(?:worth|valued)',
            ],
            CatalystType.INSIDER_BUYING: [
                r'insider\s+(?:buying|purchases?|buys?)',
                r'ceo\s+(?:buys?|purchases?)',
            ],
            CatalystType.INSIDER_SELLING: [
                r'insider\s+(?:selling|sales?|sells?)',
                r'ceo\s+(?:sells?|sold)',
            ],
        }

        for news in news_items:
            text = f"{news.title} {news.summary}".lower()

            for catalyst_type, patterns in catalyst_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        catalysts.append(Catalyst(
                            type=catalyst_type,
                            description=news.title[:100],
                            timestamp=news.published,
                            impact_score=0.7,
                            confidence=0.8,
                            source=news.source,
                        ))
                        break  # One catalyst per type per news item

        return catalysts

    def _detect_breaking_news(self, news_24h: List[NewsItem]) -> bool:
        """Detect if there's breaking/significant news."""
        breaking_indicators = [
            'breaking', 'just in', 'alert', 'urgent',
            'halted', 'suspended', 'crash', 'surge',
        ]

        for news in news_24h:
            title_lower = news.title.lower()
            if any(ind in title_lower for ind in breaking_indicators):
                return True

            # High-credibility source with recent news
            if news.credibility > 0.9:
                hours_old = (datetime.now() - news.published).total_seconds() / 3600
                if hours_old < 2:
                    return True

        return False

    def _is_analyst_news(self, news: NewsItem) -> bool:
        """Check if news is analyst-related."""
        keywords = ['upgrade', 'downgrade', 'rating', 'target', 'analyst', 'initiates']
        text_lower = news.title.lower()
        return any(kw in text_lower for kw in keywords)

    def _is_insider_news(self, news: NewsItem) -> bool:
        """Check if news is insider-related."""
        keywords = ['insider', 'ceo buy', 'ceo sell', 'director', 'form 4', '10b5-1']
        text_lower = news.title.lower()
        return any(kw in text_lower for kw in keywords)

    def _get_source_credibility(self, source: str) -> float:
        """Get credibility score for a news source."""
        source_lower = source.lower()
        for key, score in self.lexicon.SOURCE_CREDIBILITY.items():
            if key in source_lower:
                return score
        return 0.5  # Default credibility

    def _categorize_source(self, source: str) -> str:
        """Categorize news source type."""
        source_lower = source.lower()

        mainstream = ['reuters', 'bloomberg', 'wsj', 'nyt', 'ft', 'ap']
        financial = ['cnbc', 'marketwatch', 'yahoo', 'benzinga', 'seekingalpha', 'zacks']
        social = ['reddit', 'twitter', 'stocktwits', 'discord']

        if any(s in source_lower for s in mainstream):
            return 'mainstream'
        elif any(s in source_lower for s in financial):
            return 'financial'
        elif any(s in source_lower for s in social):
            return 'social'
        return 'other'

    def _score_news_flow(
        self,
        news_count_24h: int,
        velocity: float,
        sentiment: float,
    ) -> float:
        """Score overall news flow (0-100)."""
        # Base score from news count
        if news_count_24h == 0:
            count_score = 20  # No news = neutral, slightly negative
        elif news_count_24h < 3:
            count_score = 40
        elif news_count_24h < 10:
            count_score = 60
        else:
            count_score = 80  # Lots of coverage

        # Velocity bonus/penalty
        velocity_adj = velocity * 15  # ±15 points

        # Sentiment component
        sentiment_adj = sentiment * 10  # ±10 points based on sentiment direction

        # Strong sentiment (either direction) is good for day trading
        magnitude_bonus = abs(sentiment) * 10

        score = count_score + velocity_adj + magnitude_bonus
        return max(0, min(100, score))

    def _score_catalysts(
        self,
        catalysts: List[Catalyst],
        has_breaking: bool,
        has_analyst: bool,
    ) -> float:
        """Score catalyst strength (0-100)."""
        score = 30  # Base score

        # Each catalyst adds points
        for catalyst in catalysts:
            score += catalyst.impact_score * 15

        # Breaking news is significant
        if has_breaking:
            score += 25

        # Analyst action
        if has_analyst:
            score += 15

        return min(100, score)
