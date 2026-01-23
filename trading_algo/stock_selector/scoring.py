"""
Composite Scoring System for Day Trading Stock Selection.

This is the brain of the stock selector - it combines all individual
factor scores into a single composite score with intelligent weighting.

Key Innovations:
1. Regime-Adaptive Weighting: Adjusts factor importance based on market conditions
2. Cross-Factor Interactions: Detects synergies (e.g., high vol + catalyst = bonus)
3. Risk-Adjusted Scoring: Penalizes stocks with asymmetric risk
4. Confidence Scoring: Estimates reliability of the overall score
5. Direction Recommendation: Suggests long vs short bias

The system uses a quasi-ML approach with hand-tuned weights that can be
optimized via backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from trading_algo.stock_selector.models import (
    MarketRegime,
    StockCandidate,
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
    CatalystType,
)


@dataclass
class FactorWeights:
    """Weights for each factor in composite score."""
    volatility: float = 0.12
    volume: float = 0.10
    momentum: float = 0.15
    technical: float = 0.12
    news: float = 0.15
    social: float = 0.08
    options: float = 0.13
    short_interest: float = 0.10
    catalyst: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            'volatility': self.volatility,
            'volume': self.volume,
            'momentum': self.momentum,
            'technical': self.technical,
            'news': self.news,
            'social': self.social,
            'options': self.options,
            'short_interest': self.short_interest,
            'catalyst': self.catalyst,
        }

    def normalize(self) -> 'FactorWeights':
        """Normalize weights to sum to 1."""
        total = (self.volatility + self.volume + self.momentum +
                 self.technical + self.news + self.social +
                 self.options + self.short_interest + self.catalyst)

        if total == 0:
            return self

        return FactorWeights(
            volatility=self.volatility / total,
            volume=self.volume / total,
            momentum=self.momentum / total,
            technical=self.technical / total,
            news=self.news / total,
            social=self.social / total,
            options=self.options / total,
            short_interest=self.short_interest / total,
            catalyst=self.catalyst / total,
        )


class RegimeAdaptiveWeights:
    """
    Adjusts factor weights based on market regime.

    Different market conditions call for different factor emphasis:
    - Bull market: Momentum and technical breakouts matter more
    - Bear market: Short interest and put flow matter more
    - High volatility: Volume and options signals matter more
    - Low volatility: News catalysts matter more (need catalyst to move)
    """

    # Base weights for each regime
    REGIME_WEIGHTS = {
        MarketRegime.STRONG_BULL: FactorWeights(
            volatility=0.10, volume=0.08, momentum=0.20,
            technical=0.15, news=0.12, social=0.10,
            options=0.12, short_interest=0.08, catalyst=0.05,
        ),
        MarketRegime.BULL: FactorWeights(
            volatility=0.12, volume=0.10, momentum=0.18,
            technical=0.14, news=0.13, social=0.08,
            options=0.12, short_interest=0.08, catalyst=0.05,
        ),
        MarketRegime.NEUTRAL: FactorWeights(
            volatility=0.12, volume=0.10, momentum=0.12,
            technical=0.14, news=0.15, social=0.08,
            options=0.14, short_interest=0.10, catalyst=0.05,
        ),
        MarketRegime.BEAR: FactorWeights(
            volatility=0.10, volume=0.10, momentum=0.12,
            technical=0.12, news=0.15, social=0.06,
            options=0.15, short_interest=0.15, catalyst=0.05,
        ),
        MarketRegime.CRISIS: FactorWeights(
            volatility=0.08, volume=0.12, momentum=0.08,
            technical=0.10, news=0.18, social=0.04,
            options=0.18, short_interest=0.15, catalyst=0.07,
        ),
    }

    @classmethod
    def get_weights(cls, regime: MarketRegime) -> FactorWeights:
        """Get weights for the current market regime."""
        weights = cls.REGIME_WEIGHTS.get(regime, cls.REGIME_WEIGHTS[MarketRegime.NEUTRAL])
        return weights.normalize()


class CrossFactorInteractions:
    """
    Detects synergistic combinations of factors that amplify signals.

    Examples:
    - High volatility + News catalyst = Explosive move potential
    - Short squeeze setup + Call flow = Gamma squeeze potential
    - Consolidation + Volume surge = Breakout imminent
    - Oversold + Accumulation = Bottom forming
    """

    @staticmethod
    def calculate_interaction_bonus(
        volatility: VolatilityScore,
        volume: VolumeScore,
        momentum: MomentumScore,
        technical: TechnicalScore,
        news: NewsScore,
        options: OptionsScore,
        short_interest: ShortInterestScore,
        catalysts: List[Catalyst],
    ) -> Tuple[float, List[str]]:
        """
        Calculate bonus from factor interactions.

        Returns:
            (bonus_points, reasons) - Bonus to add to composite score
        """
        bonus = 0.0
        reasons = []

        # Volatility + Catalyst = Explosive potential
        if volatility.volatility_score > 70 and news.catalyst_score > 60:
            bonus += 8
            reasons.append("High vol + catalyst = explosive potential")

        # Short squeeze setup + Bullish options flow
        if short_interest.squeeze_setup_score > 60 and options.call_flow_score > 70:
            bonus += 10
            reasons.append("Squeeze setup + call flow = gamma squeeze potential")

        # Consolidation + Volume surge = Breakout imminent
        if technical.consolidation_days >= 5 and volume.volume_spike_detected:
            bonus += 7
            reasons.append("Consolidation + volume surge = breakout imminent")

        # Oversold + Accumulation = Bottom forming
        if momentum.rsi_14 < 30 and volume.accumulation_detected:
            bonus += 6
            reasons.append("Oversold + accumulation = bottom forming")

        # Overbought + Distribution = Top forming
        if momentum.rsi_14 > 70 and volume.distribution_detected:
            bonus += 6
            reasons.append("Overbought + distribution = top forming")

        # Strong momentum + Trend alignment + Breakout
        if (momentum.momentum_score > 70 and
            momentum.trend_strength > 70 and
            technical.breakout_score > 60):
            bonus += 8
            reasons.append("Momentum + trend + breakout alignment")

        # News velocity + Social buzz = Viral potential
        if news.news_velocity > 0.5 and options.unusual_activity:
            bonus += 5
            reasons.append("Rising news + unusual options = smart money moving")

        # Multiple strong catalysts
        strong_catalysts = [c for c in catalysts if c.impact_score > 0.7]
        if len(strong_catalysts) >= 2:
            bonus += 5
            reasons.append(f"Multiple strong catalysts ({len(strong_catalysts)})")

        return bonus, reasons


class CompositeScorer:
    """
    Main scoring engine that combines all factors.

    Process:
    1. Get regime-adjusted weights
    2. Calculate weighted factor scores
    3. Apply interaction bonuses
    4. Apply risk adjustments
    5. Calculate confidence
    6. Determine direction bias
    """

    def __init__(self, regime: MarketRegime = MarketRegime.NEUTRAL):
        self.regime = regime
        self.weights = RegimeAdaptiveWeights.get_weights(regime)

    def score(self, candidate: StockCandidate) -> StockCandidate:
        """
        Calculate composite score for a stock candidate.

        Modifies candidate in place and returns it.
        """
        # Extract individual scores
        scores = {
            'volatility': candidate.volatility.volatility_score,
            'volume': (candidate.volume.liquidity_score * 0.4 +
                      candidate.volume.volume_momentum_score * 0.3 +
                      candidate.volume.smart_money_score * 0.3),
            'momentum': (candidate.momentum.momentum_score * 0.5 +
                        candidate.momentum.trend_strength * 0.3 +
                        candidate.momentum.mean_reversion_score * 0.2),
            'technical': (candidate.technical.technical_setup_score * 0.4 +
                         candidate.technical.breakout_score * 0.3 +
                         candidate.technical.reversal_score * 0.3),
            'news': (candidate.news.news_score * 0.5 +
                    candidate.news.catalyst_score * 0.5),
            'social': candidate.social.retail_interest_score,
            'options': candidate.options.options_signal_score,
            'short_interest': (candidate.short_interest.squeeze_setup_score * 0.6 +
                              candidate.short_interest.short_pressure_score * 0.4),
            'catalyst': self._score_catalysts(candidate.catalysts),
        }

        # Calculate weighted base score
        weights_dict = self.weights.as_dict()
        base_score = sum(
            scores[factor] * weights_dict[factor]
            for factor in scores
        )

        # Calculate interaction bonus
        bonus, bonus_reasons = CrossFactorInteractions.calculate_interaction_bonus(
            candidate.volatility,
            candidate.volume,
            candidate.momentum,
            candidate.technical,
            candidate.news,
            candidate.options,
            candidate.short_interest,
            candidate.catalysts,
        )

        # Risk adjustment
        risk_penalty = self._calculate_risk_penalty(candidate.risk)

        # Final composite score
        composite = base_score + bonus - risk_penalty
        composite = max(0, min(100, composite))

        # Confidence score
        confidence = self._calculate_confidence(candidate, scores)

        # Expected move
        expected_move = self._estimate_expected_move(candidate)

        # Direction recommendation
        direction = self._determine_direction(candidate)

        # Top factors
        top_factors = self._get_top_factors(scores, weights_dict, bonus_reasons)

        # Risk factors
        risk_factors = self._get_risk_factors(candidate)

        # Update candidate
        candidate.composite_score = composite
        candidate.confidence = confidence
        candidate.expected_move = expected_move
        candidate.recommended_direction = direction
        candidate.top_factors = top_factors
        candidate.risk_factors = risk_factors

        return candidate

    def _score_catalysts(self, catalysts: List[Catalyst]) -> float:
        """Score catalyst presence and strength."""
        if not catalysts:
            return 30  # No catalyst = below average

        total_impact = sum(c.impact_score * c.confidence for c in catalysts)
        score = 30 + min(70, total_impact * 50)

        return score

    def _calculate_risk_penalty(self, risk: RiskMetrics) -> float:
        """Calculate penalty based on risk factors."""
        penalty = 0

        # Earnings proximity risk
        if risk.earnings_days_away is not None:
            if risk.earnings_days_away <= 1:
                penalty += 15  # Day of or next day = high risk
            elif risk.earnings_days_away <= 3:
                penalty += 8

        # Binary event risk
        if risk.has_binary_event:
            penalty += 10

        # Liquidity risk (wide spread)
        if risk.bid_ask_spread_pct > 0.5:
            penalty += 10
        elif risk.bid_ask_spread_pct > 0.2:
            penalty += 5

        # General risk score contribution
        penalty += max(0, (risk.risk_score - 50) * 0.2)

        return penalty

    def _calculate_confidence(
        self,
        candidate: StockCandidate,
        scores: Dict[str, float],
    ) -> float:
        """
        Calculate confidence in the composite score.

        Higher confidence when:
        - Multiple factors agree
        - Scores are not middling (clear signal)
        - Data quality is good
        """
        # Factor agreement (standard deviation of scores)
        score_values = list(scores.values())
        avg_score = sum(score_values) / len(score_values)
        variance = sum((s - avg_score) ** 2 for s in score_values) / len(score_values)
        std_dev = variance ** 0.5

        # Lower std dev = more agreement = higher confidence
        agreement_factor = max(0, 1 - std_dev / 30)

        # Signal strength (scores far from 50 are clearer)
        avg_distance_from_neutral = sum(abs(s - 50) for s in score_values) / len(score_values)
        strength_factor = avg_distance_from_neutral / 30

        # Data quality proxy (volume = good liquidity = reliable data)
        data_quality = min(1, candidate.volume.liquidity_score / 80)

        # Combined confidence
        confidence = (agreement_factor * 0.3 +
                     strength_factor * 0.4 +
                     data_quality * 0.3)

        return max(0.1, min(1.0, confidence))

    def _estimate_expected_move(self, candidate: StockCandidate) -> float:
        """Estimate expected percentage move."""
        # Base from volatility
        base_move = candidate.volatility.atr_percent

        # Adjust for catalyst strength
        if candidate.news.catalyst_score > 70:
            base_move *= 1.5
        elif candidate.news.catalyst_score > 50:
            base_move *= 1.2

        # Adjust for squeeze potential
        if candidate.short_interest.squeeze_setup_score > 60:
            base_move *= 1.3

        # Adjust for options activity
        if candidate.options.unusual_activity:
            base_move *= 1.2

        return base_move

    def _determine_direction(self, candidate: StockCandidate) -> int:
        """
        Determine recommended trade direction.

        Returns:
            1 = long bias, -1 = short bias, 0 = neutral/either
        """
        bullish_signals = 0
        bearish_signals = 0

        # Momentum
        if candidate.momentum.return_5d > 0.02:
            bullish_signals += 2
        elif candidate.momentum.return_5d < -0.02:
            bearish_signals += 2

        # RSI
        if candidate.momentum.rsi_14 < 35:
            bullish_signals += 1  # Oversold bounce
        elif candidate.momentum.rsi_14 > 65:
            bearish_signals += 1  # Overbought fade

        # Trend
        if candidate.momentum.price_vs_sma_20 > 0.02:
            bullish_signals += 1
        elif candidate.momentum.price_vs_sma_20 < -0.02:
            bearish_signals += 1

        # News sentiment
        if candidate.news.sentiment_score > 0.3:
            bullish_signals += 1
        elif candidate.news.sentiment_score < -0.3:
            bearish_signals += 1

        # Options flow
        if candidate.options.smart_money_direction > 0.3:
            bullish_signals += 2
        elif candidate.options.smart_money_direction < -0.3:
            bearish_signals += 2

        # Short squeeze setup (bullish)
        if candidate.short_interest.squeeze_setup_score > 60:
            bullish_signals += 2

        # Technical
        if candidate.technical.breakout_potential > 0.6:
            bullish_signals += 1
        if candidate.technical.breakdown_potential > 0.6:
            bearish_signals += 1

        # Determine direction
        net_signal = bullish_signals - bearish_signals

        if net_signal >= 3:
            return 1  # Long
        elif net_signal <= -3:
            return -1  # Short
        else:
            return 0  # Either direction

    def _get_top_factors(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
        bonus_reasons: List[str],
    ) -> List[str]:
        """Get top contributing factors."""
        # Calculate contribution of each factor
        contributions = [
            (factor, scores[factor] * weights[factor])
            for factor in scores
        ]
        contributions.sort(key=lambda x: -x[1])

        top_factors = []

        # Add top 3 factors
        for factor, contrib in contributions[:3]:
            score = scores[factor]
            if score > 60:
                top_factors.append(f"{factor.title()}: {score:.0f} (strong)")
            else:
                top_factors.append(f"{factor.title()}: {score:.0f}")

        # Add interaction bonuses
        top_factors.extend(bonus_reasons[:2])

        return top_factors

    def _get_risk_factors(self, candidate: StockCandidate) -> List[str]:
        """Get key risk factors."""
        risks = []

        if candidate.risk.earnings_days_away is not None:
            if candidate.risk.earnings_days_away <= 3:
                risks.append(f"Earnings in {candidate.risk.earnings_days_away} days")

        if candidate.risk.has_binary_event:
            risks.append("Binary event risk")

        if candidate.risk.bid_ask_spread_pct > 0.3:
            risks.append(f"Wide spread ({candidate.risk.bid_ask_spread_pct:.1%})")

        if candidate.volatility.gap_frequency > 0.3:
            risks.append("High gap frequency")

        if candidate.volume.liquidity_score < 50:
            risks.append("Low liquidity")

        return risks[:3]


def rank_candidates(
    candidates: List[StockCandidate],
    top_n: int = 10,
) -> List[StockCandidate]:
    """
    Rank scored candidates and return top N.

    Ranking considers:
    - Composite score (primary)
    - Confidence (secondary)
    - Risk-adjusted expected move (tertiary)
    """
    # Filter out untradeable candidates
    tradeable = [c for c in candidates if c.risk.tradeable]

    if not tradeable:
        return []

    # Sort by composite score, then confidence
    tradeable.sort(
        key=lambda c: (c.composite_score, c.confidence, c.expected_move),
        reverse=True
    )

    # Assign ranks
    for i, candidate in enumerate(tradeable):
        candidate.rank = i + 1
        candidate.percentile = (len(tradeable) - i) / len(tradeable) * 100

    return tradeable[:top_n]
