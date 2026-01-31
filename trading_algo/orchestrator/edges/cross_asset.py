"""
Edge 5: Cross-Asset Confirmation

Checks if related assets confirm or contradict the trade thesis.

Key insight: Moves confirmed by related assets are more likely to continue.
Divergences often signal reversals or false moves.

Examples:
- INTC up but AMD down → Divergence, be cautious
- INTC up, AMD up, SMH up → Confirmed sector strength
- Stock up but VIX up → Fear in market, potential reversal
"""

from typing import Dict, List, Tuple

from ..types import AssetState, EdgeSignal, EdgeVote


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
        # Precious metals
        "GLD": ["SIVR", "SLV", "SPY"],
        "SIVR": ["GLD", "SLV", "SPY"],
        "SLV": ["GLD", "SIVR", "SPY"],
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
