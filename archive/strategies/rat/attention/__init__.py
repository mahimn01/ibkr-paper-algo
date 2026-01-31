"""
Attention Flow Module

Tracks where market attention is flowing:
1. News velocity - Rate of news mentions
2. Order flow imbalance - Buy vs sell pressure
3. Price acceleration - Second derivative of price

Pure mathematical implementation.
"""

from trading_algo.rat.attention.flow import AttentionFlow, AttentionState
from trading_algo.rat.attention.tracker import AttentionTracker

__all__ = ["AttentionFlow", "AttentionState", "AttentionTracker"]
