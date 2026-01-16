"""
Self-Cannibalizing Alpha Module

Detects alpha decay and generates new factors before crowding destroys returns.

Key concepts:
1. Factor Decay Detection - Mathematical tracking of factor performance decay
2. Crowding Measurement - Correlation with market volume patterns
3. Factor Mutation - LLM-assisted (minimal) generation of new factor variations
4. Alpha Rotation - Automatic switching to fresh factors

LLM usage is MINIMAL - only for generating creative factor mutations
when mathematical methods are insufficient.
"""

from trading_algo.rat.alpha.tracker import (
    AlphaTracker,
    AlphaFactor,
    AlphaState,
    DecayStage,
)
from trading_algo.rat.alpha.mutator import (
    AlphaMutator,
    MutationType,
)

__all__ = [
    "AlphaTracker",
    "AlphaFactor",
    "AlphaState",
    "DecayStage",
    "AlphaMutator",
    "MutationType",
]
