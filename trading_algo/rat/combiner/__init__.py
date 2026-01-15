"""
Signal Combiner Module

Combines signals from all RAT modules into unified trading decisions:
1. Attention Flow signals
2. Reflexivity Meter signals
3. Topology Regime signals
4. Adversarial Meta-Trader signals
5. Self-Cannibalizing Alpha signals

Uses optimal weighting based on recent performance and regime context.
Pure mathematical - no LLM required.
"""

from trading_algo.rat.combiner.combiner import (
    SignalCombiner,
    CombinedDecision,
    WeightingMethod,
)
from trading_algo.rat.combiner.filters import (
    SignalFilter,
    FilterType,
)

__all__ = [
    "SignalCombiner",
    "CombinedDecision",
    "WeightingMethod",
    "SignalFilter",
    "FilterType",
]
