"""
Adversarial Meta-Trader Module

Predicts and exploits predictable behavior of other algorithms:
- Momentum algos: Chase trends with predictable patterns
- Mean reversion algos: Fade moves at predictable levels
- Index rebalancers: Extremely predictable timing/size
- Stop hunters: Exploit thin books near round numbers

Pure mathematical implementation using pattern recognition.
"""

from trading_algo.rat.adversarial.detector import (
    AdversarialDetector,
    AlgoSignature,
    AlgoArchetype,
)

__all__ = ["AdversarialDetector", "AlgoSignature", "AlgoArchetype"]
