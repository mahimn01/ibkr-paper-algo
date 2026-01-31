"""
RAT: Reflexive Attention Topology
A novel quantitative trading framework combining:
1. Attention Topology - Where market attention flows
2. Reflexivity Meter - Soros's feedback loops quantified
3. Topological Regime - Persistent homology for regime detection
4. Adversarial Meta-Trader - Exploit other algorithms
5. Self-Cannibalizing Alpha - Factors that destroy themselves

Usage:
    from trading_algo.rat import RATConfig, RATEngine

    config = RATConfig.from_env()
    engine = RATEngine(config)
    engine.start()
"""

from trading_algo.rat.config import RATConfig
from trading_algo.rat.engine import RATEngine
from trading_algo.rat.signals import Signal, SignalType, SignalSource

__all__ = [
    "RATConfig",
    "RATEngine",
    "Signal",
    "SignalType",
    "SignalSource",
]

__version__ = "1.0.0"
