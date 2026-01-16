"""
Reflexivity Meter Module

Quantifies Soros-style reflexivity feedback loops:
- Price affecting fundamentals
- Fundamentals affecting price

Uses Granger causality testing for detection.
"""

from trading_algo.rat.reflexivity.meter import (
    ReflexivityMeter,
    ReflexivityState,
    ReflexivityStage,
)

__all__ = ["ReflexivityMeter", "ReflexivityState", "ReflexivityStage"]
