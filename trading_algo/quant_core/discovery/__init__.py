"""
Signal Discovery Engine

Automated alpha signal discovery, validation, and lifecycle management.
"""

from trading_algo.quant_core.discovery.pattern_scanner import PatternScanner
from trading_algo.quant_core.discovery.alpha_monitor import AlphaDecayMonitor

__all__ = ["PatternScanner", "AlphaDecayMonitor"]
