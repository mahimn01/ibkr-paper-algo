"""
Options Trading Strategies

Systematic options strategies based on academic research:
- Variance Risk Premium (VRP) Harvesting
- Theta Harvesting (Premium Selling)
- Gamma Scalping (Delta-Neutral)
- Dispersion Trading
- FX Carry with Options
- 0DTE Iron Condors
"""

from .base_option_strategy import BaseOptionStrategy, OptionPosition, StrategyPerformance

__all__ = [
    'BaseOptionStrategy',
    'OptionPosition',
    'StrategyPerformance',
]
