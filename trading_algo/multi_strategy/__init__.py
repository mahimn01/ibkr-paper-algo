"""
Multi-Strategy Portfolio Controller

Coordinates multiple independent trading strategies into a unified
portfolio with centralized risk management and capital allocation.

Strategies:
  - Orchestrator: 6-edge ensemble day trading (core)
  - ORB: Opening Range Breakout (9:30-10:30 window)
  - Pairs Trading: Statistical arbitrage (mean reversion of spreads)
  - Pure Momentum: Daily rebalancing trend following

Architecture:
  Each strategy is wrapped in an adapter that implements the
  TradingStrategy protocol. The MultiStrategyController aggregates
  signals, resolves conflicts, and delegates to the quant_core
  RiskController and PortfolioManager for sizing and risk limits.
"""

from .protocol import StrategySignal, StrategyState, TradingStrategy
from .controller import MultiStrategyController, ControllerConfig

__all__ = [
    "StrategySignal",
    "StrategyState",
    "TradingStrategy",
    "MultiStrategyController",
    "ControllerConfig",
]
