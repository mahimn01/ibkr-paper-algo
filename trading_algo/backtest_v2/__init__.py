"""
Backtest V2 - Enterprise-level backtesting system.

Features:
- Accurate simulation with slippage and commissions
- MAE/MFE tracking for trade analysis
- Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)
- Data caching for fast reruns
- Multi-symbol support
- Full export to JSON, CSV, charts, and HTML
"""

from .models import (
    Bar,
    BacktestTrade,
    DailyResult,
    DrawdownPeriod,
    BacktestMetrics,
    EquityPoint,
    BacktestConfig,
    BacktestResults,
)
from .engine import BacktestEngine
from .data_provider import DataProvider, DataRequest
from .exporter import BacktestExporter

__all__ = [
    # Models
    "Bar",
    "BacktestTrade",
    "DailyResult",
    "DrawdownPeriod",
    "BacktestMetrics",
    "EquityPoint",
    "BacktestConfig",
    "BacktestResults",
    # Engine
    "BacktestEngine",
    # Data
    "DataProvider",
    "DataRequest",
    # Export
    "BacktestExporter",
]

__version__ = "2.0.0"
