"""
RAT Backtesting Infrastructure

Complete backtesting system for the RAT framework:
1. Data loaders (CSV, Yahoo Finance, IBKR historical)
2. Backtester engine
3. Performance analytics
4. Report generation

Designed for rigorous strategy validation.
"""

from trading_algo.rat.backtest.backtester import (
    RATBacktester,
    BacktestResult,
    BacktestConfig,
    run_walk_forward,
    aggregate_walk_forward_results,
)
from trading_algo.rat.backtest.data_loader import (
    DataLoader,
    CSVLoader,
    YahooLoader,
    Bar,
)
from trading_algo.rat.backtest.analytics import (
    PerformanceAnalytics,
    PerformanceMetrics,
)

__all__ = [
    "RATBacktester",
    "BacktestResult",
    "BacktestConfig",
    "run_walk_forward",
    "aggregate_walk_forward_results",
    "DataLoader",
    "CSVLoader",
    "YahooLoader",
    "Bar",
    "PerformanceAnalytics",
    "PerformanceMetrics",
]
