"""
Dashboard widgets for the trading TUI.
"""

from .header import HeaderWidget
from .pnl import PnLWidget
from .positions import PositionsWidget
from .trades import TradesWidget
from .signals import SignalsWidget
from .logs import LogWidget

__all__ = [
    "HeaderWidget",
    "PnLWidget",
    "PositionsWidget",
    "TradesWidget",
    "SignalsWidget",
    "LogWidget",
]
