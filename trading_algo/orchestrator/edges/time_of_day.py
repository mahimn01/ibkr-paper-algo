"""
Edge 6: Time-of-Day Patterns

Uses well-documented intraday patterns to time entries and exits.

Key patterns:
- Opening Drive (9:30-10:00): Initial momentum often continues
- Morning Reversal (10:00-10:30): First pullback opportunity
- Mid-Morning Trend (10:30-11:30): Best trending period
- Lunch Chop (11:30-13:30): Lower volume, still tradeable
- Afternoon Trend (13:30-14:30): Second trending period
- Power Hour Setup (14:30-15:00): Institutions position
- MOC Imbalance (15:30-16:00): Volume spike, use caution
"""

from datetime import datetime, time
from typing import Optional

from ..types import EdgeSignal, EdgeVote, TradeType


class TimeOfDayEngine:
    """
    Uses well-documented intraday patterns to time entries and exits.

    Key patterns:
    - Opening Drive (9:30-10:00): Initial momentum often continues
    - Morning Reversal (10:00-10:30): First pullback opportunity
    - Mid-Morning Trend (10:30-11:30): Best trending period
    - Lunch Chop (11:30-13:30): Lower volume, still tradeable
    - Afternoon Trend (13:30-14:30): Second trending period
    - Power Hour Setup (14:30-15:00): Institutions position
    - MOC Imbalance (15:30-16:00): Volume spike, use caution
    """

    def __init__(self):
        self.session_data = {
            "opening_range_high": 0.0,
            "opening_range_low": 0.0,
            "morning_trend": None,  # 'up', 'down', None
            "lunch_direction": None,
        }

    def get_time_window(self, timestamp: datetime) -> str:
        """Determine which time window we're in."""
        t = timestamp.time()

        if time(9, 30) <= t < time(10, 0):
            return "opening_drive"
        elif time(10, 0) <= t < time(10, 30):
            return "morning_reversal"
        elif time(10, 30) <= t < time(11, 30):
            return "mid_morning_trend"
        elif time(11, 30) <= t < time(13, 30):
            return "lunch_chop"
        elif time(13, 30) <= t < time(14, 30):
            return "afternoon_trend"
        elif time(14, 30) <= t < time(15, 30):
            return "power_hour"
        elif time(15, 30) <= t < time(16, 0):
            return "moc_period"
        else:
            return "outside_hours"

    def get_vote(self, timestamp: datetime, trade_type: TradeType) -> EdgeSignal:
        """Get voting signal based on time of day."""
        window = self.get_time_window(timestamp)

        # Opening drive - momentum trades good
        if window == "opening_drive":
            if trade_type == TradeType.MOMENTUM_CONTINUATION:
                return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.7,
                                "Opening drive favors momentum", {"window": window})
            else:
                return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.4,
                                "Opening drive - prefer momentum trades", {"window": window})

        # Morning reversal - mean reversion good
        elif window == "morning_reversal":
            if trade_type == TradeType.MEAN_REVERSION:
                return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.6,
                                "10 AM reversal window - good for mean reversion", {"window": window})
            else:
                return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.5,
                                "10 AM reversal possible", {"window": window})

        # Mid-morning trend - best period
        elif window == "mid_morning_trend":
            return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.7,
                            "Best trending period of day", {"window": window})

        # Lunch period - reduced confidence but still tradeable
        elif window == "lunch_chop":
            return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.4,
                            "Lunch period - lower volume but tradeable", {"window": window})

        # Afternoon trend - second best
        elif window == "afternoon_trend":
            return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.6,
                            "Afternoon trending period", {"window": window})

        # Power hour - institutions active
        elif window == "power_hour":
            if trade_type == TradeType.MOMENTUM_CONTINUATION:
                return EdgeSignal("TimeOfDay", EdgeVote.LONG, 0.6,
                                "Power hour - institutional flow", {"window": window})
            else:
                return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.5,
                                "Power hour - watch for reversals", {"window": window})

        # MOC period - high volatility, reduced confidence
        elif window == "moc_period":
            return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.3,
                            "MOC period - high volatility, use caution", {"window": window})

        return EdgeSignal("TimeOfDay", EdgeVote.NEUTRAL, 0.3,
                        f"Outside trading hours: {window}", {"window": window})
