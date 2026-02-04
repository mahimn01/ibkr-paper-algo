"""
0DTE Iron Condor Strategy

Sell same-day expiration iron condors on SPY/QQQ.

Strategy:
    - Sell OTM put spread and OTM call spread
    - Profit from high theta decay on expiration day
    - Target 1-2% daily returns
    - Win rate 70-80% but requires discipline

Performance:
    - High win rate but tail risk
    - Best on low-volatility days
    - Requires strict stop losses

Iron Condor Structure:
    - Sell put at -2 SD
    - Buy put at -3 SD (protection)
    - Sell call at +2 SD
    - Buy call at +3 SD (protection)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from .base_option_strategy import BaseOptionStrategy
from trading_algo.quant_core.models.greeks import OptionSpec, BlackScholesCalculator


class IronCondorStrategy(BaseOptionStrategy):
    """0DTE Iron Condor premium collection."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.10,
        short_strike_distance: float = 0.02,  # 2% OTM
        long_strike_distance: float = 0.03,  # 3% OTM (wings)
        profit_target: float = 0.50,
        stop_loss: float = 2.0,
        **kwargs
    ):
        super().__init__(initial_capital, max_position_size, **kwargs)
        self.short_strike_distance = short_strike_distance
        self.long_strike_distance = long_strike_distance
        self.profit_target = profit_target
        self.stop_loss = stop_loss

    def generate_signals(
        self,
        symbol: str,
        underlying_price: float,
        historical_data: Dict,
        current_date: datetime
    ) -> List[Dict]:
        """Generate iron condor signals (same-day expiration)."""
        signals = []

        # Only trade if no existing positions (one IC per day)
        if len(self.positions) > 0:
            return signals

        # Calculate strikes for iron condor
        # Short strikes (closer to price)
        short_put_strike = underlying_price * (1 - self.short_strike_distance)
        short_call_strike = underlying_price * (1 + self.short_strike_distance)

        # Long strikes (further OTM - protection)
        long_put_strike = underlying_price * (1 - self.long_strike_distance)
        long_call_strike = underlying_price * (1 + self.long_strike_distance)

        # Position size
        contracts = int((self.capital * self.max_position_size) / (underlying_price * 100))

        if contracts > 0:
            # Put spread (sell closer, buy further)
            signals.extend([
                {
                    'action': 'sell',
                    'option_type': 'put',
                    'strike': short_put_strike,
                    'quantity': contracts,
                    'expiry_days': 0  # 0DTE
                },
                {
                    'action': 'buy',
                    'option_type': 'put',
                    'strike': long_put_strike,
                    'quantity': contracts,
                    'expiry_days': 0
                }
            ])

            # Call spread (sell closer, buy further)
            signals.extend([
                {
                    'action': 'sell',
                    'option_type': 'call',
                    'strike': short_call_strike,
                    'quantity': contracts,
                    'expiry_days': 0
                },
                {
                    'action': 'buy',
                    'option_type': 'call',
                    'strike': long_call_strike,
                    'quantity': contracts,
                    'expiry_days': 0
                }
            ])

        return signals

    def manage_risk(self, current_date: datetime):
        """0DTE requires aggressive risk management."""
        to_close = []

        # Calculate total spread P&L
        total_pnl = sum(p.pnl() for p in self.positions)
        total_premium = sum(abs(p.entry_price * p.quantity * 100) for p in self.positions if p.quantity < 0)

        # Close entire IC if profit target hit
        if total_pnl > total_premium * self.profit_target:
            for position in self.positions:
                to_close.append((position, 'profit_target'))

        # Close entire IC if stop loss hit
        elif total_pnl < -total_premium * self.stop_loss:
            for position in self.positions:
                to_close.append((position, 'stop_loss'))

        return to_close
