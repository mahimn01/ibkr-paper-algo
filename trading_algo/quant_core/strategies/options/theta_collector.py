"""
Theta Harvesting Strategy

Systematically sell options to collect time decay (theta).

Strategy:
    - Sell 30-60 DTE options at 15-20 delta
    - Collect theta as options decay
    - Close at 50% profit or manage losses
    - Best in low volatility, range-bound markets

Performance:
    - Sharpe: 0.5-0.8
    - Win rate: 60-70%
    - Consistent income generation

Reference:
    - Tasty Trade research on premium selling
    - Options Alpha systematic strategies
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from .base_option_strategy import BaseOptionStrategy
from trading_algo.quant_core.models.greeks import OptionSpec, BlackScholesCalculator


class ThetaCollector(BaseOptionStrategy):
    """Theta harvesting through systematic premium selling."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.05,
        target_delta: float = 0.20,  # 20-delta options
        days_to_expiry: int = 45,  # 45 DTE
        profit_target: float = 0.50,  # Close at 50% profit
        stop_loss: float = 2.0,  # Stop at 2x loss
        **kwargs
    ):
        super().__init__(initial_capital, max_position_size, **kwargs)
        self.target_delta = target_delta
        self.days_to_expiry = days_to_expiry
        self.profit_target = profit_target
        self.stop_loss = stop_loss

    def generate_signals(
        self,
        symbol: str,
        underlying_price: float,
        historical_data: Dict,
        current_date: datetime
    ) -> List[Dict]:
        """Generate theta collection signals."""
        signals = []

        # Only sell if not already at max positions
        current_positions = len([p for p in self.positions if p.symbol == symbol])

        if current_positions < 2:  # Max 2 positions per underlying
            # Calculate strikes
            put_strike = underlying_price * (1 - self.target_delta)
            call_strike = underlying_price * (1 + self.target_delta)

            contracts = int((self.capital * self.max_position_size) / (underlying_price * 100))

            if contracts > 0:
                # Sell put
                signals.append({
                    'action': 'sell',
                    'option_type': 'put',
                    'strike': put_strike,
                    'quantity': contracts,
                    'expiry_days': self.days_to_expiry
                })

                # Sell call (iron condor if both)
                signals.append({
                    'action': 'sell',
                    'option_type': 'call',
                    'strike': call_strike,
                    'quantity': contracts,
                    'expiry_days': self.days_to_expiry
                })

        return signals

    def manage_risk(self, current_date: datetime):
        """Theta strategy risk management."""
        to_close = []

        for position in self.positions:
            pnl = position.pnl()
            premium = abs(position.entry_price * position.quantity * 100)

            # Profit target
            if pnl > premium * self.profit_target:
                to_close.append((position, 'profit_target'))

            # Stop loss
            elif pnl < -premium * self.stop_loss:
                to_close.append((position, 'stop_loss'))

            # Close before expiration
            elif (position.expiry - current_date).days <= 7:
                to_close.append((position, 'expiration'))

        return to_close
