"""
Base Options Trading Strategy

Abstract base class for all options strategies.
Provides common functionality for position management, Greeks tracking,
and performance measurement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime
import numpy as np

from trading_algo.quant_core.models.greeks import OptionSpec, Greeks, BlackScholesCalculator


@dataclass
class OptionPosition:
    """Represents an option position."""
    symbol: str
    option_type: Literal['call', 'put']
    strike: float
    expiry: datetime
    quantity: float  # Positive = long, negative = short
    entry_price: float
    entry_date: datetime
    underlying_price_at_entry: float

    # Current state
    current_price: Optional[float] = None
    current_underlying_price: Optional[float] = None
    implied_volatility: Optional[float] = None

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    def update_greeks(
        self,
        current_underlying_price: float,
        current_volatility: float,
        risk_free_rate: float = 0.05
    ):
        """Update Greeks based on current market conditions."""
        self.current_underlying_price = current_underlying_price

        # Calculate time to expiry
        time_to_expiry = max((self.expiry - datetime.now()).days / 365.0, 0.0)

        if time_to_expiry <= 0:
            # Expired
            self.delta = self.gamma = self.theta = self.vega = self.rho = 0.0
            if self.option_type == 'call':
                self.current_price = max(current_underlying_price - self.strike, 0)
            else:
                self.current_price = max(self.strike - current_underlying_price, 0)
            return

        # Calculate Greeks
        spec = OptionSpec(
            spot=current_underlying_price,
            strike=self.strike,
            time_to_expiry=time_to_expiry,
            volatility=current_volatility,
            risk_free_rate=risk_free_rate,
            option_type=self.option_type
        )

        greeks = BlackScholesCalculator.calculate_all_greeks(spec)

        self.current_price = greeks.price
        self.delta = greeks.delta * self.quantity
        self.gamma = greeks.gamma * self.quantity
        self.theta = greeks.theta * self.quantity
        self.vega = greeks.vega * self.quantity
        self.rho = greeks.rho * self.quantity
        self.implied_volatility = current_volatility

    def pnl(self) -> float:
        """Calculate current PnL."""
        if self.current_price is None:
            return 0.0
        return self.quantity * (self.current_price - self.entry_price) * 100  # $100 per contract

    def is_expired(self) -> bool:
        """Check if option has expired."""
        return datetime.now() >= self.expiry


@dataclass
class StrategyPerformance:
    """Track strategy performance metrics."""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_premium_collected: float = 0.0
    total_premium_paid: float = 0.0
    total_commissions: float = 0.0

    # Greeks
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0

    # Time series
    equity_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)

    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total

    def profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)."""
        # Simplified - would track actual win/loss amounts
        if self.losing_trades == 0:
            return float('inf') if self.winning_trades > 0 else 0.0
        return self.winning_trades / self.losing_trades

    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        return float((np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)))

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        return float(np.max(drawdown))


class BaseOptionStrategy(ABC):
    """
    Abstract base class for options trading strategies.

    All option strategies should inherit from this class and implement:
    - generate_signals(): Determine what positions to take
    - execute_trades(): Execute the positions
    - manage_risk(): Monitor and manage risk
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.10,  # 10% of capital per position
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize base strategy.

        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            risk_free_rate: Risk-free rate for Greeks calculation
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate

        # Positions
        self.positions: List[OptionPosition] = []
        self.closed_positions: List[OptionPosition] = []

        # Performance tracking
        self.performance = StrategyPerformance()

    @abstractmethod
    def generate_signals(
        self,
        symbol: str,
        underlying_price: float,
        historical_data: Dict,
        current_date: datetime
    ) -> List[Dict]:
        """
        Generate trading signals.

        Args:
            symbol: Underlying symbol
            underlying_price: Current underlying price
            historical_data: Historical price/volatility data
            current_date: Current date

        Returns:
            List of signal dictionaries with position specifications
        """
        pass

    def open_position(
        self,
        symbol: str,
        option_type: Literal['call', 'put'],
        strike: float,
        expiry: datetime,
        quantity: float,
        price: float,
        underlying_price: float,
        current_date: datetime,
        implied_vol: float = 0.25
    ) -> OptionPosition:
        """
        Open a new option position.

        Args:
            symbol: Underlying symbol
            option_type: 'call' or 'put'
            strike: Strike price
            expiry: Expiration date
            quantity: Number of contracts (negative for short)
            price: Option price
            underlying_price: Current underlying price
            current_date: Trade date
            implied_vol: Implied volatility

        Returns:
            Created position
        """
        position = OptionPosition(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            entry_price=price,
            entry_date=current_date,
            underlying_price_at_entry=underlying_price
        )

        # Update Greeks
        position.update_greeks(underlying_price, implied_vol, self.risk_free_rate)

        # Update capital and tracking
        cost = abs(quantity) * price * 100  # $100 per contract
        commission = cost * 0.0005  # 0.05% commission

        if quantity > 0:
            # Long position - pay premium
            self.capital -= cost + commission
            self.performance.total_premium_paid += cost
        else:
            # Short position - collect premium
            self.capital += cost - commission
            self.performance.total_premium_collected += cost

        self.performance.total_commissions += commission
        self.performance.total_trades += 1

        self.positions.append(position)

        return position

    def close_position(
        self,
        position: OptionPosition,
        price: float,
        current_date: datetime
    ):
        """Close an existing position."""
        # Calculate PnL
        pnl = position.quantity * (price - position.entry_price) * 100

        # Update capital
        cost = abs(position.quantity) * price * 100
        commission = cost * 0.0005

        if position.quantity > 0:
            # Closing long - sell
            self.capital += cost - commission
        else:
            # Closing short - buy back
            self.capital -= cost + commission

        self.performance.total_commissions += commission
        self.performance.realized_pnl += pnl

        if pnl > 0:
            self.performance.winning_trades += 1
        else:
            self.performance.losing_trades += 1

        # Move to closed positions
        self.positions.remove(position)
        self.closed_positions.append(position)

    def update_positions(
        self,
        symbol: str,
        underlying_price: float,
        implied_vol: float,
        current_date: datetime
    ):
        """Update all positions with current market data."""
        # Update Greeks for all positions
        for position in self.positions:
            if position.symbol == symbol:
                position.update_greeks(underlying_price, implied_vol, self.risk_free_rate)

        # Check for expired options
        expired = [p for p in self.positions if p.is_expired()]
        for position in expired:
            # Close at intrinsic value
            if position.option_type == 'call':
                close_price = max(underlying_price - position.strike, 0)
            else:
                close_price = max(position.strike - underlying_price, 0)

            self.close_position(position, close_price, current_date)

        # Update portfolio Greeks
        self.performance.portfolio_delta = sum(p.delta for p in self.positions)
        self.performance.portfolio_gamma = sum(p.gamma for p in self.positions)
        self.performance.portfolio_theta = sum(p.theta for p in self.positions)
        self.performance.portfolio_vega = sum(p.vega for p in self.positions)

        # Update unrealized PnL
        self.performance.unrealized_pnl = sum(p.pnl() for p in self.positions)

        # Total PnL
        self.performance.total_pnl = self.performance.realized_pnl + self.performance.unrealized_pnl

        # Update equity curve
        equity = self.capital + self.performance.unrealized_pnl
        self.performance.equity_curve.append(equity)
        self.performance.dates.append(current_date)

    @abstractmethod
    def manage_risk(self, current_date: datetime):
        """
        Risk management logic.

        Should check for:
        - Stop losses
        - Position limits
        - Delta hedging needs
        - Margin requirements
        """
        pass

    def get_portfolio_stats(self) -> Dict:
        """Get current portfolio statistics."""
        return {
            'capital': self.capital,
            'total_pnl': self.performance.total_pnl,
            'realized_pnl': self.performance.realized_pnl,
            'unrealized_pnl': self.performance.unrealized_pnl,
            'total_trades': self.performance.total_trades,
            'win_rate': self.performance.win_rate(),
            'sharpe_ratio': self.performance.sharpe_ratio(),
            'max_drawdown': self.performance.max_drawdown(),
            'delta': self.performance.portfolio_delta,
            'gamma': self.performance.portfolio_gamma,
            'theta': self.performance.portfolio_theta,
            'vega': self.performance.portfolio_vega,
            'open_positions': len(self.positions),
        }
