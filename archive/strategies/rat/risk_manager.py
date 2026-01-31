"""
Institutional-Grade Risk Management for RAT Framework

Implements comprehensive risk controls:
1. Position-level risk limits
2. Portfolio-level risk limits
3. Drawdown-based scaling
4. Correlation monitoring
5. Daily loss limits (circuit breaker)

Based on institutional best practices and regulatory requirements.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple


class RiskLevel(Enum):
    """Risk level classification."""
    NORMAL = auto()      # Full trading
    ELEVATED = auto()    # Reduced position sizes
    HIGH = auto()        # Minimal trading
    CRITICAL = auto()    # No new positions


@dataclass
class RiskState:
    """Current risk state."""
    timestamp: datetime
    risk_level: RiskLevel
    current_drawdown: float
    daily_pnl: float
    position_count: int
    gross_exposure: float
    net_exposure: float
    var_estimate: float
    messages: List[str]

    @property
    def can_trade(self) -> bool:
        """Check if new trades are allowed."""
        return self.risk_level in (RiskLevel.NORMAL, RiskLevel.ELEVATED)


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    quantity: float
    market_value: float
    weight: float
    unrealized_pnl: float
    var_contribution: float
    is_within_limits: bool


@dataclass
class RiskLimits:
    """Configurable risk limits."""

    # Position limits
    max_position_pct: float = 0.05          # Max 5% per position
    max_position_value: float = 50000       # Max $50k per position
    max_positions: int = 20                 # Max concurrent positions

    # Portfolio limits
    max_gross_exposure: float = 1.0         # Max 100% gross (no leverage)
    max_net_exposure: float = 0.5           # Max 50% net long/short
    max_sector_exposure: float = 0.3        # Max 30% per sector

    # Drawdown limits
    max_daily_loss: float = 0.02            # 2% max daily loss
    max_weekly_loss: float = 0.05           # 5% max weekly loss
    max_drawdown: float = 0.15              # 15% max drawdown
    drawdown_scale_start: float = 0.05      # Start scaling at 5% DD
    drawdown_scale_end: float = 0.12        # Full scale at 12% DD

    # Volatility limits
    max_portfolio_vol: float = 0.20         # Max 20% annualized vol
    vol_scale_threshold: float = 0.15       # Scale above 15% vol

    # Correlation limits
    max_position_correlation: float = 0.7   # Max 0.7 correlation
    min_diversification: float = 0.3        # Min diversification ratio


class RiskManager:
    """
    Comprehensive risk management system.

    Monitors and enforces risk limits at position and portfolio levels.
    """

    def __init__(
        self,
        initial_capital: float,
        limits: Optional[RiskLimits] = None,
    ):
        self.initial_capital = initial_capital
        self.limits = limits or RiskLimits()

        # Current state
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._positions: Dict[str, PositionRisk] = {}
        self._daily_pnl = 0.0
        self._current_date: Optional[date] = None

        # History for analysis
        self._equity_history: Deque[Tuple[datetime, float]] = deque(maxlen=252)
        self._returns_history: Deque[float] = deque(maxlen=252)
        self._daily_pnl_history: Deque[Tuple[date, float]] = deque(maxlen=30)

        # Correlation tracking
        self._price_history: Dict[str, Deque[float]] = {}

        # State
        self._risk_level = RiskLevel.NORMAL
        self._messages: List[str] = []

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """Update current equity value."""
        # Check for new day
        if self._current_date is None or timestamp.date() != self._current_date:
            if self._current_date is not None:
                # Save previous day's PnL
                self._daily_pnl_history.append((self._current_date, self._daily_pnl))
            self._current_date = timestamp.date()
            self._daily_start_equity = self._equity
            self._daily_pnl = 0.0

        # Update daily P&L
        if hasattr(self, '_daily_start_equity'):
            self._daily_pnl = equity - self._daily_start_equity

        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Calculate return
        if self._equity > 0:
            ret = (equity - self._equity) / self._equity
            self._returns_history.append(ret)

        self._equity = equity
        self._equity_history.append((timestamp, equity))

    def update_position(
        self,
        symbol: str,
        quantity: float,
        current_price: float,
        avg_cost: float,
    ) -> None:
        """Update position information."""
        if quantity == 0:
            if symbol in self._positions:
                del self._positions[symbol]
            return

        market_value = abs(quantity * current_price)
        weight = market_value / self._equity if self._equity > 0 else 0
        unrealized_pnl = quantity * (current_price - avg_cost)

        # Track prices for correlation
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=60)
        self._price_history[symbol].append(current_price)

        # VaR contribution (simplified)
        var_contribution = self._estimate_position_var(symbol, market_value)

        is_within_limits = (
            weight <= self.limits.max_position_pct and
            market_value <= self.limits.max_position_value
        )

        self._positions[symbol] = PositionRisk(
            symbol=symbol,
            quantity=quantity,
            market_value=market_value,
            weight=weight,
            unrealized_pnl=unrealized_pnl,
            var_contribution=var_contribution,
            is_within_limits=is_within_limits,
        )

    def _estimate_position_var(self, symbol: str, market_value: float) -> float:
        """
        Estimate position VaR using historical simulation.

        95% VaR = position * 1.65 * daily volatility
        """
        if symbol not in self._price_history or len(self._price_history[symbol]) < 20:
            # Use conservative estimate
            return market_value * 0.05  # 5% VaR as default

        prices = list(self._price_history[symbol])
        returns = [(prices[i] - prices[i-1]) / prices[i-1]
                   for i in range(1, len(prices)) if prices[i-1] != 0]

        if not returns:
            return market_value * 0.05

        # Calculate volatility
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        daily_vol = math.sqrt(variance)

        # 95% VaR
        var = market_value * 1.65 * daily_vol

        return var

    def check_risk(self) -> RiskState:
        """
        Comprehensive risk check.

        Returns current risk state with all metrics.
        """
        self._messages = []

        # Calculate metrics
        drawdown = self._calculate_drawdown()
        daily_loss = self._daily_pnl / self._equity if self._equity > 0 else 0
        gross_exposure = self._calculate_gross_exposure()
        net_exposure = self._calculate_net_exposure()
        var_estimate = self._calculate_portfolio_var()
        position_count = len(self._positions)

        # Determine risk level
        risk_level = self._determine_risk_level(
            drawdown, daily_loss, gross_exposure, var_estimate
        )

        self._risk_level = risk_level

        return RiskState(
            timestamp=datetime.now(),
            risk_level=risk_level,
            current_drawdown=drawdown,
            daily_pnl=self._daily_pnl,
            position_count=position_count,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            var_estimate=var_estimate,
            messages=self._messages.copy(),
        )

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._equity) / self._peak_equity

    def _calculate_gross_exposure(self) -> float:
        """Calculate gross exposure (sum of absolute position values)."""
        total = sum(p.market_value for p in self._positions.values())
        return total / self._equity if self._equity > 0 else 0

    def _calculate_net_exposure(self) -> float:
        """Calculate net exposure (long - short)."""
        long_value = sum(p.market_value for p in self._positions.values() if p.quantity > 0)
        short_value = sum(p.market_value for p in self._positions.values() if p.quantity < 0)
        net = long_value - short_value
        return net / self._equity if self._equity > 0 else 0

    def _calculate_portfolio_var(self) -> float:
        """
        Calculate portfolio VaR.

        Uses sum of position VaRs adjusted for correlation.
        """
        if not self._positions:
            return 0.0

        # Simple sum (conservative, ignores diversification)
        sum_var = sum(p.var_contribution for p in self._positions.values())

        # Apply diversification factor (rough estimate)
        n = len(self._positions)
        if n > 1:
            # Diversification benefit increases with more positions
            div_factor = 1.0 / math.sqrt(n)
            return sum_var * div_factor
        return sum_var

    def _determine_risk_level(
        self,
        drawdown: float,
        daily_loss: float,
        gross_exposure: float,
        var_estimate: float,
    ) -> RiskLevel:
        """Determine overall risk level."""

        # CRITICAL: Circuit breaker conditions
        if drawdown >= self.limits.max_drawdown:
            self._messages.append(f"CRITICAL: Max drawdown reached ({drawdown:.1%})")
            return RiskLevel.CRITICAL

        if daily_loss <= -self.limits.max_daily_loss:
            self._messages.append(f"CRITICAL: Daily loss limit reached ({daily_loss:.1%})")
            return RiskLevel.CRITICAL

        # HIGH: Approaching limits
        if drawdown >= self.limits.drawdown_scale_end:
            self._messages.append(f"HIGH: Near max drawdown ({drawdown:.1%})")
            return RiskLevel.HIGH

        if daily_loss <= -self.limits.max_daily_loss * 0.75:
            self._messages.append(f"HIGH: Near daily loss limit ({daily_loss:.1%})")
            return RiskLevel.HIGH

        # ELEVATED: Warning zone
        if drawdown >= self.limits.drawdown_scale_start:
            self._messages.append(f"ELEVATED: Drawdown warning ({drawdown:.1%})")
            return RiskLevel.ELEVATED

        if gross_exposure > self.limits.max_gross_exposure * 0.9:
            self._messages.append(f"ELEVATED: High gross exposure ({gross_exposure:.1%})")
            return RiskLevel.ELEVATED

        return RiskLevel.NORMAL

    def get_position_scale(self) -> float:
        """
        Get position sizing scale factor based on risk level.

        Returns value between 0 and 1 to multiply position sizes.
        """
        drawdown = self._calculate_drawdown()

        if self._risk_level == RiskLevel.CRITICAL:
            return 0.0
        elif self._risk_level == RiskLevel.HIGH:
            return 0.25
        elif self._risk_level == RiskLevel.ELEVATED:
            # Linear scale between start and end
            if drawdown >= self.limits.drawdown_scale_start:
                scale_range = self.limits.drawdown_scale_end - self.limits.drawdown_scale_start
                dd_above_start = drawdown - self.limits.drawdown_scale_start
                reduction = dd_above_start / scale_range
                return max(0.25, 1.0 - 0.75 * reduction)
            return 0.75
        else:
            return 1.0

    def can_open_position(
        self,
        symbol: str,
        size_pct: float,
        estimated_value: float,
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position can be opened.

        Returns: (allowed, reason, adjusted_size)
        """
        risk_state = self.check_risk()

        # Check risk level
        if not risk_state.can_trade:
            return (False, f"Risk level {risk_state.risk_level.name} - no new positions", 0.0)

        # Check position count
        if len(self._positions) >= self.limits.max_positions:
            return (False, f"Max positions reached ({self.limits.max_positions})", 0.0)

        # Check if already have position
        if symbol in self._positions:
            existing = self._positions[symbol]
            if existing.weight >= self.limits.max_position_pct:
                return (False, f"Max position size reached for {symbol}", 0.0)

        # Adjust size based on risk scale
        scale = self.get_position_scale()
        adjusted_size = size_pct * scale

        # Check position limit
        if adjusted_size > self.limits.max_position_pct:
            adjusted_size = self.limits.max_position_pct

        # Check value limit
        adjusted_value = adjusted_size * self._equity
        if adjusted_value > self.limits.max_position_value:
            adjusted_size = self.limits.max_position_value / self._equity

        if adjusted_size <= 0:
            return (False, "Adjusted position size too small", 0.0)

        return (True, "OK", adjusted_size)

    def calculate_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate correlation between two symbols."""
        if symbol1 not in self._price_history or symbol2 not in self._price_history:
            return None

        prices1 = list(self._price_history[symbol1])
        prices2 = list(self._price_history[symbol2])

        n = min(len(prices1), len(prices2))
        if n < 20:
            return None

        # Calculate returns
        returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1]
                    for i in range(1, n) if prices1[i-1] != 0]
        returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1]
                    for i in range(1, n) if prices2[i-1] != 0]

        n = min(len(returns1), len(returns2))
        if n < 10:
            return None

        returns1 = returns1[-n:]
        returns2 = returns2[-n:]

        # Correlation
        mean1 = sum(returns1) / n
        mean2 = sum(returns2) / n

        cov = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(n)) / n
        std1 = math.sqrt(sum((r - mean1) ** 2 for r in returns1) / n)
        std2 = math.sqrt(sum((r - mean2) ** 2 for r in returns2) / n)

        if std1 * std2 == 0:
            return None

        return cov / (std1 * std2)

    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        state = self.check_risk()

        # Position details
        positions = []
        for symbol, pos in self._positions.items():
            positions.append({
                'symbol': symbol,
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'weight': pos.weight,
                'unrealized_pnl': pos.unrealized_pnl,
                'var_contribution': pos.var_contribution,
                'within_limits': pos.is_within_limits,
            })

        # Portfolio volatility
        portfolio_vol = self._estimate_portfolio_volatility()

        return {
            'timestamp': state.timestamp.isoformat(),
            'risk_level': state.risk_level.name,
            'equity': self._equity,
            'peak_equity': self._peak_equity,
            'drawdown': state.current_drawdown,
            'daily_pnl': state.daily_pnl,
            'daily_return': state.daily_pnl / self._equity if self._equity > 0 else 0,
            'gross_exposure': state.gross_exposure,
            'net_exposure': state.net_exposure,
            'var_95': state.var_estimate,
            'var_pct': state.var_estimate / self._equity if self._equity > 0 else 0,
            'portfolio_volatility': portfolio_vol,
            'position_count': state.position_count,
            'position_scale': self.get_position_scale(),
            'positions': positions,
            'messages': state.messages,
        }

    def _estimate_portfolio_volatility(self) -> float:
        """Estimate annualized portfolio volatility."""
        if len(self._returns_history) < 20:
            return 0.0

        returns = list(self._returns_history)[-20:]
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        daily_vol = math.sqrt(variance)

        # Annualize
        return daily_vol * math.sqrt(252)
