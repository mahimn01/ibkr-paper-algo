"""
Risk Controller

Real-time risk management that monitors and controls trading risk.

Risk Dimensions:
    1. Position-level: Individual position limits
    2. Portfolio-level: Total exposure and concentration
    3. Drawdown: Maximum loss thresholds
    4. Tail Risk: VaR and ES monitoring
    5. Correlation: Correlation breakdown detection

Risk Responses:
    - Scale down positions
    - Halt new entries
    - Force position reduction
    - Full liquidation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from datetime import datetime, timedelta
import logging

from trading_algo.quant_core.utils.constants import (
    EPSILON, SQRT_252, MAX_DRAWDOWN_THRESHOLD
)
from trading_algo.quant_core.risk.expected_shortfall import ExpectedShortfall
from trading_algo.quant_core.risk.tail_risk import TailRiskManager, TailRiskLevel
from trading_algo.quant_core.risk.metrics import RiskMetrics


logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Risk management actions."""
    ALLOW = auto()           # Normal trading allowed
    SCALE_DOWN = auto()      # Reduce position sizes
    HALT_ENTRIES = auto()    # No new positions
    REDUCE_POSITIONS = auto() # Must reduce existing positions
    LIQUIDATE = auto()       # Liquidate everything


@dataclass
class RiskDecision:
    """
    Risk decision for trading action.

    Contains the risk verdict and any required adjustments.
    """
    action: RiskAction
    exposure_multiplier: float   # Scale factor for positions (0-1)
    max_position_size: float     # Maximum single position size
    max_portfolio_exposure: float  # Maximum total exposure
    reason: str
    triggered_limits: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def can_trade(self) -> bool:
        """Whether new trades are allowed."""
        return self.action in (RiskAction.ALLOW, RiskAction.SCALE_DOWN)

    @property
    def must_reduce(self) -> bool:
        """Whether positions must be reduced."""
        return self.action in (RiskAction.REDUCE_POSITIONS, RiskAction.LIQUIDATE)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position limits
    max_position_pct: float = 0.10          # Max 10% in single position
    max_sector_pct: float = 0.30            # Max 30% in single sector
    max_gross_exposure: float = 1.0         # Max 100% gross exposure
    max_net_exposure: float = 0.50          # Max 50% net exposure

    # Drawdown limits
    max_drawdown: float = 0.15              # 15% max drawdown
    drawdown_warning: float = 0.10          # 10% drawdown warning
    daily_loss_limit: float = 0.03          # 3% daily loss limit

    # Tail risk
    var_limit_95: float = 0.02              # 2% daily VaR limit
    es_limit_95: float = 0.03               # 3% daily ES limit

    # Volatility
    max_portfolio_vol: float = 0.20         # 20% annualized vol
    vol_target: float = 0.15                # 15% target vol

    # Correlation
    max_avg_correlation: float = 0.70       # Max average correlation

    # Time-based
    cooldown_after_loss: int = 0            # Days to wait after big loss


class RiskController:
    """
    Real-time risk management controller.

    Monitors portfolio risk across multiple dimensions and
    provides trading decisions based on current risk levels.

    Usage:
        controller = RiskController(config)

        # Before any trade
        decision = controller.evaluate(portfolio, signals)

        if decision.can_trade:
            execute_with_scaling(decision.exposure_multiplier)
        elif decision.must_reduce:
            reduce_positions(decision.max_portfolio_exposure)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk controller.

        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()

        # Risk models
        self.es_calculator = ExpectedShortfall(confidence=0.95)
        self.tail_risk_manager = TailRiskManager(
            drawdown_threshold=self.config.max_drawdown
        )

        # State
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        self._daily_pnl: float = 0.0
        self._returns_history: List[float] = []
        self._last_trade_date: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

    def evaluate(
        self,
        equity: float,
        positions: Dict[str, float],  # symbol -> market value
        returns: NDArray[np.float64],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> RiskDecision:
        """
        Evaluate current risk and provide trading decision.

        Args:
            equity: Current portfolio equity
            positions: Current position values
            returns: Historical portfolio returns
            current_prices: Current asset prices

        Returns:
            RiskDecision with action and limits
        """
        triggered_limits = []
        metrics = {}

        # Update state
        self._update_drawdown(equity)
        self._update_returns(returns)

        # 1. Check drawdown limits
        dd_action, dd_mult = self._check_drawdown()
        if dd_action != RiskAction.ALLOW:
            triggered_limits.append(f"drawdown_{self._current_drawdown:.1%}")
        metrics['current_drawdown'] = self._current_drawdown

        # 2. Check daily loss limit
        daily_action = self._check_daily_loss()
        if daily_action != RiskAction.ALLOW:
            triggered_limits.append(f"daily_loss_{self._daily_pnl:.1%}")
        metrics['daily_pnl'] = self._daily_pnl

        # 3. Check tail risk
        tail_action, tail_mult = self._check_tail_risk(returns, equity)
        if tail_action != RiskAction.ALLOW:
            triggered_limits.append("tail_risk")

        # 4. Check position concentration
        conc_action = self._check_concentration(positions, equity)
        if conc_action != RiskAction.ALLOW:
            triggered_limits.append("concentration")

        # 5. Check exposure limits
        exp_action, exp_metrics = self._check_exposure(positions, equity)
        if exp_action != RiskAction.ALLOW:
            triggered_limits.append("exposure")
        metrics.update(exp_metrics)

        # 6. Check volatility
        vol_action, vol_mult = self._check_volatility(returns)
        if vol_action != RiskAction.ALLOW:
            triggered_limits.append("volatility")
        metrics['portfolio_vol'] = self._calculate_vol(returns)

        # 7. Check cooldown
        if self._in_cooldown():
            triggered_limits.append("cooldown")

        # Determine final action (most restrictive)
        actions = [dd_action, daily_action, tail_action, conc_action, exp_action, vol_action]
        final_action = max(actions, key=lambda a: a.value)

        # Calculate exposure multiplier
        multipliers = [dd_mult, tail_mult, vol_mult]
        exposure_mult = min(multipliers) if multipliers else 1.0

        # If in cooldown, reduce further
        if self._in_cooldown():
            exposure_mult *= 0.5
            final_action = max(final_action, RiskAction.SCALE_DOWN)

        # Calculate position limits
        max_position = self.config.max_position_pct * equity * exposure_mult
        max_exposure = self.config.max_gross_exposure * exposure_mult

        # Generate reason
        if not triggered_limits:
            reason = "All risk limits within bounds"
        else:
            reason = f"Triggered: {', '.join(triggered_limits)}"

        return RiskDecision(
            action=final_action,
            exposure_multiplier=exposure_mult,
            max_position_size=max_position,
            max_portfolio_exposure=max_exposure,
            reason=reason,
            triggered_limits=triggered_limits,
            metrics=metrics,
        )

    def _update_drawdown(self, equity: float) -> None:
        """Update drawdown tracking."""
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            self._current_drawdown = 1 - equity / self._peak_equity
        else:
            self._current_drawdown = 0.0

    def _update_returns(self, returns: NDArray[np.float64]) -> None:
        """Update returns history."""
        if len(returns) > 0:
            self._returns_history.extend(returns.tolist())
            # Keep limited history
            if len(self._returns_history) > 504:  # ~2 years
                self._returns_history = self._returns_history[-504:]

            # Update daily PnL (last return)
            self._daily_pnl = float(returns[-1])

    def _check_drawdown(self) -> Tuple[RiskAction, float]:
        """Check drawdown limits."""
        dd = self._current_drawdown

        if dd >= self.config.max_drawdown:
            # Hard limit breached
            return RiskAction.LIQUIDATE, 0.0

        elif dd >= self.config.max_drawdown * 0.9:
            # Near limit, force reduction
            return RiskAction.REDUCE_POSITIONS, 0.25

        elif dd >= self.config.max_drawdown * 0.75:
            # Warning level, halt new entries
            return RiskAction.HALT_ENTRIES, 0.5

        elif dd >= self.config.drawdown_warning:
            # Scale down
            remaining = (self.config.max_drawdown - dd) / self.config.max_drawdown
            return RiskAction.SCALE_DOWN, remaining

        return RiskAction.ALLOW, 1.0

    def _check_daily_loss(self) -> RiskAction:
        """Check daily loss limit."""
        if self._daily_pnl <= -self.config.daily_loss_limit:
            return RiskAction.HALT_ENTRIES
        return RiskAction.ALLOW

    def _check_tail_risk(
        self,
        returns: NDArray[np.float64],
        equity: float,
    ) -> Tuple[RiskAction, float]:
        """Check tail risk metrics."""
        if len(returns) < 20:
            return RiskAction.ALLOW, 1.0

        try:
            # Calculate VaR and ES
            var_95 = self.es_calculator.calculate_var(returns, 0.95)
            es_95 = self.es_calculator.calculate_es(returns, 0.95)

            # Get tail risk assessment
            assessment = self.tail_risk_manager.assess_risk(returns, equity)

            if assessment.level == TailRiskLevel.CRISIS:
                return RiskAction.LIQUIDATE, 0.1

            elif assessment.level == TailRiskLevel.EXTREME:
                return RiskAction.REDUCE_POSITIONS, 0.25

            elif assessment.level == TailRiskLevel.HIGH:
                return RiskAction.HALT_ENTRIES, 0.5

            elif assessment.level == TailRiskLevel.ELEVATED:
                return RiskAction.SCALE_DOWN, 0.75

            # Check specific limits
            if var_95 > self.config.var_limit_95:
                return RiskAction.SCALE_DOWN, self.config.var_limit_95 / var_95

            if es_95 > self.config.es_limit_95:
                return RiskAction.SCALE_DOWN, self.config.es_limit_95 / es_95

        except Exception as e:
            logger.warning(f"Tail risk calculation failed: {e}")

        return RiskAction.ALLOW, 1.0

    def _check_concentration(
        self,
        positions: Dict[str, float],
        equity: float,
    ) -> RiskAction:
        """Check position concentration."""
        if equity <= 0:
            return RiskAction.ALLOW

        for symbol, value in positions.items():
            position_pct = abs(value) / equity
            if position_pct > self.config.max_position_pct * 1.5:
                return RiskAction.REDUCE_POSITIONS
            elif position_pct > self.config.max_position_pct:
                return RiskAction.SCALE_DOWN

        return RiskAction.ALLOW

    def _check_exposure(
        self,
        positions: Dict[str, float],
        equity: float,
    ) -> Tuple[RiskAction, Dict[str, float]]:
        """Check exposure limits."""
        metrics = {}

        if equity <= 0:
            return RiskAction.ALLOW, metrics

        long_exposure = sum(v for v in positions.values() if v > 0)
        short_exposure = abs(sum(v for v in positions.values() if v < 0))

        gross_exposure = (long_exposure + short_exposure) / equity
        net_exposure = (long_exposure - short_exposure) / equity

        metrics['gross_exposure'] = gross_exposure
        metrics['net_exposure'] = net_exposure
        metrics['long_exposure'] = long_exposure / equity
        metrics['short_exposure'] = short_exposure / equity

        if gross_exposure > self.config.max_gross_exposure * 1.2:
            return RiskAction.REDUCE_POSITIONS, metrics

        elif gross_exposure > self.config.max_gross_exposure:
            return RiskAction.HALT_ENTRIES, metrics

        if abs(net_exposure) > self.config.max_net_exposure * 1.2:
            return RiskAction.REDUCE_POSITIONS, metrics

        elif abs(net_exposure) > self.config.max_net_exposure:
            return RiskAction.SCALE_DOWN, metrics

        return RiskAction.ALLOW, metrics

    def _check_volatility(
        self,
        returns: NDArray[np.float64],
    ) -> Tuple[RiskAction, float]:
        """Check portfolio volatility."""
        if len(returns) < 20:
            return RiskAction.ALLOW, 1.0

        vol = self._calculate_vol(returns)

        if vol > self.config.max_portfolio_vol * 1.5:
            return RiskAction.REDUCE_POSITIONS, self.config.vol_target / vol

        elif vol > self.config.max_portfolio_vol:
            return RiskAction.SCALE_DOWN, self.config.vol_target / vol

        elif vol > self.config.vol_target:
            # Gentle scaling
            return RiskAction.ALLOW, self.config.vol_target / vol

        return RiskAction.ALLOW, 1.0

    def _calculate_vol(self, returns: NDArray[np.float64]) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.15  # Default assumption

        return float(np.std(returns, ddof=1) * SQRT_252)

    def _in_cooldown(self) -> bool:
        """Check if in trading cooldown."""
        if self._cooldown_until is None:
            return False
        return datetime.now() < self._cooldown_until

    def trigger_cooldown(self, days: Optional[int] = None) -> None:
        """Trigger trading cooldown."""
        days = days or self.config.cooldown_after_loss
        if days > 0:
            self._cooldown_until = datetime.now() + timedelta(days=days)
            logger.warning(f"Trading cooldown triggered until {self._cooldown_until}")

    def reset_peak(self, new_peak: float) -> None:
        """Reset peak equity (e.g., after capital injection)."""
        self._peak_equity = new_peak
        self._current_drawdown = 0.0

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        returns = np.array(self._returns_history) if self._returns_history else np.array([0.0])

        report = {
            'current_drawdown': self._current_drawdown,
            'peak_equity': self._peak_equity,
            'daily_pnl': self._daily_pnl,
            'in_cooldown': self._in_cooldown(),
        }

        if len(returns) >= 20:
            report['var_95'] = self.es_calculator.calculate_var(returns, 0.95)
            report['es_95'] = self.es_calculator.calculate_es(returns, 0.95)
            report['volatility'] = self._calculate_vol(returns)

        return report

    def reset(self) -> None:
        """Reset controller state."""
        self._peak_equity = 0.0
        self._current_drawdown = 0.0
        self._daily_pnl = 0.0
        self._returns_history.clear()
        self._cooldown_until = None
        self.tail_risk_manager.reset()
