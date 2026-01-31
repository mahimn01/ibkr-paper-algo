"""
Performance Analytics for RAT Backtesting

Comprehensive metrics calculation:
1. Returns analysis (total, annualized, risk-adjusted)
2. Risk metrics (volatility, max drawdown, VaR)
3. Trade analysis (win rate, profit factor, average trade)
4. Benchmark comparison

All calculations are pure mathematical - no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class Trade:
    """Record of a completed trade."""

    symbol: str
    side: str               # "long" or "short"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_period: timedelta = field(init=False)

    def __post_init__(self):
        self.holding_period = self.exit_time - self.entry_time


@dataclass
class PerformanceMetrics:
    """Complete performance metrics."""

    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    cagr: float

    # Risk
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_duration: int      # days
    var_95: float                   # Value at Risk (95%)
    cvar_95: float                  # Conditional VaR (95%)

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float]

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_trade: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float
    avg_holding_period: float       # days

    # Exposure
    time_in_market: float           # percentage
    max_position_size: float

    # Other
    start_date: datetime
    end_date: datetime
    trading_days: int


class PerformanceAnalytics:
    """Calculate comprehensive performance analytics."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

        # Track equity curve
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._daily_returns: List[float] = []
        self._trades: List[Trade] = []
        self._benchmark_returns: List[float] = []

    def record_equity(self, timestamp: datetime, equity: float) -> None:
        """Record a point on the equity curve."""
        self._equity_curve.append((timestamp, equity))

        if len(self._equity_curve) > 1:
            prev_equity = self._equity_curve[-2][1]
            if prev_equity > 0:
                daily_return = (equity - prev_equity) / prev_equity
                self._daily_returns.append(daily_return)

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade."""
        self._trades.append(trade)

    def record_benchmark_return(self, daily_return: float) -> None:
        """Record benchmark daily return for comparison."""
        self._benchmark_returns.append(daily_return)

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        if not self._equity_curve:
            raise ValueError("No equity data recorded")

        start_date = self._equity_curve[0][0]
        end_date = self._equity_curve[-1][0]
        start_equity = self._equity_curve[0][1]
        end_equity = self._equity_curve[-1][1]

        trading_days = len(self._equity_curve)
        years = trading_days / self.trading_days_per_year

        # Returns
        total_return = end_equity - start_equity
        total_return_pct = total_return / start_equity if start_equity > 0 else 0

        annualized_return = self._annualize_return(total_return_pct, years)
        cagr = self._calculate_cagr(start_equity, end_equity, years)

        # Risk metrics
        volatility = self._calculate_volatility(self._daily_returns)
        annualized_vol = volatility * math.sqrt(self.trading_days_per_year)

        max_dd, max_dd_duration = self._calculate_max_drawdown()
        var_95, cvar_95 = self._calculate_var(self._daily_returns)

        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe(annualized_return, annualized_vol)
        sortino = self._calculate_sortino(self._daily_returns)
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        info_ratio = None
        if self._benchmark_returns:
            info_ratio = self._calculate_information_ratio()

        # Trade statistics
        trade_stats = self._calculate_trade_stats()

        # Exposure
        time_in_market = self._calculate_time_in_market()
        max_position = max(t.quantity * t.entry_price for t in self._trades) if self._trades else 0

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            annualized_volatility=annualized_vol,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            **trade_stats,
            time_in_market=time_in_market,
            max_position_size=max_position,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
        )

    def _annualize_return(self, total_return: float, years: float) -> float:
        """Annualize a total return."""
        if years <= 0:
            return 0
        return total_return / years

    def _calculate_cagr(
        self, start_value: float, end_value: float, years: float
    ) -> float:
        """Calculate Compound Annual Growth Rate."""
        if start_value <= 0 or years <= 0:
            return 0
        return (end_value / start_value) ** (1 / years) - 1

    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate standard deviation of returns."""
        if len(returns) < 2:
            return 0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def _calculate_max_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if not self._equity_curve:
            return 0.0, 0

        peak = self._equity_curve[0][1]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0

        for i, (ts, equity) in enumerate(self._equity_curve):
            if equity > peak:
                peak = equity
                current_dd_start = i
            else:
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = i - current_dd_start

        return max_dd, max_dd_duration

    def _calculate_var(
        self, returns: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR."""
        if not returns:
            return 0.0, 0.0

        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[index] if index < len(sorted_returns) else 0

        # CVaR is average of returns worse than VaR
        worse_returns = sorted_returns[:index + 1]
        cvar = -sum(worse_returns) / len(worse_returns) if worse_returns else 0

        return var, cvar

    def _calculate_sharpe(
        self, annualized_return: float, annualized_vol: float
    ) -> float:
        """Calculate Sharpe ratio."""
        if annualized_vol == 0:
            return 0
        return (annualized_return - self.risk_free_rate) / annualized_vol

    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        if not returns:
            return 0

        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf')

        downside_var = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_dev = math.sqrt(downside_var) * math.sqrt(self.trading_days_per_year)

        if downside_dev == 0:
            return 0

        mean_return = sum(returns) / len(returns) * self.trading_days_per_year
        return (mean_return - self.risk_free_rate) / downside_dev

    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio vs benchmark."""
        if len(self._daily_returns) != len(self._benchmark_returns):
            return 0

        excess_returns = [
            r - b for r, b in zip(self._daily_returns, self._benchmark_returns)
        ]

        mean_excess = sum(excess_returns) / len(excess_returns)
        tracking_error = self._calculate_volatility(excess_returns)

        if tracking_error == 0:
            return 0

        return (mean_excess * self.trading_days_per_year) / (
            tracking_error * math.sqrt(self.trading_days_per_year)
        )

    def _calculate_trade_stats(self) -> Dict:
        """Calculate trade statistics."""
        if not self._trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_trade": 0,
                "average_winner": 0,
                "average_loser": 0,
                "largest_winner": 0,
                "largest_loser": 0,
                "avg_holding_period": 0,
            }

        winners = [t for t in self._trades if t.pnl > 0]
        losers = [t for t in self._trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0001

        avg_holding = sum(
            t.holding_period.total_seconds() / 86400 for t in self._trades
        ) / len(self._trades)

        return {
            "total_trades": len(self._trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(self._trades),
            "profit_factor": gross_profit / gross_loss,
            "average_trade": sum(t.pnl for t in self._trades) / len(self._trades),
            "average_winner": sum(t.pnl for t in winners) / len(winners) if winners else 0,
            "average_loser": sum(t.pnl for t in losers) / len(losers) if losers else 0,
            "largest_winner": max(t.pnl for t in self._trades) if self._trades else 0,
            "largest_loser": min(t.pnl for t in self._trades) if self._trades else 0,
            "avg_holding_period": avg_holding,
        }

    def _calculate_time_in_market(self) -> float:
        """Calculate percentage of time with open positions."""
        if not self._trades or not self._equity_curve:
            return 0

        total_time = (
            self._equity_curve[-1][0] - self._equity_curve[0][0]
        ).total_seconds()

        if total_time <= 0:
            return 0

        time_in_position = sum(
            t.holding_period.total_seconds() for t in self._trades
        )

        return min(1.0, time_in_position / total_time)

    def generate_report(self) -> str:
        """Generate a text report of performance."""
        metrics = self.calculate_metrics()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              RAT BACKTEST PERFORMANCE REPORT                 ║
╠══════════════════════════════════════════════════════════════╣
║  Period: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}
║  Trading Days: {metrics.trading_days}
╠══════════════════════════════════════════════════════════════╣
║  RETURNS                                                     ║
║  ────────────────────────────────────────────────────────────║
║  Total Return:        ${metrics.total_return:>12,.2f} ({metrics.total_return_pct:>7.2%})
║  Annualized Return:   {metrics.annualized_return:>12.2%}
║  CAGR:                {metrics.cagr:>12.2%}
╠══════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                ║
║  ────────────────────────────────────────────────────────────║
║  Volatility (Daily):  {metrics.volatility:>12.4f}
║  Volatility (Annual): {metrics.annualized_volatility:>12.2%}
║  Max Drawdown:        {metrics.max_drawdown:>12.2%}
║  Max DD Duration:     {metrics.max_drawdown_duration:>12} days
║  VaR (95%):           {metrics.var_95:>12.2%}
║  CVaR (95%):          {metrics.cvar_95:>12.2%}
╠══════════════════════════════════════════════════════════════╣
║  RISK-ADJUSTED RETURNS                                       ║
║  ────────────────────────────────────────────────────────────║
║  Sharpe Ratio:        {metrics.sharpe_ratio:>12.2f}
║  Sortino Ratio:       {metrics.sortino_ratio:>12.2f}
║  Calmar Ratio:        {metrics.calmar_ratio:>12.2f}
║  Information Ratio:   {metrics.information_ratio if metrics.information_ratio else 'N/A':>12}
╠══════════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                            ║
║  ────────────────────────────────────────────────────────────║
║  Total Trades:        {metrics.total_trades:>12}
║  Winning Trades:      {metrics.winning_trades:>12}
║  Losing Trades:       {metrics.losing_trades:>12}
║  Win Rate:            {metrics.win_rate:>12.2%}
║  Profit Factor:       {metrics.profit_factor:>12.2f}
║  Average Trade:       ${metrics.average_trade:>11,.2f}
║  Average Winner:      ${metrics.average_winner:>11,.2f}
║  Average Loser:       ${metrics.average_loser:>11,.2f}
║  Largest Winner:      ${metrics.largest_winner:>11,.2f}
║  Largest Loser:       ${metrics.largest_loser:>11,.2f}
║  Avg Holding Period:  {metrics.avg_holding_period:>12.1f} days
╠══════════════════════════════════════════════════════════════╣
║  EXPOSURE                                                    ║
║  ────────────────────────────────────────────────────────────║
║  Time in Market:      {metrics.time_in_market:>12.2%}
║  Max Position Size:   ${metrics.max_position_size:>11,.2f}
╚══════════════════════════════════════════════════════════════╝
"""
        return report

    def reset(self) -> None:
        """Reset analytics for new backtest."""
        self._equity_curve.clear()
        self._daily_returns.clear()
        self._trades.clear()
        self._benchmark_returns.clear()
