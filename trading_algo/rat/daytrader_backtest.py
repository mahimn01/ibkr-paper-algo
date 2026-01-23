"""
Intraday Backtester for Day Trading Algorithm.

Backtests the ChameleonDayTrader on 5-minute bar data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trading_algo.rat.chameleon_daytrader import (
    ChameleonDayTrader,
    DayTradeMode,
    DayTradeSignal,
    create_daytrader,
)


@dataclass
class IntradayBar:
    """5-minute OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class DayTrade:
    """Record of a completed day trade."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    hold_time_minutes: int
    exit_reason: str
    mode_at_entry: DayTradeMode


@dataclass
class BacktestResult:
    """Results from day trading backtest."""
    symbol: str
    start_time: datetime
    end_time: datetime
    num_bars: int

    # P&L
    total_pnl: float
    total_pnl_pct: float
    gross_profit: float
    gross_loss: float

    # Trade stats
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float

    # Averages
    avg_win: float
    avg_loss: float
    avg_trade: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # Trade details
    avg_hold_time_minutes: float
    longest_trade_minutes: int
    trades: List[DayTrade] = field(default_factory=list)

    # Regime breakdown
    regime_performance: Dict[str, Dict] = field(default_factory=dict)


class IntradayBacktester:
    """
    Backtest day trading strategy on intraday data.

    Simulates realistic execution with:
    - Slippage modeling
    - Commission costs
    - Position sizing with dollar cap
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_position_dollars: float = 10_000,
        slippage_pct: float = 0.0005,  # 0.05% slippage
        commission_per_share: float = 0.005,  # $0.005 per share
    ):
        self.initial_capital = initial_capital
        self.max_position_dollars = max_position_dollars
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share

    def run(
        self,
        bars: List[IntradayBar],
        symbol: str,
        aggressive: bool = True,
        warmup_bars: int = 35,
    ) -> BacktestResult:
        """
        Run backtest on intraday bars.

        Args:
            bars: List of 5-minute bars
            symbol: Stock symbol
            aggressive: Use aggressive day trader settings
            warmup_bars: Number of bars for warmup (no trading)

        Returns:
            BacktestResult with all metrics
        """
        if len(bars) < warmup_bars + 10:
            raise ValueError(f"Need at least {warmup_bars + 10} bars, got {len(bars)}")

        # Initialize
        trader = create_daytrader(
            aggressive=aggressive,
            max_position_dollars=self.max_position_dollars,
        )

        capital = self.initial_capital
        equity_curve: List[float] = []
        trades: List[DayTrade] = []

        # Current position tracking
        position: Optional[Dict] = None

        # Regime performance tracking
        regime_trades: Dict[str, List[DayTrade]] = {}

        for i, bar in enumerate(bars):
            # Update trader with bar
            signal = trader.update(
                symbol=symbol,
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

            # Track equity
            if position is not None:
                unrealized_pnl = self._calc_unrealized_pnl(position, bar.close)
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)

            # Skip warmup period
            if i < warmup_bars or signal is None:
                continue

            # Process signal
            if signal.action == 'hold':
                # Check for stop/target hits intrabar
                if position is not None:
                    exit_price, exit_reason = self._check_intrabar_exit(
                        position, bar.high, bar.low
                    )
                    if exit_price is not None:
                        trade = self._close_position(
                            position, exit_price, bar.timestamp, exit_reason
                        )
                        trades.append(trade)
                        capital += trade.pnl

                        # Track by regime
                        regime_name = trade.mode_at_entry.name
                        if regime_name not in regime_trades:
                            regime_trades[regime_name] = []
                        regime_trades[regime_name].append(trade)

                        position = None
                        trader.clear_positions()

            elif signal.action in ('buy', 'short'):
                # Close existing position if any
                if position is not None:
                    trade = self._close_position(
                        position, bar.close, bar.timestamp, "Signal reversal"
                    )
                    trades.append(trade)
                    capital += trade.pnl

                    regime_name = trade.mode_at_entry.name
                    if regime_name not in regime_trades:
                        regime_trades[regime_name] = []
                    regime_trades[regime_name].append(trade)

                # Open new position
                position = self._open_position(
                    symbol=symbol,
                    direction=1 if signal.action == 'buy' else -1,
                    price=bar.close,
                    timestamp=bar.timestamp,
                    signal=signal,
                    capital=capital,
                )

            elif signal.action in ('sell', 'cover'):
                if position is not None:
                    trade = self._close_position(
                        position, bar.close, bar.timestamp, signal.reason
                    )
                    trades.append(trade)
                    capital += trade.pnl

                    regime_name = trade.mode_at_entry.name
                    if regime_name not in regime_trades:
                        regime_trades[regime_name] = []
                    regime_trades[regime_name].append(trade)

                    position = None

        # Close any remaining position at end
        if position is not None:
            trade = self._close_position(
                position, bars[-1].close, bars[-1].timestamp, "End of backtest"
            )
            trades.append(trade)
            capital += trade.pnl

            regime_name = trade.mode_at_entry.name
            if regime_name not in regime_trades:
                regime_trades[regime_name] = []
            regime_trades[regime_name].append(trade)

        # Calculate metrics
        return self._calculate_results(
            symbol=symbol,
            bars=bars,
            trades=trades,
            equity_curve=equity_curve,
            regime_trades=regime_trades,
        )

    def _open_position(
        self,
        symbol: str,
        direction: int,
        price: float,
        timestamp: datetime,
        signal: DayTradeSignal,
        capital: float,
    ) -> Dict:
        """Open a new position."""
        # Calculate position size
        position_value = capital * signal.size
        position_value = min(position_value, self.max_position_dollars)

        # Apply slippage
        if direction > 0:
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)

        quantity = int(position_value / entry_price)
        if quantity <= 0:
            quantity = 1

        # Commission
        commission = quantity * self.commission_per_share

        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'quantity': quantity,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'mode': signal.mode,
            'commission_entry': commission,
        }

    def _close_position(
        self,
        position: Dict,
        price: float,
        timestamp: datetime,
        reason: str,
    ) -> DayTrade:
        """Close position and return trade record."""
        direction = position['direction']

        # Apply slippage
        if direction > 0:
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)

        # Commission
        commission = position['quantity'] * self.commission_per_share
        total_commission = position['commission_entry'] + commission

        # P&L
        if direction > 0:
            gross_pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            gross_pnl = (position['entry_price'] - exit_price) * position['quantity']

        net_pnl = gross_pnl - total_commission
        pnl_pct = net_pnl / (position['entry_price'] * position['quantity'])

        # Hold time
        hold_time = timestamp - position['entry_time']
        hold_minutes = int(hold_time.total_seconds() / 60)

        return DayTrade(
            symbol=position['symbol'],
            direction=direction,
            entry_time=position['entry_time'],
            entry_price=position['entry_price'],
            exit_time=timestamp,
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            hold_time_minutes=hold_minutes,
            exit_reason=reason,
            mode_at_entry=position['mode'],
        )

    def _check_intrabar_exit(
        self,
        position: Dict,
        high: float,
        low: float,
    ) -> Tuple[Optional[float], str]:
        """Check if stop or target was hit within the bar."""
        direction = position['direction']
        stop = position['stop_loss']
        target = position['take_profit']

        if direction > 0:  # Long
            if stop and low <= stop:
                return stop, "Stop loss hit"
            if target and high >= target:
                return target, "Take profit hit"
        else:  # Short
            if stop and high >= stop:
                return stop, "Stop loss hit"
            if target and low <= target:
                return target, "Take profit hit"

        return None, ""

    def _calc_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if position['direction'] > 0:
            return (current_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - current_price) * position['quantity']

    def _calculate_results(
        self,
        symbol: str,
        bars: List[IntradayBar],
        trades: List[DayTrade],
        equity_curve: List[float],
        regime_trades: Dict[str, List[DayTrade]],
    ) -> BacktestResult:
        """Calculate all backtest metrics."""

        # Basic stats
        num_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = total_pnl / self.initial_capital

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        win_rate = len(wins) / num_trades if num_trades > 0 else 0

        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        avg_trade = total_pnl / num_trades if num_trades > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        max_dd, max_dd_pct = self._calculate_drawdown(equity_curve)

        # Sharpe (annualized from 5-min returns)
        sharpe = self._calculate_sharpe(equity_curve)

        # Hold times
        if trades:
            avg_hold = sum(t.hold_time_minutes for t in trades) / len(trades)
            longest = max(t.hold_time_minutes for t in trades)
        else:
            avg_hold = 0
            longest = 0

        # Regime performance
        regime_perf = {}
        for regime, regime_trade_list in regime_trades.items():
            regime_wins = [t for t in regime_trade_list if t.pnl > 0]
            regime_pnl = sum(t.pnl for t in regime_trade_list)
            regime_perf[regime] = {
                'num_trades': len(regime_trade_list),
                'win_rate': len(regime_wins) / len(regime_trade_list) if regime_trade_list else 0,
                'total_pnl': regime_pnl,
                'avg_pnl': regime_pnl / len(regime_trade_list) if regime_trade_list else 0,
            }

        return BacktestResult(
            symbol=symbol,
            start_time=bars[0].timestamp,
            end_time=bars[-1].timestamp,
            num_bars=len(bars),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            num_trades=num_trades,
            num_wins=len(wins),
            num_losses=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            avg_hold_time_minutes=avg_hold,
            longest_trade_minutes=longest,
            trades=trades,
            regime_performance=regime_perf,
        )

    def _calculate_drawdown(self, equity_curve: List[float]) -> Tuple[float, float]:
        """Calculate max drawdown."""
        if not equity_curve:
            return 0, 0

        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    def _calculate_sharpe(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio (annualized from 5-min data)."""
        if len(equity_curve) < 2:
            return 0

        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                r = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(r)

        if not returns:
            return 0

        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std = variance ** 0.5

        if std == 0:
            return 0

        # Annualize: 78 5-min bars per day, 252 trading days
        bars_per_year = 78 * 252
        annualized_return = avg_ret * bars_per_year
        annualized_std = std * (bars_per_year ** 0.5)

        return annualized_return / annualized_std


def print_backtest_results(result: BacktestResult):
    """Pretty print backtest results."""
    print("\n" + "=" * 70)
    print(f"DAY TRADING BACKTEST RESULTS - {result.symbol}")
    print("=" * 70)
    print(f"Period: {result.start_time} to {result.end_time}")
    print(f"Bars: {result.num_bars} (5-min)")
    print()

    print("P&L SUMMARY")
    print("-" * 40)
    print(f"  Total P&L:      ${result.total_pnl:+,.2f} ({result.total_pnl_pct*100:+.2f}%)")
    print(f"  Gross Profit:   ${result.gross_profit:,.2f}")
    print(f"  Gross Loss:     ${result.gross_loss:,.2f}")
    print(f"  Profit Factor:  {result.profit_factor:.2f}")
    print()

    print("TRADE STATISTICS")
    print("-" * 40)
    print(f"  Total Trades:   {result.num_trades}")
    print(f"  Winners:        {result.num_wins} ({result.win_rate*100:.1f}%)")
    print(f"  Losers:         {result.num_losses}")
    print(f"  Avg Winner:     ${result.avg_win:+,.2f}")
    print(f"  Avg Loser:      ${result.avg_loss:,.2f}")
    print(f"  Avg Trade:      ${result.avg_trade:+,.2f}")
    print()

    print("RISK METRICS")
    print("-" * 40)
    print(f"  Max Drawdown:   ${result.max_drawdown:,.2f} ({result.max_drawdown_pct*100:.2f}%)")
    print(f"  Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print()

    print("TIMING")
    print("-" * 40)
    print(f"  Avg Hold Time:  {result.avg_hold_time_minutes:.0f} minutes")
    print(f"  Longest Trade:  {result.longest_trade_minutes} minutes")
    print()

    if result.regime_performance:
        print("PERFORMANCE BY REGIME")
        print("-" * 40)
        for regime, perf in sorted(result.regime_performance.items(), key=lambda x: -x[1]['total_pnl']):
            print(f"  {regime:25s} | {perf['num_trades']:3d} trades | "
                  f"{perf['win_rate']*100:5.1f}% win | ${perf['total_pnl']:+8.2f}")

    print()
    print("INDIVIDUAL TRADES")
    print("-" * 70)
    for i, trade in enumerate(result.trades[:20], 1):  # Show first 20
        direction = "LONG" if trade.direction > 0 else "SHORT"
        print(f"  {i:2d}. {trade.entry_time.strftime('%H:%M')} {direction:5s} "
              f"${trade.entry_price:.2f} -> ${trade.exit_price:.2f} | "
              f"${trade.pnl:+7.2f} ({trade.pnl_pct*100:+5.2f}%) | {trade.hold_time_minutes:3d}m | "
              f"{trade.exit_reason[:20]}")

    if len(result.trades) > 20:
        print(f"  ... and {len(result.trades) - 20} more trades")

    print("=" * 70)
