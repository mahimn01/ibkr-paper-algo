"""
Intraday Backtester for Day Trading Algorithm v2.

Backtests the ChameleonDayTrader on 5-minute bar data.
The trader now handles all exit logic internally (ATR stops, trailing stops,
time-of-day exits, momentum reversal), so the backtester focuses on:
- Slippage modeling
- Commission costs
- Position sizing with dollar cap
- Performance metrics
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

    # v2 additions
    num_trailing_stop_exits: int = 0
    num_take_profit_exits: int = 0
    num_stop_loss_exits: int = 0
    num_reversal_exits: int = 0
    num_eod_exits: int = 0


class IntradayBacktester:
    """
    Backtest day trading strategy on intraday data.

    Simulates realistic execution with:
    - Slippage modeling
    - Commission costs
    - Position sizing with dollar cap

    The trader handles all trading logic internally (entry filters,
    ATR-based stops, trailing stops, cooldowns, time-of-day, reversals).
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
        market: str = 'NYSE',
    ) -> BacktestResult:
        """
        Run backtest on intraday bars.

        Args:
            bars: List of 5-minute bars
            symbol: Stock symbol
            aggressive: Use aggressive day trader settings
            warmup_bars: Number of bars for warmup (no trading)
            market: Market preset ('NYSE', 'HKEX', 'TSE', 'LSE', 'ASX')

        Returns:
            BacktestResult with all metrics
        """
        if len(bars) < warmup_bars + 10:
            raise ValueError(f"Need at least {warmup_bars + 10} bars, got {len(bars)}")

        # Initialize
        trader = create_daytrader(
            aggressive=aggressive,
            max_position_dollars=self.max_position_dollars,
            market=market,
        )

        capital = self.initial_capital
        equity_curve: List[float] = []
        trades: List[DayTrade] = []

        # Current position tracking (backtester's own tracking with slippage)
        position: Optional[Dict] = None

        # Regime performance tracking
        regime_trades: Dict[str, List[DayTrade]] = {}

        # Exit type counters
        trailing_exits = 0
        tp_exits = 0
        sl_exits = 0
        reversal_exits = 0
        eod_exits = 0

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
            if signal.action in ('buy', 'short'):
                # Close existing position if any (shouldn't happen often with v2)
                if position is not None:
                    trade = self._close_position(
                        position, bar.close, bar.timestamp, "Signal reversal"
                    )
                    trades.append(trade)
                    capital += trade.pnl
                    self._track_regime(regime_trades, trade)
                    reversal_exits += 1

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
                    self._track_regime(regime_trades, trade)

                    # Categorize exit type
                    reason = signal.reason.lower()
                    if 'trailing' in reason:
                        trailing_exits += 1
                    elif 'take profit' in reason:
                        tp_exits += 1
                    elif 'stop loss' in reason:
                        sl_exits += 1
                    elif 'reversal' in reason:
                        reversal_exits += 1
                    elif 'end of day' in reason:
                        eod_exits += 1

                    position = None

            # 'hold' - do nothing

        # Close any remaining position at end
        if position is not None:
            trade = self._close_position(
                position, bars[-1].close, bars[-1].timestamp, "End of backtest"
            )
            trades.append(trade)
            capital += trade.pnl
            self._track_regime(regime_trades, trade)
            eod_exits += 1

        # Calculate metrics
        result = self._calculate_results(
            symbol=symbol,
            bars=bars,
            trades=trades,
            equity_curve=equity_curve,
            regime_trades=regime_trades,
        )
        result.num_trailing_stop_exits = trailing_exits
        result.num_take_profit_exits = tp_exits
        result.num_stop_loss_exits = sl_exits
        result.num_reversal_exits = reversal_exits
        result.num_eod_exits = eod_exits

        return result

    def _track_regime(self, regime_trades: Dict[str, List[DayTrade]], trade: DayTrade):
        """Track trade under its entry regime."""
        regime_name = trade.mode_at_entry.name
        if regime_name not in regime_trades:
            regime_trades[regime_name] = []
        regime_trades[regime_name].append(trade)

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

    print("EXIT BREAKDOWN")
    print("-" * 40)
    print(f"  Take Profit:    {result.num_take_profit_exits}")
    print(f"  Trailing Stop:  {result.num_trailing_stop_exits}")
    print(f"  Stop Loss:      {result.num_stop_loss_exits}")
    print(f"  Reversal:       {result.num_reversal_exits}")
    print(f"  End of Day:     {result.num_eod_exits}")
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
    for i, trade in enumerate(result.trades[:25], 1):  # Show first 25
        direction = "LONG" if trade.direction > 0 else "SHORT"
        # Clean exit reason (remove P&L suffix that trader adds)
        reason = trade.exit_reason.split(" | P&L")[0] if " | P&L" in trade.exit_reason else trade.exit_reason
        print(f"  {i:2d}. {trade.entry_time.strftime('%m/%d %H:%M')} {direction:5s} "
              f"${trade.entry_price:.2f} -> ${trade.exit_price:.2f} | "
              f"${trade.pnl:+7.2f} ({trade.pnl_pct*100:+5.2f}%) | {trade.hold_time_minutes:3d}m | "
              f"{reason[:30]}")

    if len(result.trades) > 25:
        print(f"  ... and {len(result.trades) - 25} more trades")

    print("=" * 70)
