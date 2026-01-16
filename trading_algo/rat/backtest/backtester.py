"""
RAT Backtester: Main backtesting engine.

Features:
1. Event-driven simulation
2. Realistic fills with slippage
3. Commission modeling
4. Multiple timeframes
5. Walk-forward optimization support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from trading_algo.rat.config import RATConfig, RATBacktestConfig
from trading_algo.rat.engine import RATEngine, RATState
from trading_algo.rat.backtest.data_loader import DataLoader, Bar, CSVLoader
from trading_algo.rat.backtest.analytics import PerformanceAnalytics, Trade, PerformanceMetrics


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    # Capital
    initial_capital: float = 100000.0

    # Trading costs
    commission_per_share: float = 0.005
    commission_minimum: float = 1.0
    slippage_pct: float = 0.0005      # 5 basis points

    # Position limits
    max_position_pct: float = 0.25     # Max 25% in single position
    max_positions: int = 10

    # Execution
    fill_on_close: bool = True         # Fill at bar close vs next bar open
    allow_partial_fills: bool = False

    # Risk limits
    max_daily_loss_pct: float = 0.02   # Stop trading after 2% daily loss
    max_drawdown_pct: float = 0.15     # Stop trading at 15% drawdown

    # Data
    warmup_bars: int = 50              # Bars needed before trading


@dataclass
class BacktestResult:
    """Complete backtest result."""

    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: List[Tuple[datetime, float]]
    trades: List[Trade]
    daily_returns: List[float]
    engine_stats: Dict[str, Any]
    report: str


@dataclass
class SimulatedPosition:
    """Track a simulated position."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    side: str                       # "long" or "short"
    unrealized_pnl: float = 0.0


class SimulatedBroker:
    """Simulated broker for backtesting."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.equity = config.initial_capital
        self.positions: Dict[str, SimulatedPosition] = {}
        self.pending_orders: List[Dict] = []
        self.filled_trades: List[Trade] = []
        self.daily_pnl = 0.0

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update position values with current prices."""
        total_position_value = 0.0

        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                if pos.side == "long":
                    pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
                total_position_value += pos.quantity * current_price

        self.equity = self.cash + sum(p.unrealized_pnl for p in self.positions.values())

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> Optional[Trade]:
        """Place and fill an order immediately."""
        # Apply slippage
        if side.upper() == "BUY":
            fill_price = price * (1 + self.config.slippage_pct)
        else:
            fill_price = price * (1 - self.config.slippage_pct)

        # Calculate commission
        commission = max(
            self.config.commission_minimum,
            quantity * self.config.commission_per_share,
        )

        # Check if closing existing position
        if symbol in self.positions:
            pos = self.positions[symbol]

            if (side.upper() == "SELL" and pos.side == "long") or \
               (side.upper() == "BUY" and pos.side == "short"):
                # Closing position
                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * pos.quantity - commission
                else:
                    pnl = (pos.entry_price - fill_price) * pos.quantity - commission

                pnl_pct = pnl / (pos.entry_price * pos.quantity)

                trade = Trade(
                    symbol=symbol,
                    side=pos.side,
                    entry_time=pos.entry_time,
                    exit_time=timestamp,
                    entry_price=pos.entry_price,
                    exit_price=fill_price,
                    quantity=pos.quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )

                self.cash += pnl + pos.entry_price * pos.quantity
                self.daily_pnl += pnl
                del self.positions[symbol]
                self.filled_trades.append(trade)

                return trade

        # Opening new position
        cost = quantity * fill_price + commission

        if cost > self.cash:
            # Not enough cash
            logger.debug(f"Insufficient cash for order: need {cost}, have {self.cash}")
            return None

        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            logger.debug("Max positions reached")
            return None

        position_value = quantity * fill_price
        if position_value / self.equity > self.config.max_position_pct:
            # Reduce to max allowed
            quantity = (self.equity * self.config.max_position_pct) / fill_price

        self.cash -= cost
        self.positions[symbol] = SimulatedPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            side="long" if side.upper() == "BUY" else "short",
        )

        return None  # No trade yet, just opened position

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self.daily_pnl = 0.0

    def close_all_positions(self, prices: Dict[str, float], timestamp: datetime) -> List[Trade]:
        """Close all positions at given prices."""
        trades = []
        symbols = list(self.positions.keys())

        for symbol in symbols:
            if symbol in prices:
                pos = self.positions[symbol]
                side = "SELL" if pos.side == "long" else "BUY"
                trade = self.place_order(symbol, side, pos.quantity, prices[symbol], timestamp)
                if trade:
                    trades.append(trade)

        return trades


class RATBacktester:
    """
    Main backtesting engine for RAT framework.

    Usage:
        config = RATConfig.from_env()
        bt_config = BacktestConfig(initial_capital=100000)

        backtester = RATBacktester(config, bt_config)
        result = backtester.run(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        print(result.report)
    """

    def __init__(
        self,
        rat_config: RATConfig,
        backtest_config: Optional[BacktestConfig] = None,
        data_loader: Optional[DataLoader] = None,
    ):
        self.rat_config = rat_config
        self.bt_config = backtest_config or BacktestConfig()
        self.data_loader = data_loader or CSVLoader()

        # Initialize components
        self.broker = SimulatedBroker(self.bt_config)
        self.analytics = PerformanceAnalytics(
            initial_capital=self.bt_config.initial_capital,
        )

        # RAT engine (initialized on run)
        self.engine: Optional[RATEngine] = None

        # State
        self._current_day: Optional[datetime] = None
        self._bar_count = 0
        self._stopped = False

    def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        benchmark_symbol: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest over date range.

        Args:
            symbols: List of symbols to trade
            start_date: Start date
            end_date: End date
            benchmark_symbol: Optional benchmark for comparison
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")

        # Reset state
        self._reset()

        # Initialize engine
        self.engine = RATEngine(
            config=self.rat_config,
            broker=None,  # No live broker
            llm_client=None,  # No LLM in backtest
        )
        self.engine.reset_for_backtest()

        # Load data
        all_bars: List[Bar] = []
        for symbol in symbols:
            try:
                bars = self.data_loader.load(symbol, start_date, end_date)
                all_bars.extend(bars)
                logger.info(f"Loaded {len(bars)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")

        if not all_bars:
            raise ValueError("No data loaded for any symbol")

        # Sort by timestamp
        all_bars.sort(key=lambda b: b.timestamp)

        # Load benchmark if provided
        benchmark_bars = []
        if benchmark_symbol:
            try:
                benchmark_bars = self.data_loader.load(
                    benchmark_symbol, start_date, end_date
                )
            except Exception:
                logger.warning(f"Could not load benchmark {benchmark_symbol}")

        # Run simulation
        equity_curve = []
        current_prices: Dict[str, float] = {}

        for bar in all_bars:
            if self._stopped:
                break

            # Track current day
            bar_date = bar.timestamp.date()
            if self._current_day != bar_date:
                # New day - reset daily tracking
                if self._current_day is not None:
                    self.broker.reset_daily()
                self._current_day = bar_date

            # Update current prices
            current_prices[bar.symbol] = bar.close

            # Update broker with prices
            self.broker.update_prices(current_prices)

            # Warmup period
            self._bar_count += 1
            if self._bar_count < self.bt_config.warmup_bars:
                continue

            # Process through RAT engine
            state = self.engine.inject_backtest_tick(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )

            # Execute any signals
            if state and state.decision and state.decision.should_trade():
                self._execute_signal(state, bar)

            # Check risk limits
            self._check_risk_limits()

            # Record equity
            equity_curve.append((bar.timestamp, self.broker.equity))
            self.analytics.record_equity(bar.timestamp, self.broker.equity)

        # Close all positions at end
        if current_prices:
            final_timestamp = all_bars[-1].timestamp if all_bars else datetime.now()
            final_trades = self.broker.close_all_positions(current_prices, final_timestamp)
            for trade in final_trades:
                self.analytics.record_trade(trade)

        # Record all trades
        for trade in self.broker.filled_trades:
            self.analytics.record_trade(trade)

        # Calculate metrics
        metrics = self.analytics.calculate_metrics()
        report = self.analytics.generate_report()

        # Get engine stats
        engine_stats = self.engine.get_stats() if self.engine else {}

        return BacktestResult(
            config=self.bt_config,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=self.broker.filled_trades,
            daily_returns=self.analytics._daily_returns,
            engine_stats=engine_stats,
            report=report,
        )

    def _execute_signal(self, state: RATState, bar: Bar) -> None:
        """Execute a trading signal."""
        decision = state.decision

        if decision.action == "buy":
            # Calculate position size
            position_value = self.broker.equity * decision.position_size_pct
            quantity = position_value / bar.close

            if quantity > 0:
                self.broker.place_order(
                    symbol=bar.symbol,
                    side="BUY",
                    quantity=quantity,
                    price=bar.close,
                    timestamp=bar.timestamp,
                )

        elif decision.action == "sell":
            # Close position if we have one
            if bar.symbol in self.broker.positions:
                pos = self.broker.positions[bar.symbol]
                self.broker.place_order(
                    symbol=bar.symbol,
                    side="SELL",
                    quantity=pos.quantity,
                    price=bar.close,
                    timestamp=bar.timestamp,
                )

    def _check_risk_limits(self) -> None:
        """Check if risk limits have been breached."""
        # Daily loss limit
        daily_loss_pct = self.broker.daily_pnl / self.bt_config.initial_capital
        if daily_loss_pct < -self.bt_config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit hit: {daily_loss_pct:.2%}")
            # Don't stop completely, just log

        # Drawdown limit
        if self.broker.equity > 0:
            drawdown = (
                self.bt_config.initial_capital - self.broker.equity
            ) / self.bt_config.initial_capital

            if drawdown > self.bt_config.max_drawdown_pct:
                logger.error(f"Max drawdown exceeded: {drawdown:.2%}")
                self._stopped = True

    def _reset(self) -> None:
        """Reset backtester state."""
        self.broker = SimulatedBroker(self.bt_config)
        self.analytics.reset()
        self._current_day = None
        self._bar_count = 0
        self._stopped = False


def run_walk_forward(
    rat_config: RATConfig,
    backtest_config: BacktestConfig,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    train_period_days: int = 252,
    test_period_days: int = 63,
    data_loader: Optional[DataLoader] = None,
) -> List[BacktestResult]:
    """
    Run walk-forward optimization.

    Splits the data into train/test periods and runs backtests.

    Args:
        rat_config: RAT configuration
        backtest_config: Backtest configuration
        symbols: Symbols to trade
        start_date: Start date
        end_date: End date
        train_period_days: Training period length
        test_period_days: Testing period length
        data_loader: Data loader to use

    Returns:
        List of backtest results for each test period
    """
    results = []

    current_start = start_date + timedelta(days=train_period_days)

    while current_start < end_date:
        test_end = min(
            current_start + timedelta(days=test_period_days),
            end_date
        )

        logger.info(f"Walk-forward period: {current_start} to {test_end}")

        # Run backtest for this period
        backtester = RATBacktester(
            rat_config=rat_config,
            backtest_config=backtest_config,
            data_loader=data_loader,
        )

        try:
            result = backtester.run(
                symbols=symbols,
                start_date=current_start,
                end_date=test_end,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Walk-forward period failed: {e}")

        current_start = test_end

    return results


def aggregate_walk_forward_results(results: List[BacktestResult]) -> Dict[str, Any]:
    """Aggregate results from walk-forward optimization."""
    if not results:
        return {}

    total_return = sum(r.metrics.total_return for r in results)
    total_trades = sum(r.metrics.total_trades for r in results)
    winning_trades = sum(r.metrics.winning_trades for r in results)

    # Combine equity curves
    combined_equity = []
    for r in results:
        combined_equity.extend(r.equity_curve)

    # Average metrics
    avg_sharpe = sum(r.metrics.sharpe_ratio for r in results) / len(results)
    avg_win_rate = winning_trades / total_trades if total_trades > 0 else 0
    max_dd = max(r.metrics.max_drawdown for r in results)

    return {
        "periods": len(results),
        "total_return": total_return,
        "total_trades": total_trades,
        "avg_sharpe": avg_sharpe,
        "avg_win_rate": avg_win_rate,
        "max_drawdown": max_dd,
        "combined_equity_points": len(combined_equity),
    }
