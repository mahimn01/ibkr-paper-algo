"""
Quantitative Trading Orchestrator

The main engine that orchestrates all components of the
quantitative trading system.

Architecture:
    1. Data Layer: TradingContext (live or backtest)
    2. Signal Layer: SignalAggregator (multi-model signals)
    3. Risk Layer: RiskController (real-time risk)
    4. Portfolio Layer: PortfolioManager (sizing and allocation)
    5. Execution Layer: ExecutionManager (optimal execution)

Flow:
    1. Fetch market data
    2. Update model states
    3. Generate aggregated signals
    4. Check risk limits
    5. Construct target portfolio
    6. Execute trades
    7. Update state and log
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import json
import os

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.engine.trading_context import (
    TradingContext, BacktestContext, LiveContext, MarketData, Position
)
from trading_algo.quant_core.engine.signal_aggregator import (
    SignalAggregator, AggregatedSignal, AggregatorConfig
)
from trading_algo.quant_core.engine.risk_controller import (
    RiskController, RiskDecision, RiskAction, RiskConfig
)
from trading_algo.quant_core.engine.portfolio_manager import (
    PortfolioManager, TargetPortfolio, PortfolioConfig
)
from trading_algo.quant_core.engine.execution_manager import (
    ExecutionManager, ExecutionResult, OrderRequest, ExecutionMethod,
    ExecutionConfig
)
from trading_algo.quant_core.execution.almgren_chriss import ExecutionUrgency
from trading_algo.quant_core.models.hmm_regime import RegimeState, MarketRegime
from trading_algo.quant_core.validation.backtest_validator import BacktestValidator


logger = logging.getLogger(__name__)


class EngineMode(Enum):
    """Engine operating mode."""
    BACKTEST = auto()
    PAPER = auto()
    LIVE = auto()


@dataclass
class EngineConfig:
    """
    Unified engine configuration.

    Combines all sub-component configurations.
    """
    # General
    mode: EngineMode = EngineMode.BACKTEST
    universe: List[str] = field(default_factory=list)
    benchmark_symbol: str = "SPY"
    bar_frequency: str = "1D"  # 1D, 1H, 15m, etc.

    # Sub-configs
    signal_config: Optional[AggregatorConfig] = None
    risk_config: Optional[RiskConfig] = None
    portfolio_config: Optional[PortfolioConfig] = None
    execution_config: Optional[ExecutionConfig] = None

    # Timing
    warmup_bars: int = 60                # Bars needed before trading
    rebalance_frequency: str = "daily"   # daily, weekly, monthly

    # Logging
    log_level: str = "INFO"
    save_trades: bool = True
    save_signals: bool = True

    def __post_init__(self):
        if self.signal_config is None:
            self.signal_config = AggregatorConfig()
        if self.risk_config is None:
            self.risk_config = RiskConfig()
        if self.portfolio_config is None:
            self.portfolio_config = PortfolioConfig()
        if self.execution_config is None:
            self.execution_config = ExecutionConfig()


@dataclass
class EngineState:
    """Current engine state."""
    bar_count: int = 0
    current_time: Optional[datetime] = None
    current_regime: MarketRegime = MarketRegime.UNKNOWN
    is_trading: bool = False
    last_rebalance: Optional[datetime] = None

    # Performance
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0

    # Recent data
    recent_signals: Dict[str, float] = field(default_factory=dict)
    recent_positions: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trade stats
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float

    # Risk stats
    volatility: float
    var_95: float
    avg_exposure: float

    # Data
    equity_curve: NDArray[np.float64]
    returns: NDArray[np.float64]
    trades: List[Dict]
    daily_stats: List[Dict]

    # Validation
    validation_passed: bool
    pbo: float
    deflated_sharpe: float


class QuantOrchestrator:
    """
    Main quantitative trading orchestrator.

    Coordinates all components into a unified trading system
    that works for both backtesting and live trading.

    Usage:
        # Initialize
        config = EngineConfig(
            mode=EngineMode.BACKTEST,
            universe=['AAPL', 'MSFT', 'GOOGL'],
        )
        engine = QuantOrchestrator(config)

        # Backtest
        result = engine.run_backtest(historical_data)

        # Live trading
        engine.run_live(ibkr_client)
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize orchestrator.

        Args:
            config: Engine configuration
        """
        self.config = config
        self.state = EngineState()

        # Initialize components
        self.signal_aggregator = SignalAggregator(config.signal_config)
        self.risk_controller = RiskController(config.risk_config)
        self.portfolio_manager = PortfolioManager(config.portfolio_config)
        self.execution_manager = ExecutionManager(config.execution_config)

        # Context (set when running)
        self.context: Optional[TradingContext] = None

        # Data storage
        self._price_history: Dict[str, List[float]] = {}
        self._returns_history: Dict[str, List[float]] = {}
        self._signal_log: List[Dict] = []
        self._trade_log: List[Dict] = []
        self._daily_stats: List[Dict] = []

        # Callbacks
        self._on_bar_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []

        # Initialize signal aggregator for universe
        if config.universe:
            self.signal_aggregator.initialize(config.universe)

        logger.info(f"Initialized QuantOrchestrator in {config.mode.name} mode")

    def run_backtest(
        self,
        historical_data: Dict[str, NDArray[np.float64]],
        timestamps: NDArray,
        initial_capital: float = 100000.0,
        validate: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            historical_data: Dict of symbol -> OHLCV arrays
            timestamps: Timestamp array
            initial_capital: Starting capital
            validate: Whether to run validation

        Returns:
            BacktestResult with comprehensive results
        """
        logger.info(f"Starting backtest with {len(timestamps)} bars")

        # Create backtest context
        self.context = BacktestContext(
            historical_data=historical_data,
            timestamps=timestamps,
            initial_capital=initial_capital,
        )
        self.execution_manager.set_context(self.context)

        # Reset state
        self._reset_state()
        self.state.is_trading = True
        self.risk_controller.reset_peak(initial_capital)

        # Main loop
        while self.context.advance():
            try:
                self._process_bar()
            except Exception as e:
                logger.error(f"Error processing bar {self.state.bar_count}: {e}")
                continue

        # Get backtest results from context
        context_results = self.context.get_results()

        # Calculate additional metrics
        returns = context_results['returns']
        equity_curve = context_results['equity_curve']

        # Trade statistics
        trades = context_results['trades']
        winning = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning) / len(trades) if trades else 0.0

        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Run validation if requested
        validation_passed = True
        pbo = 0.0
        deflated_sharpe = context_results['sharpe_ratio']

        if validate and len(returns) > 252:
            try:
                validator = BacktestValidator()
                validation = validator.validate(
                    returns=returns,
                    n_trials=1,  # Single strategy
                )
                validation_passed = validation.status.name != "FAILED"
                pbo = validation.overfitting_metrics.pbo
                deflated_sharpe = validation.overfitting_metrics.deflated_sharpe
            except Exception as e:
                logger.warning(f"Validation failed: {e}")

        result = BacktestResult(
            total_return=context_results['total_return'],
            annualized_return=context_results['total_return'] * 252 / len(returns) if len(returns) > 0 else 0,
            sharpe_ratio=context_results['sharpe_ratio'],
            sortino_ratio=context_results['sortino_ratio'],
            calmar_ratio=context_results['calmar_ratio'],
            max_drawdown=context_results['max_drawdown'],
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=np.mean([t.get('pnl', 0) for t in trades]) if trades else 0.0,
            volatility=float(np.std(returns) * SQRT_252) if len(returns) > 1 else 0.0,
            var_95=float(np.percentile(returns, 5)) if len(returns) > 20 else 0.0,
            avg_exposure=np.mean([s.get('gross_exposure', 0) for s in self._daily_stats]) if self._daily_stats else 0.0,
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            daily_stats=self._daily_stats,
            validation_passed=validation_passed,
            pbo=pbo,
            deflated_sharpe=deflated_sharpe,
        )

        logger.info(f"Backtest complete: Sharpe={result.sharpe_ratio:.2f}, "
                    f"Return={result.total_return:.2%}, MaxDD={result.max_drawdown:.2%}")

        return result

    def run_live(
        self,
        ibkr_broker,
        data_provider,
        run_duration_minutes: Optional[int] = None,
    ) -> None:
        """
        Run live trading.

        Args:
            ibkr_broker: IBKR broker interface
            data_provider: Market data provider
            run_duration_minutes: Optional time limit
        """
        logger.info("Starting live trading")

        # Create live context
        self.context = LiveContext(
            ibkr_broker=ibkr_broker,
            data_provider=data_provider,
            universe=self.config.universe,
        )
        self.execution_manager.set_context(self.context)

        # Get initial account info
        account = self.context.get_account_info()
        self.risk_controller.reset_peak(account.equity)

        self._reset_state()
        self.state.is_trading = True

        start_time = datetime.now()

        # Main loop
        while self.state.is_trading:
            try:
                # Check if market is open
                if not self.context.is_market_open():
                    logger.debug("Market closed, waiting...")
                    if not self.context.advance():
                        break
                    continue

                # Check time limit
                if run_duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= run_duration_minutes:
                        logger.info("Time limit reached, stopping")
                        break

                # Process bar
                self._process_bar()

                # Wait for next bar
                if not self.context.advance():
                    break

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down")
                break
            except Exception as e:
                logger.error(f"Error in live trading: {e}")
                continue

        # Cleanup
        self._shutdown()

    def _process_bar(self) -> None:
        """Process a single bar of data."""
        self.state.bar_count += 1
        self.state.current_time = self.context.get_current_time()

        # Skip if in warmup
        if self.state.bar_count < self.config.warmup_bars:
            self._update_history()
            return

        # 1. Update history and get market data
        self._update_history()
        market_data = self._get_market_data()

        if not market_data:
            return

        # 2. Update regime detection (using benchmark)
        benchmark_prices = self._get_price_array(self.config.benchmark_symbol)
        if benchmark_prices is not None and len(benchmark_prices) > 60:
            self.state.current_regime = self.signal_aggregator.update_regime(
                benchmark_prices
            )

        # 3. Generate signals for each symbol
        signals: Dict[str, AggregatedSignal] = {}
        for symbol in self.config.universe:
            prices = self._get_price_array(symbol)
            if prices is None or len(prices) < 60:
                continue

            signal = self.signal_aggregator.generate_signal(symbol, prices)
            signals[symbol] = signal
            self.state.recent_signals[symbol] = signal.signal

        if not signals:
            return

        # Log signals
        if self.config.save_signals:
            self._log_signals(signals)

        # 4. Get account info and check risk
        account = self.context.get_account_info()
        positions = {s: p.quantity for s, p in account.positions.items()}
        position_values = {s: p.market_value for s, p in account.positions.items()}

        returns = self._get_portfolio_returns()
        risk_decision = self.risk_controller.evaluate(
            equity=account.equity,
            positions=position_values,
            returns=returns,
        )

        # 5. Check if we should rebalance
        if not self._should_rebalance(risk_decision):
            return

        # 6. Construct target portfolio
        current_prices = {s: market_data[s].close for s in market_data}
        returns_matrix = self._get_returns_matrix()

        target = self.portfolio_manager.construct_portfolio(
            signals=signals,
            equity=account.equity,
            current_positions=positions,
            current_prices=current_prices,
            risk_decision=risk_decision,
            returns_matrix=returns_matrix,
        )

        # 7. Execute trades
        self._execute_portfolio(target, current_prices, risk_decision)

        # 8. Log daily stats
        self._log_daily_stats(account, risk_decision, target)

        # Update state
        self.state.last_rebalance = self.state.current_time

        # Call callbacks
        for callback in self._on_bar_callbacks:
            try:
                callback(self.state, signals, target)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _update_history(self) -> None:
        """Update price and returns history."""
        for symbol in self.config.universe:
            market_data = self.context.get_market_data(symbol)
            if market_data is None:
                continue

            if symbol not in self._price_history:
                self._price_history[symbol] = []
                self._returns_history[symbol] = []

            prices = self._price_history[symbol]
            prices.append(market_data.close)

            # Keep limited history
            if len(prices) > 504:
                self._price_history[symbol] = prices[-504:]

            # Calculate return
            if len(prices) >= 2:
                ret = (prices[-1] / prices[-2]) - 1
                self._returns_history[symbol].append(ret)

                if len(self._returns_history[symbol]) > 504:
                    self._returns_history[symbol] = self._returns_history[symbol][-504:]

    def _get_market_data(self) -> Dict[str, MarketData]:
        """Get current market data for all symbols."""
        data = {}
        for symbol in self.config.universe:
            md = self.context.get_market_data(symbol)
            if md:
                data[symbol] = md
        return data

    def _get_price_array(self, symbol: str) -> Optional[NDArray[np.float64]]:
        """Get price array for a symbol."""
        if symbol in self._price_history:
            return np.array(self._price_history[symbol])
        return None

    def _get_portfolio_returns(self) -> NDArray[np.float64]:
        """Get portfolio returns (equal-weighted proxy)."""
        if not self._returns_history:
            return np.array([0.0])

        # Equal-weight returns across universe
        all_returns = []
        for symbol in self.config.universe:
            if symbol in self._returns_history and self._returns_history[symbol]:
                all_returns.append(self._returns_history[symbol][-1])

        if all_returns:
            return np.array([np.mean(all_returns)])
        return np.array([0.0])

    def _get_returns_matrix(self) -> Optional[NDArray[np.float64]]:
        """Get returns matrix for portfolio optimization."""
        symbols = [s for s in self.config.universe if s in self._returns_history]
        if not symbols:
            return None

        min_len = min(len(self._returns_history[s]) for s in symbols)
        if min_len < 60:
            return None

        matrix = np.column_stack([
            self._returns_history[s][-min_len:]
            for s in symbols
        ])
        return matrix

    def _should_rebalance(self, risk_decision: RiskDecision) -> bool:
        """Check if we should rebalance."""
        # Always rebalance if risk requires it
        if risk_decision.must_reduce:
            return True

        # Check frequency
        if self.state.last_rebalance is None:
            return True

        freq = self.config.rebalance_frequency
        elapsed = self.state.current_time - self.state.last_rebalance

        if freq == "daily":
            return elapsed >= timedelta(days=1)
        elif freq == "weekly":
            return elapsed >= timedelta(weeks=1)
        elif freq == "monthly":
            return elapsed >= timedelta(days=30)

        return True

    def _execute_portfolio(
        self,
        target: TargetPortfolio,
        current_prices: Dict[str, float],
        risk_decision: RiskDecision,
    ) -> None:
        """Execute trades to reach target portfolio."""
        # Determine urgency based on risk
        if risk_decision.action == RiskAction.LIQUIDATE:
            urgency = ExecutionUrgency.IMMEDIATE
        elif risk_decision.action == RiskAction.REDUCE_POSITIONS:
            urgency = ExecutionUrgency.AGGRESSIVE
        elif risk_decision.action == RiskAction.HALT_ENTRIES:
            urgency = ExecutionUrgency.PASSIVE
        else:
            urgency = ExecutionUrgency.NEUTRAL

        for symbol, pos in target.positions.items():
            if abs(pos.delta_shares) < 1:
                continue

            # Check minimum trade value
            price = current_prices.get(symbol, 0)
            trade_value = abs(pos.delta_shares) * price
            if trade_value < self.config.portfolio_config.min_trade_value:
                continue

            # Create order request
            request = OrderRequest(
                symbol=symbol,
                shares=pos.delta_shares,
                urgency=urgency,
                method=self.config.execution_config.default_method,
            )

            # Execute
            result = self.execution_manager.execute(request)

            if result.success:
                self.state.total_trades += 1

                # Log trade
                if self.config.save_trades:
                    self._trade_log.append({
                        'timestamp': str(self.state.current_time),
                        'symbol': symbol,
                        'shares': pos.delta_shares,
                        'price': result.avg_fill_price,
                        'slippage_bps': result.slippage_bps,
                        'signal': pos.signal_strength,
                    })

                # Call trade callbacks
                for callback in self._on_trade_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.warning(f"Trade callback error: {e}")

    def _log_signals(self, signals: Dict[str, AggregatedSignal]) -> None:
        """Log current signals."""
        entry = {
            'timestamp': str(self.state.current_time),
            'regime': self.state.current_regime.name,
            'signals': {},
        }

        for symbol, signal in signals.items():
            entry['signals'][symbol] = {
                'signal': signal.signal,
                'confidence': signal.confidence,
                'direction': signal.direction,
            }

        self._signal_log.append(entry)

    def _log_daily_stats(
        self,
        account,
        risk_decision: RiskDecision,
        target: TargetPortfolio,
    ) -> None:
        """Log daily statistics."""
        self._daily_stats.append({
            'timestamp': str(self.state.current_time),
            'equity': account.equity,
            'cash': account.cash,
            'gross_exposure': target.gross_exposure,
            'net_exposure': target.net_exposure,
            'n_positions': len([p for p in target.positions.values()
                               if abs(p.target_shares) > 0]),
            'risk_action': risk_decision.action.name,
            'regime': self.state.current_regime.name,
        })

    def _reset_state(self) -> None:
        """Reset engine state."""
        self.state = EngineState()
        self._price_history.clear()
        self._returns_history.clear()
        self._signal_log.clear()
        self._trade_log.clear()
        self._daily_stats.clear()
        self.risk_controller.reset()
        self.signal_aggregator.reset()

    def _shutdown(self) -> None:
        """Shutdown engine gracefully."""
        logger.info("Shutting down engine")
        self.state.is_trading = False

        # Cancel any pending orders
        if self.context:
            self.execution_manager.cancel_all_pending()

    def add_bar_callback(self, callback: Callable) -> None:
        """Add callback for each bar processed."""
        self._on_bar_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for each trade executed."""
        self._on_trade_callbacks.append(callback)

    def save_results(self, path: str) -> None:
        """Save results to files."""
        os.makedirs(path, exist_ok=True)

        # Save signals
        if self._signal_log:
            with open(os.path.join(path, 'signals.json'), 'w') as f:
                json.dump(self._signal_log, f, indent=2)

        # Save trades
        if self._trade_log:
            with open(os.path.join(path, 'trades.json'), 'w') as f:
                json.dump(self._trade_log, f, indent=2)

        # Save daily stats
        if self._daily_stats:
            with open(os.path.join(path, 'daily_stats.json'), 'w') as f:
                json.dump(self._daily_stats, f, indent=2)

        logger.info(f"Results saved to {path}")
