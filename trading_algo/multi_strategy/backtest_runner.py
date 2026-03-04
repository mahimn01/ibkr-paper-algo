"""
Multi-Strategy Backtest Runner — Enterprise Edition.

Runs all strategies simultaneously on the same historical data to
measure combined portfolio performance and diversification benefit.

Enterprise features:
  1. NEXT-BAR OPEN execution: signals on bar N fill at bar N+1 OPEN
  2. VWAP entry prices on position adds (not overwrite)
  3. Per-strategy attribution tracking
  4. Diversification benefit measurement (combined vs individual Sharpe)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trading_algo.multi_strategy.controller import MultiStrategyController
from trading_algo.multi_strategy.protocol import StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class MultiStrategyBacktestConfig:
    """Configuration for multi-strategy backtesting."""
    initial_capital: float = 100_000
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    commission_per_share: float = 0.0035
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.045  # Annual risk-free rate (current ~4.5%)
    signal_interval_bars: int = 0  # 0 = daily only; 12 = ~hourly on 5-min data
    intraday_vol_threshold: float = 0.0  # Only intraday signals when ann vol > this (0=always)
    max_position_pct: float = 0.25  # Max % of equity per symbol
    max_gross_exposure: float = 1.0  # Max gross exposure
    trailing_stop_pct: float = 0.0  # 0 = disabled; e.g. 0.08 = 8% trailing stop


@dataclass
class StrategyAttribution:
    """Per-strategy performance attribution."""
    name: str
    n_signals: int = 0
    n_trades: int = 0
    gross_pnl: float = 0.0
    win_rate: float = 0.0
    avg_weight: float = 0.0


@dataclass
class MultiStrategyBacktestResults:
    """Results from a multi-strategy backtest."""
    # Portfolio-level metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0

    # Institutional-quality metrics
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy_per_trade: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown_duration_days: int = 0
    annual_turnover: float = 0.0

    # Benchmark-relative metrics
    beta: float = 0.0
    alpha_annual: float = 0.0
    information_ratio: float = 0.0
    benchmark_correlation: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Per-strategy attribution
    strategy_attribution: Dict[str, StrategyAttribution] = field(default_factory=dict)

    # Diversification metrics
    diversification_ratio: float = 0.0
    """Ratio of combined Sharpe to average individual Sharpe.
    >1.0 means diversification is adding value."""


class MultiStrategyBacktestRunner:
    """
    Run all strategies in the MultiStrategyController on historical bars.

    Usage::

        runner = MultiStrategyBacktestRunner(controller, config)
        results = runner.run(data)
    """

    def __init__(
        self,
        controller: MultiStrategyController,
        config: Optional[MultiStrategyBacktestConfig] = None,
    ):
        self.controller = controller
        self.config = config or MultiStrategyBacktestConfig()

        # Risk limits (from config)
        self.MAX_POSITION_PCT = self.config.max_position_pct
        self.MAX_GROSS_EXPOSURE = self.config.max_gross_exposure

        # State
        self._equity = self.config.initial_capital
        self._cash = self.config.initial_capital
        self._positions: Dict[str, float] = {}  # symbol -> shares
        self._position_prices: Dict[str, float] = {}  # symbol -> avg entry price (VWAP)
        self._position_costs: Dict[str, float] = {}  # symbol -> cumulative cost (for VWAP)
        self._position_sizes: Dict[str, float] = {}  # symbol -> cumulative abs shares (for VWAP)
        self._current_prices: Dict[str, float] = {}
        self._current_opens: Dict[str, float] = {}  # symbol -> current bar's open
        self._position_peaks: Dict[str, float] = {}  # symbol -> peak price since entry

        # Next-bar-open execution queue
        self._pending_signals: List[StrategySignal] = []

        # Tracking
        self._equity_curve: List[float] = [self.config.initial_capital]
        self._daily_returns: List[float] = []
        self._timestamps: List[datetime] = []
        self._trades: List[Dict] = []
        self._signals_by_strategy: Dict[str, int] = {}
        self._winning_trades: int = 0
        self._closed_trades: int = 0

    def run(
        self,
        data: Dict[str, List[Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MultiStrategyBacktestResults:
        """
        Run the multi-strategy backtest.

        Args:
            data: Dict of symbol -> list of bar objects.
                  Each bar must have: timestamp, open, high, low, close, volume
            progress_callback: Optional (pct, msg) callback for progress.

        Returns:
            MultiStrategyBacktestResults
        """
        symbols = self.config.symbols or list(data.keys())

        # Build a unified timeline from all bars
        # Each symbol's bars are already sorted by timestamp — use merge
        # NOTE: use a helper to avoid closure bug (symbol captured by reference)
        def _tagged(sym, bars):
            return ((bar.timestamp, sym, bar) for bar in bars)

        iterables = [_tagged(sym, bars) for sym, bars in data.items()]
        all_bars_iter = heapq.merge(*iterables, key=lambda x: x[0])
        all_bars = list(all_bars_iter)

        if not all_bars:
            return self._build_results()

        if progress_callback:
            progress_callback(0.05, f"Processing {len(all_bars)} bars across {len(data)} symbols")

        # Process bars
        total = len(all_bars)
        last_day = None
        daily_equity_open = self._equity
        bars_since_signal = 0
        signal_interval = self.config.signal_interval_bars
        day_count = 0

        for i, (ts, symbol, bar) in enumerate(all_bars):
            o = bar.open
            h = bar.high
            l = bar.low
            c = bar.close
            v = bar.volume

            # ── 1. EXECUTE pending signals for this symbol at OPEN ──
            self._current_opens[symbol] = o
            self._execute_pending_for_symbol(symbol, ts)

            # ── 2. Feed to controller ──
            self.controller.update(symbol, ts, o, h, l, c, v)
            self._current_prices[symbol] = c

            # Update position values (only when symbol has an open position)
            if symbol in self._positions:
                self._update_equity()

            # Track daily boundaries
            current_day = ts.date()
            is_new_day = last_day is not None and current_day != last_day

            # Intraday signal generation at configured interval
            bars_since_signal += 1
            if (
                signal_interval > 0
                and bars_since_signal >= signal_interval
                and self._equity > 0
                and not is_new_day
                and self._is_high_vol()
            ):
                signals = self.controller.generate_signals(
                    symbols, ts, self._equity
                )
                self._process_signals(signals, ts)
                self._sync_portfolio_state(daily_equity_open)
                bars_since_signal = 0

            if is_new_day:
                # Daily trailing stop checks (uses prior day close prices)
                if self.config.trailing_stop_pct > 0:
                    for sym in list(self._positions):
                        px = self._current_prices.get(sym, 0)
                        if px <= 0:
                            continue
                        shares = self._positions.get(sym, 0)
                        if shares > 0:
                            peak = self._position_peaks.get(sym, px)
                            if px > peak:
                                self._position_peaks[sym] = px
                                peak = px
                            if px <= peak * (1 - self.config.trailing_stop_pct):
                                self._close_position(sym, ts)
                        elif shares < 0:
                            trough = self._position_peaks.get(sym, px)
                            if px < trough:
                                self._position_peaks[sym] = px
                                trough = px
                            if px >= trough * (1 + self.config.trailing_stop_pct):
                                self._close_position(sym, ts)

                # Generate signals at day boundary for daily strategies
                if self._equity > 0:
                    signals = self.controller.generate_signals(
                        symbols, ts, self._equity
                    )
                    self._process_signals(signals, ts)
                    self._sync_portfolio_state(daily_equity_open)
                    bars_since_signal = 0

                # Record equity and daily return
                self._equity_curve.append(self._equity)
                self._timestamps.append(ts)

                if daily_equity_open > 0:
                    daily_ret = (self._equity / daily_equity_open) - 1
                    self._daily_returns.append(daily_ret)
                    self.controller.add_return(daily_ret)

                # Reset daily counters
                self.controller.new_trading_day()
                daily_equity_open = self._equity

                # Detect regime every 5 trading days
                day_count += 1
                if (
                    day_count % 5 == 0
                    and hasattr(self.controller, 'detect_regime')
                    and self.controller.config.enable_regime_adaptation
                ):
                    self.controller.detect_regime()

            last_day = current_day

            if progress_callback and i % (total // 20 + 1) == 0:
                progress_callback(0.05 + 0.90 * (i / total), f"Bar {i}/{total}")

        # Final day: close all open positions to realise P&L
        if all_bars:
            last_ts = all_bars[-1][0]
            for sym in list(self._positions):
                self._close_position(sym, last_ts)
            self._update_equity()
            self._equity_curve.append(self._equity)
            self._timestamps.append(last_ts)

        if progress_callback:
            progress_callback(0.95, "Computing metrics...")

        return self._build_results()

    def _is_high_vol(self) -> bool:
        """Check if recent volatility exceeds the intraday threshold."""
        threshold = self.config.intraday_vol_threshold
        if threshold <= 0:
            return True  # No vol gating — always use intraday signals
        if len(self._daily_returns) < 20:
            return False  # Not enough data — stay daily-only
        recent_vol = float(np.std(self._daily_returns[-20:]) * np.sqrt(252))
        return recent_vol >= threshold

    def _sync_portfolio_state(self, daily_equity_open: float) -> None:
        """Sync portfolio state back to controller for accurate risk checks."""
        pos_weights: Dict[str, float] = {}
        for sym, shares in self._positions.items():
            px = self._current_prices.get(sym, 0)
            if self._equity > 0 and px > 0:
                pos_weights[sym] = (shares * px) / self._equity
        daily_pnl = (
            (self._equity / daily_equity_open) - 1
            if daily_equity_open > 0 else 0.0
        )
        self.controller.update_portfolio_state(
            equity=self._equity,
            positions=pos_weights,
            daily_pnl=daily_pnl,
        )

    def _execute_pending_for_symbol(
        self, symbol: str, timestamp: datetime
    ) -> None:
        """Execute pending signals for a specific symbol at its bar's OPEN."""
        remaining = []
        for sig in self._pending_signals:
            if sig.symbol == symbol:
                if sig.is_exit:
                    self._close_position(sig.symbol, timestamp, use_open=True)
                elif sig.is_entry:
                    self._open_position(sig, timestamp, use_open=True)
            else:
                remaining.append(sig)
        self._pending_signals = remaining

    def _process_signals(
        self, signals: List[StrategySignal], timestamp: datetime
    ) -> None:
        """Queue signals for next-bar-open execution."""
        for sig in signals:
            self._pending_signals.append(sig)

            # Track per-strategy
            strategy = sig.strategy_name.split("+")[0]
            self._signals_by_strategy[strategy] = self._signals_by_strategy.get(strategy, 0) + 1

    def _open_position(
        self, sig: StrategySignal, timestamp: datetime, use_open: bool = False
    ) -> None:
        """Open or rebalance to a target position with risk limits."""
        if use_open:
            price = self._current_opens.get(sig.symbol)
        else:
            price = self._current_prices.get(sig.symbol)
        if price is None or price <= 0 or self._equity <= 0:
            return

        # Apply slippage
        slippage = price * (self.config.slippage_bps / 10000)
        exec_price = price + slippage if sig.direction > 0 else price - slippage

        # Compute desired dollar amount, capped at MAX_POSITION_PCT
        weight = min(abs(sig.target_weight), self.MAX_POSITION_PCT)
        target_dollar = self._equity * weight * sig.direction

        # Current position value for this symbol
        current_shares = self._positions.get(sig.symbol, 0.0)
        current_value = current_shares * price

        # Delta needed to reach target
        delta_dollar = target_dollar - current_value

        # Check gross exposure limit
        gross_exposure = sum(
            abs(s * self._current_prices.get(sym, 0))
            for sym, s in self._positions.items()
        )
        max_new_exposure = self._equity * self.MAX_GROSS_EXPOSURE - gross_exposure
        if abs(delta_dollar) > max_new_exposure and max_new_exposure > 0:
            delta_dollar = np.sign(delta_dollar) * max_new_exposure
        elif max_new_exposure <= 0:
            return  # At exposure limit

        delta_shares = delta_dollar / exec_price
        if abs(delta_shares) < 0.01:
            return

        # Ensure we have enough cash for buys
        cost = delta_shares * exec_price
        commission = abs(delta_shares) * self.config.commission_per_share
        if cost > 0 and cost + commission > self._cash:
            # Scale down to available cash
            max_cost = self._cash * 0.95 - commission
            if max_cost <= 0:
                return
            delta_shares = max_cost / exec_price
            cost = delta_shares * exec_price

        self._cash -= cost + commission

        # Update position with VWAP tracking
        new_shares = current_shares + delta_shares
        if abs(new_shares) < 0.01:
            self._positions.pop(sig.symbol, None)
            self._position_prices.pop(sig.symbol, None)
            self._position_costs.pop(sig.symbol, None)
            self._position_sizes.pop(sig.symbol, None)
            self._position_peaks.pop(sig.symbol, None)
        else:
            self._positions[sig.symbol] = new_shares
            # VWAP entry price tracking
            if current_shares != 0 and np.sign(delta_shares) == np.sign(current_shares):
                # Adding to position in same direction: compute VWAP
                prev_cost = self._position_costs.get(
                    sig.symbol, abs(current_shares) * self._position_prices.get(sig.symbol, exec_price)
                )
                prev_size = self._position_sizes.get(sig.symbol, abs(current_shares))
                new_cost = prev_cost + abs(delta_shares) * exec_price
                new_size = prev_size + abs(delta_shares)
                self._position_costs[sig.symbol] = new_cost
                self._position_sizes[sig.symbol] = new_size
                self._position_prices[sig.symbol] = new_cost / new_size
            else:
                # New position or reversal: reset VWAP
                self._position_prices[sig.symbol] = exec_price
                self._position_costs[sig.symbol] = abs(new_shares) * exec_price
                self._position_sizes[sig.symbol] = abs(new_shares)
            # Initialize peak tracking for new positions
            if sig.symbol not in self._position_peaks:
                self._position_peaks[sig.symbol] = exec_price

        self._trades.append({
            "timestamp": timestamp,
            "symbol": sig.symbol,
            "side": "BUY" if delta_shares > 0 else "SELL",
            "shares": delta_shares,
            "price": exec_price,
            "strategy": sig.strategy_name,
        })

    def _close_position(
        self, symbol: str, timestamp: datetime, use_open: bool = False
    ) -> None:
        """Close an existing position."""
        shares = self._positions.get(symbol, 0)
        if abs(shares) < 0.01:
            return

        if use_open:
            price = self._current_opens.get(symbol, self._current_prices.get(symbol))
        else:
            price = self._current_prices.get(symbol)
        if price is None:
            return

        slippage = price * (self.config.slippage_bps / 10000)
        exec_price = price - slippage if shares > 0 else price + slippage

        commission = abs(shares) * self.config.commission_per_share
        proceeds = shares * exec_price - commission
        self._cash += proceeds

        # Track win/loss
        entry_price = self._position_prices.get(symbol, exec_price)
        pnl = (exec_price - entry_price) * shares
        self._closed_trades += 1
        if pnl > 0:
            self._winning_trades += 1

        self._trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": "SELL" if shares > 0 else "COVER",
            "shares": -shares,
            "price": exec_price,
            "strategy": "exit",
        })

        self._positions.pop(symbol, None)
        self._position_prices.pop(symbol, None)
        self._position_costs.pop(symbol, None)
        self._position_sizes.pop(symbol, None)
        self._position_peaks.pop(symbol, None)

    def _update_equity(self) -> None:
        """Recalculate equity from cash + positions."""
        if not self._positions:
            self._equity = self._cash
            return
        pos_value = sum(
            shares * self._current_prices.get(sym, 0)
            for sym, shares in self._positions.items()
        )
        self._equity = self._cash + pos_value

    def _build_results(
        self,
        benchmark_daily_returns: Optional[np.ndarray] = None,
    ) -> MultiStrategyBacktestResults:
        """Compute final metrics using shared metrics module."""
        from backtest.metrics import compute_backtest_metrics
        return compute_backtest_metrics(
            equity_curve=self._equity_curve,
            daily_returns=self._daily_returns,
            timestamps=self._timestamps,
            trades=self._trades,
            signals_by_strategy=self._signals_by_strategy,
            closed_trades=self._closed_trades,
            winning_trades=self._winning_trades,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
            trading_days_per_year=252,
            benchmark_daily_returns=benchmark_daily_returns,
        )
