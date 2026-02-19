"""
Multi-Strategy Backtest Runner

Runs all strategies simultaneously on the same historical data to
measure combined portfolio performance and diversification benefit.

Unlike the single-strategy BacktestEngine in backtest_v2, this runner:
  - Feeds data to the MultiStrategyController
  - Tracks per-strategy attribution
  - Measures diversification benefit (combined vs individual Sharpe)
"""

from __future__ import annotations

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

    # Risk limits
    MAX_POSITION_PCT = 0.20   # Max 20% of equity per symbol
    MAX_GROSS_EXPOSURE = 1.0  # Max 100% gross exposure (no leverage)

    def __init__(
        self,
        controller: MultiStrategyController,
        config: Optional[MultiStrategyBacktestConfig] = None,
    ):
        self.controller = controller
        self.config = config or MultiStrategyBacktestConfig()

        # State
        self._equity = self.config.initial_capital
        self._cash = self.config.initial_capital
        self._positions: Dict[str, float] = {}  # symbol -> shares
        self._position_prices: Dict[str, float] = {}  # symbol -> avg entry price
        self._current_prices: Dict[str, float] = {}

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
        all_bars: List[Tuple[datetime, str, Any]] = []
        for symbol, bars in data.items():
            for bar in bars:
                ts = bar.timestamp if hasattr(bar, 'timestamp') else datetime.fromtimestamp(bar.timestamp_epoch_s)
                all_bars.append((ts, symbol, bar))

        all_bars.sort(key=lambda x: x[0])

        if not all_bars:
            return self._build_results()

        if progress_callback:
            progress_callback(0.05, f"Processing {len(all_bars)} bars across {len(data)} symbols")

        # Process bars
        total = len(all_bars)
        last_day = None
        daily_equity_open = self._equity
        signalled_today = False

        for i, (ts, symbol, bar) in enumerate(all_bars):
            o = bar.open if hasattr(bar, 'open') else bar.open_price
            h = bar.high
            l = bar.low
            c = bar.close
            v = bar.volume if hasattr(bar, 'volume') else 0

            # Feed to controller
            self.controller.update(symbol, ts, o, h, l, c, v)
            self._current_prices[symbol] = c

            # Update position values
            self._update_equity()

            # Track daily boundaries
            current_day = ts.date()
            is_new_day = last_day is not None and current_day != last_day

            if is_new_day:
                # Generate signals once per day (end-of-day) to avoid
                # excessive churn from intraday re-sizing
                if not signalled_today and self._equity > 0:
                    signals = self.controller.generate_signals(
                        symbols, ts, self._equity
                    )
                    self._process_signals(signals, ts)

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
                signalled_today = False

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

    def _process_signals(
        self, signals: List[StrategySignal], timestamp: datetime
    ) -> None:
        """Process signals into simulated trades."""
        for sig in signals:
            if sig.is_exit:
                self._close_position(sig.symbol, timestamp)
            elif sig.is_entry:
                self._open_position(sig, timestamp)

            # Track per-strategy
            strategy = sig.strategy_name.split("+")[0]
            self._signals_by_strategy[strategy] = self._signals_by_strategy.get(strategy, 0) + 1

    def _open_position(self, sig: StrategySignal, timestamp: datetime) -> None:
        """Open or rebalance to a target position with risk limits."""
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

        # Update position
        new_shares = current_shares + delta_shares
        if abs(new_shares) < 0.01:
            self._positions.pop(sig.symbol, None)
            self._position_prices.pop(sig.symbol, None)
        else:
            self._positions[sig.symbol] = new_shares
            self._position_prices[sig.symbol] = exec_price

        self._trades.append({
            "timestamp": timestamp,
            "symbol": sig.symbol,
            "side": "BUY" if delta_shares > 0 else "SELL",
            "shares": delta_shares,
            "price": exec_price,
            "strategy": sig.strategy_name,
        })

    def _close_position(self, symbol: str, timestamp: datetime) -> None:
        """Close an existing position."""
        shares = self._positions.get(symbol, 0)
        if abs(shares) < 0.01:
            return

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

    def _update_equity(self) -> None:
        """Recalculate equity from cash + positions."""
        pos_value = sum(
            shares * self._current_prices.get(sym, 0)
            for sym, shares in self._positions.items()
        )
        self._equity = self._cash + pos_value

    def _build_results(self) -> MultiStrategyBacktestResults:
        """Compute final metrics and build results."""
        ec = np.array(self._equity_curve) if self._equity_curve else np.array([self.config.initial_capital])
        dr = np.array(self._daily_returns) if self._daily_returns else np.array([0.0])

        total_return = (ec[-1] / ec[0]) - 1 if ec[0] > 0 else 0
        n_years = max(len(dr) / 252, 1 / 252)
        ann_return = (1 + total_return) ** (1 / n_years) - 1

        vol = float(np.std(dr) * np.sqrt(252)) if len(dr) > 1 else 0.15
        sharpe = (ann_return - 0.02) / vol if vol > 0 else 0

        # Sortino
        downside = dr[dr < 0]
        downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 1 else vol
        sortino = (ann_return - 0.02) / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(ec)
        dd = (peak - ec) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd))

        # Win rate from closed trades
        total_trades = len(self._trades)
        win_rate = (self._winning_trades / self._closed_trades
                    if self._closed_trades > 0 else 0.0)

        # Per-strategy attribution
        attribution = {}
        for name, count in self._signals_by_strategy.items():
            attribution[name] = StrategyAttribution(
                name=name,
                n_signals=count,
            )

        return MultiStrategyBacktestResults(
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            volatility=vol,
            total_trades=total_trades,
            win_rate=win_rate,
            equity_curve=ec.tolist(),
            daily_returns=dr.tolist(),
            timestamps=self._timestamps,
            strategy_attribution=attribution,
        )
