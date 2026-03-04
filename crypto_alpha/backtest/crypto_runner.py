"""
Crypto-specific backtest runner — Enterprise Edition.

Fixes from v1:
  1. NEXT-BAR OPEN execution: signals on bar N fill at bar N+1 OPEN
  2. NO max_sizing_equity cap: equity compounds naturally, returns are stationary
  3. BACKWARD-ONLY data lookups: no forward-looking funding/spot/OI
  4. VWAP entry prices on position adds (not overwrite)
  5. Proper short margin accounting on adds
  6. Settlement-aligned funding at 00:00/08:00/16:00 UTC
  7. Correct Sharpe on naturally compounded equity series

Design philosophy:
  - Position sizes proportional to CURRENT equity (standard compounding)
  - Returns = equity_t / equity_{t-1} - 1 (stationary under compounding)
  - No artificial caps that create non-stationary returns
  - Signals ALWAYS execute at the NEXT bar's open, never same-bar close
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
from trading_algo.multi_strategy.backtest_runner import MultiStrategyBacktestResults

logger = logging.getLogger(__name__)

# Binance funding settlement hours (UTC)
FUNDING_HOURS = {0, 8, 16}


@dataclass
class CryptoBacktestConfig:
    """Configuration for crypto backtesting."""
    initial_capital: float = 30_000.0
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Commission model (basis points)
    commission_bps_maker: float = 2.0
    commission_bps_taker: float = 5.0
    slippage_bps: float = 5.0

    # Funding
    funding_interval_hours: int = 8

    # Leverage
    max_leverage: float = 5.0
    maintenance_margin_ratio: float = 0.03  # 3%

    # Risk
    risk_free_rate: float = 0.045
    max_position_pct: float = 0.30  # Max 30% of equity per symbol
    max_gross_exposure: float = 3.0

    # Signal generation
    signal_interval_bars: int = 60
    intraday_vol_threshold: float = 0.0

    # Deprecated — kept for API compat but ignored
    max_sizing_equity: float = 0.0


@dataclass
class CryptoPosition:
    """Track a single crypto position."""
    symbol: str
    size: float             # Positive = long, negative = short
    entry_price: float      # VWAP of all fills
    total_cost: float       # Cumulative cost basis (for VWAP)
    total_size: float       # Cumulative abs size filled (for VWAP)
    leverage: float = 1.0
    margin_used: float = 0.0
    position_type: str = "perp"
    unrealized_pnl: float = 0.0
    funding_pnl: float = 0.0
    entry_time: Optional[datetime] = None


class CryptoBacktestRunner:
    """
    Enterprise-grade crypto backtest with:
      - Next-bar-open execution (no look-ahead)
      - Natural equity compounding (stationary returns)
      - Settlement-aligned funding
      - VWAP position tracking
    """

    def __init__(
        self,
        controller: MultiStrategyController,
        config: Optional[CryptoBacktestConfig] = None,
    ):
        self.controller = controller
        self.config = config or CryptoBacktestConfig()

        # Initialize controller equity to match
        self.controller._equity = self.config.initial_capital
        self.controller._peak_equity = self.config.initial_capital

        # State
        self._equity = self.config.initial_capital
        self._cash = self.config.initial_capital
        self._positions: Dict[str, CryptoPosition] = {}
        self._current_prices: Dict[str, float] = {}
        self._current_opens: Dict[str, float] = {}  # Track open prices for fills

        # Pending signals (generated on bar N, executed on bar N+1)
        self._pending_signals: List[StrategySignal] = []

        # Tracking
        self._equity_curve: List[float] = [self.config.initial_capital]
        self._daily_returns: List[float] = []
        self._timestamps: List[datetime] = []
        self._trades: List[Dict] = []
        self._signals_by_strategy: Dict[str, int] = {}
        self._winning_trades: int = 0
        self._closed_trades: int = 0

        # Funding tracking
        self._funding_rates: Dict[str, float] = {}
        self._last_funding_hour: Optional[int] = None
        self._total_funding_pnl: float = 0.0

        # Liquidation tracking
        self._liquidations: int = 0

    def run(
        self,
        data: Dict[str, List[Any]],
        funding_data: Optional[Dict[str, List[Tuple[datetime, float]]]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MultiStrategyBacktestResults:
        """Run the crypto backtest."""
        symbols = self.config.symbols or list(data.keys())

        # Pre-process funding rates (backward-only lookup)
        funding_lookup: Dict[str, Dict[int, float]] = {}
        if funding_data:
            for sym, entries in funding_data.items():
                funding_lookup[sym] = {
                    int(ts.timestamp()): rate for ts, rate in entries
                }

        # Build merged timeline
        def _tagged(sym, bars):
            return ((bar.timestamp, sym, bar) for bar in bars)

        iterables = [_tagged(sym, bars) for sym, bars in data.items()
                     if sym in symbols]
        all_bars = list(heapq.merge(*iterables, key=lambda x: x[0]))

        if not all_bars:
            return self._build_results()

        if progress_callback:
            progress_callback(0.05, f"Processing {len(all_bars)} bars across {len(data)} symbols")

        total = len(all_bars)
        last_day = None
        daily_equity_open = self._equity
        bars_since_signal = 0
        signal_interval = self.config.signal_interval_bars

        from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
        crypto_adapters = [
            s for s in self.controller._strategies.values()
            if isinstance(s, CryptoEdgeAdapter)
        ]

        for i, (ts, symbol, bar) in enumerate(all_bars):
            o = bar.open
            h = bar.high
            l = bar.low
            c = bar.close
            v = bar.volume

            # ── 1. EXECUTE PENDING SIGNALS at this bar's OPEN ──
            # Signals were generated on a PREVIOUS bar. Fill at this bar's open.
            self._current_opens[symbol] = o
            if self._pending_signals:
                self._execute_pending_signals(ts)

            # ── 2. Feed crypto data (BACKWARD-ONLY lookup for funding) ──
            funding_rate = getattr(bar, 'funding_rate', None)
            spot_price = getattr(bar, 'spot_price', None)
            oi = getattr(bar, 'open_interest', None)

            if funding_rate is not None:
                self._funding_rates[symbol] = funding_rate

            # Backward-only funding lookup from external data
            if funding_lookup:
                ts_epoch = int(ts.timestamp())
                sym_lookup = funding_lookup.get(symbol, {})
                # ONLY search backward in time (no future data)
                for offset in range(0, 301, 60):
                    if ts_epoch - offset in sym_lookup:
                        self._funding_rates[symbol] = sym_lookup[ts_epoch - offset]
                        break

            for adapter in crypto_adapters:
                adapter.update_crypto_data(
                    symbol,
                    funding_rate=self._funding_rates.get(symbol),
                    spot_price=spot_price,
                    open_interest=oi,
                )

            # ── 3. Feed OHLCV to controller/edges ──
            self.controller.update(symbol, ts, o, h, l, c, v)
            self._current_prices[symbol] = c

            # ── 4. Update positions, funding, liquidations ──
            self._update_positions(symbol, c)
            self._maybe_apply_funding(ts)
            self._check_liquidations(ts)

            # ── 5. Generate signals (queued for NEXT bar's open) ──
            current_day = ts.date()
            is_new_day = last_day is not None and current_day != last_day

            bars_since_signal += 1
            if (
                signal_interval > 0
                and bars_since_signal >= signal_interval
                and self._equity > 0
                and not is_new_day
            ):
                signals = self.controller.generate_signals(
                    symbols, ts, self._equity
                )
                self._process_signals(signals, ts)
                bars_since_signal = 0

            if is_new_day:
                if self._equity > 0:
                    signals = self.controller.generate_signals(
                        symbols, ts, self._equity
                    )
                    self._process_signals(signals, ts)
                    bars_since_signal = 0

                # Record daily metrics
                self._equity_curve.append(self._equity)
                self._timestamps.append(ts)

                if daily_equity_open > 0:
                    daily_ret = (self._equity / daily_equity_open) - 1
                    self._daily_returns.append(daily_ret)
                    self.controller.add_return(daily_ret)

                self.controller.new_trading_day()
                daily_equity_open = self._equity

            last_day = current_day

            if progress_callback and i % (total // 20 + 1) == 0:
                progress_callback(0.05 + 0.90 * (i / total), f"Bar {i}/{total}")

        # Close all positions at end
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

    def _execute_pending_signals(self, timestamp: datetime) -> None:
        """Execute all pending signals at current bar's OPEN prices."""
        to_execute = list(self._pending_signals)
        self._pending_signals.clear()

        for sig in to_execute:
            if sig.is_exit:
                self._close_position(sig.symbol, timestamp, use_open=True)
            elif sig.is_entry:
                self._open_position(sig, timestamp, use_open=True)

    def _maybe_apply_funding(self, timestamp: datetime) -> None:
        """Apply funding at settlement times: 00:00, 08:00, 16:00 UTC."""
        current_hour = timestamp.hour
        # Only apply at settlement hours, and only once per settlement
        if current_hour not in FUNDING_HOURS:
            return
        if self._last_funding_hour == (timestamp.date(), current_hour):
            return

        self._last_funding_hour = (timestamp.date(), current_hour)

        for sym, pos in self._positions.items():
            if pos.position_type != "perp":
                continue

            rate = self._funding_rates.get(sym, 0.0)
            if rate == 0:
                continue

            notional = abs(pos.size * self._current_prices.get(sym, pos.entry_price))

            if pos.size > 0:
                funding_pnl = -notional * rate
            else:
                funding_pnl = notional * rate

            pos.funding_pnl += funding_pnl
            self._cash += funding_pnl
            self._total_funding_pnl += funding_pnl

        self._update_equity()

    def _check_liquidations(self, timestamp: datetime) -> None:
        """Check for liquidations on leveraged positions."""
        for sym in list(self._positions):
            pos = self._positions[sym]
            if pos.leverage <= 1.0:
                continue

            price = self._current_prices.get(sym, pos.entry_price)
            pnl_pct = (price / pos.entry_price - 1) * (1 if pos.size > 0 else -1)

            liq_threshold = -(1.0 / pos.leverage - self.config.maintenance_margin_ratio)
            if pnl_pct <= liq_threshold:
                logger.warning(
                    f"LIQUIDATION: {sym} at {price:.2f} "
                    f"(entry {pos.entry_price:.2f}, leverage {pos.leverage:.1f}x)"
                )
                self._close_position(sym, timestamp, is_liquidation=True)
                self._liquidations += 1

    def _update_positions(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L."""
        if symbol not in self._positions:
            return
        pos = self._positions[symbol]
        pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
        self._update_equity()

    def _update_equity(self) -> None:
        """Equity = cash + sum(unrealized PnL)."""
        unrealized = sum(
            (self._current_prices.get(sym, p.entry_price) - p.entry_price) * p.size
            for sym, p in self._positions.items()
        )
        self._equity = self._cash + unrealized

    def _process_signals(
        self, signals: List[StrategySignal], timestamp: datetime
    ) -> None:
        """Queue signals for next-bar execution (public API for subclasses)."""
        self._pending_signals.extend(signals)
        for sig in signals:
            strategy = sig.strategy_name.split("+")[0]
            self._signals_by_strategy[strategy] = (
                self._signals_by_strategy.get(strategy, 0) + 1
            )

    def _open_position(
        self, sig: StrategySignal, timestamp: datetime, use_open: bool = False
    ) -> None:
        """Open or rebalance a position."""
        if use_open:
            price = self._current_opens.get(sig.symbol)
        else:
            price = self._current_prices.get(sig.symbol)

        if price is None or price <= 0 or self._equity <= 0:
            return

        # Apply slippage to fill price
        slippage = price * (self.config.slippage_bps / 10000)
        exec_price = price + slippage if sig.direction > 0 else price - slippage

        # Position sizing: use current equity (natural compounding)
        sizing_equity = self._equity
        weight = min(abs(sig.target_weight), self.config.max_position_pct)
        target_notional = sizing_equity * weight * sig.direction

        leverage = sig.metadata.get("leverage", 1.0)
        leverage = min(leverage, self.config.max_leverage)

        # Current position
        current_pos = self._positions.get(sig.symbol)
        current_notional = 0.0
        if current_pos:
            current_notional = current_pos.size * price

        delta_notional = target_notional - current_notional

        # Gross exposure limit
        gross_exposure = sum(
            abs(p.size * self._current_prices.get(s, p.entry_price))
            for s, p in self._positions.items()
        )
        max_new = self._equity * self.config.max_gross_exposure - gross_exposure
        if abs(delta_notional) > max_new > 0:
            delta_notional = np.sign(delta_notional) * max_new
        elif max_new <= 0:
            return

        delta_size = delta_notional / exec_price
        if abs(delta_size) < 1e-8:
            return

        commission = abs(delta_notional) * (self.config.commission_bps_taker / 10000)
        margin_change = abs(delta_notional) / leverage if leverage > 1 else abs(delta_notional)

        # Cash check for margin + commission (both longs AND shorts)
        needed = margin_change + commission
        if needed > self._cash * 0.95:
            available = self._cash * 0.95 - commission
            if available <= 0:
                return
            margin_change = available
            delta_notional = margin_change * leverage * np.sign(delta_notional)
            delta_size = delta_notional / exec_price
            commission = abs(delta_notional) * (self.config.commission_bps_taker / 10000)

        # Deduct margin + commission
        self._cash -= margin_change + commission

        # Update or create position with VWAP tracking
        if current_pos:
            old_size = current_pos.size
            new_size = old_size + delta_size
            if abs(new_size) < 1e-8:
                self._positions.pop(sig.symbol, None)
            else:
                # VWAP entry price on adds in same direction
                if np.sign(delta_size) == np.sign(old_size):
                    # Adding to position: compute VWAP
                    current_pos.total_cost += abs(delta_size) * exec_price
                    current_pos.total_size += abs(delta_size)
                    current_pos.entry_price = current_pos.total_cost / current_pos.total_size
                else:
                    # Partial close or reversal
                    if abs(new_size) < abs(old_size):
                        # Partial close: keep original entry
                        pass
                    else:
                        # Reversal: new entry at exec_price
                        current_pos.entry_price = exec_price
                        current_pos.total_cost = abs(new_size) * exec_price
                        current_pos.total_size = abs(new_size)
                current_pos.size = new_size
                current_pos.leverage = leverage
                current_pos.margin_used += margin_change
        else:
            self._positions[sig.symbol] = CryptoPosition(
                symbol=sig.symbol,
                size=delta_size,
                entry_price=exec_price,
                total_cost=abs(delta_size) * exec_price,
                total_size=abs(delta_size),
                leverage=leverage,
                margin_used=margin_change,
                position_type=sig.metadata.get("position_type", "perp"),
                entry_time=timestamp,
            )

        self._trades.append({
            "timestamp": timestamp,
            "symbol": sig.symbol,
            "side": "BUY" if delta_size > 0 else "SELL",
            "size": delta_size,
            "price": exec_price,
            "leverage": leverage,
            "commission": commission,
            "strategy": sig.strategy_name,
        })

        self._update_equity()

    def _close_position(
        self, symbol: str, timestamp: datetime,
        is_liquidation: bool = False, use_open: bool = False,
    ) -> None:
        """Close a position."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        if use_open:
            price = self._current_opens.get(symbol, self._current_prices.get(symbol, pos.entry_price))
        else:
            price = self._current_prices.get(symbol, pos.entry_price)

        slippage = price * (self.config.slippage_bps / 10000)
        exec_price = price - slippage if pos.size > 0 else price + slippage

        pnl = (exec_price - pos.entry_price) * pos.size

        notional = abs(pos.size * exec_price)
        commission = notional * (self.config.commission_bps_taker / 10000)
        liq_penalty = notional * 0.005 if is_liquidation else 0.0

        self._cash += pos.margin_used + pnl - commission - liq_penalty

        self._closed_trades += 1
        total_pnl = pnl + pos.funding_pnl
        if total_pnl > 0:
            self._winning_trades += 1

        self._trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": "CLOSE_LONG" if pos.size > 0 else "CLOSE_SHORT",
            "size": -pos.size,
            "price": exec_price,
            "pnl": pnl,
            "funding_pnl": pos.funding_pnl,
            "commission": commission,
            "is_liquidation": is_liquidation,
            "strategy": "exit",
        })

        self._positions.pop(symbol, None)
        self._update_equity()

    def _build_results(self) -> MultiStrategyBacktestResults:
        """Compute final metrics using shared metrics module."""
        from backtest.metrics import compute_backtest_metrics
        results = compute_backtest_metrics(
            equity_curve=self._equity_curve,
            daily_returns=self._daily_returns,
            timestamps=self._timestamps,
            trades=self._trades,
            signals_by_strategy=self._signals_by_strategy,
            closed_trades=self._closed_trades,
            winning_trades=self._winning_trades,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
            trading_days_per_year=365,
        )

        results.metadata = {  # type: ignore
            "total_funding_pnl": self._total_funding_pnl,
            "liquidations": self._liquidations,
            "funding_pnl_pct": self._total_funding_pnl / self.config.initial_capital,
        }

        return results
