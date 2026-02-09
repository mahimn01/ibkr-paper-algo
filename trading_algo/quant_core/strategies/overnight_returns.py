"""
Overnight Returns Strategy

Based on: Cliff, Cooper, Gulen (2008) and subsequent research showing
that a disproportionate fraction of equity returns accrues overnight
(close-to-open) rather than intraday (open-to-close).

Key finding: Buying at the close and selling at the open captures the
overnight premium, which averages ~60% of total daily returns for large
caps and is especially strong on high-uncertainty days.

Implementation:
    1. Track close-to-open (overnight) returns for each symbol.
    2. Rank symbols by recent overnight return persistence.
    3. Enter long at 15:55 (5 min before close) for top N symbols.
    4. Exit at 9:35 the next morning (shortly after open).

Expected: 10-15% standalone, ~1-2% portfolio contribution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional

import numpy as np


@dataclass
class OvernightConfig:
    """Configuration for the overnight returns strategy."""

    entry_time: time = time(15, 55)
    """Enter positions shortly before close."""

    exit_time: time = time(9, 35)
    """Exit positions shortly after next open."""

    lookback_days: int = 20
    """Days of overnight return history for ranking."""

    min_overnight_avg: float = 0.0005
    """Minimum average overnight return to consider (0.05%)."""

    top_n: int = 5
    """Number of top symbols to hold overnight."""

    position_size: float = 0.08
    """Target weight per position (8%)."""

    vol_scale: bool = True
    """Scale positions inversely by overnight return volatility."""

    target_vol: float = 0.10
    """Target annualized vol for overnight positions."""

    min_price: float = 10.0
    """Minimum stock price (avoid illiquid names)."""


@dataclass
class OvernightSignal:
    """Signal produced by the overnight returns strategy."""
    symbol: str
    direction: int           # Almost always 1 (long overnight)
    avg_overnight_return: float
    overnight_sharpe: float
    position_size: float
    confidence: float
    entry_price: float


class OvernightReturnsStrategy:
    """
    Implements the overnight returns anomaly.

    Usage::

        strategy = OvernightReturnsStrategy()

        # Track opens and closes
        strategy.record_close(symbol, close_price, timestamp)
        strategy.record_open(symbol, open_price, timestamp)

        # At 15:55, generate MOC signals
        signals = strategy.generate_signals(symbols, timestamp, prices)
    """

    def __init__(self, config: Optional[OvernightConfig] = None):
        self.config = config or OvernightConfig()

        # Daily open/close tracking
        self._closes: Dict[str, List[float]] = defaultdict(list)
        self._opens: Dict[str, List[float]] = defaultdict(list)
        self._current_prices: Dict[str, float] = {}
        self._current_date: Optional[datetime] = None
        self._day_open_recorded: Dict[str, bool] = {}

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        close: float,
        volume: float = 0,
    ) -> None:
        """Feed price data.  The strategy tracks opens and closes per day."""
        current_date = timestamp.date()

        if self._current_date is None or current_date != self._current_date:
            self._current_date = current_date
            self._day_open_recorded.clear()

        # Record first open of the day
        if symbol not in self._day_open_recorded:
            self._opens[symbol].append(open_price)
            self._day_open_recorded[symbol] = True
            # Trim history
            max_len = self.config.lookback_days + 5
            if len(self._opens[symbol]) > max_len:
                self._opens[symbol] = self._opens[symbol][-max_len:]

        # Always update close (last one of the day wins)
        self._current_prices[symbol] = close

    def record_eod(self, symbol: str, close_price: float) -> None:
        """
        Explicitly record end-of-day close.

        Call at end of day (or the update with the final bar will be used).
        """
        self._closes[symbol].append(close_price)
        max_len = self.config.lookback_days + 5
        if len(self._closes[symbol]) > max_len:
            self._closes[symbol] = self._closes[symbol][-max_len:]

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[OvernightSignal]:
        """
        Generate signals for overnight holds.

        Should be called at or near ``config.entry_time`` (default 15:55).
        """
        current_time = timestamp.time()
        if current_time < self.config.entry_time:
            return []

        prices = current_prices or self._current_prices

        # Compute overnight returns for each symbol
        on_stats: Dict[str, Dict] = {}

        for symbol in symbols:
            closes = self._closes.get(symbol, [])
            opens = self._opens.get(symbol, [])
            price = prices.get(symbol)

            if price is None or price < self.config.min_price:
                continue

            # Need at least lookback_days of paired close/open data
            n_pairs = min(len(closes), len(opens) - 1) if len(opens) > 1 else 0
            # opens[i+1] is the open after closes[i]
            if n_pairs < 5:
                continue

            overnight_returns = []
            for i in range(max(0, len(closes) - self.config.lookback_days), len(closes)):
                open_idx = i + 1
                if open_idx < len(opens) and closes[i] > 0:
                    on_ret = (opens[open_idx] - closes[i]) / closes[i]
                    overnight_returns.append(on_ret)

            if len(overnight_returns) < 5:
                continue

            arr = np.array(overnight_returns)
            avg_ret = float(np.mean(arr))
            std_ret = float(np.std(arr)) if len(arr) > 1 else 0.01
            sharpe = avg_ret / std_ret if std_ret > 0 else 0.0

            on_stats[symbol] = {
                "avg_return": avg_ret,
                "std": std_ret,
                "sharpe": sharpe,
                "price": price,
            }

        if not on_stats:
            return []

        # Rank by average overnight return
        ranked = sorted(
            on_stats.keys(),
            key=lambda s: on_stats[s]["avg_return"],
            reverse=True,
        )

        # Take top N with positive average
        selected = [
            s for s in ranked[:self.config.top_n]
            if on_stats[s]["avg_return"] >= self.config.min_overnight_avg
        ]

        signals: List[OvernightSignal] = []
        for symbol in selected:
            stats = on_stats[symbol]
            size = self.config.position_size

            # Vol scaling
            if self.config.vol_scale and stats["std"] > 0:
                # Annualize overnight vol (assume 252 trading days)
                annualized_vol = stats["std"] * np.sqrt(252)
                if annualized_vol > 0:
                    vol_scalar = min(2.0, self.config.target_vol / annualized_vol)
                    size *= vol_scalar

            # Confidence from Sharpe and consistency
            sharpe_score = min(1.0, max(0.0, stats["sharpe"] / 2.0))
            confidence = sharpe_score

            signals.append(OvernightSignal(
                symbol=symbol,
                direction=1,  # Long overnight
                avg_overnight_return=stats["avg_return"],
                overnight_sharpe=stats["sharpe"],
                position_size=size,
                confidence=confidence,
                entry_price=stats["price"],
            ))

        return signals

    def reset(self) -> None:
        """Reset all state."""
        self._closes.clear()
        self._opens.clear()
        self._current_prices.clear()
        self._day_open_recorded.clear()
        self._current_date = None
