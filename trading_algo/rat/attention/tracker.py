"""
Attention Tracker: Integrates with broker for real-time attention tracking.

Processes market data snapshots and news to update attention flow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from trading_algo.rat.attention.flow import AttentionFlow, AttentionState
from trading_algo.rat.config import AttentionConfig
from trading_algo.rat.signals import Signal


class AttentionTracker:
    """
    Track attention across multiple symbols.

    Integrates with broker for live data.
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        broker: Optional[Any] = None,
    ):
        self.config = config or AttentionConfig()
        self.broker = broker

        # Attention flow per symbol
        self._flows: Dict[str, AttentionFlow] = {}

        # Last state cache
        self._last_state: Dict[str, AttentionState] = {}

    def _get_flow(self, symbol: str) -> AttentionFlow:
        """Get or create attention flow for symbol."""
        if symbol not in self._flows:
            self._flows[symbol] = AttentionFlow(
                news_weight=self.config.news_weight,
                flow_weight=self.config.flow_weight,
                price_weight=self.config.price_weight,
                window_size=self.config.flow_window,
            )
        return self._flows[symbol]

    def process_snapshot(self, snapshot: Dict[str, Any]) -> Optional[AttentionState]:
        """
        Process a market data snapshot.

        Expected keys:
            - symbol: str
            - last: float (price)
            - bid: float
            - ask: float
            - volume: float
            - timestamp: datetime or float (epoch)
        """
        symbol = snapshot.get("symbol")
        if not symbol:
            return None

        flow = self._get_flow(symbol)

        # Extract timestamp
        ts = snapshot.get("timestamp")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts)
        elif ts is None:
            ts = datetime.now()

        # Update price
        price = snapshot.get("last") or snapshot.get("close")
        if price:
            flow.update_price(ts, float(price))

        # Estimate order flow from bid/ask activity
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        volume = snapshot.get("volume", 0)

        if bid and ask and price and volume:
            buy_vol, sell_vol = self._estimate_flow_from_quotes(price, bid, ask, volume)
            flow.update_flow(ts, buy_vol, sell_vol)

        # Compute state
        state = flow.compute_attention_state(symbol, ts)
        self._last_state[symbol] = state

        return state

    def _estimate_flow_from_quotes(
        self,
        price: float,
        bid: float,
        ask: float,
        volume: float,
    ) -> tuple[float, float]:
        """
        Estimate buy/sell flow from price position in spread.

        If price is near ask, assume buyer aggressor.
        If price is near bid, assume seller aggressor.
        """
        spread = ask - bid
        if spread <= 0:
            return volume / 2, volume / 2

        # Position in spread (0 = at bid, 1 = at ask)
        position = (price - bid) / spread
        position = max(0, min(1, position))

        buy_vol = volume * position
        sell_vol = volume * (1 - position)

        return buy_vol, sell_vol

    def process_news(self, news_item: Dict[str, Any]) -> None:
        """
        Process a news item.

        Expected keys:
            - symbol: str
            - headline: str
            - timestamp: datetime
        """
        symbol = news_item.get("symbol")
        if not symbol:
            return

        flow = self._get_flow(symbol)

        ts = news_item.get("timestamp")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts)
        elif ts is None:
            ts = datetime.now()

        flow.update_news(ts)

    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """Generate signal for a symbol."""
        if symbol not in self._flows:
            return None

        flow = self._flows[symbol]
        return flow.generate_signal(symbol)

    def get_state(self, symbol: str) -> Optional[AttentionState]:
        """Get last computed state for symbol."""
        return self._last_state.get(symbol)

    def inject_backtest_data(
        self,
        symbol: str,
        prices: list[tuple[datetime, float]],
        news_times: list[datetime],
        flow_data: list[tuple[datetime, float, float]],
    ) -> None:
        """Inject historical data for backtesting."""
        flow = self._get_flow(symbol)

        for ts, price in prices:
            flow.update_price(ts, price)

        for ts in news_times:
            flow.update_news(ts)

        for ts, buy_vol, sell_vol in flow_data:
            flow.update_flow(ts, buy_vol, sell_vol)
