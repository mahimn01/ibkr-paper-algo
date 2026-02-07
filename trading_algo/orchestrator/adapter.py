"""
Orchestrator → AutoRunner Adapter

Wraps the Orchestrator as a ``Strategy`` so it can be driven by the
AutoRunner loop.  This is the missing bridge that connects:

    AutoRunner  →  OrchestratorStrategy (this adapter)
                        ↓
                   Orchestrator.update_asset()
                   Orchestrator.generate_signal()
                        ↓
                   TradeIntent (BUY / SELL / SHORT / COVER)

The adapter:
  1. Fetches market-data snapshots for every tracked symbol.
  2. Feeds OHLCV bars into the Orchestrator.
  3. Converts OrchestratorSignals into TradeIntents the OMS can execute.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import List, Optional, Set

from trading_algo.broker.base import MarketDataSnapshot
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.strategy.base import StrategyContext

from .config import OrchestratorConfig
from .strategy import Orchestrator

logger = logging.getLogger(__name__)


class OrchestratorStrategy:
    """
    Adapter that exposes the Orchestrator as an AutoRunner-compatible Strategy.

    Parameters
    ----------
    symbols : list[str]
        Tradeable symbols (e.g. ["INTC", "AMD"]).
    reference_symbols : set[str] | None
        Extra symbols needed by the Orchestrator's edges (SPY, QQQ, etc.).
        Defaults to the Orchestrator's built-in reference_assets.
    account_equity : float
        Assumed account equity for position sizing (default 100 000).
    config : OrchestratorConfig | None
        Orchestrator configuration.
    """

    name: str = "OrchestratorStrategy"

    def __init__(
        self,
        symbols: List[str],
        reference_symbols: Optional[Set[str]] = None,
        account_equity: float = 100_000.0,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self._orchestrator = Orchestrator(config)
        self._symbols = list(symbols)
        self._reference_symbols = reference_symbols or self._orchestrator.reference_assets
        self._all_symbols = set(self._symbols) | self._reference_symbols
        self._equity = account_equity

        # Instruments cache  (STK on SMART)
        self._instruments: dict[str, InstrumentSpec] = {}
        for sym in self._all_symbols:
            self._instruments[sym] = InstrumentSpec(
                kind="STK", symbol=sym, exchange="SMART", currency="USD",
            )

    # ── Strategy protocol ─────────────────────────────────────────────

    def on_tick(self, ctx: StrategyContext) -> list[TradeIntent]:
        """
        Called every AutoRunner tick.

        1.  Fetch snapshots for all symbols.
        2.  Feed data into the Orchestrator.
        3.  Generate signals and convert to TradeIntents.
        """
        now = datetime.fromtimestamp(ctx.now_epoch_s)
        intents: list[TradeIntent] = []

        # Fetch & feed snapshots
        snapshots: dict[str, MarketDataSnapshot] = {}
        for sym in self._all_symbols:
            try:
                snap = ctx.get_snapshot(self._instruments[sym])
                snapshots[sym] = snap
            except Exception:
                continue

            price = _price_from(snap)
            if price is None or price <= 0:
                continue

            self._orchestrator.update_asset(
                symbol=sym,
                timestamp=now,
                open_price=price,
                high=price,
                low=price,
                close=price,
                volume=snap.volume or 0.0,
            )

        # Generate signals for tradeable symbols only
        for sym in self._symbols:
            if sym not in snapshots:
                continue

            signal = self._orchestrator.generate_signal(sym, now)
            intent = self._signal_to_intent(sym, signal)
            if intent is not None:
                intents.append(intent)

        return intents

    # ── internal helpers ──────────────────────────────────────────────

    def _signal_to_intent(
        self, symbol: str, signal,
    ) -> Optional[TradeIntent]:
        action = signal.action
        if action == "hold":
            return None

        instrument = self._instruments[symbol]
        price = signal.entry_price
        if price <= 0:
            return None

        if action in ("buy", "short"):
            qty = self._compute_shares(signal.size, price)
            if qty < 1:
                return None
            side = "BUY" if action == "buy" else "SELL"
            return TradeIntent(instrument=instrument, side=side, quantity=qty)

        if action == "sell":
            pos = self._orchestrator.positions  # already deleted by _check_exit
            return TradeIntent(instrument=instrument, side="SELL", quantity=1)

        if action == "cover":
            return TradeIntent(instrument=instrument, side="BUY", quantity=1)

        return None

    def _compute_shares(self, size_pct: float, price: float) -> int:
        notional = self._equity * size_pct
        return max(0, int(math.floor(notional / price)))


def _price_from(snap: MarketDataSnapshot) -> Optional[float]:
    for candidate in (snap.last, snap.close):
        if candidate is not None and candidate > 0:
            return candidate
    if snap.bid is not None and snap.ask is not None:
        mid = (snap.bid + snap.ask) / 2
        if mid > 0:
            return mid
    return None
