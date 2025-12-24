from __future__ import annotations

from dataclasses import dataclass

from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.strategy.base import StrategyContext


@dataclass
class ExampleStrategy:
    """
    Toy example: every N ticks, propose a small BUY on AAPL.
    Replace this with real signal logic and market data inputs.
    """

    name: str = "example"
    symbol: str = "AAPL"
    every_ticks: int = 12
    _tick: int = 0

    def on_tick(self, ctx: StrategyContext) -> list[TradeIntent]:
        self._tick += 1
        if self._tick % self.every_ticks != 0:
            return []

        instrument = InstrumentSpec(kind="STK", symbol=self.symbol, exchange="SMART", currency="USD")
        return [
            TradeIntent(
                instrument=instrument,
                side="BUY",
                quantity=1,
                order_type="MKT",
            )
        ]


def default_context() -> StrategyContext:
    raise RuntimeError("default_context is provided by Engine; construct StrategyContext with a snapshot getter")
