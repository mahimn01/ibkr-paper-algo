from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from trading_algo.broker.base import MarketDataSnapshot
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent


@dataclass(frozen=True)
class StrategyContext:
    now_epoch_s: float
    get_snapshot: Callable[[InstrumentSpec], MarketDataSnapshot]


class Strategy(Protocol):
    name: str

    def on_tick(self, ctx: StrategyContext) -> list[TradeIntent]: ...
