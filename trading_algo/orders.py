from __future__ import annotations

from dataclasses import dataclass

from trading_algo.broker.base import OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument


@dataclass(frozen=True)
class TradeIntent:
    instrument: InstrumentSpec
    side: str  # BUY|SELL
    quantity: float
    order_type: str = "MKT"
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "DAY"

    def to_order_request(self) -> OrderRequest:
        return OrderRequest(
            instrument=validate_instrument(self.instrument),
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            tif=self.tif,
        )
