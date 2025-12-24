from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

from trading_algo.broker.base import AccountSnapshot, Broker, MarketDataSnapshot, Position
from trading_algo.instruments import InstrumentSpec, validate_instrument
from trading_algo.orders import TradeIntent


@dataclass(frozen=True)
class RiskLimits:
    max_order_quantity: float = 100.0
    max_abs_position_per_symbol: float = 500.0
    max_notional_exposure: float = 250_000.0
    max_leverage: float = 2.0
    max_margin_utilization: float = 0.5  # MaintMarginReq / NetLiquidation
    max_daily_loss: float = 1_000.0  # based on NetLiquidation drawdown from session start
    allow_short: bool = False


def _price_from_snapshot(snap: MarketDataSnapshot) -> float:
    candidates = [snap.last]
    if snap.bid is not None and snap.ask is not None and snap.bid > 0 and snap.ask > 0:
        candidates.append((snap.bid + snap.ask) / 2.0)
    candidates.append(snap.close)
    for c in candidates:
        if c is not None and c > 0:
            return float(c)
    raise ValueError(f"No usable price in market data snapshot for {snap.instrument}")


def _instrument_key(inst: InstrumentSpec) -> tuple[str, str]:
    inst = validate_instrument(inst)
    # For risk, key by kind+symbol; exchange differences are ignored at this layer.
    return (inst.kind, inst.symbol)


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits
        self._session_start_net_liq: float | None = None

    def validate(self, intent: TradeIntent, broker: Broker, get_snapshot: Callable[[InstrumentSpec], MarketDataSnapshot]) -> None:
        intent_inst = validate_instrument(intent.instrument)

        if intent.quantity <= 0:
            raise ValueError("quantity must be positive")
        if intent.quantity > self._limits.max_order_quantity:
            raise ValueError("order quantity exceeds max_order_quantity")

        positions = broker.get_positions()
        account = broker.get_account_snapshot()

        self._update_session_start(account)

        current_qty = _position_for(positions, intent_inst)
        delta = intent.quantity if intent.side.upper() == "BUY" else -intent.quantity
        resulting_qty = current_qty + delta

        if not self._limits.allow_short and resulting_qty < 0:
            raise ValueError("order would create a short position (allow_short=false)")

        if abs(resulting_qty) > self._limits.max_abs_position_per_symbol:
            raise ValueError("resulting position exceeds max_abs_position_per_symbol")

        snap = get_snapshot(intent_inst)
        px = _price_from_snapshot(snap)
        trade_notional = abs(delta) * px

        gross_position_value = _gross_position_value(account, positions)
        if gross_position_value is not None:
            if gross_position_value + trade_notional > self._limits.max_notional_exposure:
                raise ValueError("max_notional_exposure exceeded")

        _check_leverage_and_margin(self._limits, account)
        _check_daily_loss(self._limits, self._session_start_net_liq, account)

    def _update_session_start(self, account: AccountSnapshot) -> None:
        if self._session_start_net_liq is not None:
            return
        net_liq = account.values.get("NetLiquidation")
        if net_liq is not None and net_liq > 0:
            self._session_start_net_liq = float(net_liq)


def _position_for(positions: list[Position], instrument: InstrumentSpec) -> float:
    key = _instrument_key(instrument)
    total = 0.0
    for pos in positions:
        try:
            if _instrument_key(pos.instrument) == key:
                total += float(pos.quantity)
        except Exception:
            continue
    return total


def _gross_position_value(account: AccountSnapshot, _positions: list[Position]) -> float | None:
    gpv = account.values.get("GrossPositionValue")
    if gpv is None:
        return None
    try:
        return float(gpv)
    except Exception:
        return None


def _check_leverage_and_margin(limits: RiskLimits, account: AccountSnapshot) -> None:
    net_liq = account.values.get("NetLiquidation")
    gross = account.values.get("GrossPositionValue")
    maint = account.values.get("MaintMarginReq")

    if net_liq is not None and gross is not None:
        if net_liq <= 0:
            raise ValueError("Invalid NetLiquidation")
        leverage = float(gross) / float(net_liq)
        if leverage > limits.max_leverage:
            raise ValueError("max_leverage exceeded")

    if net_liq is not None and maint is not None:
        if net_liq <= 0:
            raise ValueError("Invalid NetLiquidation")
        util = float(maint) / float(net_liq)
        if util > limits.max_margin_utilization:
            raise ValueError("max_margin_utilization exceeded")


def _check_daily_loss(limits: RiskLimits, start_net_liq: float | None, account: AccountSnapshot) -> None:
    if start_net_liq is None:
        return
    now = account.values.get("NetLiquidation")
    if now is None:
        return
    drawdown = float(start_net_liq) - float(now)
    if drawdown > limits.max_daily_loss:
        raise ValueError("max_daily_loss exceeded (circuit breaker)")
