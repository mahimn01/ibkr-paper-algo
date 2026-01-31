from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from trading_algo.broker.base import OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument

DecisionAction = Literal["PLACE", "MODIFY", "CANCEL"]


@dataclass(frozen=True)
class PlaceDecision:
    action: Literal["PLACE"]
    request: OrderRequest
    reason: str | None = None


@dataclass(frozen=True)
class ModifyDecision:
    action: Literal["MODIFY"]
    order_id: str
    request: OrderRequest
    reason: str | None = None


@dataclass(frozen=True)
class CancelDecision:
    action: Literal["CANCEL"]
    order_id: str
    reason: str | None = None


LLMDecision = PlaceDecision | ModifyDecision | CancelDecision


def parse_llm_decisions(raw_text: str) -> list[LLMDecision]:
    """
    Parse model output into a list of OMS decisions.

    Requires JSON with shape:
      {"decisions":[ ... ]}
    """
    text = _strip_code_fences(str(raw_text).strip())
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM output must be a JSON object")
    decisions = obj.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("LLM output must contain 'decisions' list")
    out: list[LLMDecision] = []
    for item in decisions:
        out.append(_parse_one(item))
    return out


def enforce_llm_limits(
    decisions: list[LLMDecision],
    *,
    allowed_kinds: set[str],
    allowed_symbols: set[str],
    max_orders: int,
    max_qty: float,
) -> list[LLMDecision]:
    if max_orders <= 0:
        raise ValueError("max_orders must be positive")
    if len(decisions) > max_orders:
        raise ValueError(f"LLM produced too many decisions: {len(decisions)} > {max_orders}")

    for d in decisions:
        if isinstance(d, CancelDecision):
            continue
        inst = d.request.instrument
        if inst.kind.upper() not in {k.upper() for k in allowed_kinds}:
            raise ValueError(f"Instrument kind not allowed: {inst.kind}")
        if allowed_symbols and inst.symbol.upper() not in {s.upper() for s in allowed_symbols}:
            raise ValueError(f"Symbol not allowed: {inst.symbol}")
        if float(d.request.quantity) > float(max_qty):
            raise ValueError(f"Quantity too large: {d.request.quantity} > {max_qty}")
    return decisions


def _parse_one(item: Any) -> LLMDecision:
    if not isinstance(item, dict):
        raise ValueError("Each decision must be an object")
    action = str(item.get("action", "")).strip().upper()
    reason = item.get("reason")
    reason_s = str(reason) if reason is not None else None
    if action == "CANCEL":
        order_id = str(item.get("order_id", "")).strip()
        if not order_id:
            raise ValueError("CANCEL requires order_id")
        return CancelDecision(action="CANCEL", order_id=order_id, reason=reason_s)

    if action == "PLACE":
        req = _parse_order_request(item.get("order"))
        return PlaceDecision(action="PLACE", request=req, reason=reason_s)

    if action == "MODIFY":
        order_id = str(item.get("order_id", "")).strip()
        if not order_id:
            raise ValueError("MODIFY requires order_id")
        req = _parse_order_request(item.get("order"))
        return ModifyDecision(action="MODIFY", order_id=order_id, request=req, reason=reason_s)

    raise ValueError(f"Unsupported action: {action}")


def _parse_order_request(obj: Any) -> OrderRequest:
    if not isinstance(obj, dict):
        raise ValueError("order must be an object")

    inst = obj.get("instrument")
    if not isinstance(inst, dict):
        raise ValueError("order.instrument must be an object")

    instrument = validate_instrument(
        InstrumentSpec(
            kind=str(inst.get("kind", "")).strip().upper(),
            symbol=str(inst.get("symbol", "")).strip().upper(),
            exchange=(str(inst["exchange"]).strip() if inst.get("exchange") else None),
            currency=(str(inst["currency"]).strip().upper() if inst.get("currency") else None),
            expiry=(str(inst["expiry"]).strip() if inst.get("expiry") else None),
        )
    )
    side = str(obj.get("side", "BUY")).strip().upper()
    order_type = str(obj.get("type", "MKT")).strip().upper()
    tif = str(obj.get("tif", "DAY")).strip().upper()
    qty = float(obj.get("qty", 0.0))
    limit_price = obj.get("limit_price")
    stop_price = obj.get("stop_price")

    return OrderRequest(
        instrument=instrument,
        side=side,
        quantity=float(qty),
        order_type=order_type,
        limit_price=(float(limit_price) if limit_price is not None else None),
        stop_price=(float(stop_price) if stop_price is not None else None),
        tif=tif,
        outside_rth=bool(obj.get("outside_rth", False)),
        good_till_date=(str(obj.get("good_till_date")).strip() if obj.get("good_till_date") else None),
        account=(str(obj.get("account")).strip() if obj.get("account") else None),
        order_ref=(str(obj.get("order_ref")).strip() if obj.get("order_ref") else None),
        oca_group=(str(obj.get("oca_group")).strip() if obj.get("oca_group") else None),
        transmit=bool(obj.get("transmit", True)),
    ).normalized()


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        # ```json\n...\n```
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text

