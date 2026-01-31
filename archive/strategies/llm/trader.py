from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from trading_algo.broker.base import Broker
from trading_algo.config import TradingConfig
from trading_algo.market_data import MarketDataClient, MarketDataConfig
from trading_algo.oms import OrderManager
from trading_algo.risk import RiskManager
from trading_algo.llm.config import LLMConfig
from trading_algo.llm.decision import CancelDecision, ModifyDecision, PlaceDecision, enforce_llm_limits, parse_llm_decisions
from trading_algo.llm.gemini import LLMClient

log = logging.getLogger(__name__)


@dataclass
class LLMTrader:
    """
    LLM-driven trading loop.

    The LLM proposes PLACE/MODIFY/CANCEL decisions (JSON), but execution is always
    performed via the OMS so existing safety gates apply.
    """

    broker: Broker
    trading: TradingConfig
    llm: LLMConfig
    client: LLMClient
    risk: RiskManager
    confirm_token: str | None = None
    market_data: MarketDataConfig = MarketDataConfig()
    sleep_seconds: float = 5.0
    max_ticks: int | None = None

    def run(self) -> None:
        if not self.llm.enabled or self.llm.provider != "gemini":
            raise RuntimeError("LLM trader is disabled (set LLM_ENABLED=true and LLM_PROVIDER=gemini)")
        if not self.llm.allowed_symbols():
            raise RuntimeError("LLM_ALLOWED_SYMBOLS must be set (comma-separated) when LLM is enabled")

        self.broker.connect()
        md = MarketDataClient(self.broker, self.market_data)
        oms = OrderManager(self.broker, self.trading, confirm_token=self.confirm_token)
        try:
            tick = 0
            while True:
                tick += 1
                self._run_once(md, oms)
                if self.max_ticks is not None and tick >= int(self.max_ticks):
                    return
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
        finally:
            oms.close()
            self.broker.disconnect()

    def run_once(self) -> None:
        if not self.llm.enabled or self.llm.provider != "gemini":
            raise RuntimeError("LLM trader is disabled (set LLM_ENABLED=true and LLM_PROVIDER=gemini)")
        if not self.llm.allowed_symbols():
            raise RuntimeError("LLM_ALLOWED_SYMBOLS must be set (comma-separated) when LLM is enabled")
        self.broker.connect()
        md = MarketDataClient(self.broker, self.market_data)
        oms = OrderManager(self.broker, self.trading, confirm_token=self.confirm_token)
        try:
            self._run_once(md, oms)
        finally:
            oms.close()
            self.broker.disconnect()

    def _run_once(self, md: MarketDataClient, oms: OrderManager) -> None:
        # Build minimal context for the model.
        account = self.broker.get_account_snapshot()
        positions = self.broker.get_positions()
        open_orders = self.broker.list_open_order_statuses()
        symbols = sorted(self.llm.allowed_symbols())

        snapshots = {}
        for sym in symbols:
            try:
                from trading_algo.instruments import InstrumentSpec

                snap = md.get_snapshot(InstrumentSpec(kind="STK", symbol=sym))
                snapshots[sym] = {
                    "bid": snap.bid,
                    "ask": snap.ask,
                    "last": snap.last,
                    "close": snap.close,
                    "volume": snap.volume,
                    "ts": snap.timestamp_epoch_s,
                }
            except Exception as exc:
                snapshots[sym] = {"error": str(exc)}

        prompt = _build_prompt(
            now_epoch_s=time.time(),
            allowed_symbols=symbols,
            account=account.values,
            positions=[{"symbol": p.instrument.symbol, "kind": p.instrument.kind, "qty": p.quantity, "avg_cost": p.avg_cost} for p in positions],
            open_orders=[{"order_id": o.order_id, "status": o.status, "filled": o.filled, "remaining": o.remaining} for o in open_orders],
            snapshots=snapshots,
        )
        raw = self.client.generate(
            prompt=prompt,
            system=_SYSTEM_INSTRUCTIONS,
            use_google_search=bool(self.llm.gemini_use_google_search),
        )

        try:
            decisions = parse_llm_decisions(raw)
            decisions = enforce_llm_limits(
                decisions,
                allowed_kinds=self.llm.allowed_kinds(),
                allowed_symbols=self.llm.allowed_symbols(),
                max_orders=int(self.llm.max_orders_per_tick),
                max_qty=float(self.llm.max_qty),
            )
        except Exception as exc:
            oms.log_action("llm.parse", payload={"raw": raw}, accepted=False, reason=str(exc))
            log.warning("LLM output rejected: %s", exc)
            return

        for d in decisions:
            if isinstance(d, PlaceDecision):
                self._exec_place(d, oms)
            elif isinstance(d, ModifyDecision):
                self._exec_modify(d, oms)
            elif isinstance(d, CancelDecision):
                self._exec_cancel(d, oms)

    def _exec_place(self, d: PlaceDecision, oms: OrderManager) -> None:
        try:
            # Reuse risk validation by converting to TradeIntent (place-only).
            from trading_algo.orders import TradeIntent

            intent = TradeIntent(
                instrument=d.request.instrument,
                side=d.request.side,
                quantity=float(d.request.quantity),
                order_type=d.request.order_type,
                limit_price=d.request.limit_price,
                stop_price=d.request.stop_price,
                tif=d.request.tif,
            )
            self.risk.validate(intent, self.broker, self.broker.get_market_data_snapshot)
            res = oms.submit(d.request)
            oms.log_action("llm.place", payload=_decision_to_json(d), accepted=(res.status != "DryRun"), reason=res.status)
        except Exception as exc:
            oms.log_action("llm.place", payload=_decision_to_json(d), accepted=False, reason=str(exc))

    def _exec_modify(self, d: ModifyDecision, oms: OrderManager) -> None:
        try:
            res = oms.modify(d.order_id, d.request)
            oms.log_action("llm.modify", payload=_decision_to_json(d), accepted=(res.status != "DryRun"), reason=res.status)
        except Exception as exc:
            oms.log_action("llm.modify", payload=_decision_to_json(d), accepted=False, reason=str(exc))

    def _exec_cancel(self, d: CancelDecision, oms: OrderManager) -> None:
        try:
            oms.cancel(d.order_id)
            oms.log_action("llm.cancel", payload=_decision_to_json(d), accepted=not bool(self.trading.dry_run), reason="cancelled")
        except Exception as exc:
            oms.log_action("llm.cancel", payload=_decision_to_json(d), accepted=False, reason=str(exc))


_SYSTEM_INSTRUCTIONS = (
    "You are an execution assistant for a PAPER-trading only system. "
    "Return ONLY valid JSON (no markdown). "
    "You must not exceed max_orders_per_tick, and must only trade allowed_symbols. "
    "Output schema: {\"decisions\":[ ... ]} where each decision is one of:\n"
    "  {\"action\":\"PLACE\",\"reason\":...,\"order\":{"
    "     \"instrument\":{\"kind\":\"STK\",\"symbol\":\"AAPL\",\"exchange\":\"SMART\",\"currency\":\"USD\"},"
    "     \"side\":\"BUY\",\"qty\":1,\"type\":\"MKT\",\"tif\":\"DAY\""
    "  }}\n"
    "  {\"action\":\"MODIFY\",\"order_id\":\"...\",\"reason\":...,\"order\":{...same order fields...}}\n"
    "  {\"action\":\"CANCEL\",\"order_id\":\"...\",\"reason\":...}\n"
)


def _build_prompt(
    *,
    now_epoch_s: float,
    allowed_symbols: list[str],
    account: dict[str, float],
    positions: list[dict[str, object]],
    open_orders: list[dict[str, object]],
    snapshots: dict[str, object],
) -> str:
    return json.dumps(
        {
            "now_epoch_s": float(now_epoch_s),
            "allowed_symbols": list(allowed_symbols),
            "account": dict(account),
            "positions": list(positions),
            "open_orders": list(open_orders),
            "market_snapshots": snapshots,
        },
        sort_keys=True,
    )


def _decision_to_json(d: object) -> dict[str, object]:
    if hasattr(d, "__dict__"):
        return {k: v for k, v in dict(getattr(d, "__dict__")).items()}
    return {"decision": str(d)}
