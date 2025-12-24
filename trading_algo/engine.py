from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from trading_algo.broker.base import Broker
from trading_algo.config import TradingConfig
from trading_algo.market_data import MarketDataClient
from trading_algo.orders import TradeIntent
from trading_algo.persistence import SqliteStore
from trading_algo.risk import RiskLimits, RiskManager
from trading_algo.strategy.base import Strategy, StrategyContext

log = logging.getLogger(__name__)


@dataclass
class Engine:
    broker: Broker
    config: TradingConfig
    strategy: Strategy
    risk: RiskManager
    confirm_token: str | None = None
    _md: MarketDataClient | None = None
    _store: SqliteStore | None = None
    _run_id: int | None = None

    def run_forever(self) -> None:
        if not self.config.live_enabled and self.config.broker == "ibkr":
            log.warning("Live trading disabled: set TRADING_LIVE_ENABLED=true to allow placing orders on IBKR.")

        self.broker.connect()
        self._md = MarketDataClient(self.broker)
        if self.config.db_path:
            self._store = SqliteStore(self.config.db_path)
            self._run_id = self._store.start_run(self.config)
        try:
            while True:
                ctx = self._build_context()
                intents = self.strategy.on_tick(ctx)
                self._handle_intents(ctx, intents)
                time.sleep(self.config.poll_seconds)
        finally:
            self._md = None
            if self._store is not None:
                if self._run_id is not None:
                    self._store.end_run(self._run_id)
                self._store.close()
            self._store = None
            self._run_id = None
            self.broker.disconnect()

    def run_once(self, ctx: StrategyContext | None = None) -> None:
        self.broker.connect()
        self._md = MarketDataClient(self.broker)
        if self.config.db_path:
            self._store = SqliteStore(self.config.db_path)
            self._run_id = self._store.start_run(self.config)
        try:
            ctx = ctx or self._build_context()
            intents = self.strategy.on_tick(ctx)
            self._handle_intents(ctx, intents)
        finally:
            self._md = None
            if self._store is not None:
                if self._run_id is not None:
                    self._store.end_run(self._run_id)
                self._store.close()
            self._store = None
            self._run_id = None
            self.broker.disconnect()

    def _build_context(self) -> StrategyContext:
        if self._md is None:
            raise RuntimeError("MarketDataClient is not initialized")
        return StrategyContext(
            now_epoch_s=time.time(),
            get_snapshot=self._md.get_snapshot,
        )

    def _handle_intents(self, ctx: StrategyContext, intents: list[TradeIntent]) -> None:
        if not intents:
            log.info("No intents from strategy=%s", getattr(self.strategy, "name", "unknown"))
            return

        strategy_name = getattr(self.strategy, "name", "unknown")

        for intent in intents:
            try:
                if self._md is None:
                    raise RuntimeError("MarketDataClient is not initialized")
                self.risk.validate(intent, self.broker, self._md.get_snapshot)
                placed, info = self._place_intent(intent)
                self._persist_decision(strategy_name, intent, accepted=placed, reason=None if placed else info)
            except Exception as exc:
                log.error(
                    "Intent rejected kind=%s symbol=%s side=%s qty=%s err=%s",
                    intent.instrument.kind,
                    intent.instrument.symbol,
                    intent.side,
                    intent.quantity,
                    exc,
                )
                self._persist_decision(strategy_name, intent, accepted=False, reason=str(exc))
                self._persist_error("handle_intents", str(exc))

    def _place_intent(self, intent: TradeIntent) -> tuple[bool, str]:
        if self.config.dry_run:
            log.info("DRY RUN staged intent: %s", intent)
            return False, "dry_run"

        if self.config.broker == "ibkr" and not self.config.live_enabled:
            log.warning("Blocked order (live disabled): %s", intent)
            return False, "live_disabled"

        if self.config.broker == "ibkr":
            if not self.config.order_token:
                log.warning("Blocked order (missing TRADING_ORDER_TOKEN): %s", intent)
                return False, "missing_order_token"
            if self.confirm_token != self.config.order_token:
                log.warning("Blocked order (confirm token mismatch): %s", intent)
                return False, "confirm_token_mismatch"

        result = self.broker.place_order(intent.to_order_request())
        log.info("Order result: orderId=%s status=%s", result.order_id, result.status)
        self._persist_order(result.order_id, intent, result.status)
        self._persist_order_status_event(result.order_id)
        return True, result.order_id

    def _persist_decision(self, strategy: str, intent: TradeIntent, *, accepted: bool, reason: str | None) -> None:
        if self._store is None or self._run_id is None:
            return
        self._store.log_decision(self._run_id, strategy=strategy, intent=intent, accepted=accepted, reason=reason)

    def _persist_order(self, order_id: str, intent: TradeIntent, status: str) -> None:
        if self._store is None or self._run_id is None:
            return
        self._store.log_order(
            self._run_id,
            broker=self.config.broker,
            order_id=order_id,
            request=intent.to_order_request(),
            status=status,
        )

    def _persist_order_status_event(self, order_id: str) -> None:
        if self._store is None or self._run_id is None:
            return
        try:
            st = self.broker.get_order_status(order_id)
        except Exception as exc:
            self._persist_error("order_status", str(exc))
            return
        self._store.log_order_status_event(self._run_id, self.config.broker, st)

    def _persist_error(self, where: str, message: str) -> None:
        if self._store is None or self._run_id is None:
            return
        self._store.log_error(self._run_id, where=where, message=message)


def default_risk_manager() -> RiskManager:
    return RiskManager(RiskLimits())
