from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass

from trading_algo.broker.base import Broker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.market_data import MarketDataClient, MarketDataConfig
from trading_algo.orders import TradeIntent
from trading_algo.oms import OrderManager
from trading_algo.risk import RiskLimits, RiskManager
from trading_algo.strategy.base import Strategy, StrategyContext

log = logging.getLogger(__name__)


@dataclass
class AutoRunner:
    """
    Always-on runner that:
    - connects broker
    - runs OMS reconciliation on startup (if db enabled)
    - executes strategy ticks
    - submits orders via OMS (so all gating applies)
    - persists decisions/orders/status transitions if db enabled
    - tracks open orders periodically

    For tests, set `max_ticks` and `sleep_seconds=0`.
    """

    broker: Broker
    config: TradingConfig
    strategy: Strategy
    risk: RiskManager
    confirm_token: str | None = None
    market_data: MarketDataConfig = MarketDataConfig()
    sleep_seconds: float = 1.0
    track_every_ticks: int = 1
    track_poll_seconds: float = 1.0
    track_timeout_seconds: float | None = 5.0
    max_ticks: int | None = None

    def run(self) -> None:
        self.broker.connect()
        md = MarketDataClient(self.broker, self.market_data)
        oms = OrderManager(self.broker, self.config, confirm_token=self.confirm_token)
        try:
            if self.config.db_path:
                try:
                    oms.reconcile()
                except Exception as exc:
                    log.error("OMS reconcile failed: %s", exc)

            tick = 0
            while True:
                tick += 1
                ctx = StrategyContext(now_epoch_s=time.time(), get_snapshot=md.get_snapshot)
                self._ctx_snapshot_getter = ctx.get_snapshot
                intents = list(self.strategy.on_tick(ctx))
                self._handle_intents(oms, intents)
                self._ctx_snapshot_getter = None

                if self.track_every_ticks > 0 and (tick % self.track_every_ticks == 0):
                    try:
                        oms.track_open_orders(
                            poll_seconds=self.track_poll_seconds,
                            timeout_seconds=self.track_timeout_seconds,
                        )
                    except Exception as exc:
                        log.error("OMS track failed: %s", exc)

                if self.max_ticks is not None and tick >= self.max_ticks:
                    return
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
        finally:
            oms.close()
            self.broker.disconnect()

    def _handle_intents(self, oms: OrderManager, intents: list[TradeIntent]) -> None:
        strategy_name = getattr(self.strategy, "name", "unknown")
        for intent in intents:
            try:
                snapshot_getter = getattr(self, "_ctx_snapshot_getter", None)
                get_snapshot = snapshot_getter if callable(snapshot_getter) else self.broker.get_market_data_snapshot
                self.risk.validate(intent, self.broker, get_snapshot)
                res = oms.submit(intent.to_order_request())
                accepted = (res.status != "DryRun")
                reason = res.status if res.status == "DryRun" else f"order_id={res.order_id} status={res.status}"
                oms.log_decision(strategy_name, intent, accepted=accepted, reason=reason)
            except Exception as exc:
                oms.log_decision(strategy_name, intent, accepted=False, reason=str(exc))
                log.warning(
                    "Intent rejected kind=%s symbol=%s side=%s qty=%s err=%s",
                    intent.instrument.kind,
                    intent.instrument.symbol,
                    intent.side,
                    intent.quantity,
                    exc,
                )


def _load_dotenv_if_present() -> None:
    if not os.path.exists(".env"):
        return
    with open(".env", "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading_algo.autorun", description="Always-on trading runner (paper-only enforced)")
    p.add_argument("--broker", choices=["ibkr", "sim"], default=None, help="Override TRADING_BROKER")
    p.add_argument("--symbol", default="AAPL", help="Symbol for ExampleStrategy")
    p.add_argument("--confirm-token", default=None, help="Must match TRADING_ORDER_TOKEN to send IBKR orders")
    p.add_argument("--ibkr-host", default=None)
    p.add_argument("--ibkr-port", default=None)
    p.add_argument("--ibkr-client-id", default=None)
    p.add_argument("--db-path", default=None, help="Override TRADING_DB_PATH")
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--track-every-ticks", type=int, default=1)
    p.add_argument("--track-poll-seconds", type=float, default=1.0)
    p.add_argument("--track-timeout-seconds", type=float, default=5.0)
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_present()
    cfg = TradingConfig.from_env()
    args = build_parser().parse_args(argv)

    ibkr = IBKRConfig(
        host=args.ibkr_host or cfg.ibkr.host,
        port=int(args.ibkr_port or cfg.ibkr.port),
        client_id=int(args.ibkr_client_id or cfg.ibkr.client_id),
    )
    cfg = TradingConfig(
        broker=args.broker or cfg.broker,
        live_enabled=cfg.live_enabled,
        require_paper=True,
        dry_run=cfg.dry_run,
        order_token=cfg.order_token,
        db_path=args.db_path or cfg.db_path,
        poll_seconds=cfg.poll_seconds,
        ibkr=ibkr,
    )

    if cfg.broker == "sim":
        from trading_algo.broker.sim import SimBroker
        from trading_algo.instruments import InstrumentSpec

        sim = SimBroker()
        sim.set_market_data(InstrumentSpec(kind="STK", symbol=args.symbol), last=100.0)
        broker = sim
    else:
        from trading_algo.broker.ibkr import IBKRBroker

        broker = IBKRBroker(cfg.ibkr, require_paper=True)

    from trading_algo.strategy.example import ExampleStrategy

    runner = AutoRunner(
        broker=broker,
        config=cfg,
        strategy=ExampleStrategy(symbol=args.symbol),
        risk=RiskManager(RiskLimits()),
        confirm_token=args.confirm_token,
        sleep_seconds=float(args.sleep_seconds),
        max_ticks=args.max_ticks,
        track_every_ticks=int(args.track_every_ticks),
        track_poll_seconds=float(args.track_poll_seconds),
        track_timeout_seconds=float(args.track_timeout_seconds) if args.track_timeout_seconds is not None else None,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
