from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7


@dataclass(frozen=True)
class TradingConfig:
    broker: str = "ibkr"  # "ibkr" | "sim"
    live_enabled: bool = False
    require_paper: bool = True  # forced-on safety rail
    dry_run: bool = True
    order_token: str | None = None
    db_path: str | None = None
    poll_seconds: int = 5
    ibkr: IBKRConfig = IBKRConfig()

    @staticmethod
    def from_env() -> "TradingConfig":
        ibkr = IBKRConfig(
            host=_get_env("IBKR_HOST", "127.0.0.1"),
            port=_get_env_int("IBKR_PORT", 7497),
            client_id=_get_env_int("IBKR_CLIENT_ID", 7),
        )
        return TradingConfig(
            broker=_get_env("TRADING_BROKER", "ibkr"),
            live_enabled=_get_env_bool("TRADING_LIVE_ENABLED", False),
            # Intentionally forced on: paper-only guard should not be disabled by env.
            require_paper=True,
            dry_run=_get_env_bool("TRADING_DRY_RUN", True),
            order_token=(_get_env("TRADING_ORDER_TOKEN", "").strip() or None),
            db_path=(_get_env("TRADING_DB_PATH", "").strip() or None),
            poll_seconds=_get_env_int("TRADING_POLL_SECONDS", 5),
            ibkr=ibkr,
        )
