from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "off"  # off|gemini

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3"
    gemini_use_google_search: bool = False

    allowed_kinds_csv: str = "STK"
    allowed_symbols_csv: str = ""  # required when enabled, for safety
    max_orders_per_tick: int = 3
    max_qty: float = 10.0

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
            enabled=_get_env_bool("LLM_ENABLED", False),
            provider=_get_env("LLM_PROVIDER", "off").strip().lower(),
            gemini_api_key=(_get_env("GEMINI_API_KEY", "").strip() or None),
            gemini_model=_get_env("GEMINI_MODEL", "gemini-3").strip(),
            gemini_use_google_search=_get_env_bool("LLM_USE_GOOGLE_SEARCH", False),
            allowed_kinds_csv=_get_env("LLM_ALLOWED_KINDS", "STK").strip(),
            allowed_symbols_csv=_get_env("LLM_ALLOWED_SYMBOLS", "").strip(),
            max_orders_per_tick=_get_env_int("LLM_MAX_ORDERS_PER_TICK", 3),
            max_qty=float(_get_env("LLM_MAX_QTY", "10.0")),
        )

    def allowed_kinds(self) -> set[str]:
        kinds = {k.strip().upper() for k in self.allowed_kinds_csv.split(",") if k.strip()}
        return kinds or {"STK"}

    def allowed_symbols(self) -> set[str]:
        return {s.strip().upper() for s in self.allowed_symbols_csv.split(",") if s.strip()}

