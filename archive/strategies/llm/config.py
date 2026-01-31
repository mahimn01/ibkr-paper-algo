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

def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "off"  # off|gemini

    gemini_api_key: str | None = None
    # Gemini 3 model IDs are currently preview IDs.
    gemini_model: str = "gemini-3-pro-preview"
    # Enable built-in Google Search grounding by default.
    gemini_use_google_search: bool = True
    # Prefer the official google-genai SDK when installed (falls back to REST).
    gemini_prefer_sdk: bool = True
    # Include streaming thought summaries (includeThoughts).
    gemini_include_thoughts: bool = True
    # Enable URL Context tool during research (when URLs are provided).
    gemini_use_url_context: bool = True
    # Enable Code Execution tool during research when useful (explicitly requested).
    gemini_use_code_execution: bool = False

    allowed_kinds_csv: str = "STK"
    # Empty means "allow any symbol" (still paper-only at the broker layer).
    allowed_symbols_csv: str = ""
    # If true, tool calls are restricted to allowed_kinds/allowed_symbols.
    # Default is false to avoid unexpected "Symbol not allowed" errors.
    enforce_allowlist: bool = False
    max_orders_per_tick: int = 3
    max_qty: float = 10.0

    # Context management (token-aware).
    # Max total prompt tokens (including history) before summarization/trimming kicks in.
    max_context_tokens: int = 200_000
    # Keep the most recent N content blocks verbatim.
    keep_recent_contents: int = 30
    # Summarization triggers at max_context_tokens * trigger_ratio.
    summarize_trigger_ratio: float = 0.90
    # Model to use for summarization (defaults to the main model).
    summarize_model: str = ""

    # Explicit context caching (optional; may incur billing depending on your plan).
    gemini_explicit_caching: bool = False
    gemini_cache_ttl_seconds: int = 600
    gemini_cache_min_tokens: int = 2048
    gemini_cache_delete_after_turn: bool = True

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
            enabled=_get_env_bool("LLM_ENABLED", False),
            provider=_get_env("LLM_PROVIDER", "off").strip().lower(),
            gemini_api_key=(_get_env("GEMINI_API_KEY", "").strip() or None),
            gemini_model=_get_env("GEMINI_MODEL", "gemini-3").strip(),
            gemini_use_google_search=_get_env_bool("LLM_USE_GOOGLE_SEARCH", True),
            gemini_prefer_sdk=_get_env_bool("GEMINI_PREFER_SDK", True),
            gemini_include_thoughts=_get_env_bool("LLM_INCLUDE_THOUGHTS", True),
            gemini_use_url_context=_get_env_bool("LLM_USE_URL_CONTEXT", True),
            gemini_use_code_execution=_get_env_bool("LLM_USE_CODE_EXECUTION", False),
            allowed_kinds_csv=_get_env("LLM_ALLOWED_KINDS", "STK").strip(),
            allowed_symbols_csv=_get_env("LLM_ALLOWED_SYMBOLS", "").strip(),
            enforce_allowlist=_get_env_bool("LLM_ENFORCE_ALLOWLIST", False),
            max_orders_per_tick=_get_env_int("LLM_MAX_ORDERS_PER_TICK", 3),
            max_qty=float(_get_env("LLM_MAX_QTY", "10.0")),
            max_context_tokens=_get_env_int("LLM_MAX_CONTEXT_TOKENS", 200_000),
            keep_recent_contents=_get_env_int("LLM_KEEP_RECENT_CONTENTS", 30),
            summarize_trigger_ratio=_get_env_float("LLM_SUMMARIZE_TRIGGER_RATIO", 0.90),
            summarize_model=_get_env("LLM_SUMMARIZE_MODEL", "").strip(),
            gemini_explicit_caching=_get_env_bool("GEMINI_EXPLICIT_CACHING", False),
            gemini_cache_ttl_seconds=_get_env_int("GEMINI_CACHE_TTL_SECONDS", 600),
            gemini_cache_min_tokens=_get_env_int("GEMINI_CACHE_MIN_TOKENS", 2048),
            gemini_cache_delete_after_turn=_get_env_bool("GEMINI_CACHE_DELETE_AFTER_TURN", True),
        )

    def allowed_kinds(self) -> set[str]:
        kinds = {k.strip().upper() for k in self.allowed_kinds_csv.split(",") if k.strip()}
        return kinds or {"STK"}

    def allowed_symbols(self) -> set[str]:
        return {s.strip().upper() for s in self.allowed_symbols_csv.split(",") if s.strip()}

    def normalized_gemini_model(self) -> str:
        """
        Normalize common (but invalid) model names to supported Gemini 3 identifiers.

        Users commonly guess suffixes like "-pro" which may not exist on the v1beta endpoint.
        """
        m = str(self.gemini_model or "").strip()
        aliases = {
            # Common shorthand / guesses -> preview IDs
            "gemini-3": "gemini-3-pro-preview",
            "gemini-3-pro": "gemini-3-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
        }
        m = aliases.get(m, m)
        return m or "gemini-3-pro-preview"
