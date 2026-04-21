"""Secret redaction for logs, error messages, and any stderr output.

Threats this module addresses:

1. IBKR Flex tokens, TWS/Gateway session tokens, or TRADING_ORDER_TOKEN
   appearing in stringified errors or logs.
2. Gemini / OpenAI / Claude API keys leaking through LLM subprocess stderr.
3. Generic `Authorization: Bearer ...` headers from upstream HTTP clients.
4. Long token-shaped strings whose provenance we don't know at logging time.

Usage:
    from trading_algo.redaction import redact_text, install_logging_filter

    # One-shot redaction at any user-visible surface:
    print(redact_text(str(exc)), file=sys.stderr)

    # At process start, attach a filter to every logger so TRADING_DEBUG=1
    # is safe:
    install_logging_filter()

The filter is idempotent. It reads only static patterns plus known secrets
from the current process environment — it never exfiltrates secrets.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

REDACTED = "***REDACTED***"


# Regex patterns for secret-shaped tokens. These cast a wide net and may
# over-redact — that is the desired failure mode.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("auth-header", re.compile(r"(?i)(authorization:\s*(?:token|bearer)\s+)\S+")),
    # "access_token": "...", 'order_token': '...', api_secret=... etc.
    ("credential-kv", re.compile(
        r"""(?ix)
        (                                                     # group 1: key + separator (kept)
          ["']?
          (?:
            access_token | refresh_token | public_token | request_token |
            api_secret | api_key | order_token | flex_token |
            confirm_token | ibkr_order_token | trading_order_token |
            password | pin | gemini_api_key | openai_api_key |
            anthropic_api_key | claude_api_key
          )
          ["']?
          \s*[:=]\s*
        )
        ( "[^"]+" | '[^']+' | \S+ )                           # group 2: value (replaced)
        """
    )),
    # Bearer prefix.
    ("bearer", re.compile(r"(?i)(bearer\s+)\S+")),
    # IBKR account numbers — U1234567, DU1234567. Redact to prevent
    # correlation between paper and live account IDs leaking.
    ("ibkr-account", re.compile(r"\b(?:U|DU|DF|F|DI|I)\d{6,10}\b")),
    # Long hex / base64 strings (>=32 chars, mostly alnum). Heuristic — may
    # over-redact. Acceptable trade for an IBKR audit trail.
    ("long-token", re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")),
)


def _patterns_sub(text: str) -> str:
    out = text
    for name, rx in _PATTERNS:
        if name in ("credential-kv", "auth-header", "bearer"):
            out = rx.sub(lambda m: f"{m.group(1)}{REDACTED}", out)
        else:
            out = rx.sub(REDACTED, out)
    return out


def known_secrets() -> list[str]:
    """Collect secrets we already know from the current process env.

    Returns candidate secret strings. Short strings (< 8 chars) are dropped —
    literal replacement with short strings would false-positive on ordinary
    numbers or words.
    """
    candidates: list[str] = []
    for var in (
        "IBKR_FLEX_TOKEN",
        "FLEX_TOKEN",
        "TRADING_ORDER_TOKEN",
        "TRADING_CONFIRM_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
    ):
        v = os.getenv(var)
        if v and len(v) >= 8:
            candidates.append(v)
    return candidates


def redact_text(text: str, *, extra_secrets: Iterable[str] = ()) -> str:
    """Redact secrets from `text`.

    Two layers:
      1. Known secrets (from env + `extra_secrets`) are replaced literally.
      2. Regex patterns catch anything that *looks* like a token.

    Empty/non-string inputs pass through (str() cast on non-strings).
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else str(text)

    out = text
    all_secrets = list(known_secrets()) + [s for s in extra_secrets if s]
    # Sort by length descending so longer secrets redact first (prevents
    # replacing a prefix that's shared with a shorter secret).
    for s in sorted({s for s in all_secrets if s and len(s) >= 8}, key=len, reverse=True):
        out = out.replace(s, REDACTED)
    out = _patterns_sub(out)
    return out


def redact_dict(d: Any) -> Any:
    """Recursively redact a dict/list structure. String leaves pass through
    `redact_text`; known credential-typed keys have values wholesale
    replaced.
    """
    _CREDENTIAL_KEYS = frozenset({
        "access_token", "refresh_token", "request_token", "public_token",
        "api_secret", "api_key", "order_token", "flex_token",
        "confirm_token", "password", "pin",
        "TRADING_ORDER_TOKEN", "IBKR_FLEX_TOKEN", "FLEX_TOKEN",
        "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    })
    if isinstance(d, dict):
        return {
            k: (REDACTED if (isinstance(k, str) and k in _CREDENTIAL_KEYS) else redact_dict(v))
            for k, v in d.items()
        }
    if isinstance(d, (list, tuple)):
        return [redact_dict(x) for x in d]
    if isinstance(d, str):
        return redact_text(d)
    return d


# ---------------------------------------------------------------------------
# Global logging filter
# ---------------------------------------------------------------------------

class _SecretRedactingFilter(logging.Filter):
    """Rewrite LogRecord.msg / args so emitted log lines carry no secrets.

    Performance: the expensive env read is cached. If secrets rotate
    mid-process, call `install_logging_filter(reset=True)` to refresh.
    """

    def __init__(self) -> None:
        super().__init__()
        self._secrets_cache: list[str] | None = None

    def _secrets(self) -> list[str]:
        if self._secrets_cache is None:
            self._secrets_cache = known_secrets()
        return self._secrets_cache

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            secrets = self._secrets()
            if isinstance(record.msg, str):
                redacted = _sub_many(record.msg, secrets)
                redacted = _patterns_sub(redacted)
                record.msg = redacted
            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(
                        _sub_many_any(a, secrets) for a in record.args
                    )
                elif isinstance(record.args, dict):
                    record.args = {
                        k: _sub_many_any(v, secrets)
                        for k, v in record.args.items()
                    }
        except Exception:
            pass  # never break logging
        return True


def _sub_many(text: str, secrets: list[str]) -> str:
    out = text
    for s in sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True):
        out = out.replace(s, REDACTED)
    return out


def _sub_many_any(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        return _patterns_sub(_sub_many(value, secrets))
    return value


_FILTER_INSTALLED: _SecretRedactingFilter | None = None


def install_logging_filter(reset: bool = False) -> None:
    """Attach the redacting filter to the root logger (idempotent).

    Use `reset=True` after token rotation to force the cache refresh.
    """
    global _FILTER_INSTALLED
    root = logging.getLogger()
    if _FILTER_INSTALLED is not None and not reset:
        return
    if _FILTER_INSTALLED is not None:
        try:
            root.removeFilter(_FILTER_INSTALLED)
        except Exception:
            pass
    _FILTER_INSTALLED = _SecretRedactingFilter()
    root.addFilter(_FILTER_INSTALLED)
    for h in root.handlers:
        h.addFilter(_FILTER_INSTALLED)
