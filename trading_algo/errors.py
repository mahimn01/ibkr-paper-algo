"""Structured error emission for agent-driven IBKR recovery.

Every top-level CLI command wraps its body in `with_error_envelope(cmd)` so
that any uncaught exception becomes a structured stderr JSON blob with the
correct exit code. Agents never have to regex free-text error messages.

Error JSON shape:

    {
      "ok": false,
      "cmd": "place-order",
      "schema_version": "1",
      "request_id": "...",
      "data": null,
      "warnings": [],
      "meta": {...},
      "error": {
        "code": "HARD_REJECT",
        "class": "IBKRDependencyError",
        "message": "Order rejected: insufficient buying power",
        "retryable": false,
        "ib_error_code": 201,
        "field_errors": [...],
        "suggested_action": "...",
        "exit_code": 4,
        "exit_code_name": "HARD_REJECT"
      }
    }

stdout carries the success envelope; stderr carries the error envelope on
failure. An agent reads stderr's final JSON blob to branch.
"""

from __future__ import annotations

import functools
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from trading_algo.envelope import (
    Envelope,
    envelope_to_json,
    finalize_envelope,
    new_envelope,
)
from trading_algo.exit_codes import (
    ClassifiedError,
    _extract_ib_error_code,
    classify_exception,
    exit_code_name,
)


# ---------------------------------------------------------------------------
# IBKR-flavored suggested actions
# ---------------------------------------------------------------------------

# Canned remediation text per error code. Concrete to IBKR's operational
# world — an agent reading these knows exactly which lever to pull next.
_SUGGESTED_ACTIONS: dict[str, str] = {
    "AUTH": (
        "TWS / IB Gateway is not connected, or the session is invalid. "
        "Verify the gateway is running on the configured host:port, the "
        "client_id is not in use by another process, and 2FA has been "
        "completed. Restart TWS/Gateway if the connection won't re-establish."
    ),
    "HARD_REJECT": (
        "IBKR OMS rejected the request. NOT retryable. Inspect "
        "`error.message` and the IBKR error code; common causes: no "
        "security definition, insufficient buying power, contract not "
        "tradable in session, duplicate order id. Fix params before retry."
    ),
    "PERMISSION": (
        "Your account lacks the required market-data / news / scanner "
        "subscription. See Client Portal → Settings → Market Data "
        "Subscriptions. Some actions (delayed data, US bundle) are free; "
        "others are per-month."
    ),
    "VALIDATION": (
        "Pre-flight validation rejected the request. Inspect "
        "`error.field_errors`; each entry names the field and rule that "
        "was violated. No IBKR call was made. Fix params and retry."
    ),
    "USAGE": (
        "Re-run with `--help` to see expected flags. Every destructive "
        "command requires explicit confirmation (`--yes`, and distinct "
        "tokens like `--confirm-panic` for cancel-all)."
    ),
    "UNAVAILABLE": (
        "IBKR connectivity lost or circuit-breaker open. Retry with "
        "exponential backoff. If it persists, check TWS status + IBKR "
        "system-status page (status.interactivebrokers.com)."
    ),
    "TRANSIENT": (
        "Transient error — typically historical-data pacing (~60s per-"
        "contract throttle) or rate-limit. Wait 10-60s and retry."
    ),
    "INTERNAL": (
        "Unexpected error inside trading-algo. Capture `request_id`, "
        "`message`, and file a bug. Do NOT silently retry."
    ),
    "SIGINT": (
        "User interrupt (Ctrl+C). No retry."
    ),
    "LEASE": (
        "Another process holds the trading lease. Retry after backoff "
        "or coordinate with the lease holder."
    ),
    "HALTED": (
        "Trading is administratively halted. Resume with "
        "`trading-algo resume --confirm-resume`."
    ),
    "MARKET_CLOSED": (
        "Regular session is closed for this exchange. Retry at next "
        "market open; for extended-hours trading, set outside_rth=True "
        "if the instrument and order type permit."
    ),
    "OUT_OF_WINDOW": (
        "Order falls outside the configured live-trade window. Adjust "
        "window configuration or retry later."
    ),
    "TIMEOUT": (
        "Deadline elapsed while waiting for a terminal state. The "
        "operation may still be in progress — poll order status "
        "separately before retrying."
    ),
    "GENERIC": (
        "Partial or non-specific failure. Inspect `error.message` for "
        "detail."
    ),
}


def suggested_action(error_code: str) -> str:
    return _SUGGESTED_ACTIONS.get(error_code, "See `error.message`.")


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _field_errors(exc: BaseException) -> list[dict]:
    """Extract per-field validation errors from known exception types."""
    errs = getattr(exc, "field_errors", None)
    if isinstance(errs, list):
        out = []
        for e in errs:
            if is_dataclass(e):
                out.append(asdict(e))
            elif isinstance(e, dict):
                out.append(e)
            else:
                out.append({"field": "unknown", "message": str(e)})
        return out
    return []


def build_error_payload(
    exc: BaseException,
    *,
    classified: ClassifiedError | None = None,
) -> dict:
    """Build the `error` sub-object for the envelope."""
    if classified is None:
        classified = classify_exception(exc)

    from trading_algo.redaction import redact_text
    message = redact_text(str(exc) or type(exc).__name__)
    payload: dict[str, Any] = {
        "code": classified.error_code,
        "exit_code": classified.exit_code,
        "exit_code_name": exit_code_name(classified.exit_code),
        "class": type(exc).__name__,
        "message": message,
        "retryable": classified.retryable,
        "suggested_action": suggested_action(classified.error_code),
    }

    ib_code = _extract_ib_error_code(exc)
    if ib_code is not None:
        payload["ib_error_code"] = ib_code

    fe = _field_errors(exc)
    if fe:
        payload["field_errors"] = fe

    return payload


# ---------------------------------------------------------------------------
# Emission + decorator
# ---------------------------------------------------------------------------

def emit_error(
    exc: BaseException,
    *,
    env: Envelope,
    stream=None,
) -> int:
    """Populate `env` with the structured error, emit to stderr, return the
    exit code. Caller pattern:

        try:
            ...
        except BaseException as e:
            return emit_error(e, env=env)
    """
    classified = classify_exception(exc)
    env.ok = False
    env.data = None
    env.error = build_error_payload(exc, classified=classified)
    finalize_envelope(env)

    out = stream if stream is not None else sys.stderr
    try:
        out.write(envelope_to_json(env))
        out.write("\n")
        out.flush()
    except BrokenPipeError:
        pass

    return classified.exit_code


def with_error_envelope(cmd: str) -> Callable:
    """Decorator: any exception in the wrapped command → structured stderr
    JSON + mapped exit code. SystemExit (argparse, gates) still propagates —
    the outer runner deals with it.

    Usage:

        @with_error_envelope("place-order")
        def cmd_place_order(args, *, env):
            ...
            env.data = {"order_id": 123}
            return 0
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(args) -> int:
            env = new_envelope(cmd)
            try:
                return fn(args, env=env)
            except SystemExit:
                raise
            except BaseException as exc:
                return emit_error(exc, env=env)
        return wrapper
    return decorator
