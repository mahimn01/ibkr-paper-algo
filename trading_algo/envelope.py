"""Structured output envelope for agent consumption.

Every non-streaming CLI command wraps its output as:

    {
      "ok": true,
      "cmd": "place-order",
      "schema_version": "1",
      "request_id": "01HNKXXXX...",       // ULID-ish, time-sortable
      "data": { ... },                     // command-specific payload
      "warnings": [{"code": "...", "message": "..."}],
      "meta": {
        "elapsed_ms": 423,
        "retries": 0,
        "replayed": false,
        "parent_request_id": null         // inherited from TRADING_PARENT_REQUEST_ID
      }
    }

On error, `ok=false`, `data=null`, and a top-level `error` object (see
`trading_algo.errors`) is added.

Environment:
  - `TRADING_NO_ENVELOPE=1`  — backwards-compat shim: emit raw data instead.
  - `TRADING_JSON=1`         — force JSON even when stdout is a TTY.
  - `TRADING_PARENT_REQUEST_ID` — propagated to `meta.parent_request_id`.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1"

ENV_NO_ENVELOPE = "TRADING_NO_ENVELOPE"
ENV_PARENT_REQUEST_ID = "TRADING_PARENT_REQUEST_ID"
ENV_FORCE_JSON = "TRADING_JSON"


# ---------------------------------------------------------------------------
# ULID-ish request IDs (48-bit time + 80-bit randomness, Crockford base32)
# ---------------------------------------------------------------------------

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_base32(n: int, length: int) -> str:
    out: list[str] = []
    for _ in range(length):
        out.append(_ULID_ALPHABET[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def new_request_id(clock_ms: int | None = None) -> str:
    """Return a 26-char Crockford-base32 time-sortable ID.

    First 10 chars encode milliseconds since epoch (48 bits); the last 16
    chars are crypto-random (80 bits). Sorting the strings preserves
    creation-time order. Collisions within the same millisecond are
    ~1-in-2^80 — effectively zero.
    """
    ms = clock_ms if clock_ms is not None else int(time.time() * 1000)
    rand = secrets.randbits(80)
    return _encode_base32(ms, 10) + _encode_base32(rand, 16)


def parent_request_id() -> str | None:
    """Caller's parent request ID (from env), if any.

    Lets a top-level agent propagate its turn-ID to every tool
    subprocess it spawns, so the full execution tree is reconstructable
    from audit logs.
    """
    v = os.getenv(ENV_PARENT_REQUEST_ID)
    return v.strip() if v else None


# ---------------------------------------------------------------------------
# Envelope dataclass
# ---------------------------------------------------------------------------

@dataclass
class Envelope:
    """Mutable envelope — handlers can attach warnings / meta during
    execution before serialising for emission."""
    ok: bool
    cmd: str
    request_id: str
    data: Any = None
    error: dict | None = None
    warnings: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        out: dict = {
            "ok": self.ok,
            "cmd": self.cmd,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "data": self.data,
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }
        if self.error is not None:
            out["error"] = self.error
        return out

    def add_warning(self, code: str, message: str, **extra: Any) -> None:
        entry = {"code": code, "message": message}
        entry.update(extra)
        self.warnings.append(entry)


def new_envelope(cmd: str) -> Envelope:
    rid = new_request_id()
    meta: dict = {"started_at_epoch_ms": int(time.time() * 1000)}
    parent = parent_request_id()
    if parent:
        meta["parent_request_id"] = parent
    return Envelope(ok=True, cmd=cmd, request_id=rid, meta=meta)


def finalize_envelope(env: Envelope) -> Envelope:
    """Populate `meta.elapsed_ms`. Idempotent."""
    start = env.meta.get("started_at_epoch_ms")
    if start is not None and "elapsed_ms" not in env.meta:
        env.meta["elapsed_ms"] = max(0, int(time.time() * 1000) - int(start))
    return env


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------

def envelopes_disabled() -> bool:
    raw = os.getenv(ENV_NO_ENVELOPE) or ""
    return raw.strip().lower() in ("1", "true", "yes", "on")


def json_is_default_for(stream=None) -> bool:
    """Default output format resolution.

    - `TRADING_JSON=1` forces JSON regardless of TTY.
    - If `stream` (default stdout) is NOT a TTY, default to JSON.
    - Otherwise default to table / native for human terminals.
    """
    if (os.getenv(ENV_FORCE_JSON) or "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    s = stream if stream is not None else sys.stdout
    try:
        return not s.isatty()
    except (AttributeError, ValueError):
        return True  # closed / exotic stream → safer to emit JSON


def envelope_to_json(env: Envelope) -> str:
    """Serialise to pretty JSON. Falls through to str() on unknown objects
    so a single unserialisable value never crashes emission — the agent
    needs to receive *something*, even if degraded."""
    return json.dumps(env.to_dict(), indent=2, default=str, ensure_ascii=False)
