"""Shared CLI entry-point runner.

Every top-level `main()` in `cli.py`, `ibkr_tool.py`, `flex_tool.py`
delegates to `run_command()`. It handles, uniformly:

- Minting a ULID `request_id` for the invocation.
- Capturing argparse args (redacted) for the NDJSON audit log.
- Executing the command's handler.
- Classifying ANY exception via `exit_codes.classify_exception`.
- On failure: emitting a structured stderr JSON blob (see
  `trading_algo.errors`) with error code + suggested_action, so agents
  consuming stderr never have to regex.
- Writing exactly one audit line per invocation, whether success or
  failure — so the audit log is the ground truth for what happened,
  visible even if stdout / stderr were redirected away.
- Returning the correct exit code.

Agents branch on exit code. Humans read stderr. Compliance reads the
audit log. Three audiences; one runner.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Callable

from trading_algo.audit import log_command
from trading_algo.envelope import new_envelope, new_request_id, parent_request_id
from trading_algo.errors import emit_error
from trading_algo.exit_codes import (
    ClassifiedError,
    classify_exception,
    exit_code_name,
    GENERIC,
    SIGINT,
    USAGE,
)

log = logging.getLogger(__name__)


def _args_to_dict(args: argparse.Namespace) -> dict:
    """argparse Namespace → plain dict, excluding the `func` handler."""
    out: dict = {}
    for k, v in vars(args).items():
        if k == "func" or callable(v):
            continue
        out[k] = v
    return out


def _resolve_cmd_name(args: argparse.Namespace, default: str) -> str:
    """Best-effort identification of which subcommand ran.

    argparse conventionally sets `args.cmd` or `args.subcommand` via
    `add_subparsers(dest="cmd")`. Fall back to the caller-supplied
    default if nothing else works.
    """
    for attr in ("cmd", "command", "subcommand"):
        v = getattr(args, attr, None)
        if isinstance(v, str) and v:
            return v
    # Derive from the handler function name: `_cmd_place_order` → `place_order`.
    fn = getattr(args, "func", None)
    if fn is not None:
        name = getattr(fn, "__name__", "")
        if name.startswith("_cmd_"):
            return name[5:].replace("_", "-")
        if name.startswith("cmd_"):
            return name[4:].replace("_", "-")
    return default


def run_command(
    args: argparse.Namespace,
    *,
    default_cmd_name: str = "unknown",
    strategy_id_env: str = "TRADING_STRATEGY_ID",
    agent_id_env: str = "TRADING_AGENT_ID",
) -> int:
    """Execute `args.func(args)` wrapped in audit + error classification.

    Returns the exit code. Never re-raises (SystemExit from argparse is
    caught and passed through preserving its code).
    """
    cmd_name = _resolve_cmd_name(args, default_cmd_name)
    request_id = new_request_id()
    args_dict = _args_to_dict(args)

    # Expose request_id to the handler so it can propagate to sub-
    # operations (Kite-style parent_request_id chain).
    os.environ["TRADING_CURRENT_REQUEST_ID"] = request_id

    started_ms = int(time.time() * 1000)
    exit_code: int = 0
    error_code: str | None = None

    try:
        rc = args.func(args)
        exit_code = int(rc) if isinstance(rc, int) else 0
    except SystemExit as exc:
        # argparse / explicit `raise SystemExit` — preserve exit code.
        if isinstance(exc.code, int):
            exit_code = exc.code
        elif isinstance(exc.code, str):
            # String SystemExit is how we signal "Refusing to place
            # without --yes" etc. — usage-level, with message to stderr.
            import sys as _sys
            try:
                _sys.stderr.write(str(exc.code) + "\n")
                _sys.stderr.flush()
            except Exception:
                pass
            exit_code = USAGE
            error_code = "USAGE"
        else:
            exit_code = 0 if exc.code is None else GENERIC
    except KeyboardInterrupt:
        exit_code = SIGINT
        error_code = "SIGINT"
    except BaseException as exc:
        # Route every other exception through the structured-error pipeline.
        env = new_envelope(cmd_name)
        env.request_id = request_id
        classified: ClassifiedError = classify_exception(exc)
        error_code = classified.error_code
        # emit_error writes to stderr and returns the classified exit code.
        exit_code = emit_error(exc, env=env)

    # Audit — always, even on failure. Wrap in try so an audit I/O error
    # doesn't propagate into the user-visible exit code.
    try:
        elapsed_ms = int(time.time() * 1000) - started_ms
        log_command(
            cmd=cmd_name,
            request_id=request_id,
            args=args_dict,
            exit_code=exit_code,
            error_code=error_code,
            elapsed_ms=elapsed_ms,
            parent_request_id=parent_request_id(),
            strategy_id=os.getenv(strategy_id_env),
            agent_id=os.getenv(agent_id_env),
        )
    except Exception as log_exc:
        # Never let audit I/O crash the process — the command already ran.
        log.warning("audit log_command failed: %s", log_exc)

    return exit_code
