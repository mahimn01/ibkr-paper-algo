"""NDJSON audit log — every CLI invocation appends one JSON line.

For an IBKR trading operation handling real capital, the audit log is the
compliance artifact: per US broker-dealer rules (SEC 17a-4 / FINRA) and
IBKR's own record-keeping, every algorithmic order must have a retained
audit trail with timestamp, account identifier, request parameters, and
outcome. Retention default: 7 years (longer than 17a-4 requires, shorter
than FINRA best practice of 8).

Why NDJSON + daily rotation (rather than only SQLite):
- **Atomic appends**. POSIX guarantees writes under PIPE_BUF (4096 bytes
  on Linux/macOS) are atomic when the file is opened O_APPEND — concurrent
  writes never interleave partial lines.
- **Trivial archival**. One file per trading day → `tar` one file per day.
- **jq / grep / awk**-compatible out of the box.
- **Independent of SqliteStore**. The SQLite audit captures engine-driven
  orders + decisions; this NDJSON log captures EVERY CLI invocation
  (including failures that never reached engine / OMS), which is the
  audit surface the compliance review cares about.

Audit line shape:

    {
      "ts": "2026-04-21T10:15:30.123-04:00",   // IBKR US-local
      "ts_epoch_ms": 1745234130123,
      "request_id": "01JBP...",
      "parent_request_id": "01JBO..." | null,
      "cmd": "place-order",
      "args": {... redacted ...},
      "exit_code": 0,
      "error_code": null | "HARD_REJECT" | ...,
      "elapsed_ms": 234,
      "ib_order_id": 42 | null,
      "perm_id": "...",
      "order_ref": "TA....",
      "account": "U1234567" | null,
      "strategy_id": null,
      "agent_id": null
    }
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_AUDIT_DIR = Path("data/audit")
DEFAULT_RETENTION_YEARS = 7  # SEC 17a-4 minimum is 6; we use 7 as enterprise default


def audit_dir() -> Path:
    raw = os.getenv("TRADING_AUDIT_DIR")
    return Path(raw) if raw else DEFAULT_AUDIT_DIR


def _utc_date_from_wall(when: datetime | date | None) -> date:
    """Normalise to calendar date. Default: today's local date.

    We rotate by local date rather than UTC because US market operations
    align with local days — a US trader's 2026-04-21 trading day is one
    file, not split by a 20:00 local = 00:00 UTC boundary.
    """
    if when is None:
        return datetime.now().date()
    if isinstance(when, datetime):
        return when.date()
    return when


def audit_path_for(when: datetime | date | None = None, *, root: Path | None = None) -> Path:
    d = _utc_date_from_wall(when)
    root = root or audit_dir()
    return root / f"{d.isoformat()}.jsonl"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

# Keys of common credentials we redact wholesale if they appear in args.
_REDACT_KEYS = frozenset({
    "api_secret", "access_token", "refresh_token", "password", "pin",
    "flex_token", "IBKR_FLEX_TOKEN", "FLEX_TOKEN",
    "order_token", "confirm_token", "TRADING_ORDER_TOKEN",
})


def _redact_value(v: Any) -> Any:
    """Shallow-redact token-shaped strings in args values.

    We don't recurse JSON deeply — args are argparse Namespaces which are
    flat. If an agent passes a dict/list (e.g. `--basket-json`), we
    serialise → redact any known-key values → re-parse.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        # Avoid `isinstance(True, int)` catching here.
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        # Strings are just emitted as-is; known-secret strings can be
        # handled at the key level below.
        return v
    if isinstance(v, (list, tuple, set)):
        return [_redact_value(x) for x in v]
    if isinstance(v, dict):
        return {
            str(k): (
                "***REDACTED***"
                if (isinstance(k, str) and k in _REDACT_KEYS)
                else _redact_value(val)
            )
            for k, val in v.items()
        }
    # Unknown — stringify.
    return str(v)


def _redact_args(args: dict) -> dict:
    """Shallow redaction pass over known credential keys, plus a pattern-
    based pass (via `trading_algo.redaction`) over every string value so
    tokens embedded in free-form args (e.g. `--note "used token XYZ"`)
    still get scrubbed.
    """
    from trading_algo.redaction import redact_text

    def _scrub_string_leaves(v: Any) -> Any:
        if isinstance(v, str):
            return redact_text(v)
        if isinstance(v, (list, tuple)):
            return [_scrub_string_leaves(x) for x in v]
        if isinstance(v, dict):
            return {k: _scrub_string_leaves(val) for k, val in v.items()}
        return v

    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(k, str) and k in _REDACT_KEYS:
            out[k] = "***REDACTED***"
        else:
            out[k] = _scrub_string_leaves(_redact_value(v))
    return out


@dataclass
class AuditEntry:
    ts: str
    ts_epoch_ms: int
    request_id: str
    cmd: str
    args: dict
    exit_code: int | None = None
    error_code: str | None = None
    elapsed_ms: int | None = None
    ib_order_id: int | None = None
    perm_id: str | None = None
    order_ref: str | None = None
    account: str | None = None
    parent_request_id: str | None = None
    strategy_id: str | None = None
    agent_id: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {
            "ts": self.ts,
            "ts_epoch_ms": self.ts_epoch_ms,
            "request_id": self.request_id,
            "parent_request_id": self.parent_request_id,
            "cmd": self.cmd,
            "args": self.args,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "elapsed_ms": self.elapsed_ms,
            "ib_order_id": self.ib_order_id,
            "perm_id": self.perm_id,
            "order_ref": self.order_ref,
            "account": self.account,
            "strategy_id": self.strategy_id,
            "agent_id": self.agent_id,
        }
        if self.extra:
            out.update(self.extra)
        return out


def _atomic_append_line(path: Path, line: str) -> None:
    """Append a single newline-terminated line via one write() syscall.

    POSIX guarantees writes <PIPE_BUF atomic under O_APPEND. Lines exceeding
    4 KB could theoretically interleave; we don't truncate because losing
    audit data is a bigger harm than the negligible odds of interleave at
    that size.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(path), flags, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def write_entry(entry: AuditEntry, *, root: Path | None = None) -> Path:
    now_dt = datetime.now().astimezone()
    if not entry.ts:
        entry.ts = now_dt.isoformat(timespec="milliseconds")
    if not entry.ts_epoch_ms:
        entry.ts_epoch_ms = int(now_dt.timestamp() * 1000)
    path = audit_path_for(now_dt, root=root)
    line = json.dumps(entry.to_dict(), default=str, ensure_ascii=False) + "\n"
    _atomic_append_line(path, line)
    return path


def log_command(
    *,
    cmd: str,
    request_id: str,
    args: dict,
    exit_code: int | None = None,
    error_code: str | None = None,
    elapsed_ms: int | None = None,
    ib_order_id: int | None = None,
    perm_id: str | None = None,
    order_ref: str | None = None,
    account: str | None = None,
    parent_request_id: str | None = None,
    strategy_id: str | None = None,
    agent_id: str | None = None,
    extra: dict | None = None,
    root: Path | None = None,
) -> Path:
    entry = AuditEntry(
        ts="",
        ts_epoch_ms=0,
        request_id=request_id,
        cmd=cmd,
        args=_redact_args(args),
        exit_code=exit_code,
        error_code=error_code,
        elapsed_ms=elapsed_ms,
        ib_order_id=ib_order_id,
        perm_id=perm_id,
        order_ref=order_ref,
        account=account,
        parent_request_id=parent_request_id,
        strategy_id=strategy_id,
        agent_id=agent_id,
        extra=extra or {},
    )
    return write_entry(entry, root=root)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def iter_entries(
    *,
    since: datetime | date | None = None,
    until: datetime | date | None = None,
    cmd: str | None = None,
    outcome: str | None = None,   # "ok" | "error"
    root: Path | None = None,
) -> Iterable[dict]:
    """Iterate audit entries in chronological order.

    Date filters are inclusive. Outcome: 'ok' → exit_code 0; 'error' →
    non-zero exit_code (None also treated as error — rare crash-before-log
    case).
    """
    root = root or audit_dir()
    if not root.exists():
        return
    files = sorted(root.glob("*.jsonl"))
    for f in files:
        try:
            day = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if since is not None:
            since_d = since.date() if isinstance(since, datetime) else since
            if day < since_d:
                continue
        if until is not None:
            until_d = until.date() if isinstance(until, datetime) else until
            if day > until_d:
                continue

        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if cmd is not None and entry.get("cmd") != cmd:
                        continue
                    if outcome == "ok" and entry.get("exit_code") != 0:
                        continue
                    if outcome == "error" and entry.get("exit_code") == 0:
                        continue
                    yield entry
        except OSError:
            continue


def tail(
    n: int = 100,
    *,
    cmd: str | None = None,
    outcome: str | None = None,
    root: Path | None = None,
) -> list[dict]:
    from collections import deque
    buf: deque[dict] = deque(maxlen=max(1, n))
    for e in iter_entries(cmd=cmd, outcome=outcome, root=root):
        buf.append(e)
    return list(buf)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def purge_older_than(days: int, *, root: Path | None = None) -> int:
    """Delete audit files older than `days` days. Default retention 7y."""
    from datetime import timedelta
    root = root or audit_dir()
    if not root.exists():
        return 0
    cutoff = datetime.now().date() - timedelta(days=days)
    deleted = 0
    for f in root.glob("*.jsonl"):
        try:
            day = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if day < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted
