"""Durable idempotency cache for IBKR write commands.

Problem: IBKR has no strong server-side dedup on `placeOrder`. If an agent
invokes `place-order --order-ref ABC`, the call succeeds, the broker
accepts the order, but the CLI process crashes before writing the
response, a naive retry re-transmits the order. Double fill.

This module provides the cross-process crash-recovery layer:

- Agent supplies `--idempotency-key KEY` (or the CLI derives one from
  `strategy_id + intent`).
- On first invocation: `INSERT OR IGNORE` a row into `writes` with cmd,
  request args (JSON), and `first_seen_at` timestamp.
- Deterministically derive `orderRef` = `BLAKE2b(key)` (30-char hex) and
  pass it to IBKR.
- On completion: `UPDATE` the row with `result_json`, `completed_at`,
  `exit_code`.
- Subsequent invocations with the same key short-circuit with the stored
  result and `meta.replayed=true`.
- Crash between start and completion: row has no `completed_at`. Retry
  with same key sees "started but not completed" — the
  `IdempotentOrderPlacer` will also query IBKR's `reqOpenOrders` /
  `reqCompletedOrders` for our derived orderRef before re-transmitting.

Storage: `data/idempotency.sqlite`; override with `TRADING_IDEMPOTENCY_PATH`.
WAL mode for concurrent-reader safety.

IBKR's `orderRef` field:
- Free-text, passed through to the exchange as a user identifier.
- Max ~40 chars (some exchanges truncate further).
- Survives TWS restarts (part of the order record, not client state).
- Reported in `openTrade.order.orderRef` + in Flex statements.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_PATH = Path("data/idempotency.sqlite")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS writes (
  key                 TEXT PRIMARY KEY,
  cmd                 TEXT NOT NULL,
  request_json        TEXT NOT NULL,
  first_seen_at_ms    INTEGER NOT NULL,
  result_json         TEXT,
  completed_at_ms     INTEGER,
  exit_code           INTEGER,
  ib_order_id         INTEGER,
  perm_id             TEXT,
  order_ref           TEXT,
  request_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_writes_cmd ON writes(cmd);
CREATE INDEX IF NOT EXISTS idx_writes_completed ON writes(completed_at_ms);
CREATE INDEX IF NOT EXISTS idx_writes_order_ref ON writes(order_ref);
"""


def default_path() -> Path:
    raw = os.getenv("TRADING_IDEMPOTENCY_PATH")
    return Path(raw) if raw else DEFAULT_PATH


@dataclass(frozen=True)
class WriteRecord:
    key: str
    cmd: str
    request_json: str
    first_seen_at_ms: int
    result_json: str | None = None
    completed_at_ms: int | None = None
    exit_code: int | None = None
    ib_order_id: int | None = None
    perm_id: str | None = None
    order_ref: str | None = None
    request_id: str | None = None

    @property
    def completed(self) -> bool:
        return self.completed_at_ms is not None

    @property
    def result(self) -> Any:
        if self.result_json is None:
            return None
        try:
            return json.loads(self.result_json)
        except json.JSONDecodeError:
            return self.result_json


class IdempotencyStore:
    """Thread-safe SQLite-backed idempotency cache."""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path is not None else default_path()
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            try:
                os.chmod(self._path.parent, 0o700)
            except OSError:
                pass
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(str(self._path), timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def lookup(self, key: str) -> WriteRecord | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT key, cmd, request_json, first_seen_at_ms, result_json, "
                "completed_at_ms, exit_code, ib_order_id, perm_id, order_ref, request_id "
                "FROM writes WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return WriteRecord(
                key=row[0], cmd=row[1], request_json=row[2],
                first_seen_at_ms=int(row[3]),
                result_json=row[4],
                completed_at_ms=int(row[5]) if row[5] is not None else None,
                exit_code=int(row[6]) if row[6] is not None else None,
                ib_order_id=int(row[7]) if row[7] is not None else None,
                perm_id=row[8],
                order_ref=row[9],
                request_id=row[10],
            )

    def record_attempt(
        self,
        *,
        key: str,
        cmd: str,
        request: Any,
        order_ref: str | None = None,
        request_id: str | None = None,
    ) -> bool:
        """Register an in-flight attempt. Returns True on first insert,
        False on duplicate key."""
        now_ms = int(time.time() * 1000)
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO writes "
                "(key, cmd, request_json, first_seen_at_ms, order_ref, request_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, cmd, json.dumps(request, default=str), now_ms,
                 order_ref, request_id),
            )
            return cur.rowcount == 1

    def record_completion(
        self,
        *,
        key: str,
        result: Any,
        exit_code: int,
        ib_order_id: int | None = None,
        perm_id: str | None = None,
    ) -> None:
        """Mark the attempt complete. Safe to call twice."""
        now_ms = int(time.time() * 1000)
        with self._conn() as c:
            c.execute(
                "UPDATE writes SET result_json=?, completed_at_ms=?, "
                "exit_code=?, "
                "ib_order_id=COALESCE(?, ib_order_id), "
                "perm_id=COALESCE(?, perm_id) "
                "WHERE key=?",
                (json.dumps(result, default=str), now_ms, exit_code,
                 ib_order_id, perm_id, key),
            )

    def find_by_order_ref(self, order_ref: str) -> WriteRecord | None:
        """Reverse lookup: we received an orderRef from IBKR, find the
        local record. Useful when reconciling a crashed retry.

        We resolve the key inside the connection scope, release the lock,
        then call `lookup` fresh. Calling `lookup` while still holding
        `self._lock` would deadlock — `threading.Lock` is not reentrant.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT key FROM writes WHERE order_ref = ? LIMIT 1",
                (order_ref,),
            ).fetchone()
        # Lock released here before the second lookup() call.
        if row is None:
            return None
        return self.lookup(row[0])

    def purge_older_than(self, cutoff_ms: int) -> int:
        """Delete completed rows older than cutoff. In-flight rows are
        never purged — they represent potential ghost orders needing
        reconciliation."""
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM writes WHERE completed_at_ms IS NOT NULL AND "
                "first_seen_at_ms < ?",
                (cutoff_ms,),
            )
            return int(cur.rowcount)


# ---------------------------------------------------------------------------
# orderRef derivation
# ---------------------------------------------------------------------------

def derive_order_ref(
    key: str,
    *,
    prefix: str = "TA",
    length: int = 30,
) -> str:
    """Deterministically derive an IBKR `orderRef` from an idempotency key.

    IBKR orderRef is free-text and travels with the order through the
    exchange. We use BLAKE2b for a collision-resistant, reproducible
    mapping. Same key → same orderRef, which makes the orderbook
    pre-check (find_trade_by_ref) able to match a prior-attempt's order
    across process restarts.

    Default length 30 keeps well under IBKR's ~40-char limit.
    `prefix="TA"` (trading-algo) distinguishes from Kite's `"KA"` tags.
    """
    if length < 6 or length > 40:
        raise ValueError(f"length must be in [6, 40], got {length}")
    body_len = length - len(prefix)
    if body_len < 4:
        raise ValueError(f"prefix {prefix!r} too long for length {length}")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=32).hexdigest().upper()
    ref = f"{prefix}{digest[:body_len]}"
    assert len(ref) == length, f"bad derivation: {ref!r}"
    assert all(c.isalnum() for c in ref), f"non-alnum in ref: {ref!r}"
    return ref
