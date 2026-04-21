from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from typing import Any

from trading_algo.broker.base import OrderRequest, OrderStatus
from trading_algo.config import TradingConfig
from trading_algo.orders import TradeIntent


class SqliteStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def start_run(self, cfg: TradingConfig) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO runs(started_epoch_s, config_json) VALUES(?, ?)",
            (time.time(), json.dumps(asdict(cfg), sort_keys=True)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def end_run(self, run_id: int) -> None:
        self._conn.execute("UPDATE runs SET ended_epoch_s=? WHERE id=?", (time.time(), int(run_id)))
        self._conn.commit()

    def log_decision(
        self,
        run_id: int,
        *,
        strategy: str,
        intent: TradeIntent,
        accepted: bool,
        reason: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO decisions(run_id, ts_epoch_s, strategy, intent_json, accepted, reason) VALUES(?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(strategy),
                json.dumps(_to_jsonable(asdict(intent)), sort_keys=True),
                1 if accepted else 0,
                reason,
            ),
        )
        self._conn.commit()

    def log_order(
        self,
        run_id: int,
        *,
        broker: str,
        order_id: str,
        request: OrderRequest,
        status: str,
        perm_id: str | None = None,
        order_ref: str | None = None,
        account: str | None = None,
        strategy_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        req_n = request.normalized()
        inst = req_n.instrument
        self._conn.execute(
            "INSERT INTO orders("
            "run_id, ts_epoch_s, broker, order_id, instrument_kind, instrument_symbol, "
            "side, quantity, order_type, request_json, status, "
            "perm_id, order_ref, account, strategy_id, agent_id, group_id, idempotency_key"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(broker),
                str(order_id),
                str(inst.kind),
                str(inst.symbol),
                str(req_n.side),
                float(req_n.quantity),
                str(req_n.order_type),
                json.dumps(_to_jsonable(asdict(req_n)), sort_keys=True),
                str(status),
                perm_id,
                order_ref,
                account,
                strategy_id,
                agent_id,
                group_id,
                idempotency_key,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # T4.1 accessors for richer order lookups
    # ------------------------------------------------------------------

    def orders_by_group(self, group_id: str) -> list[dict]:
        """Return all order rows in a group ordered by ts ascending."""
        cur = self._conn.execute(
            "SELECT id, run_id, ts_epoch_s, broker, order_id, "
            "instrument_kind, instrument_symbol, side, quantity, order_type, "
            "status, perm_id, order_ref, account, strategy_id, agent_id, "
            "group_id, idempotency_key "
            "FROM orders WHERE group_id=? ORDER BY ts_epoch_s ASC, id ASC",
            (str(group_id),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def list_groups(self) -> list[dict]:
        """Aggregate summary of every distinct group_id seen."""
        cur = self._conn.execute(
            "SELECT group_id, COUNT(*) AS n, "
            "MIN(ts_epoch_s) AS first_ts, MAX(ts_epoch_s) AS last_ts, "
            "GROUP_CONCAT(DISTINCT status) AS statuses "
            "FROM orders WHERE group_id IS NOT NULL GROUP BY group_id "
            "ORDER BY MAX(ts_epoch_s) DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def order_by_idempotency_key(self, key: str) -> dict | None:
        """Return the most recent order row matching `idempotency_key`."""
        cur = self._conn.execute(
            "SELECT id, order_id, status, ts_epoch_s FROM orders "
            "WHERE idempotency_key=? ORDER BY ts_epoch_s DESC, id DESC LIMIT 1",
            (str(key),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def update_order_status(self, order_id: str, status: str) -> None:
        self._conn.execute("UPDATE orders SET status=? WHERE order_id=?", (str(status), str(order_id)))
        self._conn.commit()

    def list_non_terminal_order_ids(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT order_id, status FROM orders")
        order_ids: list[str] = []
        for oid, st in cur.fetchall():
            if oid and not _is_terminal_status(str(st)):
                order_ids.append(str(oid))
        return order_ids

    def get_latest_status(self, order_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT status FROM orders WHERE order_id=? ORDER BY ts_epoch_s DESC, id DESC LIMIT 1",
            (str(order_id),),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None

    def log_order_status_event(self, run_id: int, broker: str, st: OrderStatus) -> None:
        self._conn.execute(
            "INSERT INTO order_status_events(run_id, ts_epoch_s, broker, order_id, status, filled, remaining, avg_fill_price) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(broker),
                str(st.order_id),
                str(st.status),
                st.filled,
                st.remaining,
                st.avg_fill_price,
            ),
        )
        self._conn.commit()

    def log_error(self, run_id: int, *, where: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO errors(run_id, ts_epoch_s, where_text, message) VALUES(?, ?, ?, ?)",
            (int(run_id), time.time(), str(where), str(message)),
        )
        self._conn.commit()

    def log_action(
        self,
        run_id: int,
        *,
        actor: str,
        payload: dict[str, Any],
        accepted: bool,
        reason: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO actions(run_id, ts_epoch_s, actor, payload_json, accepted, reason) VALUES(?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                time.time(),
                str(actor),
                json.dumps(_to_jsonable(payload), sort_keys=True),
                1 if accepted else 0,
                reason,
            ),
        )
        self._conn.commit()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version(
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_epoch_s REAL NOT NULL,
                ended_epoch_s REAL,
                config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                strategy TEXT NOT NULL,
                intent_json TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS orders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                broker TEXT NOT NULL,
                order_id TEXT NOT NULL,
                instrument_kind TEXT,
                instrument_symbol TEXT,
                side TEXT,
                quantity REAL,
                order_type TEXT,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                perm_id TEXT,
                order_ref TEXT,
                account TEXT,
                strategy_id TEXT,
                agent_id TEXT,
                group_id TEXT,
                idempotency_key TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(instrument_symbol);

            CREATE TABLE IF NOT EXISTS order_status_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                broker TEXT NOT NULL,
                order_id TEXT NOT NULL,
                status TEXT NOT NULL,
                filled REAL,
                remaining REAL,
                avg_fill_price REAL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_order_status_events_order_id ON order_status_events(order_id);

            CREATE TABLE IF NOT EXISTS errors(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                where_text TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS actions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ts_epoch_s REAL NOT NULL,
                actor TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        # T4.1 — idempotent migration: older DBs won't have these columns yet.
        self._add_column_if_missing("orders", "perm_id", "TEXT")
        self._add_column_if_missing("orders", "order_ref", "TEXT")
        self._add_column_if_missing("orders", "account", "TEXT")
        self._add_column_if_missing("orders", "strategy_id", "TEXT")
        self._add_column_if_missing("orders", "agent_id", "TEXT")
        self._add_column_if_missing("orders", "group_id", "TEXT")
        self._add_column_if_missing("orders", "idempotency_key", "TEXT")
        # Indexes (safe once columns exist).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_group_id ON orders(group_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_idempotency_key ON orders(idempotency_key)"
        )

        cur = self._conn.execute("SELECT COUNT(*) FROM schema_version")
        if int(cur.fetchone()[0]) == 0:
            self._conn.execute("INSERT INTO schema_version(version) VALUES(2)")
        self._conn.commit()

    def _add_column_if_missing(self, table: str, col: str, col_type: str) -> None:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        if col not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    return str(obj)


def _is_terminal_status(status: str) -> bool:
    s = str(status).strip()
    return s in {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}
