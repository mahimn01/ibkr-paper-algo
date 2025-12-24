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
    ) -> None:
        req_n = request.normalized()
        inst = req_n.instrument
        self._conn.execute(
            "INSERT INTO orders(run_id, ts_epoch_s, broker, order_id, instrument_kind, instrument_symbol, side, quantity, order_type, request_json, status) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        self._conn.commit()

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
            """
        )
        cur = self._conn.execute("SELECT COUNT(*) FROM schema_version")
        if int(cur.fetchone()[0]) == 0:
            self._conn.execute("INSERT INTO schema_version(version) VALUES(1)")
        self._conn.commit()


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
