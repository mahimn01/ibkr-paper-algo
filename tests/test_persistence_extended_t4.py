"""Tests for T4.1 SqliteStore extended columns + migration."""

from __future__ import annotations

import sqlite3

import pytest

from trading_algo.broker.base import OrderRequest
from trading_algo.config import TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.persistence import SqliteStore


def _make_request(symbol: str = "AAPL", side: str = "BUY", qty: float = 10.0) -> OrderRequest:
    return OrderRequest(
        instrument=InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD"),
        side=side, quantity=qty, order_type="MKT", limit_price=None,
        stop_price=None, tif="DAY",
    )


class TestSchemaExtension:
    def test_new_db_has_all_columns(self, tmp_path) -> None:
        db = tmp_path / "new.db"
        store = SqliteStore(str(db))
        try:
            cur = store._conn.execute("PRAGMA table_info(orders)")
            cols = {row[1] for row in cur.fetchall()}
            for expected in ("perm_id", "order_ref", "account", "strategy_id",
                             "agent_id", "group_id", "idempotency_key"):
                assert expected in cols
        finally:
            store.close()

    def test_old_db_migrated_idempotently(self, tmp_path) -> None:
        db = tmp_path / "old.db"
        # Simulate an old-schema DB (only original columns).
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE schema_version(version INTEGER);
            CREATE TABLE runs(id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_epoch_s REAL NOT NULL, ended_epoch_s REAL,
                config_json TEXT NOT NULL);
            CREATE TABLE orders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, ts_epoch_s REAL NOT NULL,
                broker TEXT NOT NULL, order_id TEXT NOT NULL,
                instrument_kind TEXT, instrument_symbol TEXT,
                side TEXT, quantity REAL, order_type TEXT,
                request_json TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE decisions(id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, ts_epoch_s REAL NOT NULL,
                strategy TEXT NOT NULL, intent_json TEXT NOT NULL,
                accepted INTEGER NOT NULL, reason TEXT);
            CREATE TABLE order_status_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, ts_epoch_s REAL NOT NULL,
                broker TEXT NOT NULL, order_id TEXT NOT NULL,
                status TEXT NOT NULL, filled REAL, remaining REAL,
                avg_fill_price REAL);
            CREATE TABLE errors(id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, ts_epoch_s REAL NOT NULL,
                where_text TEXT NOT NULL, message TEXT NOT NULL);
            CREATE TABLE actions(id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, ts_epoch_s REAL NOT NULL,
                actor TEXT NOT NULL, payload_json TEXT NOT NULL,
                accepted INTEGER NOT NULL, reason TEXT);
            INSERT INTO schema_version(version) VALUES(1);
        """)
        con.commit()
        con.close()

        # Open with new SqliteStore — should ALTER TABLE idempotently.
        store = SqliteStore(str(db))
        try:
            cur = store._conn.execute("PRAGMA table_info(orders)")
            cols = {row[1] for row in cur.fetchall()}
            for expected in ("perm_id", "order_ref", "account", "group_id",
                             "idempotency_key"):
                assert expected in cols
            # Re-open is idempotent.
        finally:
            store.close()

        store2 = SqliteStore(str(db))
        try:
            pass  # re-opens fine
        finally:
            store2.close()


class TestLogOrderWithExtendedFields:
    def test_writes_all_fields(self, tmp_path) -> None:
        store = SqliteStore(str(tmp_path / "x.db"))
        run_id = store.start_run(TradingConfig.from_env())
        try:
            store.log_order(
                run_id, broker="ibkr", order_id="42",
                request=_make_request(), status="Submitted",
                perm_id="P999", order_ref="TA_abc123",
                account="DU1234567", strategy_id="wheel",
                agent_id="agent-01", group_id="basket-7",
                idempotency_key="deadbeef",
            )
            cur = store._conn.execute(
                "SELECT perm_id, order_ref, account, strategy_id, agent_id, "
                "group_id, idempotency_key FROM orders WHERE order_id='42'"
            )
            row = cur.fetchone()
            assert row == ("P999", "TA_abc123", "DU1234567", "wheel",
                           "agent-01", "basket-7", "deadbeef")
        finally:
            store.close()

    def test_omitting_fields_stores_null(self, tmp_path) -> None:
        store = SqliteStore(str(tmp_path / "y.db"))
        run_id = store.start_run(TradingConfig.from_env())
        try:
            store.log_order(
                run_id, broker="sim", order_id="7",
                request=_make_request(), status="Filled",
            )
            cur = store._conn.execute(
                "SELECT perm_id, order_ref, account, strategy_id, agent_id, "
                "group_id, idempotency_key FROM orders WHERE order_id='7'"
            )
            row = cur.fetchone()
            assert row == (None, None, None, None, None, None, None)
        finally:
            store.close()


class TestOrdersByGroup:
    def test_group_aggregation(self, tmp_path) -> None:
        store = SqliteStore(str(tmp_path / "g.db"))
        run_id = store.start_run(TradingConfig.from_env())
        try:
            store.log_order(run_id, broker="sim", order_id="1",
                            request=_make_request(), status="Submitted",
                            group_id="G1")
            store.log_order(run_id, broker="sim", order_id="2",
                            request=_make_request("MSFT"), status="Filled",
                            group_id="G1")
            store.log_order(run_id, broker="sim", order_id="3",
                            request=_make_request("TSLA"), status="Submitted",
                            group_id="G2")

            g1 = store.orders_by_group("G1")
            assert len(g1) == 2
            assert {r["order_id"] for r in g1} == {"1", "2"}

            groups = store.list_groups()
            names = {g["group_id"] for g in groups}
            assert names == {"G1", "G2"}
            # G1 has 2 rows.
            g1_entry = next(g for g in groups if g["group_id"] == "G1")
            assert g1_entry["n"] == 2
        finally:
            store.close()


class TestIdempotencyKeyLookup:
    def test_most_recent_row_returned(self, tmp_path) -> None:
        store = SqliteStore(str(tmp_path / "idem.db"))
        run_id = store.start_run(TradingConfig.from_env())
        try:
            store.log_order(run_id, broker="sim", order_id="first",
                            request=_make_request(), status="Submitted",
                            idempotency_key="K1")
            store.log_order(run_id, broker="sim", order_id="second",
                            request=_make_request(), status="Filled",
                            idempotency_key="K1")

            row = store.order_by_idempotency_key("K1")
            assert row is not None
            assert row["order_id"] == "second"  # most recent

            assert store.order_by_idempotency_key("unknown") is None
        finally:
            store.close()
