"""Tests for T4.2 groups-list / groups-show CLI commands."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

import pytest

from trading_algo import cli
from trading_algo.broker.base import OrderRequest
from trading_algo.config import TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.persistence import SqliteStore


def _request(symbol: str = "AAPL") -> OrderRequest:
    return OrderRequest(
        instrument=InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD"),
        side="BUY", quantity=10, order_type="MKT",
        limit_price=None, stop_price=None, tif="DAY",
    )


def _args(**kw) -> argparse.Namespace:
    defaults = {
        "broker": "sim", "ibkr_host": None, "ibkr_port": None,
        "ibkr_client_id": None, "paper": False, "live": False,
        "dry_run": False, "no_dry_run": False, "allow_live": False,
        "confirm_token": None, "explain": False,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


@pytest.fixture
def populated_db(tmp_path):
    path = tmp_path / "oms.db"
    store = SqliteStore(str(path))
    try:
        rid = store.start_run(TradingConfig.from_env())
        store.log_order(rid, broker="sim", order_id="A1",
                        request=_request("AAPL"), status="Submitted",
                        group_id="basket-1")
        store.log_order(rid, broker="sim", order_id="A2",
                        request=_request("MSFT"), status="Filled",
                        group_id="basket-1")
        store.log_order(rid, broker="sim", order_id="B1",
                        request=_request("TSLA"), status="Cancelled",
                        group_id="basket-2")
        store.log_order(rid, broker="sim", order_id="X1",
                        request=_request("NVDA"), status="Submitted")
        store.end_run(rid)
    finally:
        store.close()
    return path


class TestGroupsList:
    def test_lists_groups(self, populated_db, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_DB_PATH", str(populated_db))
        args = _args()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_groups_list(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["count"] == 2
        names = {g["group_id"] for g in p["groups"]}
        assert names == {"basket-1", "basket-2"}

    def test_requires_db_path(self, monkeypatch) -> None:
        monkeypatch.delenv("TRADING_DB_PATH", raising=False)
        with pytest.raises(SystemExit):
            cli._cmd_groups_list(_args())


class TestGroupsShow:
    def test_shows_group_orders(self, populated_db, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_DB_PATH", str(populated_db))
        args = _args(group_id="basket-1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_groups_show(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["group_id"] == "basket-1"
        assert p["count"] == 2
        assert {o["order_id"] for o in p["orders"]} == {"A1", "A2"}

    def test_empty_group_returns_empty_list(self, populated_db, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_DB_PATH", str(populated_db))
        args = _args(group_id="nonexistent")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_groups_show(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["count"] == 0


class TestParserRegisters:
    def test_subcommands_present(self) -> None:
        parser = cli.build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        assert "groups-list" in sub.choices
        assert "groups-show" in sub.choices


class TestExplainShortCircuit:
    def test_list_explain(self) -> None:
        args = _args(explain=True, cmd="groups-list")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_groups_list(args)
        assert rc == 0  # no DB touched
