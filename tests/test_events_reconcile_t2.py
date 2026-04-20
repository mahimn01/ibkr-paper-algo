"""Tests for T2.6 events + reconcile commands."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_algo import cli


class TestCmdEvents:
    def test_reads_empty_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))
        args = argparse.Namespace(
            since=None, until=None, cmd_filter=None, outcome=None, tail=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_events(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["count"] == 0
        assert p["entries"] == []

    def test_reads_and_filters(self, tmp_path, monkeypatch) -> None:
        audit_dir = tmp_path / "audit"
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(audit_dir))
        from trading_algo.audit import log_command

        log_command(cmd="place-order", request_id="AAA", args={"symbol": "AAPL"}, exit_code=0)
        log_command(cmd="place-order", request_id="BBB", args={"symbol": "MSFT"}, exit_code=1)
        log_command(cmd="cancel-order", request_id="CCC", args={"order_id": 42}, exit_code=0)

        # All
        args = argparse.Namespace(since=None, until=None, cmd_filter=None, outcome=None, tail=None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_events(args)
        all_entries = json.loads(buf.getvalue())
        assert all_entries["count"] == 3

        # Filter by cmd
        args.cmd_filter = "place-order"
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_events(args)
        place = json.loads(buf.getvalue())
        assert place["count"] == 2

        # Filter by outcome
        args.cmd_filter = None
        args.outcome = "error"
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_events(args)
        errs = json.loads(buf.getvalue())
        assert errs["count"] == 1
        assert errs["entries"][0]["request_id"] == "BBB"

    def test_tail_limits(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))
        from trading_algo.audit import log_command
        for i in range(5):
            log_command(cmd="foo", request_id=f"R{i}", args={}, exit_code=0)

        args = argparse.Namespace(since=None, until=None, cmd_filter=None, outcome=None, tail=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_events(args)
        p = json.loads(buf.getvalue())
        assert p["count"] == 2

    def test_invalid_date_raises_systemexit(self) -> None:
        args = argparse.Namespace(
            since="not-a-date", until=None, cmd_filter=None, outcome=None, tail=None,
        )
        with pytest.raises(SystemExit):
            cli._cmd_events(args)


class TestCmdReconcile:
    def test_requires_db_path(self, monkeypatch) -> None:
        monkeypatch.delenv("TRADING_DB_PATH", raising=False)
        args = argparse.Namespace(
            broker="sim", ibkr_host=None, ibkr_port=None, ibkr_client_id=None,
            paper=False, live=False, dry_run=True, no_dry_run=False,
            allow_live=False, confirm_token=None,
        )
        with pytest.raises(SystemExit):
            cli._cmd_reconcile(args)

    def test_emits_structured_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "oms.db"))

        fake_broker = MagicMock()
        fake_oms = MagicMock()
        fake_oms.reconcile.return_value = {42: "Filled", 43: "Cancelled"}

        monkeypatch.setattr(cli, "_make_broker", lambda *a, **k: fake_broker)
        monkeypatch.setattr(cli, "OrderManager", lambda *a, **k: fake_oms)

        args = argparse.Namespace(
            broker="sim", ibkr_host=None, ibkr_port=None, ibkr_client_id=None,
            paper=False, live=False, dry_run=True, no_dry_run=False,
            allow_live=False, confirm_token=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_reconcile(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["reconciled_count"] == 2
        ids = {e["order_id"] for e in p["orders"]}
        assert ids == {42, 43}


class TestParserHasT26:
    def test_events_registered(self) -> None:
        p = cli.build_parser()
        sub_action = next(
            a for a in p._actions if isinstance(a, argparse._SubParsersAction)
        )
        assert "events" in sub_action.choices
        assert "reconcile" in sub_action.choices
