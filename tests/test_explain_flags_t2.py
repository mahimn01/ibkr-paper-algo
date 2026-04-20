"""Tests for T2.7 --explain / --fields / --summary / tools-describe."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

import pytest

from trading_algo import cli


class TestExplainFlag:
    def test_time_explain_shortcircuits(self) -> None:
        args = argparse.Namespace(explain=True, cmd="time")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_time(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["cmd"] == "time"
        assert "explanation" in p

    def test_status_explain_does_not_connect(self, monkeypatch) -> None:
        # If --explain short-circuits, _make_broker should never be called.
        monkeypatch.setattr(cli, "_make_broker", lambda *a, **k: pytest.fail("should not connect"))
        args = argparse.Namespace(
            explain=True, cmd="status", broker="ibkr",
            ibkr_host=None, ibkr_port=None, ibkr_client_id=None,
            paper=False, live=False, dry_run=False, no_dry_run=False,
            allow_live=False, skip_broker=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_status(args)
        assert rc == 0

    def test_watch_explain_does_not_poll(self, monkeypatch) -> None:
        monkeypatch.setattr(cli, "_make_broker", lambda *a, **k: pytest.fail("should not poll"))
        args = argparse.Namespace(
            explain=True, cmd="watch", broker="sim",
            resource="quote", symbol="AAPL", kind="STK",
            exchange=None, currency=None, expiry=None,
            right=None, strike=None, multiplier=None, order_id=None,
            until="last > 1", every=0.2, timeout=2.0,
            ibkr_host=None, ibkr_port=None, ibkr_client_id=None,
            paper=False, live=False, dry_run=False, no_dry_run=False, allow_live=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_watch(args)
        assert rc == 0


class TestEventsFieldsAndSummary:
    def test_fields_projects_keys(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))
        from trading_algo.audit import log_command
        log_command(cmd="place-order", request_id="XYZ", args={"symbol": "AAPL"}, exit_code=0)

        args = argparse.Namespace(
            since=None, until=None, cmd_filter=None, outcome=None, tail=None,
            fields="request_id,cmd", summary=False, explain=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_events(args)
        p = json.loads(buf.getvalue())
        entry = p["entries"][0]
        assert set(entry.keys()) == {"request_id", "cmd"}

    def test_summary_rollup(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))
        from trading_algo.audit import log_command
        log_command(cmd="place-order", request_id="A", args={}, exit_code=0)
        log_command(cmd="place-order", request_id="B", args={}, exit_code=1)
        log_command(cmd="cancel-order", request_id="C", args={}, exit_code=0)

        args = argparse.Namespace(
            since=None, until=None, cmd_filter=None, outcome=None, tail=None,
            fields=None, summary=True, explain=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_events(args)
        p = json.loads(buf.getvalue())
        assert p["count"] == 3
        assert p["by_cmd"]["place-order"] == 2
        assert p["by_cmd"]["cancel-order"] == 1
        assert p["outcome"] == {"ok": 2, "error": 1}


class TestToolsDescribe:
    def test_emits_tools_array(self) -> None:
        args = argparse.Namespace()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_tools_describe(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["count"] > 0
        names = {t["name"] for t in p["tools"]}
        assert "watch" in names
        assert "status" in names
        assert "events" in names
        assert "reconcile" in names
        assert "tools-describe" in names

    def test_registered_as_subcommand(self) -> None:
        p = cli.build_parser()
        sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        assert "tools-describe" in sub.choices
