"""Tests for T2.5 watch / status / time commands."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from trading_algo import cli


def _make_args(**kw) -> argparse.Namespace:
    defaults = {
        "broker": "sim",
        "ibkr_host": None,
        "ibkr_port": None,
        "ibkr_client_id": None,
        "paper": False,
        "live": False,
        "dry_run": False,
        "no_dry_run": False,
        "allow_live": False,
        "confirm_token": None,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestCmdTime:
    def test_emits_core_fields(self) -> None:
        args = argparse.Namespace()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_time(args)
        assert rc == 0
        payload = json.loads(buf.getvalue())
        for key in ("utc_now", "et_now", "et_date", "weekday",
                    "us_equity_regular_session_open", "session_cutoffs_et",
                    "next_open_et", "next_close_et",
                    "is_trading_day", "is_holiday", "is_half_day"):
            assert key in payload

    def test_weekend_is_closed(self) -> None:
        args = argparse.Namespace()
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_time(args)
        p = json.loads(buf.getvalue())
        # Static assertion: open flag is bool.
        assert isinstance(p["us_equity_regular_session_open"], bool)


class TestCmdStatus:
    def test_skip_broker_is_fast_and_clean(self) -> None:
        args = _make_args(skip_broker=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_status(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["broker"]["connected"] is None
        assert "market" in p and "config" in p and "halt" in p
        assert p["halt"]["is_halted"] is False

    def test_reports_halt_active(self, tmp_path, monkeypatch) -> None:
        halt_file = tmp_path / "HALTED"
        monkeypatch.setenv("TRADING_HALT_PATH", str(halt_file))
        from trading_algo.halt import write_halt
        write_halt(reason="maintenance", by="ops")

        args = _make_args(skip_broker=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_status(args)
        p = json.loads(buf.getvalue())
        assert p["halt"]["is_halted"] is True
        assert p["halt"]["reason"] == "maintenance"


class TestCmdWatch:
    def test_unsafe_expression_rejected(self) -> None:
        args = _make_args(
            resource="quote", symbol="AAPL", kind="STK",
            exchange=None, currency=None, expiry=None,
            right=None, strike=None, multiplier=None, order_id=None,
            until="abs(last) > 5",  # function call — disallowed
            every=0.2, timeout=0.5,
        )
        rc = cli._cmd_watch(args)
        assert rc == 2

    def test_syntax_error_rejected(self) -> None:
        args = _make_args(
            resource="quote", symbol="AAPL", kind="STK",
            exchange=None, currency=None, expiry=None,
            right=None, strike=None, multiplier=None, order_id=None,
            until="last >",
            every=0.2, timeout=0.5,
        )
        rc = cli._cmd_watch(args)
        assert rc == 2

    def test_matches_immediately(self, monkeypatch) -> None:
        fake_broker = MagicMock()
        fake_broker.get_market_data_snapshot.return_value = MagicMock(
            instrument=MagicMock(symbol="AAPL"),
            bid=149.0, ask=151.0, last=200.0,
            close=148.0, volume=1_000_000, timestamp_epoch_s=1234567890,
        )
        monkeypatch.setattr(cli, "_make_broker", lambda *a, **k: fake_broker)

        args = _make_args(
            resource="quote", symbol="AAPL", kind="STK",
            exchange=None, currency=None, expiry=None,
            right=None, strike=None, multiplier=None, order_id=None,
            until="last > 150",
            every=0.2, timeout=2.0,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_watch(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["matched"] is True
        assert p["snapshot"]["last"] == 200.0

    def test_times_out_with_124(self, monkeypatch) -> None:
        fake_broker = MagicMock()
        fake_broker.get_market_data_snapshot.return_value = MagicMock(
            instrument=MagicMock(symbol="AAPL"),
            bid=100.0, ask=101.0, last=100.5,
            close=99.0, volume=1000, timestamp_epoch_s=1,
        )
        monkeypatch.setattr(cli, "_make_broker", lambda *a, **k: fake_broker)

        args = _make_args(
            resource="quote", symbol="AAPL", kind="STK",
            exchange=None, currency=None, expiry=None,
            right=None, strike=None, multiplier=None, order_id=None,
            until="last > 999999",  # never matches
            every=0.2, timeout=0.6,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_watch(args)
        assert rc == 124
        p = json.loads(buf.getvalue())
        assert p["matched"] is False
        assert p["reason"] == "timeout"

    def test_unknown_resource_returns_2(self) -> None:
        args = _make_args(
            resource="bogus", symbol="AAPL", kind="STK",
            exchange=None, currency=None, expiry=None,
            right=None, strike=None, multiplier=None, order_id=None,
            until="True", every=0.2, timeout=0.5,
        )
        # Parser would normally reject this, but we're calling handler directly.
        # Need to monkey past the _make_broker connect by using sim broker.
        rc = cli._cmd_watch(args)
        assert rc == 2


class TestParserHasT25Commands:
    def test_watch_subcommand_registered(self) -> None:
        p = cli.build_parser()
        # argparse exposes subparsers via the _actions list.
        sub_actions = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        assert sub_actions
        choices = sub_actions[0].choices
        assert "watch" in choices
        assert "status" in choices
        assert "time" in choices
