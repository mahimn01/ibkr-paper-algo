"""Tests that confirm every IBKR write command checks the halt sentinel.

We use `inspect.getsource` so the test stays green even when the handler
implementations change — all we care about is: the call to
`assert_not_halted` is present.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


WRITE_HANDLERS_CLI = [
    "_cmd_place_order",
    "_cmd_place_bracket",
    "_cmd_cancel_order",
    "_cmd_modify_order",
    "_cmd_run",
]

WRITE_HANDLERS_IBKR_TOOL = [
    "cmd_place",
    "cmd_combo",
    "cmd_cancel",
    "cmd_cancel_all",
]


class TestCliWriteHandlers:
    @pytest.mark.parametrize("name", WRITE_HANDLERS_CLI)
    def test_handler_has_halt_guard(self, name: str) -> None:
        from trading_algo import cli
        fn = getattr(cli, name)
        src = inspect.getsource(fn)
        assert "assert_not_halted" in src, (
            f"{name} is a write-path command but does not call "
            f"assert_not_halted() — a halt sentinel would NOT block it."
        )


class TestIBKRToolWriteHandlers:
    @pytest.mark.parametrize("name", WRITE_HANDLERS_IBKR_TOOL)
    def test_handler_has_halt_guard(self, name: str) -> None:
        from trading_algo import ibkr_tool
        fn = getattr(ibkr_tool, name)
        src = inspect.getsource(fn)
        assert "assert_not_halted" in src, f"{name} missing assert_not_halted()"


class TestReadHandlersUnchanged:
    """Read commands (positions, orders, quotes) must NOT halt-gate.

    Halt blocks WRITES. Agents still need to query state during a halt
    so they can reconcile + reason about whether to resume.
    """
    @pytest.mark.parametrize("name", [
        "cmd_positions", "cmd_portfolio", "cmd_pnl", "cmd_summary",
        "cmd_quote", "cmd_quotes", "cmd_history", "cmd_accounts",
    ])
    def test_read_handler_unchanged(self, name: str) -> None:
        from trading_algo import ibkr_tool
        fn = getattr(ibkr_tool, name)
        src = inspect.getsource(fn)
        assert "assert_not_halted" not in src, (
            f"{name} is a READ-only command — it should not halt-gate"
        )


class TestHaltClassifies:
    """When a halted command is hit, HaltActive raises, classified as
    exit code 11 (HALTED) via cli_runner.
    """
    def test_halted_cmd_exits_11(self, tmp_path, monkeypatch, capsys) -> None:
        halt_file = tmp_path / "HALTED"
        monkeypatch.setenv("TRADING_HALT_PATH", str(halt_file))
        from trading_algo.halt import write_halt
        write_halt(reason="emergency", by="ops")

        # Simulate invoking _cmd_place_order via cli_runner.
        from trading_algo.cli_runner import run_command
        import trading_algo.cli as cli

        # Monkeypatch audit root so we don't pollute real dir.
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))

        # Call the REAL _cmd_place_order through run_command; it should
        # raise HaltActive before touching any broker logic.
        ns = argparse.Namespace(
            func=cli._cmd_place_order,
            cmd="place-order",
            # Stub everything the handler reads — but it never gets there.
            broker="ibkr", symbol="AAPL", kind="STK",
        )
        rc = run_command(ns)
        assert rc == 11  # HALTED

        import json
        err = capsys.readouterr().err
        parsed = json.loads(err)
        assert parsed["error"]["code"] == "HALTED"
