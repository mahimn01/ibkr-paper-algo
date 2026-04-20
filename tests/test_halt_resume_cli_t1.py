"""Tests for `halt` / `resume` CLI subcommands in trading_algo.cli."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


@pytest.fixture
def halt_file(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "HALTED"
    monkeypatch.setenv("TRADING_HALT_PATH", str(p))
    monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))
    return p


class TestHaltCmd:
    def test_halt_writes_sentinel(self, halt_file) -> None:
        from trading_algo.cli import _cmd_halt, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "halt", "--reason", "circuit breaker", "--by", "agent-A",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_halt(args)
        assert rc == 0
        assert halt_file.exists()
        parsed = json.loads(buf.getvalue())
        assert parsed["halted"] is True
        assert parsed["reason"] == "circuit breaker"
        assert parsed["by"] == "agent-A"

    def test_halt_with_expires_in(self, halt_file) -> None:
        from trading_algo.cli import _cmd_halt, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "halt", "--reason", "x", "--expires-in", "30s",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_halt(args)
        assert rc == 0
        from trading_algo.halt import read_halt
        state = read_halt()
        assert state is not None
        assert state.expires_epoch_ms is not None

    def test_halt_bad_duration_returns_2(self, halt_file, capsys) -> None:
        from trading_algo.cli import _cmd_halt, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "halt", "--reason", "x", "--expires-in", "garbage",
        ])
        rc = _cmd_halt(args)
        assert rc == 2

    def test_reason_required(self) -> None:
        from trading_algo.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["halt"])


class TestResumeCmd:
    def test_resume_requires_confirm_resume(self, halt_file, capsys) -> None:
        from trading_algo.cli import _cmd_halt, _cmd_resume, build_parser
        parser = build_parser()
        # Halt first.
        _cmd_halt(parser.parse_args(["halt", "--reason", "x"]))
        # Resume without --confirm-resume.
        args = parser.parse_args(["resume"])
        rc = _cmd_resume(args)
        assert rc == 2
        assert halt_file.exists()  # still halted

    def test_resume_with_confirm_clears(self, halt_file) -> None:
        from trading_algo.cli import _cmd_halt, _cmd_resume, build_parser
        parser = build_parser()
        _cmd_halt(parser.parse_args(["halt", "--reason", "x"]))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_resume(parser.parse_args(["resume", "--confirm-resume"]))
        assert rc == 0
        assert not halt_file.exists()
        parsed = json.loads(buf.getvalue())
        assert parsed["resumed"] is True

    def test_resume_yes_is_not_an_alias(self) -> None:
        """An accidental `resume --yes` must be rejected — argparse
        doesn't know `--yes` for resume, so it errors out at parse time.
        """
        from trading_algo.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["resume", "--yes"])


class TestHaltBlocksWrites:
    """End-to-end: halt + try to place-order → HALTED exit code."""

    def test_halt_then_place_returns_halted(self, halt_file, tmp_path, monkeypatch, capsys) -> None:
        from trading_algo.cli import _cmd_halt, _cmd_place_order, build_parser

        parser = build_parser()
        _cmd_halt(parser.parse_args(["halt", "--reason", "emergency"]))

        # Try to place an order — should raise HaltActive.
        args = parser.parse_args([
            "place-order", "--broker", "sim", "--kind", "STK",
            "--symbol", "AAPL", "--side", "BUY", "--qty", "10",
            "--type", "MKT",
        ])
        from trading_algo.halt import HaltActive
        with pytest.raises(HaltActive):
            _cmd_place_order(args)

    def test_halt_cleared_unblocks_writes(self, halt_file, monkeypatch) -> None:
        from trading_algo.cli import _cmd_halt, _cmd_resume, build_parser
        parser = build_parser()
        _cmd_halt(parser.parse_args(["halt", "--reason", "x"]))
        _cmd_resume(parser.parse_args(["resume", "--confirm-resume"]))

        from trading_algo.halt import is_halted
        assert not is_halted()
