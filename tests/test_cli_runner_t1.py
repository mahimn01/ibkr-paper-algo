"""Tests for the shared CLI runner (trading-algo T1)."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import pytest

from trading_algo.cli_runner import _args_to_dict, _resolve_cmd_name, run_command


@pytest.fixture
def audit_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "audit"
    monkeypatch.setenv("TRADING_AUDIT_DIR", str(root))
    return root


def _make_args(fn, **extras) -> argparse.Namespace:
    ns = argparse.Namespace(func=fn, **extras)
    return ns


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

class TestHelpers:
    def test_args_to_dict_drops_func(self) -> None:
        ns = argparse.Namespace(func=lambda x: 0, symbol="AAPL", quantity=10)
        out = _args_to_dict(ns)
        assert "func" not in out
        assert out == {"symbol": "AAPL", "quantity": 10}

    def test_resolve_from_attr(self) -> None:
        ns = argparse.Namespace(cmd="place-order", func=None)
        assert _resolve_cmd_name(ns, "default") == "place-order"

    def test_resolve_from_func_name(self) -> None:
        def _cmd_place_order(args): return 0
        ns = argparse.Namespace(func=_cmd_place_order)
        assert _resolve_cmd_name(ns, "default") == "place-order"

    def test_resolve_fallback_default(self) -> None:
        ns = argparse.Namespace(func=lambda a: 0)
        assert _resolve_cmd_name(ns, "fallback") == "fallback"


# -----------------------------------------------------------------------------
# Success
# -----------------------------------------------------------------------------

class TestSuccess:
    def test_exit_code_from_handler(self, audit_root) -> None:
        def fn(args): return 0
        ns = _make_args(fn, cmd="x")
        assert run_command(ns) == 0

    def test_non_int_handler_return_becomes_0(self, audit_root) -> None:
        def fn(args): return None
        ns = _make_args(fn, cmd="x")
        assert run_command(ns) == 0

    def test_audit_entry_written(self, audit_root) -> None:
        def fn(args): return 0
        ns = _make_args(fn, cmd="my-cmd", symbol="AAPL")
        run_command(ns)
        files = list(audit_root.glob("*.jsonl"))
        assert len(files) == 1
        entry = json.loads(files[0].read_text().strip())
        assert entry["cmd"] == "my-cmd"
        assert entry["exit_code"] == 0
        assert entry["args"]["symbol"] == "AAPL"
        assert entry["error_code"] is None
        assert "request_id" in entry
        assert entry["elapsed_ms"] is not None


# -----------------------------------------------------------------------------
# Exception classification
# -----------------------------------------------------------------------------

class TestExceptions:
    def test_value_error_becomes_validation(self, audit_root, capsys) -> None:
        def fn(args): raise ValueError("bad qty")
        ns = _make_args(fn, cmd="place-order")
        rc = run_command(ns)
        assert rc == 3  # VALIDATION
        # Structured error on stderr.
        err = capsys.readouterr().err
        parsed = json.loads(err)
        assert parsed["ok"] is False
        assert parsed["error"]["code"] == "VALIDATION"
        # Audit entry captures the error_code.
        files = list(audit_root.glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert entry["exit_code"] == 3
        assert entry["error_code"] == "VALIDATION"

    def test_ibkr_connection_error_unavailable(self, audit_root, capsys) -> None:
        IBKRConnectionError = type("IBKRConnectionError", (RuntimeError,), {})
        def fn(args): raise IBKRConnectionError("TWS disconnected")
        ns = _make_args(fn, cmd="place-order")
        rc = run_command(ns)
        assert rc == 69  # UNAVAILABLE
        err = capsys.readouterr().err
        parsed = json.loads(err)
        assert parsed["error"]["code"] == "UNAVAILABLE"
        assert parsed["error"]["retryable"] is True

    def test_ibkr_error_201_hard_reject(self, audit_root, capsys) -> None:
        def fn(args):
            exc = RuntimeError("order rejected")
            exc.errorCode = 201  # type: ignore[attr-defined]
            raise exc
        ns = _make_args(fn, cmd="place-order")
        rc = run_command(ns)
        assert rc == 4  # HARD_REJECT
        parsed = json.loads(capsys.readouterr().err)
        assert parsed["error"]["code"] == "HARD_REJECT"
        assert parsed["error"]["ib_error_code"] == 201

    def test_keyboard_interrupt_sigint(self, audit_root) -> None:
        def fn(args): raise KeyboardInterrupt()
        ns = _make_args(fn, cmd="x")
        rc = run_command(ns)
        assert rc == 130
        entry = json.loads(next(audit_root.glob("*.jsonl")).read_text().strip())
        assert entry["exit_code"] == 130
        assert entry["error_code"] == "SIGINT"

    def test_system_exit_int_preserved(self, audit_root) -> None:
        def fn(args): raise SystemExit(42)
        ns = _make_args(fn, cmd="x")
        assert run_command(ns) == 42

    def test_system_exit_string_is_usage(self, audit_root, capsys) -> None:
        def fn(args): raise SystemExit("Refusing without --yes")
        ns = _make_args(fn, cmd="x")
        rc = run_command(ns)
        assert rc == 2  # USAGE
        err = capsys.readouterr().err
        assert "Refusing without --yes" in err

    def test_halt_active_classified(self, audit_root, capsys) -> None:
        from trading_algo.halt import HaltActive, HaltState
        state = HaltState(reason="emergency", since_epoch_ms=0, by="op")
        def fn(args): raise HaltActive(state)
        ns = _make_args(fn, cmd="place-order")
        rc = run_command(ns)
        assert rc == 11  # HALTED
        parsed = json.loads(capsys.readouterr().err)
        assert parsed["error"]["code"] == "HALTED"


# -----------------------------------------------------------------------------
# Audit resilience
# -----------------------------------------------------------------------------

class TestAuditResilience:
    def test_audit_failure_does_not_crash(self, tmp_path, monkeypatch) -> None:
        """If the audit log can't be written (e.g. read-only disk), the
        command's exit code is still returned correctly.
        """
        # Point audit at an unwritable path — /proc on Linux; on macOS
        # we force a broken directory by using a file as parent.
        broken = tmp_path / "file-not-dir"
        broken.write_text("")
        # Can't mkdir inside a file → will throw in atomic-append.
        monkeypatch.setenv("TRADING_AUDIT_DIR", str(broken / "nested"))

        def fn(args): return 0
        ns = _make_args(fn, cmd="x")
        # Must not raise.
        rc = run_command(ns)
        assert rc == 0


# -----------------------------------------------------------------------------
# Parent request ID propagation
# -----------------------------------------------------------------------------

class TestParentRequestID:
    def test_parent_env_recorded(self, audit_root, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_PARENT_REQUEST_ID", "PARENT_AGENT_TURN_42")
        def fn(args): return 0
        ns = _make_args(fn, cmd="x")
        run_command(ns)
        entry = json.loads(next(audit_root.glob("*.jsonl")).read_text().strip())
        assert entry["parent_request_id"] == "PARENT_AGENT_TURN_42"

    def test_strategy_id_recorded(self, audit_root, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_STRATEGY_ID", "WHEEL_01")
        monkeypatch.setenv("TRADING_AGENT_ID", "claude-turn-X")
        def fn(args): return 0
        ns = _make_args(fn, cmd="x")
        run_command(ns)
        entry = json.loads(next(audit_root.glob("*.jsonl")).read_text().strip())
        assert entry["strategy_id"] == "WHEEL_01"
        assert entry["agent_id"] == "claude-turn-X"
