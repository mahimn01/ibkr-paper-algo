"""Tests for trading-algo T1 structured error emitter."""

from __future__ import annotations

import io
import json

import pytest

from trading_algo.envelope import new_envelope
from trading_algo.errors import (
    build_error_payload,
    emit_error,
    suggested_action,
    with_error_envelope,
)


def _fake(name: str, message: str = "boom", **attrs) -> Exception:
    cls = type(name, (Exception,), {})
    exc = cls(message)
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


# -----------------------------------------------------------------------------
# build_error_payload
# -----------------------------------------------------------------------------

class TestBuildPayload:
    def test_basic_fields(self) -> None:
        payload = build_error_payload(_fake("IBKRConnectionError", "lost"))
        assert payload["code"] == "UNAVAILABLE"
        assert payload["class"] == "IBKRConnectionError"
        assert payload["retryable"] is True
        assert "suggested_action" in payload
        assert payload["exit_code_name"] == "UNAVAILABLE"
        assert payload["exit_code"] == 69

    def test_ib_error_code_extracted(self) -> None:
        exc = Exception("error 1100, reqId 0")
        payload = build_error_payload(exc)
        assert payload.get("ib_error_code") == 1100

    def test_no_ib_error_code_for_our_exceptions(self) -> None:
        payload = build_error_payload(_fake("IBKRCircuitOpenError"))
        # No errorCode attribute → not surfaced.
        assert "ib_error_code" not in payload

    def test_field_errors_from_dataclass(self) -> None:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class V:
            field: str
            message: str

        exc = _fake("ValidationFailure")
        exc.field_errors = [V("qty", "must be positive"), V("lmt", "required")]
        payload = build_error_payload(exc)
        assert payload["field_errors"] == [
            {"field": "qty", "message": "must be positive"},
            {"field": "lmt", "message": "required"},
        ]

    def test_empty_message_fallback(self) -> None:
        """str() on a no-arg Exception gives empty; we fall back to class."""
        payload = build_error_payload(_fake("IBKRConnectionError", ""))
        assert payload["message"]


# -----------------------------------------------------------------------------
# emit_error
# -----------------------------------------------------------------------------

class TestEmitError:
    def test_writes_json_to_stream(self) -> None:
        env = new_envelope("place-order")
        buf = io.StringIO()
        code = emit_error(_fake("IBKRConnectionError", "TWS down"), env=env, stream=buf)
        parsed = json.loads(buf.getvalue())
        assert parsed["ok"] is False
        assert parsed["cmd"] == "place-order"
        assert parsed["data"] is None
        assert parsed["error"]["code"] == "UNAVAILABLE"
        assert code == 69

    def test_includes_elapsed_ms(self) -> None:
        env = new_envelope("x")
        buf = io.StringIO()
        emit_error(_fake("IBKRConnectionError"), env=env, stream=buf)
        parsed = json.loads(buf.getvalue())
        assert "elapsed_ms" in parsed["meta"]

    def test_broken_pipe_tolerated(self) -> None:
        class BrokenStream:
            def write(self, s):
                raise BrokenPipeError()
            def flush(self):
                pass
        env = new_envelope("x")
        # Must not raise.
        emit_error(_fake("IBKRConnectionError"), env=env, stream=BrokenStream())

    def test_unknown_exception_internal(self) -> None:
        env = new_envelope("x")
        buf = io.StringIO()
        code = emit_error(Exception("???"), env=env, stream=buf)
        parsed = json.loads(buf.getvalue())
        assert parsed["error"]["code"] == "INTERNAL"
        assert code == 70

    def test_ibkr_error_201_hard_reject(self) -> None:
        exc = Exception("order rejected")
        exc.errorCode = 201  # type: ignore[attr-defined]
        env = new_envelope("place-order")
        buf = io.StringIO()
        code = emit_error(exc, env=env, stream=buf)
        parsed = json.loads(buf.getvalue())
        assert parsed["error"]["code"] == "HARD_REJECT"
        assert parsed["error"]["ib_error_code"] == 201
        assert parsed["error"]["retryable"] is False
        assert code == 4


# -----------------------------------------------------------------------------
# with_error_envelope decorator
# -----------------------------------------------------------------------------

class TestDecorator:
    def test_success(self) -> None:
        @with_error_envelope("x")
        def cmd(args, *, env):
            env.data = {"ok": True}
            return 0
        assert cmd(None) == 0

    def test_exception_becomes_structured(self, capsys) -> None:
        @with_error_envelope("place-order")
        def cmd(args, *, env):
            raise _fake("IBKRDependencyError", "bad contract")
        rc = cmd(None)
        assert rc == 70  # INTERNAL
        parsed = json.loads(capsys.readouterr().err)
        assert parsed["ok"] is False
        assert parsed["error"]["class"] == "IBKRDependencyError"

    def test_system_exit_propagates(self) -> None:
        @with_error_envelope("x")
        def cmd(args, *, env):
            raise SystemExit(2)
        with pytest.raises(SystemExit) as exc_info:
            cmd(None)
        assert exc_info.value.code == 2

    def test_keyboard_interrupt_sigint(self, capsys) -> None:
        @with_error_envelope("x")
        def cmd(args, *, env):
            raise KeyboardInterrupt()
        rc = cmd(None)
        assert rc == 130


# -----------------------------------------------------------------------------
# suggested_action
# -----------------------------------------------------------------------------

class TestSuggestedAction:
    def test_auth_mentions_tws(self) -> None:
        assert "TWS" in suggested_action("AUTH") or "Gateway" in suggested_action("AUTH")

    def test_permission_mentions_subscription(self) -> None:
        assert "subscription" in suggested_action("PERMISSION").lower()

    def test_transient_mentions_pacing(self) -> None:
        text = suggested_action("TRANSIENT").lower()
        assert "pacing" in text or "rate" in text

    def test_unknown_fallback(self) -> None:
        assert suggested_action("MADE_UP_CODE")
