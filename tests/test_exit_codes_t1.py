"""Tests for the T1 exit-code taxonomy + IBKR classifier."""

from __future__ import annotations

import pytest

from trading_algo import exit_codes as ec
from trading_algo.exit_codes import (
    ClassifiedError,
    _extract_ib_error_code,
    classify_exception,
    exit_code_name,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class TestConstants:
    def test_ok_zero(self) -> None:
        assert ec.OK == 0

    def test_all_codes_unique(self) -> None:
        codes = [
            ec.OK, ec.GENERIC, ec.USAGE, ec.VALIDATION, ec.HARD_REJECT,
            ec.AUTH, ec.PERMISSION, ec.LEASE, ec.HALTED, ec.OUT_OF_WINDOW,
            ec.MARKET_CLOSED, ec.UNAVAILABLE, ec.INTERNAL, ec.TRANSIENT,
            ec.TIMEOUT, ec.SIGINT,
        ]
        assert len(codes) == len(set(codes))

    def test_codes_within_8bit(self) -> None:
        for c in ec.ALL_CODES:
            assert 0 <= c <= 255

    def test_sigint_130(self) -> None:
        assert ec.SIGINT == 130

    def test_timeout_124(self) -> None:
        assert ec.TIMEOUT == 124


# -----------------------------------------------------------------------------
# Helper to synthesize exceptions by name
# -----------------------------------------------------------------------------

def _fake(name: str, message: str = "boom", **attrs) -> Exception:
    cls = type(name, (Exception,), {})
    exc = cls(message)
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


# -----------------------------------------------------------------------------
# Class-name mapping
# -----------------------------------------------------------------------------

class TestOurExceptions:
    def test_ibkr_connection_error_unavailable(self) -> None:
        cls = classify_exception(_fake("IBKRConnectionError", "disconnected"))
        assert cls.exit_code == ec.UNAVAILABLE
        assert cls.retryable is True

    def test_ibkr_circuit_open_unavailable(self) -> None:
        cls = classify_exception(_fake("IBKRCircuitOpenError"))
        assert cls.exit_code == ec.UNAVAILABLE

    def test_ibkr_rate_limit_transient(self) -> None:
        cls = classify_exception(_fake("IBKRRateLimitError"))
        assert cls.exit_code == ec.TRANSIENT
        assert cls.retryable is True

    def test_ibkr_dependency_internal(self) -> None:
        cls = classify_exception(_fake("IBKRDependencyError"))
        assert cls.exit_code == ec.INTERNAL
        assert cls.retryable is False

    def test_halt_active(self) -> None:
        cls = classify_exception(_fake("HaltActive"))
        assert cls.exit_code == ec.HALTED

    def test_env_parse_usage(self) -> None:
        cls = classify_exception(_fake("EnvParseError"))
        assert cls.exit_code == ec.USAGE


# -----------------------------------------------------------------------------
# ib_async errorCode mapping
# -----------------------------------------------------------------------------

class TestIBKRErrorCodes:
    def test_1100_connectivity_lost_unavailable(self) -> None:
        exc = Exception("boom")
        exc.errorCode = 1100  # type: ignore[attr-defined]
        cls = classify_exception(exc)
        assert cls.exit_code == ec.UNAVAILABLE
        assert cls.retryable is True

    def test_200_no_security_definition_validation(self) -> None:
        exc = Exception("x")
        exc.errorCode = 200  # type: ignore[attr-defined]
        cls = classify_exception(exc)
        assert cls.exit_code == ec.VALIDATION

    def test_201_order_rejected_hard_reject(self) -> None:
        exc = Exception("x")
        exc.errorCode = 201  # type: ignore[attr-defined]
        assert classify_exception(exc).exit_code == ec.HARD_REJECT

    def test_103_duplicate_order_hard_reject(self) -> None:
        exc = Exception("x")
        exc.errorCode = 103  # type: ignore[attr-defined]
        assert classify_exception(exc).exit_code == ec.HARD_REJECT

    def test_162_historical_pacing_transient(self) -> None:
        exc = Exception("x")
        exc.errorCode = 162  # type: ignore[attr-defined]
        cls = classify_exception(exc)
        assert cls.exit_code == ec.TRANSIENT
        assert cls.retryable is True

    def test_354_market_data_not_subscribed_permission(self) -> None:
        exc = Exception("x")
        exc.errorCode = 354  # type: ignore[attr-defined]
        assert classify_exception(exc).exit_code == ec.PERMISSION

    def test_502_not_connected_auth(self) -> None:
        exc = Exception("x")
        exc.errorCode = 502  # type: ignore[attr-defined]
        assert classify_exception(exc).exit_code == ec.AUTH

    def test_extract_from_string(self) -> None:
        """ib_async often stringifies errors as 'error 1100, reqId 0: ...'"""
        exc = Exception("error 1100, reqId 0: Connectivity between IB and TWS has been lost.")
        cls = classify_exception(exc)
        assert cls.exit_code == ec.UNAVAILABLE

    def test_extract_errorcode_equals(self) -> None:
        exc = Exception("errorCode=354: market data not subscribed")
        cls = classify_exception(exc)
        assert cls.exit_code == ec.PERMISSION

    def test_extract_code_attribute_also_works(self) -> None:
        exc = Exception("x")
        exc.code = 1100  # type: ignore[attr-defined]
        assert classify_exception(exc).exit_code == ec.UNAVAILABLE


# -----------------------------------------------------------------------------
# Python builtins
# -----------------------------------------------------------------------------

class TestBuiltins:
    def test_keyboard_interrupt(self) -> None:
        assert classify_exception(KeyboardInterrupt()).exit_code == ec.SIGINT

    def test_system_exit_int(self) -> None:
        assert classify_exception(SystemExit(42)).exit_code == 42

    def test_system_exit_string_usage(self) -> None:
        assert classify_exception(SystemExit("Refusing")).exit_code == ec.USAGE

    def test_value_error_validation(self) -> None:
        assert classify_exception(ValueError("bad")).exit_code == ec.VALIDATION

    def test_unknown_internal(self) -> None:
        assert classify_exception(Exception("???")).exit_code == ec.INTERNAL


# -----------------------------------------------------------------------------
# String markers
# -----------------------------------------------------------------------------

class TestMessageMarkers:
    def test_timeout_word_transient(self) -> None:
        assert classify_exception(Exception("read timed out")).exit_code == ec.TRANSIENT

    def test_502_transient(self) -> None:
        assert classify_exception(Exception("502 bad gateway")).exit_code == ec.TRANSIENT

    def test_pacing_word_transient(self) -> None:
        assert classify_exception(Exception("pacing violation")).exit_code == ec.TRANSIENT


# -----------------------------------------------------------------------------
# extract_ib_error_code behaviors
# -----------------------------------------------------------------------------

class TestExtract:
    def test_returns_none_when_no_code(self) -> None:
        assert _extract_ib_error_code(Exception("no digits here")) is None

    def test_prefers_attribute(self) -> None:
        exc = Exception("error 999")
        exc.errorCode = 1100  # type: ignore[attr-defined]
        assert _extract_ib_error_code(exc) == 1100

    def test_ignores_tiny_numbers(self) -> None:
        """'x 5' shouldn't be mistaken for errorCode=5. Our regex requires
        ≥2 digits."""
        assert _extract_ib_error_code(Exception("x 5 y")) is None


# -----------------------------------------------------------------------------
# Reverse lookup
# -----------------------------------------------------------------------------

class TestExitCodeName:
    def test_known(self) -> None:
        assert exit_code_name(ec.OK) == "OK"
        assert exit_code_name(ec.AUTH) == "AUTH"
        assert exit_code_name(ec.TIMEOUT) == "TIMEOUT"

    def test_unknown(self) -> None:
        assert exit_code_name(200) == "UNKNOWN_200"
