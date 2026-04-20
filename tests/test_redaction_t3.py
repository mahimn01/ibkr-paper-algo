"""Tests for T3.1 trading_algo/redaction.py."""

from __future__ import annotations

import logging

import pytest

from trading_algo.redaction import (
    REDACTED,
    install_logging_filter,
    known_secrets,
    redact_dict,
    redact_text,
)


class TestRedactText:
    def test_bearer(self) -> None:
        out = redact_text("Authorization: Bearer eyJ.abc.def12345678901234567")
        assert "eyJ" not in out
        assert REDACTED in out

    def test_auth_token(self) -> None:
        out = redact_text("Authorization: Token 12345abcdef67890fedcba0987654321")
        assert REDACTED in out

    def test_access_token_kv(self) -> None:
        out = redact_text('{"access_token": "xyz_SHOULD_NOT_APPEAR_LONG_ENOUGH_123"}')
        assert "xyz_SHOULD_NOT_APPEAR" not in out
        assert REDACTED in out

    def test_api_secret_kv(self) -> None:
        out = redact_text("api_secret=SECRET_ABCDEFGH12345678")
        assert "SECRET_ABCDEFGH" not in out
        assert REDACTED in out

    def test_order_token(self) -> None:
        out = redact_text('order_token="MY_PROD_TOKEN_12345678"')
        assert "MY_PROD_TOKEN" not in out
        assert REDACTED in out

    def test_ibkr_account_redacted(self) -> None:
        out = redact_text("Account U1234567 has NetLiquidation 1000")
        assert "U1234567" not in out
        assert REDACTED in out

    def test_long_token(self) -> None:
        out = redact_text("error code: ABCdef123456789_abcdef_ghijkl_mnop_qrstuv")
        assert REDACTED in out

    def test_short_string_unchanged(self) -> None:
        assert redact_text("hello") == "hello"

    def test_none_and_non_string(self) -> None:
        assert redact_text("") == ""
        # Non-string input: return stringified.
        assert redact_text(42) == "42"  # type: ignore[arg-type]

    def test_known_env_secret_is_redacted(self, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_ORDER_TOKEN", "PROD_ORDER_TOKEN_ABC_12345")
        out = redact_text("attempt with PROD_ORDER_TOKEN_ABC_12345 failed")
        assert "PROD_ORDER_TOKEN_ABC" not in out

    def test_extra_secrets_param(self) -> None:
        out = redact_text("custom: SUPERSECRETVALUE12345", extra_secrets=["SUPERSECRETVALUE12345"])
        assert "SUPERSECRETVALUE" not in out


class TestRedactDict:
    def test_nested_credential_keys(self) -> None:
        d = {
            "access_token": "abc123",
            "inner": {"api_secret": "def456", "safe": "hello"},
            "list": ["Authorization: Bearer ABCdef123456789_abcdef_ghijkl_mnop_qrstuv"],
            "normal_key": "normal value",
        }
        out = redact_dict(d)
        assert out["access_token"] == REDACTED
        assert out["inner"]["api_secret"] == REDACTED
        assert out["inner"]["safe"] == "hello"
        assert REDACTED in out["list"][0]
        assert out["normal_key"] == "normal value"

    def test_list_passthrough(self) -> None:
        out = redact_dict(["hello", "world"])
        assert out == ["hello", "world"]

    def test_scalar_passthrough(self) -> None:
        assert redact_dict(42) == 42
        assert redact_dict(None) is None


class TestLoggingFilter:
    def test_filter_redacts_at_record_level(self, monkeypatch) -> None:
        """Exercise the filter directly on a synthetic LogRecord — avoids
        pytest's internal logging-handler churn which can swallow records.
        """
        monkeypatch.setenv("TRADING_ORDER_TOKEN", "SECRET_TOKEN_ABCDEFGH12345")
        install_logging_filter(reset=True)
        root = logging.getLogger()
        from trading_algo.redaction import _SecretRedactingFilter
        f = next(x for x in root.filters if isinstance(x, _SecretRedactingFilter))
        f._secrets_cache = None  # force re-read with new env

        record = logging.LogRecord(
            name="t", level=logging.WARNING, pathname="", lineno=0,
            msg="leaking token: SECRET_TOKEN_ABCDEFGH12345",
            args=None, exc_info=None,
        )
        assert f.filter(record) is True
        assert "SECRET_TOKEN_ABCDEFGH" not in record.msg
        assert REDACTED in record.msg

    def test_install_is_idempotent(self) -> None:
        install_logging_filter(reset=True)
        install_logging_filter()
        install_logging_filter()
        # No assertion — we just verify no exception and no exploding
        # number of filters on the root logger.
        root = logging.getLogger()
        from trading_algo.redaction import _SecretRedactingFilter
        filters = [f for f in root.filters if isinstance(f, _SecretRedactingFilter)]
        assert len(filters) == 1


class TestKnownSecrets:
    def test_reads_env(self, monkeypatch) -> None:
        monkeypatch.setenv("IBKR_FLEX_TOKEN", "FLX_ABCDEFGH12345678")
        monkeypatch.setenv("GEMINI_API_KEY", "GEM_API_ABCDEFGH12345")
        secrets = known_secrets()
        assert "FLX_ABCDEFGH12345678" in secrets
        assert "GEM_API_ABCDEFGH12345" in secrets

    def test_filters_short(self, monkeypatch) -> None:
        monkeypatch.setenv("TRADING_ORDER_TOKEN", "abc")  # too short
        secrets = known_secrets()
        assert "abc" not in secrets
