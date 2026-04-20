"""Tests for envelope + ULID request IDs (trading-algo T1)."""

from __future__ import annotations

import io
import json
import time

import pytest

from trading_algo.envelope import (
    ENV_FORCE_JSON,
    ENV_NO_ENVELOPE,
    ENV_PARENT_REQUEST_ID,
    SCHEMA_VERSION,
    Envelope,
    envelope_to_json,
    envelopes_disabled,
    finalize_envelope,
    json_is_default_for,
    new_envelope,
    new_request_id,
    parent_request_id,
)


class TestRequestId:
    def test_length_26(self) -> None:
        assert len(new_request_id()) == 26

    def test_alphabet_is_crockford(self) -> None:
        ok = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        assert all(c in ok for c in new_request_id())

    def test_excludes_i_l_o_u(self) -> None:
        r = new_request_id()
        for bad in "ILOU":
            assert bad not in r

    def test_time_ordering(self) -> None:
        a = new_request_id(clock_ms=1_000_000_000_000)
        b = new_request_id(clock_ms=1_000_000_001_000)
        assert a < b

    def test_uniqueness_same_ms(self) -> None:
        ids = {new_request_id(clock_ms=1_700_000_000_000) for _ in range(300)}
        assert len(ids) == 300

    def test_monotonic_wall_clock(self) -> None:
        a = new_request_id()
        time.sleep(0.002)
        b = new_request_id()
        assert a < b


class TestParent:
    def test_unset_none(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_PARENT_REQUEST_ID, raising=False)
        assert parent_request_id() is None

    def test_reads_env(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_PARENT_REQUEST_ID, "PARENT_X")
        assert parent_request_id() == "PARENT_X"

    def test_strips_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_PARENT_REQUEST_ID, "  P  ")
        assert parent_request_id() == "P"


class TestEnvelope:
    def test_new_envelope_shape(self) -> None:
        env = new_envelope("place-order")
        assert env.ok is True
        assert env.cmd == "place-order"
        assert env.schema_version == SCHEMA_VERSION
        assert len(env.request_id) == 26
        assert "started_at_epoch_ms" in env.meta

    def test_to_dict_has_required_keys(self) -> None:
        env = new_envelope("x")
        env.data = {"n": 1}
        out = env.to_dict()
        for k in ("ok", "cmd", "schema_version", "request_id", "data", "warnings", "meta"):
            assert k in out
        assert "error" not in out  # omitted when unset

    def test_error_included_when_set(self) -> None:
        env = Envelope(ok=False, cmd="x", request_id="R", error={"code": "AUTH"})
        assert env.to_dict()["error"] == {"code": "AUTH"}

    def test_add_warning(self) -> None:
        env = new_envelope("x")
        env.add_warning("PACING", "historical pacing — backing off", severity="warn")
        assert env.warnings[0] == {
            "code": "PACING", "message": "historical pacing — backing off",
            "severity": "warn",
        }

    def test_parent_request_id_propagates(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_PARENT_REQUEST_ID, "PARENT_ABC")
        env = new_envelope("x")
        assert env.meta.get("parent_request_id") == "PARENT_ABC"


class TestFinalize:
    def test_adds_elapsed(self) -> None:
        env = new_envelope("x")
        time.sleep(0.005)
        finalize_envelope(env)
        assert env.meta["elapsed_ms"] >= 4

    def test_idempotent(self) -> None:
        env = new_envelope("x")
        finalize_envelope(env)
        first = env.meta["elapsed_ms"]
        finalize_envelope(env)
        assert env.meta["elapsed_ms"] == first

    def test_missing_start_skipped(self) -> None:
        env = Envelope(ok=True, cmd="x", request_id="R")
        finalize_envelope(env)
        assert "elapsed_ms" not in env.meta


class TestEnvelopesDisabled:
    def test_unset_false(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_NO_ENVELOPE, raising=False)
        assert envelopes_disabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy(self, monkeypatch, val) -> None:
        monkeypatch.setenv(ENV_NO_ENVELOPE, val)
        assert envelopes_disabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy(self, monkeypatch, val) -> None:
        monkeypatch.setenv(ENV_NO_ENVELOPE, val)
        assert envelopes_disabled() is False


class TestJsonDefault:
    def test_forced_by_env(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_FORCE_JSON, "1")
        tty = io.StringIO()
        tty.isatty = lambda: True  # type: ignore[method-assign]
        assert json_is_default_for(tty) is True

    def test_non_tty_json(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_FORCE_JSON, raising=False)
        pipe = io.StringIO()
        pipe.isatty = lambda: False  # type: ignore[method-assign]
        assert json_is_default_for(pipe) is True

    def test_tty_native(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_FORCE_JSON, raising=False)
        tty = io.StringIO()
        tty.isatty = lambda: True  # type: ignore[method-assign]
        assert json_is_default_for(tty) is False

    def test_broken_stream_defaults_json(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_FORCE_JSON, raising=False)
        class Broken:
            def isatty(self):
                raise ValueError("closed")
        assert json_is_default_for(Broken()) is True


class TestSerialization:
    def test_round_trip(self) -> None:
        env = new_envelope("orders")
        env.data = [{"order_id": 1}]
        parsed = json.loads(envelope_to_json(env))
        assert parsed["cmd"] == "orders"
        assert parsed["data"] == [{"order_id": 1}]

    def test_handles_unserializable_via_str(self) -> None:
        env = new_envelope("x")
        class Weird:
            def __str__(self):
                return "weird-thing"
        env.data = {"x": Weird()}
        text = envelope_to_json(env)
        assert "weird-thing" in text
