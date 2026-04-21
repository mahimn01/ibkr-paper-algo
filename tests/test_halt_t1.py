"""Tests for the trading-algo halt sentinel."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from trading_algo.halt import (
    HaltActive,
    HaltState,
    assert_not_halted,
    clear_halt,
    is_halted,
    parse_duration,
    read_halt,
    write_halt,
)


@pytest.fixture
def halt_file(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "HALTED"
    monkeypatch.setenv("TRADING_HALT_PATH", str(p))
    return p


class TestWriteReadClear:
    def test_write_then_read(self, halt_file) -> None:
        write_halt(reason="circuit breaker", by="ops")
        state = read_halt()
        assert state is not None
        assert state.reason == "circuit breaker"
        assert state.by == "ops"
        assert state.expires_epoch_ms is None

    def test_clear_existing(self, halt_file) -> None:
        write_halt(reason="x", by="me")
        assert clear_halt() is True
        assert read_halt() is None

    def test_clear_absent_returns_false(self, halt_file) -> None:
        assert clear_halt() is False

    def test_is_halted(self, halt_file) -> None:
        assert is_halted() is False
        write_halt(reason="x", by="me")
        assert is_halted() is True

    def test_overwrite(self, halt_file) -> None:
        write_halt(reason="first", by="A")
        write_halt(reason="second", by="B")
        state = read_halt()
        assert state.reason == "second"
        assert state.by == "B"


class TestExpiry:
    def test_unexpired_halted(self, halt_file) -> None:
        write_halt(reason="x", by="me", expires_in_seconds=60)
        assert read_halt() is not None

    def test_expired_auto_clears(self, halt_file) -> None:
        state = HaltState(reason="x", since_epoch_ms=0, by="me", expires_epoch_ms=1)
        halt_file.parent.mkdir(parents=True, exist_ok=True)
        halt_file.write_text(json.dumps(state.to_dict()))
        assert read_halt() is None
        assert not halt_file.exists()

    def test_expiry_serialises(self, halt_file) -> None:
        write_halt(reason="x", by="me", expires_in_seconds=30)
        raw = json.loads(halt_file.read_text())
        assert raw.get("expires_epoch_ms", 0) > 0


class TestCorrupt:
    def test_malformed_json_stays_halted(self, halt_file) -> None:
        halt_file.parent.mkdir(parents=True, exist_ok=True)
        halt_file.write_text("not-json")
        state = read_halt()
        assert state is not None
        assert "corrupt" in state.reason.lower()

    def test_empty_file_stays_halted(self, halt_file) -> None:
        halt_file.parent.mkdir(parents=True, exist_ok=True)
        halt_file.write_text("")
        assert is_halted() is True


class TestParseDuration:
    def test_seconds(self) -> None:
        assert parse_duration("30s") == 30

    def test_minutes(self) -> None:
        assert parse_duration("5m") == 300
        assert parse_duration("0.5m") == 30

    def test_hours(self) -> None:
        assert parse_duration("1h") == 3600

    def test_days(self) -> None:
        assert parse_duration("2d") == 172800

    def test_bare_float(self) -> None:
        assert parse_duration("42.5") == 42.5

    def test_malformed(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("five minutes")
        with pytest.raises(ValueError):
            parse_duration("")
        with pytest.raises(ValueError):
            parse_duration("30x")


class TestAssertNotHalted:
    def test_no_sentinel_ok(self, halt_file) -> None:
        assert_not_halted()

    def test_raises_when_halted(self, halt_file) -> None:
        write_halt(reason="emergency", by="ops")
        with pytest.raises(HaltActive) as exc_info:
            assert_not_halted()
        assert exc_info.value.state.reason == "emergency"

    def test_halt_active_classifies_as_halted_exit_code(self, halt_file) -> None:
        from trading_algo.exit_codes import classify_exception, HALTED
        write_halt(reason="x", by="me")
        try:
            assert_not_halted()
        except HaltActive as exc:
            assert classify_exception(exc).exit_code == HALTED
