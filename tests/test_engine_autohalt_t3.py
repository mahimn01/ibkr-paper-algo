"""Tests for T3.6 — Engine auto-halts on IBKR connection-lost conditions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trading_algo.engine import Engine, _should_auto_halt


class _FakeIBKRConnectionError(Exception):
    pass

_FakeIBKRConnectionError.__name__ = "IBKRConnectionError"


class _FakeIBKRCircuitOpenError(Exception):
    pass

_FakeIBKRCircuitOpenError.__name__ = "IBKRCircuitOpenError"


class TestShouldAutoHalt:
    def test_ibkr_connection_error(self) -> None:
        assert _should_auto_halt(_FakeIBKRConnectionError("lost"))

    def test_ibkr_circuit_open(self) -> None:
        assert _should_auto_halt(_FakeIBKRCircuitOpenError("tripped"))

    def test_connection_error(self) -> None:
        assert _should_auto_halt(ConnectionError("reset"))

    def test_error_code_1100(self) -> None:
        e = Exception("error 1100, reqId 0: Connectivity between IB and TWS has been lost.")
        assert _should_auto_halt(e)

    def test_error_code_1300(self) -> None:
        e = Exception("error 1300, reqId 0: socket port reset")
        assert _should_auto_halt(e)

    def test_error_code_502(self) -> None:
        e = Exception("error 502: Couldn't connect to TWS")
        assert _should_auto_halt(e)

    def test_unrelated_error_ignored(self) -> None:
        assert not _should_auto_halt(ValueError("qty must be positive"))

    def test_rate_limit_not_halt(self) -> None:
        e = Exception("error 162, reqId 5: Historical Market Data Service error message: pacing")
        assert not _should_auto_halt(e)


class TestEngineAutoHalt:
    def test_run_forever_halts_on_connection_lost(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("TRADING_HALT_PATH", str(tmp_path / "HALTED"))

        # Build a minimal Engine with mocks that raise connection-lost on tick.
        broker = MagicMock()
        broker.connect = MagicMock()
        broker.disconnect = MagicMock()

        class _FailStrategy:
            name = "fail"
            def on_tick(self, ctx):
                raise _FakeIBKRConnectionError("lost during tick")

        from trading_algo.config import TradingConfig
        cfg = TradingConfig.from_env()

        engine = Engine(
            broker=broker, config=cfg, strategy=_FailStrategy(),
            risk=MagicMock(), confirm_token=None,
        )

        # run_forever should catch and auto-halt, then break out of the loop.
        engine.run_forever()

        from trading_algo.halt import read_halt
        state = read_halt()
        assert state is not None
        assert state.reason.startswith("auto:ConnectionLost")
        assert state.by == "engine-auto"
        broker.disconnect.assert_called_once()

    def test_unrelated_exception_propagates(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("TRADING_HALT_PATH", str(tmp_path / "HALTED"))

        broker = MagicMock()
        broker.connect = MagicMock()
        broker.disconnect = MagicMock()

        class _OtherFailStrategy:
            name = "other-fail"
            def on_tick(self, ctx):
                raise RuntimeError("programmer error")

        from trading_algo.config import TradingConfig
        cfg = TradingConfig.from_env()
        engine = Engine(
            broker=broker, config=cfg, strategy=_OtherFailStrategy(),
            risk=MagicMock(), confirm_token=None,
        )

        with pytest.raises(RuntimeError, match="programmer error"):
            engine.run_forever()

        from trading_algo.halt import read_halt
        assert read_halt() is None  # no auto-halt
