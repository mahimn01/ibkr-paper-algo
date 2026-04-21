"""End-to-end tests for --idempotency-key on cli.py place-order."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def isolated_stores(tmp_path, monkeypatch):
    """Point all on-disk stores at tmp so tests don't pollute real data."""
    idem_path = tmp_path / "idem.sqlite"
    monkeypatch.setenv("TRADING_IDEMPOTENCY_PATH", str(idem_path))
    monkeypatch.setenv("TRADING_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("TRADING_HALT_PATH", str(tmp_path / "HALTED"))
    return tmp_path


def _parse_place_args(parser, argv) -> argparse.Namespace:
    return parser.parse_args(["place-order"] + argv)


class TestArgparseFlag:
    def test_flag_is_registered(self) -> None:
        from trading_algo.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "place-order", "--broker", "sim", "--kind", "STK",
            "--symbol", "AAPL", "--side", "BUY", "--qty", "10",
            "--type", "MKT",
            "--idempotency-key", "TEST_AGENT_KEY",
        ])
        assert args.idempotency_key == "TEST_AGENT_KEY"

    def test_flag_defaults_none(self) -> None:
        from trading_algo.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "place-order", "--broker", "sim", "--kind", "STK",
            "--symbol", "AAPL", "--side", "BUY", "--qty", "10",
            "--type", "MKT",
        ])
        assert args.idempotency_key is None


class TestReplayFromCache:
    """Same key twice → second invocation replays, broker.place_order
    called exactly ONCE across both runs."""

    def test_replay_on_second_call(self, isolated_stores, monkeypatch) -> None:
        from trading_algo import cli

        # Stub the sim broker so we can count place_order calls.
        class StubBroker:
            def __init__(self):
                self.place_calls = 0
            def connect(self): return None
            def disconnect(self): return None
            def place_order(self, req):
                self.place_calls += 1
                from trading_algo.broker.base import OrderResult
                return OrderResult(order_id="1234", status="Submitted")
            def get_order_status(self, order_id):
                from trading_algo.broker.base import OrderStatus
                return OrderStatus(order_id=order_id, status="Submitted",
                                   filled=0, remaining=10, avg_fill_price=0)

        broker = StubBroker()
        monkeypatch.setattr(cli, "_make_broker", lambda kind, cfg: broker)
        # Force non-dry-run.
        monkeypatch.setenv("TRADING_DRY_RUN", "false")

        parser = cli.build_parser()
        argv = [
            "place-order", "--broker", "sim", "--kind", "STK",
            "--symbol", "AAPL", "--side", "BUY", "--qty", "10",
            "--type", "MKT",
            "--idempotency-key", "CRASHED_AGENT_KEY_01",
        ]

        # First call — real placement.
        args1 = parser.parse_args(argv)
        buf1 = io.StringIO()
        with redirect_stdout(buf1):
            rc1 = cli._cmd_place_order(args1)
        assert rc1 == 0
        assert broker.place_calls == 1

        # Second call — same key. Must REPLAY, not re-transmit.
        args2 = parser.parse_args(argv)
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = cli._cmd_place_order(args2)
        assert rc2 == 0
        assert broker.place_calls == 1, "place_order was re-called — double fill!"
        assert "replayed=true" in buf2.getvalue()


class TestOrderRefDerivation:
    """Deterministic orderRef: same idempotency-key → same orderRef across
    runs. That's how the orderbook-based dedup works.
    """

    def test_deterministic_order_ref(self, isolated_stores, monkeypatch) -> None:
        from trading_algo import cli
        from trading_algo.idempotency import derive_order_ref

        refs_captured: list[str] = []

        class StubBroker:
            def connect(self): return None
            def disconnect(self): return None
            def place_order(self, req):
                refs_captured.append(req.order_ref)
                from trading_algo.broker.base import OrderResult
                return OrderResult(order_id="1", status="Submitted")
            def get_order_status(self, order_id):
                from trading_algo.broker.base import OrderStatus
                return OrderStatus(order_id, "Submitted", 0, 0, 0)

        monkeypatch.setattr(cli, "_make_broker", lambda kind, cfg: StubBroker())
        monkeypatch.setenv("TRADING_DRY_RUN", "false")

        parser = cli.build_parser()
        argv = [
            "place-order", "--broker", "sim", "--kind", "STK",
            "--symbol", "AAPL", "--side", "BUY", "--qty", "10",
            "--type", "MKT",
            "--idempotency-key", "STABLE_KEY_07",
        ]
        args = parser.parse_args(argv)
        cli._cmd_place_order(args)

        assert len(refs_captured) == 1
        assert refs_captured[0] == derive_order_ref("STABLE_KEY_07")


class TestNoKeyLegacyPath:
    """Without --idempotency-key, the command goes through the direct
    broker.place_order path (backwards-compatible).
    """

    def test_no_key_direct_path(self, isolated_stores, monkeypatch) -> None:
        from trading_algo import cli

        class StubBroker:
            def __init__(self):
                self.place_calls = 0
            def connect(self): return None
            def disconnect(self): return None
            def place_order(self, req):
                self.place_calls += 1
                from trading_algo.broker.base import OrderResult
                return OrderResult(order_id="1", status="Submitted")
            def get_order_status(self, order_id):
                from trading_algo.broker.base import OrderStatus
                return OrderStatus(order_id, "Submitted", 0, 0, 0)

        broker = StubBroker()
        monkeypatch.setattr(cli, "_make_broker", lambda kind, cfg: broker)
        monkeypatch.setenv("TRADING_DRY_RUN", "false")

        parser = cli.build_parser()
        argv = [
            "place-order", "--broker", "sim", "--kind", "STK",
            "--symbol", "AAPL", "--side", "BUY", "--qty", "10",
            "--type", "MKT",
        ]
        args = parser.parse_args(argv)
        rc = cli._cmd_place_order(args)
        assert rc == 0
        assert broker.place_calls == 1
        # Idempotency SQLite should be empty (no rows written).
        db = isolated_stores / "idem.sqlite"
        if db.exists():
            import sqlite3
            con = sqlite3.connect(str(db))
            count = con.execute("SELECT COUNT(*) FROM writes").fetchone()[0]
            con.close()
            assert count == 0
