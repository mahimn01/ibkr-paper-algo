"""Tests for T4.3 stream / tail-ticks."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import pytest

from trading_algo import cli


def _args(**kw) -> argparse.Namespace:
    defaults = {
        "broker": "sim", "ibkr_host": None, "ibkr_port": None,
        "ibkr_client_id": None, "paper": False, "live": False,
        "dry_run": False, "no_dry_run": False, "allow_live": False,
        "confirm_token": None, "explain": False,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestStream:
    def test_emits_ticks_to_buffer(self, monkeypatch, tmp_path) -> None:
        fake_broker = MagicMock()
        tick_price = [100.0]

        def _snap(instrument):
            tick_price[0] += 0.1
            return MagicMock(
                instrument=instrument, bid=tick_price[0] - 0.01,
                ask=tick_price[0] + 0.01, last=tick_price[0],
                close=tick_price[0] - 1, volume=1_000_000,
                timestamp_epoch_s=123,
            )

        fake_broker.get_market_data_snapshot.side_effect = _snap
        monkeypatch.setattr(cli, "_make_broker", lambda *a, **k: fake_broker)

        buffer = tmp_path / "ticks.ndjson"
        args = _args(
            resource=None,
            symbol="AAPL", kind="STK", exchange=None, currency=None,
            expiry=None, right=None, strike=None, multiplier=None,
            every=0.2, duration=0.6, buffer_to=str(buffer),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_stream(args)
        assert rc == 0

        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert len(lines) >= 2
        first = json.loads(lines[0])
        assert first["_seq"] == 1
        assert first["symbol"] == "AAPL"

        # Buffer file also got them.
        file_lines = [l for l in buffer.read_text().splitlines() if l.strip()]
        assert len(file_lines) == len(lines)

    def test_stream_explain_short_circuits(self) -> None:
        args = _args(
            explain=True, cmd="stream",
            symbol="AAPL", kind="STK", exchange=None, currency=None,
            expiry=None, right=None, strike=None, multiplier=None,
            every=0.2, duration=10.0, buffer_to=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_stream(args)
        assert rc == 0


class TestTailTicks:
    def test_empty_file_returns_no_ticks(self, tmp_path) -> None:
        args = argparse.Namespace(
            file=str(tmp_path / "missing.ndjson"),
            from_seq=0, max=None, explain=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli._cmd_tail_ticks(args)
        assert rc == 0
        p = json.loads(buf.getvalue())
        assert p["count"] == 0

    def test_reads_and_filters(self, tmp_path) -> None:
        path = tmp_path / "ticks.ndjson"
        with open(path, "w") as fh:
            for i in range(1, 6):
                fh.write(json.dumps({"_seq": i, "symbol": "X", "last": 100 + i}) + "\n")
        args = argparse.Namespace(
            file=str(path), from_seq=2, max=None, explain=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_tail_ticks(args)
        p = json.loads(buf.getvalue())
        assert p["count"] == 3  # seq 3,4,5
        assert p["last_seq"] == 5

    def test_respects_max(self, tmp_path) -> None:
        path = tmp_path / "ticks.ndjson"
        with open(path, "w") as fh:
            for i in range(1, 11):
                fh.write(json.dumps({"_seq": i, "last": i}) + "\n")
        args = argparse.Namespace(
            file=str(path), from_seq=0, max=3, explain=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_tail_ticks(args)
        p = json.loads(buf.getvalue())
        assert p["count"] == 3
        # Last 3 tail.
        assert [t["_seq"] for t in p["ticks"]] == [8, 9, 10]

    def test_malformed_lines_ignored(self, tmp_path) -> None:
        path = tmp_path / "ticks.ndjson"
        path.write_text('{"_seq":1}\nnot-json\n{"_seq":2}\n')
        args = argparse.Namespace(
            file=str(path), from_seq=0, max=None, explain=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._cmd_tail_ticks(args)
        p = json.loads(buf.getvalue())
        assert p["count"] == 2  # malformed skipped


class TestParserRegistration:
    def test_subcommands(self) -> None:
        parser = cli.build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        assert "stream" in sub.choices
        assert "tail-ticks" in sub.choices
