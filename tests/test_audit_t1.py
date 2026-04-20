"""Tests for trading-algo T1 NDJSON audit log."""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from pathlib import Path

import pytest

from trading_algo.audit import (
    iter_entries,
    log_command,
    purge_older_than,
    tail,
)


@pytest.fixture
def audit_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "audit"
    monkeypatch.setenv("TRADING_AUDIT_DIR", str(root))
    return root


class TestWriteRead:
    def test_single_line_per_entry(self, audit_root) -> None:
        log_command(cmd="place-order", request_id="R1", args={"a": 1}, exit_code=0, root=audit_root)
        log_command(cmd="cancel", request_id="R2", args={}, exit_code=0, root=audit_root)
        files = list(audit_root.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2

    def test_valid_json_per_line(self, audit_root) -> None:
        log_command(cmd="x", request_id="R1", args={"a": 1}, exit_code=0, root=audit_root)
        entry = json.loads(next(audit_root.glob("*.jsonl")).read_text().strip())
        assert entry["cmd"] == "x"
        assert entry["args"] == {"a": 1}

    def test_ib_fields_persisted(self, audit_root) -> None:
        log_command(
            cmd="place-order", request_id="R", args={}, exit_code=0,
            ib_order_id=42, perm_id="PERM_X", order_ref="TA_ORDERREF",
            account="DU1234567", root=audit_root,
        )
        entry = json.loads(next(audit_root.glob("*.jsonl")).read_text().strip())
        assert entry["ib_order_id"] == 42
        assert entry["perm_id"] == "PERM_X"
        assert entry["order_ref"] == "TA_ORDERREF"
        assert entry["account"] == "DU1234567"

    def test_filename_is_local_date(self, audit_root) -> None:
        log_command(cmd="x", request_id="R", args={}, root=audit_root)
        today = datetime.now().date()
        assert (audit_root / f"{today.isoformat()}.jsonl").exists()


class TestRedaction:
    def test_key_level_secret_redacted(self, audit_root) -> None:
        log_command(
            cmd="x", request_id="R",
            args={"api_secret": "THIS_IS_SECRET", "symbol": "AAPL"},
            root=audit_root,
        )
        content = next(audit_root.glob("*.jsonl")).read_text()
        assert "THIS_IS_SECRET" not in content
        assert "REDACTED" in content
        assert "AAPL" in content  # non-secret preserved

    def test_nested_dict_secret_redacted(self, audit_root) -> None:
        log_command(
            cmd="x", request_id="R",
            args={"creds": {"password": "MYPASS", "user": "u1"}},
            root=audit_root,
        )
        content = next(audit_root.glob("*.jsonl")).read_text()
        assert "MYPASS" not in content
        assert "u1" in content

    def test_flex_token_redacted(self, audit_root) -> None:
        log_command(
            cmd="flex-pull", request_id="R",
            args={"flex_token": "12345XYZ"},
            root=audit_root,
        )
        content = next(audit_root.glob("*.jsonl")).read_text()
        assert "12345XYZ" not in content


class TestConcurrency:
    def test_concurrent_writes_all_well_formed(self, audit_root) -> None:
        def w(i: int) -> None:
            for j in range(25):
                log_command(
                    cmd="place-order", request_id=f"T{i}_J{j}",
                    args={"i": i, "j": j}, exit_code=0, root=audit_root,
                )
        threads = [threading.Thread(target=w, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        files = list(audit_root.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 250
        for line in lines:
            json.loads(line)  # every line well-formed


class TestIter:
    def test_iter_all(self, audit_root) -> None:
        for i in range(5):
            log_command(cmd="x", request_id=f"R{i}", args={}, exit_code=0, root=audit_root)
        got = list(iter_entries(root=audit_root))
        assert len(got) == 5

    def test_filter_by_cmd(self, audit_root) -> None:
        log_command(cmd="place-order", request_id="A", args={}, exit_code=0, root=audit_root)
        log_command(cmd="cancel", request_id="B", args={}, exit_code=0, root=audit_root)
        got = list(iter_entries(cmd="cancel", root=audit_root))
        assert len(got) == 1 and got[0]["request_id"] == "B"

    def test_outcome_error_filter(self, audit_root) -> None:
        log_command(cmd="x", request_id="A", args={}, exit_code=0, root=audit_root)
        log_command(cmd="x", request_id="B", args={}, exit_code=4, root=audit_root)
        log_command(cmd="x", request_id="C", args={}, exit_code=5, root=audit_root)
        got = list(iter_entries(outcome="error", root=audit_root))
        assert {e["request_id"] for e in got} == {"B", "C"}

    def test_date_range(self, audit_root) -> None:
        old = audit_root / "2025-01-01.jsonl"
        new = audit_root / "2026-04-21.jsonl"
        old.parent.mkdir(parents=True, exist_ok=True)
        for f, rid in [(old, "OLD"), (new, "NEW")]:
            f.write_text(json.dumps({
                "cmd": "x", "request_id": rid, "exit_code": 0,
                "ts_epoch_ms": 0, "ts": "", "args": {},
            }) + "\n")
        got = list(iter_entries(since=date(2026, 1, 1), root=audit_root))
        assert [e["request_id"] for e in got] == ["NEW"]

    def test_malformed_lines_skipped(self, audit_root) -> None:
        f = audit_root / "2026-04-21.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            '{"cmd":"x","request_id":"A","exit_code":0,"ts_epoch_ms":1,"ts":"","args":{}}\n'
            'not-json-here\n'
            '{"cmd":"x","request_id":"B","exit_code":0,"ts_epoch_ms":2,"ts":"","args":{}}\n'
        )
        got = list(iter_entries(root=audit_root))
        assert [e["request_id"] for e in got] == ["A", "B"]


class TestTail:
    def test_last_n(self, audit_root) -> None:
        for i in range(10):
            log_command(cmd="x", request_id=f"R{i}", args={}, exit_code=0, root=audit_root)
        got = tail(3, root=audit_root)
        assert [e["request_id"] for e in got] == ["R7", "R8", "R9"]


class TestPerms:
    def test_file_mode_0o600(self, audit_root) -> None:
        log_command(cmd="x", request_id="R", args={}, root=audit_root)
        f = next(audit_root.glob("*.jsonl"))
        if os.name == "posix":
            assert f.stat().st_mode & 0o777 == 0o600


class TestPurge:
    def test_purges_old(self, audit_root) -> None:
        old = audit_root / "2018-01-01.jsonl"
        audit_root.mkdir(parents=True, exist_ok=True)
        old.write_text("")
        deleted = purge_older_than(days=365, root=audit_root)
        assert deleted == 1
        assert not old.exists()

    def test_keeps_recent(self, audit_root) -> None:
        log_command(cmd="x", request_id="R", args={}, root=audit_root)
        deleted = purge_older_than(days=30, root=audit_root)
        assert deleted == 0
