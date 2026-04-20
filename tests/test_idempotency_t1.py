"""Tests for the trading-algo T1 idempotency store + orderRef derivation."""

from __future__ import annotations

import json
import threading
import time

import pytest

from trading_algo.idempotency import (
    IdempotencyStore,
    WriteRecord,
    derive_order_ref,
)


@pytest.fixture
def store(tmp_path) -> IdempotencyStore:
    return IdempotencyStore(tmp_path / "idem.sqlite")


class TestRecordAttempt:
    def test_first_insert_true(self, store) -> None:
        assert store.record_attempt(
            key="K1", cmd="place-order", request={"qty": 10},
            order_ref="TA0123", request_id="R1",
        ) is True

    def test_duplicate_insert_false(self, store) -> None:
        store.record_attempt(key="K1", cmd="place-order", request={"qty": 10})
        assert store.record_attempt(
            key="K1", cmd="place-order", request={"qty": 99},  # different payload
        ) is False
        # Original preserved.
        rec = store.lookup("K1")
        assert json.loads(rec.request_json) == {"qty": 10}

    def test_lookup_absent(self, store) -> None:
        assert store.lookup("NOPE") is None

    def test_lookup_started_not_completed(self, store) -> None:
        store.record_attempt(key="K1", cmd="x", request={}, order_ref="TA_X")
        rec = store.lookup("K1")
        assert rec is not None
        assert rec.completed is False
        assert rec.order_ref == "TA_X"


class TestRecordCompletion:
    def test_marks_completed(self, store) -> None:
        store.record_attempt(key="K1", cmd="place-order", request={})
        store.record_completion(
            key="K1", result={"order_id": 42}, exit_code=0,
            ib_order_id=42, perm_id="PERM_X",
        )
        rec = store.lookup("K1")
        assert rec.completed is True
        assert rec.exit_code == 0
        assert rec.result == {"order_id": 42}
        assert rec.ib_order_id == 42
        assert rec.perm_id == "PERM_X"

    def test_completion_idempotent(self, store) -> None:
        store.record_attempt(key="K", cmd="x", request={})
        store.record_completion(key="K", result={"r": 1}, exit_code=0)
        rec1 = store.lookup("K")
        time.sleep(0.005)
        store.record_completion(key="K", result={"r": 1}, exit_code=0)
        rec2 = store.lookup("K")
        assert rec2.completed_at_ms >= rec1.completed_at_ms

    def test_completion_without_attempt_is_noop(self, store) -> None:
        store.record_completion(key="NEW", result={}, exit_code=0)
        assert store.lookup("NEW") is None


class TestFindByOrderRef:
    def test_forward_then_reverse(self, store) -> None:
        store.record_attempt(
            key="K", cmd="place-order", request={},
            order_ref="TA_ABCDEF",
        )
        rec = store.find_by_order_ref("TA_ABCDEF")
        assert rec is not None
        assert rec.key == "K"

    def test_missing_ref(self, store) -> None:
        assert store.find_by_order_ref("NOSUCH") is None


class TestDurability:
    def test_two_instances_see_same_data(self, tmp_path) -> None:
        path = tmp_path / "idem.sqlite"
        s1 = IdempotencyStore(path)
        s1.record_attempt(key="K", cmd="x", request={})
        s1.record_completion(key="K", result={"r": 42}, exit_code=0)

        s2 = IdempotencyStore(path)
        rec = s2.lookup("K")
        assert rec is not None
        assert rec.result == {"r": 42}


class TestPurge:
    def test_purges_old_completed(self, store) -> None:
        store.record_attempt(key="OLD", cmd="x", request={})
        store.record_completion(key="OLD", result={}, exit_code=0)
        cutoff_ms = int(time.time() * 1000) + 60 * 60 * 1000
        deleted = store.purge_older_than(cutoff_ms)
        assert deleted == 1
        assert store.lookup("OLD") is None

    def test_keeps_incomplete(self, store) -> None:
        store.record_attempt(key="GHOST", cmd="x", request={})
        cutoff_ms = int(time.time() * 1000) + 60 * 60 * 1000
        store.purge_older_than(cutoff_ms)
        assert store.lookup("GHOST") is not None


class TestDeriveOrderRef:
    def test_deterministic(self) -> None:
        assert derive_order_ref("agent-turn-42") == derive_order_ref("agent-turn-42")

    def test_default_length_30(self) -> None:
        ref = derive_order_ref("x")
        assert len(ref) == 30
        assert ref.startswith("TA")
        assert all(c.isalnum() for c in ref)

    def test_different_keys_different_refs(self) -> None:
        assert derive_order_ref("a") != derive_order_ref("b")

    def test_custom_prefix(self) -> None:
        ref = derive_order_ref("x", prefix="AA", length=20)
        assert ref.startswith("AA")
        assert len(ref) == 20

    def test_collision_resistance(self) -> None:
        """10k random keys → 10k unique refs."""
        import secrets
        keys = [secrets.token_hex(8) for _ in range(10_000)]
        refs = {derive_order_ref(k) for k in keys}
        assert len(refs) == len(keys)

    def test_length_bounds(self) -> None:
        with pytest.raises(ValueError):
            derive_order_ref("x", length=3)  # too short
        with pytest.raises(ValueError):
            derive_order_ref("x", length=50)  # too long for IBKR

    def test_prefix_too_long(self) -> None:
        with pytest.raises(ValueError):
            derive_order_ref("x", prefix="TOOLONGPREFIX", length=10)


class TestConcurrency:
    def test_many_threads_same_key(self, store) -> None:
        results: list[bool] = []

        def w() -> None:
            inserted = store.record_attempt(key="RACE", cmd="x", request={})
            results.append(inserted)

        threads = [threading.Thread(target=w) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert results.count(True) == 1
        assert results.count(False) == 19
