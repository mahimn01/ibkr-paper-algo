"""Tests for T3.2 rate_limits.py."""

from __future__ import annotations

import threading
import time

import pytest

from trading_algo.rate_limits import (
    HistoricalDedupeGuard,
    IBKRRateLimiter,
    SlidingWindowLimiter,
    TokenBucket,
    get_default_limiter,
    reset_default_limiter,
)


class TestTokenBucket:
    def test_immediate_acquire_when_full(self) -> None:
        b = TokenBucket(rate_per_sec=10.0, capacity=10)
        start = time.monotonic()
        assert b.acquire() is True
        assert time.monotonic() - start < 0.05

    def test_blocks_when_empty(self) -> None:
        b = TokenBucket(rate_per_sec=10.0, capacity=1)
        b.acquire()  # drain
        start = time.monotonic()
        assert b.acquire() is True
        # Refill at 10/s → should wait ~0.1s.
        elapsed = time.monotonic() - start
        assert 0.08 < elapsed < 0.3

    def test_non_blocking_returns_false(self) -> None:
        b = TokenBucket(rate_per_sec=10.0, capacity=1)
        b.acquire()
        assert b.acquire(block=False) is False

    def test_reject_over_capacity(self) -> None:
        b = TokenBucket(rate_per_sec=10.0, capacity=5)
        with pytest.raises(ValueError):
            b.acquire(tokens=10)

    def test_reject_zero_rate(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate_per_sec=0.0)


class TestSlidingWindow:
    def test_allows_up_to_max(self) -> None:
        w = SlidingWindowLimiter(max_requests=3, window_seconds=1.0)
        for _ in range(3):
            assert w.acquire() is True

    def test_blocks_over_max_non_blocking(self) -> None:
        w = SlidingWindowLimiter(max_requests=2, window_seconds=10.0)
        assert w.acquire()
        assert w.acquire()
        assert w.acquire(block=False) is False


class TestHistoricalDedupe:
    def test_first_call_passes(self) -> None:
        g = HistoricalDedupeGuard(gap_seconds=60.0)
        assert g.wait("key1") is True

    def test_second_identical_within_gap_blocks(self) -> None:
        g = HistoricalDedupeGuard(gap_seconds=60.0)
        g.wait("key1")
        # Non-blocking check — should refuse because 60s hasn't passed.
        assert g.wait("key1", block=False) is False

    def test_different_key_passes(self) -> None:
        g = HistoricalDedupeGuard(gap_seconds=60.0)
        g.wait("key1")
        assert g.wait("key2") is True  # distinct key

    def test_gap_expires(self) -> None:
        g = HistoricalDedupeGuard(gap_seconds=0.05)
        g.wait("k")
        time.sleep(0.08)
        assert g.wait("k") is True


class TestIBKRRateLimiter:
    def test_instantiation(self) -> None:
        lim = IBKRRateLimiter(general_rate=10.0, orders_rate=5.0)
        assert lim.general.rate == 10.0
        assert lim.orders.rate == 5.0

    def test_wait_order_goes_through(self) -> None:
        lim = IBKRRateLimiter()
        lim.wait_order()  # fresh bucket, immediate

    def test_wait_historical_without_dedupe_key(self) -> None:
        lim = IBKRRateLimiter()
        lim.wait_historical()  # no dedupe_key — should pass

    def test_wait_historical_with_dedupe_blocks_second_identical(self) -> None:
        lim = IBKRRateLimiter(historical_dedupe_seconds=60.0)
        lim.wait_historical(dedupe_key=("AAPL", "1 D", "5 mins", ""))
        # Check that dedupe guard now refuses in non-blocking mode.
        assert lim.historical_dedupe.wait(
            ("AAPL", "1 D", "5 mins", ""), block=False
        ) is False


class TestDefaultLimiter:
    def test_module_singleton(self) -> None:
        reset_default_limiter()
        a = get_default_limiter()
        b = get_default_limiter()
        assert a is b

    def test_reset(self) -> None:
        reset_default_limiter()
        a = get_default_limiter()
        reset_default_limiter()
        b = get_default_limiter()
        assert a is not b


class TestThreadSafety:
    def test_bucket_concurrent_acquire(self) -> None:
        b = TokenBucket(rate_per_sec=100.0, capacity=100)
        results = []

        def worker():
            results.append(b.acquire())

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(results) and len(results) == 50
