"""IBKR-specific rate-limit buckets.

Unlike Kite (hard per-endpoint limits), IBKR's TWS/Gateway caps throughput
on a per-connection basis:

* **General API messages**: ~50 msg/s soft cap (more and you get 162 PACING)
* **Historical data requests**:
  - 60s gap required between *identical* requests
  - 6 identical requests per 2s triggers error 162
  - 60 simultaneous open reqs max
* **Market-data snapshots**: counted toward market-data lines
  (default 100 lines for paper / most live accounts)
* **Order placements**: no hard server cap but IBKR recommends staggering
  at 50/s or below. Violations produce 162 PACING (retry) or account flag.

This module centralises those limits behind a per-endpoint token-bucket
façade so callers can say `rate_limits.wait_historical(symbol)` and the
correct guard is applied.

Reference:
https://interactivebrokers.github.io/tws-api/historical_limitations.html
https://interactivebrokers.github.io/tws-api/introduction.html#pacing_violation
"""

from __future__ import annotations

import collections
import logging
import random
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives (copied from kite-algo/resilience for self-containment)
# ---------------------------------------------------------------------------

class TokenBucket:
    """Thread-safe token bucket with Condition-variable waits.

    Consumers block until refill or non-blocking callers get False. Clamps
    floating-point deficit to >=0 to prevent spinloop inflation.
    """

    def __init__(self, rate_per_sec: float, capacity: float | None = None):
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be positive, got {rate_per_sec}")
        self.rate = rate_per_sec
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_sec)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._cond = threading.Condition()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def acquire(self, tokens: float = 1.0, block: bool = True) -> bool:
        if tokens > self.capacity:
            raise ValueError(f"requested {tokens} > capacity {self.capacity}")
        with self._cond:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._cond.notify_all()
                    return True
                if not block:
                    return False
                deficit = max(0.0, tokens - self._tokens)
                expected = deficit / self.rate if self.rate > 0 else 0.1
                wait = max(0.001, min(expected, self.capacity / self.rate)) + random.random() * 0.005
                self._cond.wait(timeout=wait)


class SlidingWindowLimiter:
    """N requests per M seconds window."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max = max_requests
        self.window = window_seconds
        self._events: collections.deque[float] = collections.deque(maxlen=max_requests * 2)
        self._lock = threading.Lock()

    def acquire(self, block: bool = True) -> bool:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window
                while self._events and self._events[0] < cutoff:
                    self._events.popleft()
                if len(self._events) < self.max:
                    self._events.append(now)
                    return True
                wait = self._events[0] + self.window - now
            if not block:
                return False
            time.sleep(max(0.01, wait) + 0.001)


# ---------------------------------------------------------------------------
# Historical-specific: 60s gap between identical requests
# ---------------------------------------------------------------------------

class HistoricalDedupeGuard:
    """Enforces IBKR's "60s between identical historical requests" rule.

    A request is identified by a hashable tuple: (contract_key, end_datetime,
    duration, bar_size, what_to_show, use_rth). Callers pass the key and we
    remember the last-sent timestamp. A second request with the same key
    within 60s blocks until enough time elapses.
    """

    def __init__(self, gap_seconds: float = 60.0) -> None:
        self.gap = gap_seconds
        self._last_seen: dict[Any, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: Any, block: bool = True) -> bool:
        while True:
            with self._lock:
                now = time.monotonic()
                last = self._last_seen.get(key)
                if last is None or now - last >= self.gap:
                    self._last_seen[key] = now
                    return True
                wait = self.gap - (now - last)
            if not block:
                return False
            time.sleep(max(0.1, wait))


# ---------------------------------------------------------------------------
# IBKR rate-limit facade
# ---------------------------------------------------------------------------

class IBKRRateLimiter:
    """IBKR-specific buckets. Defaults are conservative — agents that know
    they're on a higher-tier account can pass overrides.

    Buckets:
      - general:         50 msg/s (default TWS API soft cap)
      - historical:      6 msg / 2s sliding window + 60s dedupe guard
      - snapshot:        30 msg/s (market-data snapshots — soft)
      - orders:          25 msg/s (stagger recommended; above this,
                         IBKR risk review can flag the account)
    """

    def __init__(
        self,
        *,
        general_rate: float = 50.0,
        historical_max_per_window: int = 6,
        historical_window_seconds: float = 2.0,
        historical_dedupe_seconds: float = 60.0,
        snapshot_rate: float = 30.0,
        orders_rate: float = 25.0,
    ) -> None:
        self.general = TokenBucket(general_rate, capacity=general_rate)
        self.historical_window = SlidingWindowLimiter(
            max_requests=historical_max_per_window,
            window_seconds=historical_window_seconds,
        )
        self.historical_dedupe = HistoricalDedupeGuard(gap_seconds=historical_dedupe_seconds)
        self.snapshot = TokenBucket(snapshot_rate, capacity=snapshot_rate)
        self.orders = TokenBucket(orders_rate, capacity=orders_rate)

    def wait_general(self) -> None:
        self.general.acquire()

    def wait_snapshot(self) -> None:
        self.snapshot.acquire()
        self.general.acquire()

    def wait_historical(self, dedupe_key: Any | None = None) -> None:
        """Block until the historical-data request can proceed.

        Pass `dedupe_key` to also enforce the 60s-between-identical rule.
        If dedupe_key is None, the dedupe check is skipped — useful for
        deliberately chunked retrieval of long histories where each chunk
        has a distinct end_datetime.
        """
        if dedupe_key is not None:
            self.historical_dedupe.wait(dedupe_key)
        self.historical_window.acquire()
        self.general.acquire()

    def wait_order(self) -> None:
        self.orders.acquire()
        self.general.acquire()


# ---------------------------------------------------------------------------
# Module-global default limiter
# ---------------------------------------------------------------------------

_DEFAULT_LIMITER: IBKRRateLimiter | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_limiter() -> IBKRRateLimiter:
    """Module-singleton limiter — use this unless you need custom rates."""
    global _DEFAULT_LIMITER
    with _DEFAULT_LOCK:
        if _DEFAULT_LIMITER is None:
            _DEFAULT_LIMITER = IBKRRateLimiter()
        return _DEFAULT_LIMITER


def reset_default_limiter() -> None:
    """Test-only: discard the module singleton so tests get a fresh one."""
    global _DEFAULT_LIMITER
    with _DEFAULT_LOCK:
        _DEFAULT_LIMITER = None
