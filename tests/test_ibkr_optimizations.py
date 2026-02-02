"""
Comprehensive tests for IBKR speed optimizations.

Tests cover:
1. Contract cache (LRU with TTL)
2. Rate limiter (token bucket)
3. Circuit breaker pattern
4. Connection health monitor
5. Event-based order confirmation
6. Parallel data fetching
7. Compressed cache
8. Integration tests
"""

import gzip
import json
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# CONTRACT CACHE TESTS
# =============================================================================

class TestContractCache:
    """Tests for ContractCache with LRU and TTL."""

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=100, ttl=3600.0)
        instrument = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")
        contract = Mock(conId=12345)

        cache.put(instrument, contract)
        result = cache.get(instrument)

        assert result is not None
        assert result.conId == 12345

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=100, ttl=3600.0)
        instrument = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")

        result = cache.get(instrument)
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=2, ttl=3600.0)

        # Add 3 items to cache with capacity 2
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
            cache.put(instrument, Mock(symbol=symbol))

        # First item should be evicted
        aapl = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")
        assert cache.get(aapl) is None

        # Last two should still be present
        googl = InstrumentSpec(kind="STK", symbol="GOOGL", exchange="SMART", currency="USD")
        msft = InstrumentSpec(kind="STK", symbol="MSFT", exchange="SMART", currency="USD")
        assert cache.get(googl) is not None
        assert cache.get(msft) is not None

    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=100, ttl=0.1)  # 100ms TTL
        instrument = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")

        cache.put(instrument, Mock(conId=12345))

        # Should be present immediately
        assert cache.get(instrument) is not None

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired
        assert cache.get(instrument) is None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=100, ttl=3600.0)
        instrument = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")

        # One miss
        cache.get(instrument)

        # One put + one hit
        cache.put(instrument, Mock())
        cache.get(instrument)

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1

    def test_cache_thread_safety(self):
        """Test thread-safe cache operations."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=100, ttl=3600.0)
        errors = []

        def worker(symbol: str):
            try:
                instrument = InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")
                for _ in range(100):
                    cache.put(instrument, Mock(symbol=symbol))
                    cache.get(instrument)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"SYM{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Tests for token bucket rate limiter."""

    def test_rate_limiter_allows_burst(self):
        """Test that burst requests are allowed."""
        from trading_algo.broker.ibkr import RateLimiter

        limiter = RateLimiter(max_rate=10.0, burst_size=5, min_interval=0.01)

        # Should allow burst of 5 immediately
        for _ in range(5):
            assert limiter.acquire(timeout=0.1) is True

    def test_rate_limiter_enforces_rate(self):
        """Test rate limiting after burst."""
        from trading_algo.broker.ibkr import RateLimiter

        limiter = RateLimiter(max_rate=100.0, burst_size=2, min_interval=0.001)

        # Use up burst
        limiter.acquire(timeout=1.0)
        limiter.acquire(timeout=1.0)

        # Third request should wait
        start = time.monotonic()
        limiter.acquire(timeout=1.0)
        elapsed = time.monotonic() - start

        # Should have waited for token replenishment
        assert elapsed >= 0.005  # At least some wait

    def test_rate_limiter_timeout(self):
        """Test rate limiter timeout."""
        from trading_algo.broker.ibkr import RateLimiter

        limiter = RateLimiter(max_rate=1.0, burst_size=1, min_interval=0.5)

        # Use the one token
        assert limiter.acquire(timeout=1.0) is True

        # Should timeout quickly
        start = time.monotonic()
        result = limiter.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.2

    def test_rate_limiter_min_interval(self):
        """Test minimum interval between requests."""
        from trading_algo.broker.ibkr import RateLimiter

        limiter = RateLimiter(max_rate=1000.0, burst_size=100, min_interval=0.05)

        t1 = time.monotonic()
        limiter.acquire(timeout=1.0)
        t2 = time.monotonic()
        limiter.acquire(timeout=1.0)
        t3 = time.monotonic()

        # Second request should wait for min_interval
        assert (t3 - t2) >= 0.04  # Allow small tolerance


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_circuit_starts_closed(self):
        """Test circuit starts in closed state."""
        from trading_algo.broker.ibkr import CircuitBreaker, CircuitState

        cb = CircuitBreaker(threshold=3, timeout=1.0)
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after threshold failures."""
        from trading_algo.broker.ibkr import CircuitBreaker, CircuitState

        cb = CircuitBreaker(threshold=3, timeout=1.0)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        from trading_algo.broker.ibkr import CircuitBreaker, CircuitState

        cb = CircuitBreaker(threshold=1, timeout=0.1)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_closes_on_success(self):
        """Test circuit closes on successful call."""
        from trading_algo.broker.ibkr import CircuitBreaker, CircuitState

        cb = CircuitBreaker(threshold=1, timeout=0.1)

        cb.record_failure()
        time.sleep(0.15)  # Transition to half-open

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_protect_context_manager(self):
        """Test protect context manager."""
        from trading_algo.broker.ibkr import CircuitBreaker, IBKRCircuitOpenError

        cb = CircuitBreaker(threshold=1, timeout=1.0)

        # Should work when closed
        with cb.protect():
            pass

        # Open the circuit
        cb.record_failure()

        # Should raise when open
        with pytest.raises(IBKRCircuitOpenError):
            with cb.protect():
                pass


# =============================================================================
# CONNECTION HEALTH MONITOR TESTS
# =============================================================================

class TestConnectionHealthMonitor:
    """Tests for connection health monitoring."""

    def test_health_check_healthy(self):
        """Test health check for healthy connection."""
        from trading_algo.broker.ibkr import ConnectionHealthMonitor

        monitor = ConnectionHealthMonitor(check_interval=0.0)
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        assert monitor.check_health(mock_ib) is True

    def test_health_check_unhealthy(self):
        """Test health check for unhealthy connection."""
        from trading_algo.broker.ibkr import ConnectionHealthMonitor

        monitor = ConnectionHealthMonitor(check_interval=0.0)
        mock_ib = Mock()
        mock_ib.isConnected.return_value = False

        assert monitor.check_health(mock_ib) is False

    def test_health_check_none_connection(self):
        """Test health check with None connection."""
        from trading_algo.broker.ibkr import ConnectionHealthMonitor

        monitor = ConnectionHealthMonitor(check_interval=0.0)
        assert monitor.check_health(None) is False

    def test_reconnect_callback(self):
        """Test reconnection callback."""
        from trading_algo.broker.ibkr import ConnectionHealthMonitor

        monitor = ConnectionHealthMonitor(
            max_reconnect_attempts=2,
            reconnect_base_delay=0.01
        )

        reconnect_called = [0]

        def reconnect():
            reconnect_called[0] += 1
            return reconnect_called[0] >= 2  # Succeed on second attempt

        monitor.set_reconnect_callback(reconnect)
        result = monitor.attempt_reconnect()

        assert result is True
        assert reconnect_called[0] == 2


# =============================================================================
# DATA PROVIDER TESTS
# =============================================================================

class TestCacheManager:
    """Tests for compressed cache manager."""

    def test_cache_put_and_get_compressed(self):
        """Test compressed cache put and get."""
        from trading_algo.backtest_v2.data_provider import CacheManager, DataRequest
        from trading_algo.backtest_v2.models import Bar

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(
                cache_dir=Path(tmpdir),
                compression=True,
                ttl_days=1
            )

            request = DataRequest(
                symbol="AAPL",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                bar_size="5 mins"
            )

            bars = [
                Bar(timestamp=datetime(2024, 1, 2, 10, 0), open=100, high=101, low=99, close=100.5, volume=1000),
                Bar(timestamp=datetime(2024, 1, 2, 10, 5), open=100.5, high=102, low=100, close=101, volume=1500),
            ]

            cache.put(request, bars)
            result = cache.get(request)

            assert result is not None
            assert len(result) == 2
            assert result[0].open == 100

    def test_cache_compression_reduces_size(self):
        """Test that compression reduces file size."""
        from trading_algo.backtest_v2.data_provider import CacheManager, DataRequest
        from trading_algo.backtest_v2.models import Bar

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create caches with and without compression
            compressed_cache = CacheManager(Path(tmpdir) / "compressed", compression=True)
            uncompressed_cache = CacheManager(Path(tmpdir) / "uncompressed", compression=False)

            request = DataRequest(
                symbol="AAPL",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            # Create many bars (use different days to avoid minute overflow)
            bars = [
                Bar(
                    timestamp=datetime(2024, 1, 2 + i // 60, 10, i % 60),
                    open=100+i, high=101+i, low=99+i, close=100.5+i, volume=1000+i
                )
                for i in range(100)
            ]

            compressed_cache.put(request, bars)
            uncompressed_cache.put(request, bars)

            # Get file sizes
            compressed_files = list((Path(tmpdir) / "compressed").glob("*.json.gz"))
            uncompressed_files = list((Path(tmpdir) / "uncompressed").glob("*.json"))

            assert len(compressed_files) == 1
            assert len(uncompressed_files) == 1

            compressed_size = compressed_files[0].stat().st_size
            uncompressed_size = uncompressed_files[0].stat().st_size

            # Compressed should be significantly smaller
            assert compressed_size < uncompressed_size * 0.5

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        import os
        from trading_algo.backtest_v2.data_provider import CacheManager, DataRequest
        from trading_algo.backtest_v2.models import Bar

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(
                cache_dir=Path(tmpdir),
                compression=False,
                ttl_days=1  # 1 day TTL
            )

            request = DataRequest(symbol="AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
            bars = [Bar(timestamp=datetime(2024, 1, 2, 10, 0), open=100, high=101, low=99, close=100.5, volume=1000)]

            cache.put(request, bars)

            # Should be present immediately
            result = cache.get(request)
            assert result is not None

            # Modify file timestamp to make it 2 days old
            cache_files = list(Path(tmpdir).glob("*.json"))
            assert len(cache_files) == 1
            old_time = time.time() - (2 * 24 * 60 * 60)  # 2 days ago
            os.utime(cache_files[0], (old_time, old_time))

            # Should be expired now
            result = cache.get(request)
            assert result is None


class TestParallelFetcher:
    """Tests for parallel data fetching."""

    def test_parallel_fetch_multiple_symbols(self):
        """Test parallel fetching of multiple symbols."""
        from trading_algo.backtest_v2.data_provider import ParallelFetcher, DataRequest
        from trading_algo.backtest_v2.models import Bar

        fetcher = ParallelFetcher(max_workers=3, min_interval=0.01)

        requests = [
            DataRequest(symbol=f"SYM{i}", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
            for i in range(5)
        ]

        def mock_fetch(req: DataRequest):
            time.sleep(0.05)  # Simulate API delay
            return [Bar(timestamp=datetime.now(), open=100, high=101, low=99, close=100.5, volume=1000)]

        start = time.time()
        result = fetcher.fetch_parallel(requests, mock_fetch)
        elapsed = time.time() - start

        # Should complete faster than sequential (5 * 0.05 = 0.25s)
        assert elapsed < 0.2  # Allow some overhead
        assert len(result) == 5

    def test_parallel_fetch_progress_callback(self):
        """Test progress callback is called."""
        from trading_algo.backtest_v2.data_provider import ParallelFetcher, DataRequest
        from trading_algo.backtest_v2.models import Bar

        fetcher = ParallelFetcher(max_workers=2, min_interval=0.01)

        requests = [
            DataRequest(symbol=f"SYM{i}", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
            for i in range(3)
        ]

        progress_calls = []

        def mock_fetch(req):
            return [Bar(timestamp=datetime.now(), open=100, high=101, low=99, close=100.5, volume=1000)]

        def progress_callback(pct, msg):
            progress_calls.append((pct, msg))

        fetcher.fetch_parallel(requests, mock_fetch, progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[-1][0] == 1.0  # Final progress is 100%

    def test_parallel_fetch_handles_errors(self):
        """Test graceful handling of fetch errors."""
        from trading_algo.backtest_v2.data_provider import ParallelFetcher, DataRequest
        from trading_algo.backtest_v2.models import Bar

        fetcher = ParallelFetcher(max_workers=2, min_interval=0.01)

        requests = [
            DataRequest(symbol=f"SYM{i}", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
            for i in range(3)
        ]

        def mock_fetch(req):
            if req.symbol == "SYM1":
                raise Exception("Simulated error")
            return [Bar(timestamp=datetime.now(), open=100, high=101, low=99, close=100.5, volume=1000)]

        result = fetcher.fetch_parallel(requests, mock_fetch)

        # Should have 2 results (SYM0 and SYM2), SYM1 failed
        assert len(result) == 2
        assert "SYM0" in result
        assert "SYM2" in result
        assert "SYM1" not in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIBKRBrokerIntegration:
    """Integration tests for IBKRBroker with mocked IB."""

    def test_broker_initialization(self):
        """Test broker initializes with optimization components."""
        from trading_algo.broker.ibkr import IBKRBroker, IBKROptimizationConfig
        from trading_algo.config import IBKRConfig

        config = IBKRConfig(host="127.0.0.1", port=7497, client_id=1)
        opt_config = IBKROptimizationConfig(
            contract_cache_size=100,
            max_requests_per_second=50.0
        )

        broker = IBKRBroker(
            config=config,
            require_paper=False,
            optimization_config=opt_config
        )

        assert broker._contract_cache is not None
        assert broker._rate_limiter is not None
        assert broker._circuit_breaker is not None
        assert broker._health_monitor is not None

    def test_contract_caching_on_order(self):
        """Test that contracts are cached during order placement."""
        from trading_algo.broker.ibkr import IBKRBroker
        from trading_algo.broker.base import OrderRequest
        from trading_algo.config import IBKRConfig
        from trading_algo.instruments import InstrumentSpec

        config = IBKRConfig(host="127.0.0.1", port=7497, client_id=1)

        # Create mock IB
        mock_ib = MagicMock()
        mock_ib.qualifyContracts.return_value = [Mock(conId=12345)]
        mock_ib.placeOrder.return_value = Mock(
            orderStatus=Mock(status="Submitted"),
            order=Mock(orderId=1)
        )

        broker = IBKRBroker(
            config=config,
            require_paper=False,
            ib_factory=lambda: mock_ib
        )
        broker._ib = mock_ib

        instrument = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")
        order = OrderRequest(
            instrument=instrument,
            side="BUY",
            quantity=100,
            order_type="MKT",
            tif="DAY"
        )

        # First order should qualify contract
        broker.place_order(order)
        first_qualify_count = mock_ib.qualifyContracts.call_count

        # Second order should use cache
        broker.place_order(order)
        second_qualify_count = mock_ib.qualifyContracts.call_count

        # Should only have qualified once (cached)
        assert first_qualify_count == 1
        assert second_qualify_count == 1

        # Verify cache hit
        stats = broker.get_cache_stats()
        assert stats["hits"] >= 1

    def test_event_based_order_confirmation(self):
        """Test event-based order confirmation (no blind sleep)."""
        from trading_algo.broker.ibkr import IBKRBroker
        from trading_algo.broker.base import OrderRequest
        from trading_algo.config import IBKRConfig
        from trading_algo.instruments import InstrumentSpec

        config = IBKRConfig(host="127.0.0.1", port=7497, client_id=1)

        # Create mock IB that returns status immediately
        mock_ib = MagicMock()
        mock_ib.qualifyContracts.return_value = [Mock(conId=12345)]

        # Mock trade with immediate "Submitted" status
        mock_trade = Mock(
            orderStatus=Mock(status="Submitted"),
            order=Mock(orderId=1)
        )
        mock_ib.placeOrder.return_value = mock_trade
        mock_ib.sleep = lambda x: time.sleep(x)

        broker = IBKRBroker(
            config=config,
            require_paper=False,
            ib_factory=lambda: mock_ib
        )
        broker._ib = mock_ib

        instrument = InstrumentSpec(kind="STK", symbol="AAPL", exchange="SMART", currency="USD")
        order = OrderRequest(
            instrument=instrument,
            side="BUY",
            quantity=100,
            order_type="MKT",
            tif="DAY"
        )

        start = time.time()
        result = broker.place_order(order)
        elapsed = time.time() - start

        # Should return quickly (status was already "Submitted")
        assert elapsed < 0.5  # Much faster than old 250ms sleep
        assert result.status == "Submitted"


class TestDataProviderIntegration:
    """Integration tests for DataProvider."""

    def test_data_provider_with_cache(self):
        """Test data provider caches fetched data."""
        from trading_algo.backtest_v2.data_provider import DataProvider, DataRequest, DataProviderConfig
        from trading_algo.backtest_v2.models import Bar

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DataProviderConfig(cache_compression=True)
            provider = DataProvider(cache_dir=Path(tmpdir), config=config)

            # Mock Yahoo Finance
            mock_bars = [
                Bar(timestamp=datetime(2024, 1, 2, 10, 0), open=100, high=101, low=99, close=100.5, volume=1000)
            ]

            with patch.object(provider, '_fetch_from_yahoo', return_value=mock_bars):
                request = DataRequest(symbol="AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

                # First fetch - should call Yahoo
                result1 = provider.get_data([request])
                assert "AAPL" in result1

            # Second fetch - should use cache (no Yahoo mock needed)
            result2 = provider.get_data([request])
            assert "AAPL" in result2
            assert len(result2["AAPL"]) == 1

    def test_cache_stats(self):
        """Test cache statistics."""
        from trading_algo.backtest_v2.data_provider import DataProvider, DataProviderConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DataProviderConfig(cache_compression=True)
            provider = DataProvider(cache_dir=Path(tmpdir), config=config)

            stats = provider.get_cache_stats()

            assert "files" in stats
            assert "total_size_mb" in stats
            assert "compression" in stats
            assert stats["compression"] is True


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_contract_cache_performance(self):
        """Benchmark contract cache performance."""
        from trading_algo.broker.ibkr import ContractCache
        from trading_algo.instruments import InstrumentSpec

        cache = ContractCache(maxsize=1000, ttl=3600.0)

        # Warm up cache
        for i in range(100):
            instrument = InstrumentSpec(kind="STK", symbol=f"SYM{i}", exchange="SMART", currency="USD")
            cache.put(instrument, Mock(symbol=f"SYM{i}"))

        # Benchmark cache hits
        start = time.time()
        for _ in range(10000):
            instrument = InstrumentSpec(kind="STK", symbol="SYM50", exchange="SMART", currency="USD")
            cache.get(instrument)
        elapsed = time.time() - start

        # Should complete 10k lookups very quickly
        assert elapsed < 1.0  # Less than 100us per lookup

    def test_rate_limiter_performance(self):
        """Benchmark rate limiter performance."""
        from trading_algo.broker.ibkr import RateLimiter

        limiter = RateLimiter(max_rate=1000.0, burst_size=100, min_interval=0.001)

        start = time.time()
        for _ in range(100):
            limiter.acquire(timeout=1.0)
        elapsed = time.time() - start

        # 100 acquisitions should complete in reasonable time
        assert elapsed < 2.0

    def test_parallel_fetch_speedup(self):
        """Verify parallel fetching provides speedup."""
        from trading_algo.backtest_v2.data_provider import ParallelFetcher, DataRequest
        from trading_algo.backtest_v2.models import Bar

        # Sequential timing
        def slow_fetch(req):
            time.sleep(0.02)
            return [Bar(timestamp=datetime.now(), open=100, high=101, low=99, close=100.5, volume=1000)]

        requests = [
            DataRequest(symbol=f"SYM{i}", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
            for i in range(10)
        ]

        # Sequential baseline
        seq_start = time.time()
        for req in requests:
            slow_fetch(req)
        seq_elapsed = time.time() - seq_start

        # Parallel execution
        fetcher = ParallelFetcher(max_workers=5, min_interval=0.001)
        par_start = time.time()
        fetcher.fetch_parallel(requests, slow_fetch)
        par_elapsed = time.time() - par_start

        # Parallel should be faster
        assert par_elapsed < seq_elapsed * 0.5  # At least 2x speedup
