"""Tests for T4.4 MarketDataClient enhancements."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from trading_algo.broker.base import MarketDataSnapshot
from trading_algo.instruments import InstrumentSpec
from trading_algo.market_data import MarketDataClient, MarketDataConfig


def _inst(symbol: str = "AAPL") -> InstrumentSpec:
    return InstrumentSpec(kind="STK", symbol=symbol, exchange="SMART", currency="USD")


def _snap(inst: InstrumentSpec, price: float = 100.0, ts: float | None = None) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        instrument=inst, bid=price - 0.01, ask=price + 0.01,
        last=price, close=price, volume=1_000_000,
        timestamp_epoch_s=ts if ts is not None else time.time(),
    )


class TestPerSymbolTTL:
    def test_per_symbol_override(self, monkeypatch) -> None:
        # AAPL TTL 5s, default TTL 0.1s — fetch MSFT twice, should re-fetch;
        # fetch AAPL twice, should reuse cache.
        cfg = MarketDataConfig(
            ttl_seconds=0.1, min_interval_seconds=0.0,
            per_symbol_ttl={"AAPL": 5.0},
        )
        broker = MagicMock()
        call_count = [0]

        def _get(inst):
            call_count[0] += 1
            return _snap(inst, price=100 + call_count[0])

        broker.get_market_data_snapshot.side_effect = _get

        client = MarketDataClient(broker, cfg)
        aapl = _inst("AAPL")
        msft = _inst("MSFT")

        s1 = client.get_snapshot(aapl)
        s2 = client.get_snapshot(aapl)
        assert s1 is s2  # AAPL cached, one broker call
        assert call_count[0] == 1

        # MSFT TTL = 0.1 → wait 0.2 and re-fetch.
        client.get_snapshot(msft)
        time.sleep(0.15)
        client.get_snapshot(msft)
        assert call_count[0] == 3  # AAPL 1 + MSFT 2


class TestGetSnapshotsBatch:
    def test_returns_in_order(self) -> None:
        cfg = MarketDataConfig(min_interval_seconds=0.0)
        broker = MagicMock()

        def _get(inst):
            return _snap(inst, price=10.0)

        broker.get_market_data_snapshot.side_effect = _get
        client = MarketDataClient(broker, cfg)
        res = client.get_snapshots([_inst("A"), _inst("B"), _inst("C")])
        assert [s.instrument.symbol for s in res] == ["A", "B", "C"]
        assert broker.get_market_data_snapshot.call_count == 3

    def test_reuses_cache_within_ttl(self) -> None:
        cfg = MarketDataConfig(ttl_seconds=10.0, min_interval_seconds=0.0)
        broker = MagicMock()
        broker.get_market_data_snapshot.side_effect = lambda inst: _snap(inst)
        client = MarketDataClient(broker, cfg)
        client.get_snapshots([_inst("A"), _inst("B")])
        client.get_snapshots([_inst("A"), _inst("B")])
        # Second batch should hit cache entirely.
        assert broker.get_market_data_snapshot.call_count == 2


class TestSubscriptionHeadroom:
    def test_warns_past_threshold(self, caplog) -> None:
        cfg = MarketDataConfig(
            ttl_seconds=100.0, min_interval_seconds=0.0,
            soft_subscription_limit=10, warn_subscription_at=0.8,
        )
        broker = MagicMock()
        broker.get_market_data_snapshot.side_effect = lambda inst: _snap(inst)
        client = MarketDataClient(broker, cfg)

        with caplog.at_level(logging.WARNING, logger="trading_algo.market_data"):
            for i in range(10):
                client.get_snapshot(_inst(f"SYM{i}"))
        assert any("subscription soft limit" in rec.message for rec in caplog.records)

    def test_does_not_warn_twice(self, caplog) -> None:
        cfg = MarketDataConfig(
            ttl_seconds=100.0, min_interval_seconds=0.0,
            soft_subscription_limit=5, warn_subscription_at=0.5,
        )
        broker = MagicMock()
        broker.get_market_data_snapshot.side_effect = lambda inst: _snap(inst)
        client = MarketDataClient(broker, cfg)

        with caplog.at_level(logging.WARNING, logger="trading_algo.market_data"):
            for i in range(10):
                client.get_snapshot(_inst(f"S{i}"))
        warnings = [r for r in caplog.records if "subscription soft limit" in r.message]
        assert len(warnings) == 1  # idempotent


class TestValidation:
    def test_invalid_bid_ask_rejected(self) -> None:
        broker = MagicMock()
        broker.get_market_data_snapshot.return_value = MarketDataSnapshot(
            instrument=_inst(), bid=100.0, ask=99.0, last=99.5,
            close=100.0, volume=1, timestamp_epoch_s=time.time(),
        )
        client = MarketDataClient(broker)
        with pytest.raises(ValueError, match="bid > ask"):
            client.get_snapshot(_inst())
