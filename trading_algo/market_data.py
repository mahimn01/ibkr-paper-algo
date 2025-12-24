from __future__ import annotations

import time
from dataclasses import dataclass

from trading_algo.broker.base import Broker, MarketDataSnapshot
from trading_algo.instruments import InstrumentSpec, validate_instrument


@dataclass(frozen=True)
class MarketDataConfig:
    ttl_seconds: float = 1.0
    min_interval_seconds: float = 0.2


class MarketDataClient:
    """
    Small caching + rate-limiting wrapper around Broker market data snapshots.
    """

    def __init__(self, broker: Broker, cfg: MarketDataConfig | None = None) -> None:
        self._broker = broker
        self._cfg = cfg or MarketDataConfig()
        self._cache: dict[InstrumentSpec, MarketDataSnapshot] = {}
        self._last_fetch_epoch_s: float = 0.0

    def get_snapshot(self, instrument: InstrumentSpec) -> MarketDataSnapshot:
        instrument = validate_instrument(instrument)
        now = time.time()
        cached = self._cache.get(instrument)
        if cached is not None and (now - cached.timestamp_epoch_s) <= self._cfg.ttl_seconds:
            return cached

        elapsed = now - self._last_fetch_epoch_s
        if elapsed < self._cfg.min_interval_seconds:
            time.sleep(self._cfg.min_interval_seconds - elapsed)

        snap = self._broker.get_market_data_snapshot(instrument)
        self._validate_snapshot(snap)
        self._cache[instrument] = snap
        self._last_fetch_epoch_s = time.time()
        return snap

    @staticmethod
    def _validate_snapshot(snap: MarketDataSnapshot) -> None:
        if snap.bid is not None and snap.bid < 0:
            raise ValueError("Invalid bid in snapshot")
        if snap.ask is not None and snap.ask < 0:
            raise ValueError("Invalid ask in snapshot")
        if snap.bid is not None and snap.ask is not None and snap.bid > snap.ask:
            raise ValueError("Invalid snapshot (bid > ask)")
        for field in (snap.last, snap.close):
            if field is not None and field < 0:
                raise ValueError("Invalid price in snapshot")

