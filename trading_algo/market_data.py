from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from trading_algo.broker.base import Broker, MarketDataSnapshot
from trading_algo.instruments import InstrumentSpec, validate_instrument

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDataConfig:
    """Shared defaults. Override TTL per symbol via `per_symbol_ttl`.

    IBKR issues a default of ~100 simultaneous market-data subscriptions
    per connection (paper + most retail live). `soft_subscription_limit`
    warns if we're within 10% of that; hard enforcement lives in the
    broker layer.
    """
    ttl_seconds: float = 1.0
    min_interval_seconds: float = 0.2
    per_symbol_ttl: dict[str, float] = field(default_factory=dict)
    soft_subscription_limit: int = 100
    warn_subscription_at: float = 0.9  # 90% of soft limit


class MarketDataClient:
    """
    Caching + rate-limiting wrapper around Broker market-data snapshots.

    Extensions (T4.4):
      - Per-symbol cache TTL override via `cfg.per_symbol_ttl['AAPL']=5.0`.
        Useful when an agent is polling one symbol at 1s cadence but
        another at 30s.
      - `get_snapshots([...])` batches N snapshots, reusing cache where
        possible and staggering un-cached fetches by `min_interval_seconds`.
      - Subscription headroom warning: logs a warning when the unique
        instruments we've fetched exceeds `warn_subscription_at *
        soft_subscription_limit`. Prevents silent 354 / 10197 errors.
    """

    def __init__(self, broker: Broker, cfg: MarketDataConfig | None = None) -> None:
        self._broker = broker
        self._cfg = cfg or MarketDataConfig()
        self._cache: dict[InstrumentSpec, MarketDataSnapshot] = {}
        self._last_fetch_epoch_s: float = 0.0
        self._warned_subscription: bool = False

    def _ttl_for(self, instrument: InstrumentSpec) -> float:
        override = self._cfg.per_symbol_ttl.get(instrument.symbol)
        return float(override) if override is not None else float(self._cfg.ttl_seconds)

    def _maybe_warn_subscription(self) -> None:
        if self._warned_subscription:
            return
        if len(self._cache) >= int(
            self._cfg.soft_subscription_limit * self._cfg.warn_subscription_at
        ):
            log.warning(
                "MarketDataClient approaching subscription soft limit: "
                "%d unique instruments cached (soft cap %d). IBKR will "
                "start rejecting with error 10197 / 354 near the hard "
                "cap — consider recycling subscriptions.",
                len(self._cache), self._cfg.soft_subscription_limit,
            )
            self._warned_subscription = True

    def get_snapshot(self, instrument: InstrumentSpec) -> MarketDataSnapshot:
        instrument = validate_instrument(instrument)
        now = time.time()
        ttl = self._ttl_for(instrument)
        cached = self._cache.get(instrument)
        if cached is not None and (now - cached.timestamp_epoch_s) <= ttl:
            return cached

        elapsed = now - self._last_fetch_epoch_s
        if elapsed < self._cfg.min_interval_seconds:
            time.sleep(self._cfg.min_interval_seconds - elapsed)

        snap = self._broker.get_market_data_snapshot(instrument)
        self._validate_snapshot(snap)
        self._cache[instrument] = snap
        self._last_fetch_epoch_s = time.time()
        self._maybe_warn_subscription()
        return snap

    def get_snapshots(
        self, instruments: list[InstrumentSpec]
    ) -> list[MarketDataSnapshot]:
        """Fetch snapshots for several instruments. Cached entries return
        immediately; uncached entries are fetched serially with the
        `min_interval_seconds` stagger applied between broker calls.
        """
        out: list[MarketDataSnapshot] = []
        for inst in instruments:
            out.append(self.get_snapshot(inst))
        return out

    @staticmethod
    def _validate_snapshot(snap: MarketDataSnapshot) -> None:
        if snap.bid is not None and snap.bid < 0:
            raise ValueError("Invalid bid in snapshot")
        if snap.ask is not None and snap.ask < 0:
            raise ValueError("Invalid ask in snapshot")
        if snap.bid is not None and snap.ask is not None and snap.bid > snap.ask:
            raise ValueError("Invalid snapshot (bid > ask)")
        for f in (snap.last, snap.close):
            if f is not None and f < 0:
                raise ValueError("Invalid price in snapshot")

