"""
CCXT-based crypto data loader.

Downloads historical OHLCV bars and funding rates from any
CCXT-supported exchange. Caches locally as numpy archives.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crypto_alpha.data.cache_manager import (
    load_funding_rates,
    load_ohlcv,
    save_funding_rates,
    save_ohlcv,
)
from crypto_alpha.types import CryptoBar

logger = logging.getLogger(__name__)

# Map bar_size string to CCXT timeframe + milliseconds
TIMEFRAME_MAP = {
    "1m": ("1m", 60_000),
    "5m": ("5m", 300_000),
    "15m": ("15m", 900_000),
    "1h": ("1h", 3_600_000),
    "4h": ("4h", 14_400_000),
    "1d": ("1d", 86_400_000),
}


def _init_exchange(exchange_id: str) -> Any:
    """Initialize a CCXT exchange instance."""
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt is required: pip install ccxt")

    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},  # For perpetual futures
    })
    return exchange


def download_ohlcv(
    symbol: str,
    exchange_id: str = "binance",
    bar_size: str = "5m",
    start_date: str = "2022-01-01",
    end_date: str = "2026-03-01",
    use_cache: bool = True,
) -> Tuple[List[datetime], np.ndarray]:
    """
    Download historical OHLCV bars via CCXT.

    Returns:
        (timestamps, ohlcv_array) where ohlcv is (N, 5) [O, H, L, C, V]
    """
    if use_cache:
        cached = load_ohlcv(symbol, exchange_id, start_date, end_date)
        if cached is not None:
            logger.info(f"Using cached {symbol} from {exchange_id}")
            return cached

    exchange = _init_exchange(exchange_id)

    timeframe, tf_ms = TIMEFRAME_MAP.get(bar_size, ("5m", 300_000))
    since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    all_bars: List[list] = []
    current_since = since
    limit = 1000  # Most exchanges support 1000 bars per request

    logger.info(f"Downloading {symbol} {bar_size} from {exchange_id} "
                f"({start_date} to {end_date})...")

    while current_since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=current_since, limit=limit
            )
        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}, retrying...")
            time.sleep(2)
            continue

        if not bars:
            break

        all_bars.extend(bars)
        current_since = bars[-1][0] + tf_ms

        # Rate limit
        time.sleep(exchange.rateLimit / 1000)

        if len(all_bars) % 10000 < limit:
            logger.info(f"  ... {len(all_bars)} bars downloaded")

    if not all_bars:
        raise ValueError(f"No data returned for {symbol} on {exchange_id}")

    # Convert to numpy
    timestamps = [datetime.fromtimestamp(b[0] / 1000, tz=timezone.utc) for b in all_bars]
    ohlcv = np.array([[b[1], b[2], b[3], b[4], b[5]] for b in all_bars],
                     dtype=np.float64)

    # Cache
    save_ohlcv(symbol, exchange_id, timestamps, ohlcv, start_date, end_date)

    logger.info(f"Downloaded {len(timestamps)} bars for {symbol}")
    return timestamps, ohlcv


def download_funding_rates(
    symbol: str,
    exchange_id: str = "binance",
    start_date: str = "2022-01-01",
    end_date: str = "2026-03-01",
    use_cache: bool = True,
) -> Tuple[List[datetime], np.ndarray]:
    """
    Download historical funding rates via CCXT.

    Returns:
        (timestamps, rates_array) where rates is (N,) of funding rates
    """
    if use_cache:
        cached = load_funding_rates(symbol, exchange_id, start_date, end_date)
        if cached is not None:
            logger.info(f"Using cached funding rates for {symbol}")
            return cached

    exchange = _init_exchange(exchange_id)

    since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    all_rates: List[dict] = []
    current_since = since

    logger.info(f"Downloading funding rates for {symbol} from {exchange_id}...")

    while current_since < end_ms:
        try:
            rates = exchange.fetch_funding_rate_history(
                symbol, since=current_since, limit=1000
            )
        except Exception as e:
            logger.warning(f"Error fetching funding rates: {e}, retrying...")
            time.sleep(2)
            continue

        if not rates:
            break

        all_rates.extend(rates)
        current_since = rates[-1]["timestamp"] + 1

        time.sleep(exchange.rateLimit / 1000)

    if not all_rates:
        logger.warning(f"No funding rate data for {symbol} on {exchange_id}")
        return [], np.array([])

    timestamps = [datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc)
                  for r in all_rates]
    rates = np.array([r.get("fundingRate", 0.0) for r in all_rates],
                     dtype=np.float64)

    save_funding_rates(symbol, exchange_id, timestamps, rates, start_date, end_date)

    logger.info(f"Downloaded {len(timestamps)} funding rates for {symbol}")
    return timestamps, rates


def ohlcv_to_crypto_bars(
    timestamps: List[datetime],
    ohlcv: np.ndarray,
    funding_timestamps: Optional[List[datetime]] = None,
    funding_rates: Optional[np.ndarray] = None,
    spot_timestamps: Optional[List[datetime]] = None,
    spot_prices: Optional[np.ndarray] = None,
) -> List[CryptoBar]:
    """
    Convert raw arrays to CryptoBar objects with aligned funding rates
    and spot prices for basis calculation.
    """
    # Pre-build funding rate lookup (timestamp -> rate)
    fr_lookup: Dict[int, float] = {}
    if funding_timestamps and funding_rates is not None:
        for ft, fr in zip(funding_timestamps, funding_rates):
            fr_lookup[int(ft.timestamp())] = float(fr)

    # Pre-build spot price lookup (closest timestamp -> price)
    spot_lookup: Dict[int, float] = {}
    if spot_timestamps and spot_prices is not None:
        for st, sp in zip(spot_timestamps, spot_prices):
            spot_lookup[int(st.timestamp())] = float(sp)

    bars = []
    last_funding = 0.0
    last_spot = 0.0

    for i, ts in enumerate(timestamps):
        ts_epoch = int(ts.timestamp())

        # Find closest funding rate (BACKWARD-ONLY — no future data)
        if ts_epoch in fr_lookup:
            last_funding = fr_lookup[ts_epoch]
        else:
            for offset in range(60, 301, 60):
                if ts_epoch - offset in fr_lookup:
                    last_funding = fr_lookup[ts_epoch - offset]
                    break

        # Find closest spot price (BACKWARD-ONLY)
        if ts_epoch in spot_lookup:
            last_spot = spot_lookup[ts_epoch]
        else:
            for offset in range(60, 301, 60):
                if ts_epoch - offset in spot_lookup:
                    last_spot = spot_lookup[ts_epoch - offset]
                    break

        bars.append(CryptoBar(
            timestamp=ts,
            open=ohlcv[i, 0],
            high=ohlcv[i, 1],
            low=ohlcv[i, 2],
            close=ohlcv[i, 3],
            volume=ohlcv[i, 4],
            funding_rate=last_funding if last_funding != 0 else None,
            spot_price=last_spot if last_spot != 0 else None,
        ))

    return bars


def download_universe(
    symbols: List[str],
    exchange_id: str = "binance",
    bar_size: str = "5m",
    start_date: str = "2022-01-01",
    end_date: str = "2026-03-01",
) -> Dict[str, Tuple[List[datetime], np.ndarray]]:
    """Download OHLCV data for multiple symbols."""
    result = {}
    for symbol in symbols:
        try:
            ts, ohlcv = download_ohlcv(
                symbol, exchange_id, bar_size, start_date, end_date
            )
            result[symbol] = (ts, ohlcv)
            logger.info(f"  {symbol}: {len(ts)} bars")
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
    return result
