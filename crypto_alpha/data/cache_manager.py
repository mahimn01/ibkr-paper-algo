"""
Parquet-based cache for crypto market data.

Stores OHLCV bars and funding rates as parquet files for fast
backtesting without re-downloading from exchanges.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / "crypto_data_cache"


def _ensure_cache_dir(subdir: str = "") -> Path:
    d = CACHE_DIR / subdir if subdir else CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(symbol: str, data_type: str, exchange: str,
                start: str, end: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    subdir = _ensure_cache_dir(data_type)
    return subdir / f"{safe_symbol}_{exchange}_{start}_{end}.npz"


def save_ohlcv(
    symbol: str,
    exchange: str,
    timestamps: List[datetime],
    ohlcv: np.ndarray,
    start: str,
    end: str,
) -> Path:
    """Save OHLCV data as compressed numpy archive."""
    path = _cache_path(symbol, "ohlcv", exchange, start, end)
    ts_epochs = np.array([t.timestamp() for t in timestamps], dtype=np.float64)
    np.savez_compressed(path, timestamps=ts_epochs, ohlcv=ohlcv)
    logger.info(f"Cached {len(timestamps)} bars -> {path}")
    return path


def load_ohlcv(
    symbol: str,
    exchange: str,
    start: str,
    end: str,
) -> Optional[Tuple[List[datetime], np.ndarray]]:
    """Load cached OHLCV data. Returns (timestamps, ohlcv_array) or None."""
    path = _cache_path(symbol, "ohlcv", exchange, start, end)
    if not path.exists():
        return None
    data = np.load(path)
    timestamps = [datetime.fromtimestamp(t) for t in data["timestamps"]]
    return timestamps, data["ohlcv"]


def save_funding_rates(
    symbol: str,
    exchange: str,
    timestamps: List[datetime],
    rates: np.ndarray,
    start: str,
    end: str,
) -> Path:
    """Save funding rate data."""
    path = _cache_path(symbol, "funding_rates", exchange, start, end)
    ts_epochs = np.array([t.timestamp() for t in timestamps], dtype=np.float64)
    np.savez_compressed(path, timestamps=ts_epochs, rates=rates)
    logger.info(f"Cached {len(timestamps)} funding rates -> {path}")
    return path


def load_funding_rates(
    symbol: str,
    exchange: str,
    start: str,
    end: str,
) -> Optional[Tuple[List[datetime], np.ndarray]]:
    """Load cached funding rates. Returns (timestamps, rates_array) or None."""
    path = _cache_path(symbol, "funding_rates", exchange, start, end)
    if not path.exists():
        return None
    data = np.load(path)
    timestamps = [datetime.fromtimestamp(t) for t in data["timestamps"]]
    return timestamps, data["rates"]


def list_cached(data_type: str = "ohlcv") -> List[str]:
    """List all cached files of a given type."""
    d = CACHE_DIR / data_type
    if not d.exists():
        return []
    return [f.name for f in d.glob("*.npz")]
