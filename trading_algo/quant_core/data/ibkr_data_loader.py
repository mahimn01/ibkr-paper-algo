"""
IBKR Historical Data Loader

Loads cached IBKR historical data for backtesting.
Supports 5-minute and daily bar data.
"""

from __future__ import annotations

import json
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cache"


def load_ibkr_bars(
    symbol: str,
    start_date: str = "2024-01-01",
    end_date: str = "2026-01-31",
    bar_size: str = "5mins",
    cache_dir: Optional[Path] = None,
) -> Tuple[NDArray[np.float64], List[datetime]]:
    """
    Load IBKR historical bars from cache.

    Args:
        symbol: Stock symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        bar_size: Bar size ('5mins' or 'daily')
        cache_dir: Directory containing cached data

    Returns:
        Tuple of (OHLCV array shape (T, 5), list of timestamps)
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Find matching cache file
    pattern = f"{symbol}_{bar_size}_{start_date}_{end_date}.json"
    cache_file = cache_dir / pattern

    if not cache_file.exists():
        # Try to find any file for this symbol
        matching = list(cache_dir.glob(f"{symbol}_{bar_size}_*.json"))
        if matching:
            cache_file = max(matching, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using cached file: {cache_file.name}")
        else:
            raise FileNotFoundError(f"No cached data for {symbol}")

    # Load JSON data
    with open(cache_file, 'r') as f:
        bars = json.load(f)

    if not bars:
        raise ValueError(f"Empty data for {symbol}")

    # Convert to arrays
    ohlcv = np.zeros((len(bars), 5))
    timestamps = []

    for i, bar in enumerate(bars):
        ohlcv[i] = [
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
        ]
        # Parse timestamp
        ts = bar['timestamp']
        if isinstance(ts, str):
            timestamps.append(datetime.fromisoformat(ts))
        else:
            timestamps.append(datetime.fromtimestamp(ts))

    logger.info(f"Loaded {len(bars)} bars for {symbol} from {timestamps[0]} to {timestamps[-1]}")
    return ohlcv, timestamps


def load_universe_data(
    symbols: List[str],
    start_date: str = "2024-01-01",
    end_date: str = "2026-01-31",
    bar_size: str = "5mins",
    cache_dir: Optional[Path] = None,
) -> Tuple[Dict[str, NDArray[np.float64]], List[datetime]]:
    """
    Load IBKR data for multiple symbols.

    Returns aligned data for all symbols.

    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        bar_size: Bar size
        cache_dir: Cache directory

    Returns:
        Tuple of (dict of symbol -> OHLCV array, common timestamps)
    """
    all_data = {}
    all_timestamps = {}

    for symbol in symbols:
        try:
            ohlcv, ts = load_ibkr_bars(
                symbol, start_date, end_date, bar_size, cache_dir
            )
            all_data[symbol] = ohlcv
            all_timestamps[symbol] = ts
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
            continue

    if not all_data:
        raise ValueError("No data loaded for any symbol")

    # Use the symbol with the most data as reference
    ref_symbol = max(all_data.keys(), key=lambda s: len(all_data[s]))
    ref_timestamps = all_timestamps[ref_symbol]

    # Create timestamp index for alignment
    ts_to_idx = {ts: i for i, ts in enumerate(ref_timestamps)}

    # Align all data to reference timestamps
    aligned_data = {}
    for symbol, ohlcv in all_data.items():
        if symbol == ref_symbol:
            aligned_data[symbol] = ohlcv
            continue

        # Align this symbol's data
        aligned = np.full((len(ref_timestamps), 5), np.nan)
        for i, ts in enumerate(all_timestamps[symbol]):
            if ts in ts_to_idx:
                aligned[ts_to_idx[ts]] = ohlcv[i]

        # Forward fill NaN values
        for col in range(5):
            mask = np.isnan(aligned[:, col])
            if mask.any():
                # Forward fill
                idx = np.where(~mask, np.arange(len(aligned)), 0)
                np.maximum.accumulate(idx, out=idx)
                aligned[:, col] = aligned[idx, col]

        aligned_data[symbol] = aligned

    return aligned_data, ref_timestamps


def resample_to_daily(
    ohlcv: NDArray[np.float64],
    timestamps: List[datetime],
) -> Tuple[NDArray[np.float64], List[datetime]]:
    """
    Resample 5-minute bars to daily bars.

    Args:
        ohlcv: 5-minute OHLCV data
        timestamps: 5-minute timestamps

    Returns:
        Tuple of (daily OHLCV, daily timestamps)
    """
    # Group by date
    daily_data = {}

    for i, ts in enumerate(timestamps):
        date = ts.date()
        if date not in daily_data:
            daily_data[date] = {
                'open': ohlcv[i, 0],
                'high': ohlcv[i, 1],
                'low': ohlcv[i, 2],
                'close': ohlcv[i, 3],
                'volume': 0.0,
            }

        # Update OHLCV
        daily_data[date]['high'] = max(daily_data[date]['high'], ohlcv[i, 1])
        daily_data[date]['low'] = min(daily_data[date]['low'], ohlcv[i, 2])
        daily_data[date]['close'] = ohlcv[i, 3]
        daily_data[date]['volume'] += ohlcv[i, 4]

    # Convert to arrays
    dates = sorted(daily_data.keys())
    daily_ohlcv = np.zeros((len(dates), 5))
    daily_timestamps = []

    for i, date in enumerate(dates):
        d = daily_data[date]
        daily_ohlcv[i] = [d['open'], d['high'], d['low'], d['close'], d['volume']]
        daily_timestamps.append(datetime.combine(date, datetime.min.time()))

    return daily_ohlcv, daily_timestamps


def get_available_symbols(cache_dir: Optional[Path] = None) -> List[str]:
    """Get list of symbols with cached data."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    if not cache_dir.exists():
        return []

    symbols = set()
    for f in cache_dir.glob("*_5mins_*.json"):
        symbol = f.name.split("_")[0]
        symbols.add(symbol)

    return sorted(symbols)
