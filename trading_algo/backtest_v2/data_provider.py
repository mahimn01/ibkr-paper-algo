"""
Historical data provider for backtesting.

Fetches data from IBKR or CSV files, with caching for fast reruns.
Handles 1+ year of historical data efficiently.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional
import csv

from .models import Bar


@dataclass
class DataRequest:
    """Request for historical data."""
    symbol: str
    start_date: date
    end_date: date
    bar_size: str = "5 mins"


class DataProvider:
    """
    Historical data provider with caching.

    Supports:
    - IBKR historical data API
    - CSV files
    - Local cache for fast reruns
    """

    # Bar size to IBKR duration mapping
    BAR_SIZE_MAP = {
        "1 min": ("1 D", 60),
        "5 mins": ("1 D", 300),
        "15 mins": ("2 D", 900),
        "30 mins": ("1 W", 1800),
        "1 hour": ("1 M", 3600),
        "1 day": ("1 Y", 86400),
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        broker: Optional[Any] = None,
    ):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.broker = broker

    def get_data(
        self,
        requests: List[DataRequest],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, List[Bar]]:
        """
        Get historical data for multiple symbols.

        Args:
            requests: List of data requests
            progress_callback: Optional progress callback

        Returns:
            Dict mapping symbol to list of bars
        """
        result: Dict[str, List[Bar]] = {}

        for i, req in enumerate(requests):
            if progress_callback:
                pct = i / len(requests)
                progress_callback(pct, f"Loading {req.symbol}...")

            bars = self._get_symbol_data(req)
            if bars:
                result[req.symbol] = bars

        if progress_callback:
            progress_callback(1.0, "Data loaded")

        return result

    def _get_symbol_data(self, request: DataRequest) -> List[Bar]:
        """Get data for a single symbol."""
        # Check cache first
        cached = self._load_from_cache(request)
        if cached:
            return cached

        # Try IBKR
        if self.broker:
            bars = self._fetch_from_ibkr(request)
            if bars:
                self._save_to_cache(request, bars)
                return bars

        # Try CSV
        csv_path = self._find_csv(request.symbol)
        if csv_path:
            bars = self._load_from_csv(csv_path, request.start_date, request.end_date)
            if bars:
                return bars

        return []

    def _get_cache_path(self, request: DataRequest) -> Path:
        """Get cache file path for a request."""
        filename = f"{request.symbol}_{request.bar_size.replace(' ', '')}_{request.start_date}_{request.end_date}.json"
        return self.cache_dir / filename

    def _load_from_cache(self, request: DataRequest) -> Optional[List[Bar]]:
        """Load data from cache if available and fresh."""
        cache_path = self._get_cache_path(request)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            bars = []
            for item in data:
                bars.append(Bar(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item.get("volume", 0),
                    vwap=item.get("vwap", 0),
                ))
            return bars
        except Exception:
            return None

    def _save_to_cache(self, request: DataRequest, bars: List[Bar]) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(request)
        try:
            data = [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                }
                for bar in bars
            ]
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _fetch_from_ibkr(self, request: DataRequest) -> List[Bar]:
        """Fetch data from IBKR."""
        if not self.broker:
            return []

        try:
            from trading_algo.instruments import InstrumentSpec

            instrument = InstrumentSpec(
                kind="STK",
                symbol=request.symbol,
                exchange="SMART",
                currency="USD",
            )

            # IBKR limits how much data we can get per request
            # For 5-min bars, max is about 1 month at a time
            all_bars = []
            current_end = request.end_date
            duration_str, _ = self.BAR_SIZE_MAP.get(request.bar_size, ("1 D", 300))

            while current_end > request.start_date:
                try:
                    end_dt = datetime.combine(current_end, datetime.max.time())

                    bars = self.broker.get_historical_bars(
                        instrument,
                        duration="30 D",
                        bar_size=request.bar_size,
                        what_to_show="TRADES",
                        use_rth=True,
                        end_datetime=end_dt,
                    )

                    if not bars:
                        break

                    for bar in bars:
                        ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
                        if request.start_date <= ts.date() <= request.end_date:
                            all_bars.append(Bar(
                                timestamp=ts,
                                open=bar.open,
                                high=bar.high,
                                low=bar.low,
                                close=bar.close,
                                volume=bar.volume or 0,
                            ))

                    # Move to earlier period
                    earliest = min(datetime.fromtimestamp(b.timestamp_epoch_s) for b in bars)
                    current_end = (earliest - timedelta(days=1)).date()

                    # Rate limiting
                    import time
                    time.sleep(1)

                except Exception as e:
                    print(f"Warning: Error fetching {request.symbol}: {e}")
                    break

            # Sort by timestamp
            all_bars.sort(key=lambda x: x.timestamp)
            return all_bars

        except Exception as e:
            print(f"Error fetching from IBKR: {e}")
            return []

    def _find_csv(self, symbol: str) -> Optional[Path]:
        """Find CSV file for a symbol."""
        data_dir = Path("data")
        if not data_dir.exists():
            return None

        # Try different naming conventions
        patterns = [
            f"{symbol}.csv",
            f"{symbol}_5m.csv",
            f"{symbol}_5min.csv",
            f"{symbol.lower()}.csv",
        ]

        for pattern in patterns:
            path = data_dir / pattern
            if path.exists():
                return path

        return None

    def _load_from_csv(
        self,
        path: Path,
        start_date: date,
        end_date: date,
    ) -> List[Bar]:
        """Load data from CSV file."""
        bars = []

        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different timestamp formats
                    ts_str = row.get("timestamp") or row.get("date") or row.get("datetime")
                    if not ts_str:
                        continue

                    try:
                        if "T" in ts_str:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        else:
                            # Try common formats
                            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M"]:
                                try:
                                    ts = datetime.strptime(ts_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                continue
                    except Exception:
                        continue

                    if not (start_date <= ts.date() <= end_date):
                        continue

                    bars.append(Bar(
                        timestamp=ts,
                        open=float(row.get("open", 0)),
                        high=float(row.get("high", 0)),
                        low=float(row.get("low", 0)),
                        close=float(row.get("close", 0)),
                        volume=int(float(row.get("volume", 0))),
                        vwap=float(row.get("vwap", 0)),
                    ))

        except Exception as e:
            print(f"Error loading CSV {path}: {e}")
            return []

        bars.sort(key=lambda x: x.timestamp)
        return bars

    def generate_sample_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        bar_size: str = "5 mins",
    ) -> List[Bar]:
        """
        Generate sample random walk data for testing.

        This is useful for testing the backtest system without real data.
        """
        import random

        bars = []
        price = 100.0  # Starting price
        current = datetime.combine(start_date, datetime.min.time().replace(hour=9, minute=30))
        end = datetime.combine(end_date, datetime.min.time().replace(hour=16, minute=0))

        # Parse bar size
        if "min" in bar_size:
            minutes = int(bar_size.split()[0])
        elif "hour" in bar_size:
            minutes = int(bar_size.split()[0]) * 60
        else:
            minutes = 5

        while current <= end:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                current = current.replace(hour=9, minute=30)
                continue

            # Skip outside market hours
            if current.hour < 9 or current.hour >= 16:
                current += timedelta(days=1)
                current = current.replace(hour=9, minute=30)
                continue
            if current.hour == 9 and current.minute < 30:
                current = current.replace(minute=30)
                continue

            # Generate bar with random walk
            volatility = 0.001  # 0.1% per bar
            change = random.gauss(0, volatility)
            open_price = price
            close_price = price * (1 + change)

            # High/low around open/close
            high = max(open_price, close_price) * (1 + random.uniform(0, 0.002))
            low = min(open_price, close_price) * (1 - random.uniform(0, 0.002))

            volume = int(random.uniform(10000, 100000))

            bars.append(Bar(
                timestamp=current,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=volume,
            ))

            price = close_price
            current += timedelta(minutes=minutes)

            # Jump to next day if past market close
            if current.hour >= 16:
                current += timedelta(days=1)
                current = current.replace(hour=9, minute=30)

        return bars
