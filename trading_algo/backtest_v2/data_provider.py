"""
Historical data provider for backtesting.

Fetches data from multiple sources with priority:
1. Local cache (fastest)
2. IBKR API (comprehensive, requires account)
3. Yahoo Finance (FREE, good for stocks)
4. CSV files (manual import)
5. Sample data (realistic simulation for testing)

Handles 1+ year of historical data efficiently with caching.

DATA SOURCE COMPARISON:
=======================
| Source        | Cost    | Data Quality | Intraday | Notes                          |
|---------------|---------|--------------|----------|--------------------------------|
| IBKR          | Free*   | Excellent    | Yes      | *With brokerage account        |
| Yahoo Finance | Free    | Good         | Limited  | 7 days intraday, years daily   |
| Alpha Vantage | Free**  | Good         | Yes      | **5 calls/min free tier        |
| Polygon.io    | Free*** | Excellent    | Yes      | ***Basic tier, $29/mo for full |
| Alpaca        | Free*   | Good         | Yes      | *With brokerage account        |
| Twelve Data   | Free**  | Good         | Yes      | **800 calls/day free           |
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
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

    Supports (in priority order):
    1. Local cache - Fastest, no API calls
    2. IBKR API - Best for comprehensive data with active account
    3. Yahoo Finance - FREE, no API key needed
    4. CSV files - Manual data import
    5. Realistic sample data - For testing without API

    Usage:
        provider = DataProvider()  # Uses cache + Yahoo Finance

        # With IBKR:
        provider = DataProvider(broker=ibkr_broker)

        # Force Yahoo even if IBKR available:
        provider = DataProvider(prefer_yahoo=True)
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

    # Yahoo Finance interval mapping
    YAHOO_INTERVAL_MAP = {
        "1 min": "1m",
        "5 mins": "5m",
        "15 mins": "15m",
        "30 mins": "30m",
        "1 hour": "1h",
        "1 day": "1d",
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        broker: Optional[Any] = None,
        prefer_yahoo: bool = False,
    ):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.broker = broker
        self.prefer_yahoo = prefer_yahoo
        self._yf = None  # Lazy-loaded yfinance

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

        bars = None

        # If broker is provided, use IBKR exclusively (no Yahoo fallback)
        if self.broker:
            bars = self._fetch_from_ibkr(request)
            if bars:
                self._save_to_cache(request, bars)
                return bars
            else:
                # IBKR is the primary source when broker is connected
                # Don't fall back to Yahoo for consistency in backtests
                print(f"Warning: Could not fetch {request.symbol} from IBKR")
                return []

        # No broker - try Yahoo Finance (FREE) as primary source
        bars = self._fetch_from_yahoo(request)
        if bars:
            self._save_to_cache(request, bars)
            return bars

        # Try CSV as last resort
        csv_path = self._find_csv(request.symbol)
        if csv_path:
            bars = self._load_from_csv(csv_path, request.start_date, request.end_date)
            if bars:
                return bars

        return []

    def _get_yfinance(self):
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                return None
        return self._yf

    def _fetch_from_yahoo(self, request: DataRequest) -> Optional[List[Bar]]:
        """
        Fetch data from Yahoo Finance (FREE).

        Limitations:
        - Intraday data (1m, 5m, 15m, 30m): Last 7 days only
        - Hourly data: Last 730 days
        - Daily data: Full history available

        For backtesting 1+ year, use daily bars or IBKR for intraday.
        """
        yf = self._get_yfinance()
        if yf is None:
            return None

        try:
            # Map bar size to Yahoo interval
            interval = self.YAHOO_INTERVAL_MAP.get(request.bar_size, "1d")

            ticker = yf.Ticker(request.symbol)

            # For intraday, Yahoo limits history to 7 days
            # Adjust if needed
            if interval in ["1m", "5m", "15m", "30m"]:
                # Warn if requesting more than 7 days of intraday
                days_requested = (request.end_date - request.start_date).days
                if days_requested > 7:
                    print(f"Warning: Yahoo Finance only provides 7 days of {interval} data. "
                          f"Use IBKR or daily bars for longer backtests.")

            df = ticker.history(
                start=request.start_date.strftime("%Y-%m-%d"),
                end=(request.end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
            )

            if df.empty:
                return None

            bars = []
            for idx, row in df.iterrows():
                ts = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
                # Remove timezone info for consistency
                if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)

                bars.append(Bar(
                    timestamp=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]) if not math.isnan(row["Volume"]) else 0,
                ))

            return bars

        except Exception as e:
            print(f"Yahoo Finance error for {request.symbol}: {e}")
            return None

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
            print(f"IBKR fetch failed for {request.symbol}: No broker connection")
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
            all_bars = []
            current_end = request.end_date

            # Determine duration and chunk size based on bar size
            if request.bar_size == "1 day":
                # For daily bars, we can fetch 1 year at a time
                duration = "1 Y"
                chunk_days = 365
            else:
                # For intraday, fetch 30 days at a time
                duration = "30 D"
                chunk_days = 30

            fetch_count = 0
            max_fetches = 20  # Prevent infinite loops

            while current_end > request.start_date and fetch_count < max_fetches:
                fetch_count += 1
                try:
                    end_dt = datetime.combine(current_end, datetime.max.time())

                    print(f"  IBKR: Fetching {request.symbol} ending {current_end}, duration={duration}, bar_size={request.bar_size}")

                    bars = self.broker.get_historical_bars(
                        instrument,
                        duration=duration,
                        bar_size=request.bar_size,
                        what_to_show="TRADES",
                        use_rth=True,
                        end_datetime=end_dt,
                    )

                    if not bars:
                        print(f"  IBKR: No bars returned for {request.symbol}")
                        break

                    print(f"  IBKR: Got {len(bars)} bars for {request.symbol}")

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
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  IBKR Error fetching {request.symbol}: {type(e).__name__}: {e}")
                    break

            # Sort by timestamp
            all_bars.sort(key=lambda x: x.timestamp)
            print(f"  IBKR: Total {len(all_bars)} bars for {request.symbol}")
            return all_bars

        except Exception as e:
            print(f"IBKR fetch failed for {request.symbol}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
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
        Generate sample random walk data for UNIT TESTING ONLY.

        WARNING: This method generates synthetic data with random walk behavior.
        DO NOT use for actual backtesting - use IBKR real historical data instead.

        This is only useful for testing the backtest engine mechanics without
        requiring an IBKR connection.
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
