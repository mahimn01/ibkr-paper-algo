"""
Data Loaders for RAT Backtesting

Supports multiple data sources:
1. CSV files (local)
2. Yahoo Finance (via yfinance)
3. IBKR historical data

All data normalized to common Bar format.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass
class Bar:
    """Normalized OHLCV bar."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

    # Optional order flow data
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None

    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """High-low range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Candle body (close - open)."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish."""
        return self.close > self.open


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Bar]:
        """Load historical data for a symbol."""
        pass

    @abstractmethod
    def stream(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[Bar]:
        """Stream historical data bar by bar."""
        pass


class CSVLoader(DataLoader):
    """
    Load historical data from CSV files.

    Expected CSV format:
    date,open,high,low,close,volume[,vwap,trade_count,buy_volume,sell_volume]

    Date format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    """

    def __init__(
        self,
        data_dir: str = "data",
        date_column: str = "date",
        date_format: str = "%Y-%m-%d",
    ):
        self.data_dir = Path(data_dir)
        self.date_column = date_column
        self.date_format = date_format

        # Column mapping
        self._column_map = {
            "open": ["open", "Open", "OPEN", "o"],
            "high": ["high", "High", "HIGH", "h"],
            "low": ["low", "Low", "LOW", "l"],
            "close": ["close", "Close", "CLOSE", "c", "adj_close", "Adj Close"],
            "volume": ["volume", "Volume", "VOLUME", "v", "vol"],
            "vwap": ["vwap", "VWAP", "Vwap"],
            "trade_count": ["trade_count", "trades", "count"],
            "buy_volume": ["buy_volume", "buy_vol", "bid_volume"],
            "sell_volume": ["sell_volume", "sell_vol", "ask_volume"],
        }

    def _find_column(self, headers: List[str], field: str) -> Optional[int]:
        """Find column index for a field."""
        possible_names = self._column_map.get(field, [field])
        for name in possible_names:
            if name in headers:
                return headers.index(name)
        return None

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string with multiple format attempts."""
        formats = [
            self.date_format,
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d-%m-%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        raise ValueError(f"Could not parse date: {date_str}")

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Bar]:
        """Load all data within date range."""
        return list(self.stream(symbol, start_date, end_date))

    def stream(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[Bar]:
        """Stream data bar by bar."""
        # Try different file naming conventions
        possible_files = [
            self.data_dir / f"{symbol}.csv",
            self.data_dir / f"{symbol.lower()}.csv",
            self.data_dir / f"{symbol.upper()}.csv",
            self.data_dir / symbol / "daily.csv",
            self.data_dir / symbol / "1d.csv",
        ]

        file_path = None
        for path in possible_files:
            if path.exists():
                file_path = path
                break

        if file_path is None:
            raise FileNotFoundError(
                f"No data file found for {symbol} in {self.data_dir}"
            )

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Find column indices
            date_idx = self._find_column(headers, self.date_column)
            if date_idx is None:
                date_idx = 0  # Assume first column

            open_idx = self._find_column(headers, "open")
            high_idx = self._find_column(headers, "high")
            low_idx = self._find_column(headers, "low")
            close_idx = self._find_column(headers, "close")
            volume_idx = self._find_column(headers, "volume")
            vwap_idx = self._find_column(headers, "vwap")
            trade_count_idx = self._find_column(headers, "trade_count")
            buy_vol_idx = self._find_column(headers, "buy_volume")
            sell_vol_idx = self._find_column(headers, "sell_volume")

            for row in reader:
                try:
                    ts = self._parse_date(row[date_idx])

                    if ts < start_date or ts > end_date:
                        continue

                    bar = Bar(
                        timestamp=ts,
                        symbol=symbol,
                        open=float(row[open_idx]) if open_idx else 0.0,
                        high=float(row[high_idx]) if high_idx else 0.0,
                        low=float(row[low_idx]) if low_idx else 0.0,
                        close=float(row[close_idx]) if close_idx else 0.0,
                        volume=float(row[volume_idx]) if volume_idx else 0.0,
                        vwap=float(row[vwap_idx]) if vwap_idx else None,
                        trade_count=int(row[trade_count_idx]) if trade_count_idx else None,
                        buy_volume=float(row[buy_vol_idx]) if buy_vol_idx else None,
                        sell_volume=float(row[sell_vol_idx]) if sell_vol_idx else None,
                    )

                    yield bar

                except (ValueError, IndexError) as e:
                    continue  # Skip malformed rows


class YahooLoader(DataLoader):
    """
    Load historical data from Yahoo Finance.

    Requires: pip install yfinance
    """

    def __init__(self, interval: str = "1d"):
        """
        Args:
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
        """
        self.interval = interval
        self._yf = None

    def _get_yfinance(self):
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance not installed. Run: pip install yfinance"
                )
        return self._yf

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Bar]:
        """Load data from Yahoo Finance."""
        yf = self._get_yfinance()

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=self.interval,
        )

        bars = []
        for idx, row in df.iterrows():
            bar = Bar(
                timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                symbol=symbol,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            bars.append(bar)

        return bars

    def stream(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[Bar]:
        """Stream data from Yahoo Finance."""
        bars = self.load(symbol, start_date, end_date)
        for bar in bars:
            yield bar


class IBKRLoader(DataLoader):
    """
    Load historical data from IBKR.

    Requires active IBKR connection.
    """

    def __init__(self, broker):
        """
        Args:
            broker: IBKR broker connection
        """
        self.broker = broker

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Bar]:
        """Load data from IBKR historical data API."""
        if not self.broker:
            raise RuntimeError("No broker connection")

        # Calculate duration
        duration = end_date - start_date
        duration_str = f"{duration.days} D"

        try:
            # Request historical data
            # This is simplified - actual implementation depends on broker interface
            bars_data = self.broker.get_historical_data(
                symbol=symbol,
                duration=duration_str,
                bar_size="1 day",
                end_datetime=end_date,
            )

            bars = []
            for data in bars_data:
                bar = Bar(
                    timestamp=data.get("timestamp", datetime.now()),
                    symbol=symbol,
                    open=float(data.get("open", 0)),
                    high=float(data.get("high", 0)),
                    low=float(data.get("low", 0)),
                    close=float(data.get("close", 0)),
                    volume=float(data.get("volume", 0)),
                    vwap=data.get("vwap"),
                    trade_count=data.get("trade_count"),
                )
                bars.append(bar)

            return bars

        except Exception as e:
            raise RuntimeError(f"Failed to load IBKR data: {e}")

    def stream(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[Bar]:
        """Stream data from IBKR."""
        bars = self.load(symbol, start_date, end_date)
        for bar in bars:
            yield bar


class MultiSymbolLoader:
    """Load data for multiple symbols and merge into time-ordered stream."""

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def load_all(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, List[Bar]]:
        """Load data for all symbols."""
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.loader.load(symbol, start_date, end_date)
            except Exception as e:
                print(f"Warning: Could not load {symbol}: {e}")
                data[symbol] = []
        return data

    def stream_merged(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[Bar]:
        """Stream all symbols merged by timestamp."""
        # Load all data
        all_data = self.load_all(symbols, start_date, end_date)

        # Merge and sort by timestamp
        all_bars = []
        for symbol, bars in all_data.items():
            all_bars.extend(bars)

        all_bars.sort(key=lambda b: b.timestamp)

        for bar in all_bars:
            yield bar
