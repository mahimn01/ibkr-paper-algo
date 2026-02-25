#!/usr/bin/env python3
"""
Download 10 years of 5-minute IBKR bar data for backtesting.

Downloads data for 11 symbols using round-robin pacing to stay within IBKR's
historical data rate limits (~60 requests per 10 minutes). Supports resume
from interruption via a manifest file.

Prerequisites:
    1. pip install ib_insync
    2. IB Gateway running on port 4002 (paper trading)
    3. API access enabled in Gateway settings

Usage:
    python scripts/download_10yr_data.py
    python scripts/download_10yr_data.py --symbols SPY QQQ --sleep 12
    python scripts/download_10yr_data.py --validate-only
    python scripts/download_10yr_data.py --resume --port 4002
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Fix Python 3.14 asyncio event loop issue BEFORE importing ib_insync.
# ib_insync probes for a running loop at import time; on 3.14 there may not
# be one yet in the main thread.
# ---------------------------------------------------------------------------
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "SMCI", "IWM", "HYG", "LQD", "TLT", "GLD"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MANIFEST_PATH = DATA_DIR / "download_manifest.json"

ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")

# How many chunks between incremental saves of partial bar data to disk.
INCREMENTAL_SAVE_INTERVAL = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cache_filename(symbol: str, start_date: str, end_date: str) -> str:
    """Return the cache filename for a symbol and date range."""
    return f"{symbol}_5mins_{start_date}_{end_date}.json"


def bars_to_dicts(bars) -> list[dict]:
    """Convert ib_insync BarData list to list of dicts matching cache format.

    IBKR returns bar timestamps in UTC.  We convert to US/Eastern local time
    and store as an ISO-8601 string without timezone suffix (matching the
    existing cache convention).
    """
    result = []
    for bar in bars:
        dt = bar.date
        # bar.date is a datetime object; ensure it is timezone-aware (UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        # Convert to Eastern
        dt_et = dt.astimezone(ET)
        result.append({
            "timestamp": dt_et.strftime("%Y-%m-%dT%H:%M:%S"),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": 0.0,
        })
    return result


def load_manifest() -> dict:
    """Load the download manifest (resume state)."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    """Persist the download manifest atomically."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(MANIFEST_PATH)


def load_partial_bars(symbol: str) -> list[dict]:
    """Load bars already saved to the partial-progress file."""
    path = CACHE_DIR / f".{symbol}_partial.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_partial_bars(symbol: str, bars: list[dict]) -> None:
    """Save partial bar data so progress is not lost on crash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f".{symbol}_partial.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(bars, f)
    tmp.replace(path)


def remove_partial_bars(symbol: str) -> None:
    """Remove the partial progress file after final save."""
    path = CACHE_DIR / f".{symbol}_partial.json"
    if path.exists():
        path.unlink()


def save_final_bars(symbol: str, bars: list[dict], start_date: str, end_date: str) -> Path:
    """Deduplicate, sort, and save the final bar file for a symbol."""
    # Deduplicate by timestamp (keep first occurrence)
    seen = set()
    unique = []
    for b in bars:
        ts = b["timestamp"]
        if ts not in seen:
            seen.add(ts)
            unique.append(b)

    # Sort chronologically
    unique.sort(key=lambda b: b["timestamp"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = cache_filename(symbol, start_date, end_date)
    out_path = CACHE_DIR / fname
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(unique, f)
    tmp.replace(out_path)

    # Clean up partial file
    remove_partial_bars(symbol)

    return out_path


def format_eta(seconds: float) -> str:
    """Human-readable ETA string."""
    if seconds < 0 or seconds > 1e8:
        return "unknown"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def connect_ib(host: str, port: int, client_id: int, max_retries: int = 3) -> IB:
    """Connect to IB Gateway with exponential backoff."""
    ib = IB()
    for attempt in range(1, max_retries + 1):
        try:
            ib.connect(host, port, clientId=client_id, readonly=True)
            print(f"  Connected to IB Gateway at {host}:{port} (attempt {attempt})")
            return ib
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise ConnectionError(
                    f"Failed to connect after {max_retries} attempts: {e}"
                ) from e
    # Should not reach here, but satisfy type checker
    raise ConnectionError("Failed to connect")


def ensure_connected(ib: IB, host: str, port: int, client_id: int) -> IB:
    """Ensure we have a live connection; reconnect if needed."""
    if ib.isConnected():
        return ib
    print("  Connection lost. Reconnecting...")
    try:
        ib.disconnect()
    except Exception:
        pass
    return connect_ib(host, port, client_id)


# ---------------------------------------------------------------------------
# Chunk generation
# ---------------------------------------------------------------------------

def generate_chunks(start_date: date, end_date: date) -> list[datetime]:
    """Generate a list of end-datetimes walking backward from end_date to
    start_date in 1-week steps.

    Each entry is the endDateTime to pass to reqHistoricalData.  We use
    16:00:00 (4 PM ET) as the time component because RTH ends at 16:00 ET.
    The returned datetimes are in US/Eastern.
    """
    chunks = []
    current = datetime(end_date.year, end_date.month, end_date.day, 16, 0, 0, tzinfo=ET)
    limit = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=ET)

    while current > limit:
        chunks.append(current)
        current -= timedelta(weeks=1)

    return chunks


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------

class DownloadEngine:
    """Manages the round-robin download of historical bars for all symbols."""

    def __init__(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        host: str,
        port: int,
        client_id: int,
        sleep_secs: float,
        resume: bool,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.host = host
        self.port = port
        self.client_id = client_id
        self.sleep_secs = sleep_secs
        self.resume = resume

        # Per-symbol state
        self.chunks: dict[str, list[datetime]] = {}
        self.cursor: dict[str, int] = {}  # next chunk index to download
        self.bars: dict[str, list[dict]] = {}  # accumulated bars
        self.completed: dict[str, bool] = {}
        self.chunks_since_save: dict[str, int] = {}

        self.manifest = load_manifest() if resume else {}
        self.ib: IB | None = None
        self._shutdown_requested = False

        # Timing
        self._request_count = 0
        self._start_time: float = 0.0

        # Round-robin index: tracks which symbol to serve next
        self._rr_index = 0

    # ---- Setup -----------------------------------------------------------

    def setup(self) -> None:
        """Prepare chunk lists and load resume state."""
        all_chunks = generate_chunks(self.start_date, self.end_date)

        for sym in self.symbols:
            self.chunks[sym] = list(all_chunks)  # copy
            self.bars[sym] = []
            self.completed[sym] = False
            self.chunks_since_save[sym] = 0

            # Resume: skip already-downloaded chunks
            if self.resume and sym in self.manifest:
                info = self.manifest[sym]
                last_end_dt_str = info.get("last_end_dt")
                if last_end_dt_str:
                    last_end_dt = datetime.fromisoformat(last_end_dt_str)
                    if last_end_dt.tzinfo is None:
                        last_end_dt = last_end_dt.replace(tzinfo=ET)
                    # Find the cursor position: skip chunks whose end_dt >= last_end_dt
                    # Chunks are ordered newest-first (walking backward).
                    idx = 0
                    for i, chunk_dt in enumerate(self.chunks[sym]):
                        if chunk_dt <= last_end_dt:
                            idx = i
                            break
                    else:
                        # All chunks already done
                        idx = len(self.chunks[sym])

                    self.cursor[sym] = idx
                    # Load partial bars from disk
                    partial = load_partial_bars(sym)
                    self.bars[sym] = partial
                    chunks_done = info.get("chunks_done", idx)
                    print(
                        f"  Resuming {sym}: skipping {idx} chunks "
                        f"({len(partial)} bars loaded from partial save)"
                    )
                else:
                    self.cursor[sym] = 0
            else:
                self.cursor[sym] = 0

            # Check if already fully done
            if self.cursor[sym] >= len(self.chunks[sym]):
                self.completed[sym] = True

    def total_remaining_chunks(self) -> int:
        return sum(
            len(self.chunks[sym]) - self.cursor[sym]
            for sym in self.symbols
            if not self.completed[sym]
        )

    def total_chunks(self) -> int:
        return sum(len(self.chunks[sym]) for sym in self.symbols)

    def total_done_chunks(self) -> int:
        return sum(self.cursor[sym] for sym in self.symbols)

    # ---- Round-robin -----------------------------------------------------

    def _next_active_symbol(self) -> str | None:
        """Return the next symbol that still has chunks to download,
        cycling round-robin through all symbols.  Returns None when
        every symbol is completed."""
        n = len(self.symbols)
        for _ in range(n):
            sym = self.symbols[self._rr_index % n]
            self._rr_index = (self._rr_index + 1) % n
            if not self.completed[sym]:
                return sym
        return None

    # ---- Signal handling -------------------------------------------------

    def _handle_signal(self, signum, frame):
        print(f"\n  Signal {signum} received. Finishing current request then saving...")
        self._shutdown_requested = True

    # ---- Main download loop ----------------------------------------------

    def run(self) -> None:
        """Execute the full download."""
        self.setup()

        remaining = self.total_remaining_chunks()
        if remaining == 0:
            print("\nAll chunks already downloaded. Nothing to do.")
            return

        total = self.total_chunks()
        done = self.total_done_chunks()
        print(f"\nDownload plan: {remaining} chunks remaining out of {total} total")
        print(f"Estimated time: {format_eta(remaining * self.sleep_secs)}")
        print()

        # Install signal handlers for graceful shutdown
        old_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        old_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)

        try:
            self.ib = connect_ib(self.host, self.port, self.client_id)
            self._start_time = time.monotonic()

            while not self._shutdown_requested:
                # Round-robin: find the next symbol that still has work
                sym = self._next_active_symbol()
                if sym is None:
                    break  # all symbols completed

                if self.cursor[sym] >= len(self.chunks[sym]):
                    self._finalize_symbol(sym)
                    continue

                # Download one chunk for this symbol
                self._download_chunk(sym)

                if self._shutdown_requested:
                    break

                # Sleep between requests to respect pacing
                time.sleep(self.sleep_secs)

        finally:
            # Restore signal handlers
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

            # Save all progress
            self._save_all_progress()

            # Disconnect
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                print("  Disconnected from IB Gateway")

        if self._shutdown_requested:
            print("\nShutdown requested. Progress saved. Re-run with --resume to continue.")
        else:
            self._print_completion_summary()

    def _download_chunk(self, sym: str) -> bool:
        """Download a single chunk for the given symbol. Returns True on success."""
        chunk_idx = self.cursor[sym]
        end_dt = self.chunks[sym][chunk_idx]
        chunk_num = chunk_idx + 1
        total_sym_chunks = len(self.chunks[sym])

        # Format endDateTime for IBKR: 'YYYYMMDD-HH:MM:SS' in local (ET) time
        end_dt_str = end_dt.strftime("%Y%m%d-%H:%M:%S")

        contract = Stock(sym, "SMART", "USD")

        done_global = self.total_done_chunks()
        total_global = self.total_chunks()
        bars_so_far = len(self.bars[sym])

        # ETA calculation
        elapsed = time.monotonic() - self._start_time
        if self._request_count > 0:
            avg_per_req = elapsed / self._request_count
            remaining_chunks = self.total_remaining_chunks()
            eta = avg_per_req * remaining_chunks
        else:
            eta = -1

        print(
            f"  [{done_global + 1}/{total_global}] {sym} chunk {chunk_num}/{total_sym_chunks} "
            f"| {bars_so_far} bars | ETA: {format_eta(eta)}"
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.ib = ensure_connected(self.ib, self.host, self.port, self.client_id)

                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt_str,
                    durationStr="1 W",
                    barSizeSetting="5 mins",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                )

                bar_dicts = bars_to_dicts(bars)
                self.bars[sym].extend(bar_dicts)
                self.cursor[sym] += 1
                self._request_count += 1
                self.chunks_since_save[sym] += 1

                # Update manifest
                if sym not in self.manifest:
                    self.manifest[sym] = {}
                self.manifest[sym]["last_end_dt"] = end_dt.isoformat()
                self.manifest[sym]["chunks_done"] = self.cursor[sym]
                self.manifest[sym]["bars_total"] = len(self.bars[sym])
                save_manifest(self.manifest)

                # Incremental save every N chunks
                if self.chunks_since_save[sym] >= INCREMENTAL_SAVE_INTERVAL:
                    save_partial_bars(sym, self.bars[sym])
                    self.chunks_since_save[sym] = 0
                    print(f"    (saved {len(self.bars[sym])} partial bars for {sym})")

                return True

            except Exception as e:
                err_str = str(e)

                # Check for IBKR pacing violation (error 162)
                if "162" in err_str or "pacing" in err_str.lower():
                    print(f"    Pacing violation on {sym}. Sleeping 60s...")
                    time.sleep(60)
                    continue  # retry same chunk

                print(f"    Error on {sym} attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)
                    # Force reconnect
                    try:
                        self.ib.disconnect()
                    except Exception:
                        pass
                    self.ib = IB()  # fresh instance
                else:
                    print(f"    Skipping chunk {chunk_num} for {sym} after {max_retries} failures")
                    self.cursor[sym] += 1  # skip this chunk
                    # Update manifest even for skipped chunk so we don't retry forever
                    if sym not in self.manifest:
                        self.manifest[sym] = {}
                    self.manifest[sym]["last_end_dt"] = end_dt.isoformat()
                    self.manifest[sym]["chunks_done"] = self.cursor[sym]
                    self.manifest[sym]["bars_total"] = len(self.bars[sym])
                    save_manifest(self.manifest)
                    return False

        return False

    def _finalize_symbol(self, sym: str) -> None:
        """Save final bar file for a completed symbol."""
        if self.completed[sym]:
            return

        self.completed[sym] = True
        start_str = self.start_date.isoformat()
        end_str = self.end_date.isoformat()

        out_path = save_final_bars(sym, self.bars[sym], start_str, end_str)
        total_bars = len(self.bars[sym])

        # Determine date range of actual data
        if total_bars > 0:
            sorted_bars = sorted(self.bars[sym], key=lambda b: b["timestamp"])
            first_ts = sorted_bars[0]["timestamp"]
            last_ts = sorted_bars[-1]["timestamp"]
        else:
            first_ts = "N/A"
            last_ts = "N/A"

        print(f"\n  ** Completed {sym}: {total_bars} bars, {first_ts} to {last_ts}")
        print(f"     Saved to {out_path}\n")

        # Update manifest to mark completion
        if sym not in self.manifest:
            self.manifest[sym] = {}
        self.manifest[sym]["completed"] = True
        self.manifest[sym]["bars_total"] = total_bars
        self.manifest[sym]["file"] = str(out_path)
        save_manifest(self.manifest)

    def _save_all_progress(self) -> None:
        """Save progress for all symbols (called on shutdown)."""
        for sym in self.symbols:
            if not self.completed[sym] and self.bars[sym]:
                save_partial_bars(sym, self.bars[sym])
                print(f"  Saved {len(self.bars[sym])} partial bars for {sym}")
        save_manifest(self.manifest)
        print("  Manifest saved.")

    def _print_completion_summary(self) -> None:
        """Print a summary table after all symbols are downloaded."""
        elapsed = time.monotonic() - self._start_time
        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"Total time: {format_eta(elapsed)}")
        print(f"Total requests: {self._request_count}")
        print()
        print(f"{'Symbol':<8} {'Bars':>8} {'First Date':<22} {'Last Date':<22} {'File'}")
        print("-" * 80)

        start_str = self.start_date.isoformat()
        end_str = self.end_date.isoformat()

        for sym in self.symbols:
            bars = self.bars.get(sym, [])
            total = len(bars)
            if total > 0:
                sorted_bars = sorted(bars, key=lambda b: b["timestamp"])
                first_ts = sorted_bars[0]["timestamp"]
                last_ts = sorted_bars[-1]["timestamp"]
            else:
                first_ts = "N/A"
                last_ts = "N/A"
            fname = cache_filename(sym, start_str, end_str)
            print(f"{sym:<8} {total:>8} {first_ts:<22} {last_ts:<22} {fname}")

        print("=" * 80)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cache_files(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> None:
    """Validate existing cache files and report statistics."""
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    print("\n" + "=" * 80)
    print("DATA VALIDATION REPORT")
    print("=" * 80)
    print()

    # Expected trading days: ~252/year for 10 years ~ 2520 days
    # Expected bars per day (RTH, 5-min): 9:30-16:00 = 6.5 hours = 78 bars
    # Total expected ~ 2520 * 78 ~ 196,560 per symbol
    years = (end_date - start_date).days / 365.25
    expected_days = int(years * 252)
    expected_bars_per_day = 78  # 6.5 hours * 12 bars/hour
    expected_total = expected_days * expected_bars_per_day

    five_years_ago = end_date - timedelta(days=5 * 365)

    print(f"Date range: {start_str} to {end_str} ({years:.1f} years)")
    print(f"Expected ~{expected_days} trading days, ~{expected_total} bars per symbol")
    print()
    print(f"{'Symbol':<8} {'Bars':>8} {'Expected':>8} {'Pct':>6} {'First Date':<22} {'Last Date':<22} {'Gaps':>5} {'Issues'}")
    print("-" * 110)

    for sym in symbols:
        fname = cache_filename(sym, start_str, end_str)
        fpath = CACHE_DIR / fname

        if not fpath.exists():
            print(f"{sym:<8} {'MISSING':>8} {expected_total:>8} {'0%':>6} {'N/A':<22} {'N/A':<22} {'N/A':>5} FILE NOT FOUND")
            continue

        try:
            with open(fpath, "r") as f:
                bars = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"{sym:<8} {'ERROR':>8} {expected_total:>8} {'0%':>6} {'N/A':<22} {'N/A':<22} {'N/A':>5} {e}")
            continue

        total = len(bars)
        if total == 0:
            print(f"{sym:<8} {0:>8} {expected_total:>8} {'0%':>6} {'N/A':<22} {'N/A':<22} {'N/A':>5} EMPTY FILE")
            continue

        # Sort by timestamp
        bars.sort(key=lambda b: b["timestamp"])
        first_ts = bars[0]["timestamp"]
        last_ts = bars[-1]["timestamp"]
        pct = total / expected_total * 100

        # Find gaps > 1 week
        gap_count = 0
        prev_dt = datetime.fromisoformat(bars[0]["timestamp"])
        for b in bars[1:]:
            curr_dt = datetime.fromisoformat(b["timestamp"])
            diff = curr_dt - prev_dt
            if diff > timedelta(weeks=1):
                gap_count += 1
            prev_dt = curr_dt

        # Check for < 5 years of data
        issues = []
        first_date = datetime.fromisoformat(first_ts).date()
        if first_date > five_years_ago:
            issues.append(f"<5yr (starts {first_date})")
        if gap_count > 0:
            issues.append(f"{gap_count} gaps>1wk")

        issue_str = "; ".join(issues) if issues else "OK"

        print(
            f"{sym:<8} {total:>8} {expected_total:>8} {pct:>5.1f}% "
            f"{first_ts:<22} {last_ts:<22} {gap_count:>5} {issue_str}"
        )

    print("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10 years of 5-minute IBKR data for backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Download all 11 symbols
    %(prog)s --symbols SPY QQQ            # Download only SPY and QQQ
    %(prog)s --validate-only              # Validate existing cache files
    %(prog)s --sleep 12 --port 4002       # Custom pacing and port
    %(prog)s --no-resume                  # Start fresh (ignore manifest)
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4002,
        help="IB Gateway port (default: 4002)",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=55,
        help="IBKR client ID (default: 55)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="Start date YYYY-MM-DD (default: 2016-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-02-25",
        help="End date YYYY-MM-DD (default: 2026-02-25)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help=f"Symbols to download (default: {' '.join(SYMBOLS)})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=11.0,
        help="Seconds to sleep between requests (default: 11)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from manifest (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Start fresh, ignore existing manifest",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Only validate existing cache files, do not download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    symbols = [s.upper() for s in args.symbols]

    # Handle --no-resume
    if args.no_resume:
        args.resume = False

    print("=" * 80)
    print("IBKR 10-Year 5-Minute Bar Data Downloader")
    print("=" * 80)
    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Port:       {args.port}")
    print(f"Client ID:  {args.client_id}")
    print(f"Sleep:      {args.sleep}s between requests")
    print(f"Resume:     {args.resume}")
    print(f"Cache dir:  {CACHE_DIR}")
    print(f"Manifest:   {MANIFEST_PATH}")

    if args.validate_only:
        validate_cache_files(symbols, start_date, end_date)
        return

    # Ensure directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Run the download
    engine = DownloadEngine(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        host="127.0.0.1",
        port=args.port,
        client_id=args.client_id,
        sleep_secs=args.sleep,
        resume=args.resume,
    )
    engine.run()

    # Validate after download
    print("\nRunning post-download validation...")
    validate_cache_files(symbols, start_date, end_date)


if __name__ == "__main__":
    main()
