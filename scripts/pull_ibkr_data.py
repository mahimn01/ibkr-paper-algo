#!/usr/bin/env python3
"""
Pull historical data from IBKR TWS/Gateway.

Prerequisites:
1. pip install ib_insync
2. TWS or IB Gateway running (paper: port 7497, live: port 7496)
3. API access enabled in TWS/Gateway settings

Usage:
    python scripts/pull_ibkr_data.py AAPL MSFT GOOG --days 30
    python scripts/pull_ibkr_data.py AAPL --days 365 --bar-size "1 hour"
    python scripts/pull_ibkr_data.py AAPL --output data/
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path


def pull_data(
    symbols: list[str],
    days: int = 30,
    bar_size: str = "1 day",
    output_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
) -> dict[str, list]:
    """
    Pull historical data from IBKR.

    Args:
        symbols: List of stock symbols
        days: Number of days of history
        bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins")
        output_dir: Directory to save CSV files (optional)
        host: TWS/Gateway host
        port: TWS/Gateway port (7497=paper, 7496=live)
        client_id: IBKR client ID

    Returns:
        Dictionary of symbol -> list of bars
    """
    from trading_algo.broker.ibkr import IBKRBroker
    from trading_algo.config import IBKRConfig
    from trading_algo.instruments import InstrumentSpec

    # Connect to IBKR
    config = IBKRConfig(host=host, port=port, client_id=client_id)
    broker = IBKRBroker(config=config, require_paper=(port in [7497, 4002]))

    print(f"Connecting to IBKR at {host}:{port}...")
    try:
        broker.connect()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  1. Is TWS/IB Gateway running?")
        print("  2. Is API access enabled? (File > Global Configuration > API > Settings)")
        print("  3. Is the port correct? (Paper=7497/4002, Live=7496/4001)")
        return {}

    all_data = {}

    try:
        for symbol in symbols:
            print(f"\nFetching {days} days of {bar_size} data for {symbol}...")

            instrument = InstrumentSpec(
                symbol=symbol,
                sec_type="STK",
                exchange="SMART",
                currency="USD",
            )

            try:
                bars = broker.get_historical_bars(
                    instrument,
                    duration=f"{days} D",
                    bar_size=bar_size,
                    what_to_show="TRADES",
                    use_rth=False,
                )

                print(f"  Retrieved {len(bars)} bars")
                all_data[symbol] = bars

                # Save to CSV if output directory specified
                if output_dir and bars:
                    save_to_csv(symbol, bars, output_dir)

            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                all_data[symbol] = []

    finally:
        broker.disconnect()
        print("\nDisconnected from IBKR")

    return all_data


def save_to_csv(symbol: str, bars: list, output_dir: str) -> None:
    """Save bars to CSV file."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{symbol}.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])

        for bar in bars:
            ts = datetime.fromtimestamp(bar.timestamp_epoch_s)
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume or 0,
            ])

    print(f"  Saved to {filepath}")


def print_summary(data: dict[str, list]) -> None:
    """Print summary of fetched data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    for symbol, bars in data.items():
        if not bars:
            print(f"{symbol}: No data")
            continue

        first = bars[0]
        last = bars[-1]
        start_dt = datetime.fromtimestamp(first.timestamp_epoch_s)
        end_dt = datetime.fromtimestamp(last.timestamp_epoch_s)

        # Calculate returns
        start_price = first.close
        end_price = last.close
        total_return = (end_price - start_price) / start_price * 100

        # Calculate volatility
        if len(bars) > 1:
            returns = []
            for i in range(1, len(bars)):
                r = (bars[i].close - bars[i-1].close) / bars[i-1].close
                returns.append(r)
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            volatility = (variance ** 0.5) * (252 ** 0.5) * 100  # Annualized
        else:
            volatility = 0

        print(f"\n{symbol}:")
        print(f"  Period:     {start_dt.date()} to {end_dt.date()}")
        print(f"  Bars:       {len(bars)}")
        print(f"  Start:      ${start_price:,.2f}")
        print(f"  End:        ${end_price:,.2f}")
        print(f"  Return:     {total_return:+.2f}%")
        print(f"  Volatility: {volatility:.2f}% (annualized)")


def main():
    parser = argparse.ArgumentParser(
        description="Pull historical data from IBKR TWS/Gateway"
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbols to fetch (e.g., AAPL MSFT GOOG)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history (default: 30)"
    )
    parser.add_argument(
        "--bar-size",
        default="1 day",
        help="Bar size: '1 day', '1 hour', '5 mins', etc. (default: '1 day')"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TWS/Gateway host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="TWS/Gateway port (default: 7497 for paper trading)"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        help="IBKR client ID (default: 1)"
    )

    args = parser.parse_args()

    # Pull data
    data = pull_data(
        symbols=[s.upper() for s in args.symbols],
        days=args.days,
        bar_size=args.bar_size,
        output_dir=args.output,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )

    # Print summary
    if data:
        print_summary(data)


if __name__ == "__main__":
    main()
