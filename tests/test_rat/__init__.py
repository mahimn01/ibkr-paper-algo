"""Test utilities and fixtures for RAT framework tests."""

from datetime import datetime, timedelta
from typing import List, Tuple
import random

from trading_algo.rat.backtest.data_loader import Bar


def generate_trending_prices(
    start_price: float = 100.0,
    num_bars: int = 100,
    trend: float = 0.001,  # Daily drift
    volatility: float = 0.02,
) -> List[float]:
    """Generate trending price series."""
    random.seed(42)
    prices = [start_price]
    for _ in range(num_bars - 1):
        change = trend + volatility * (random.random() - 0.5) * 2
        prices.append(prices[-1] * (1 + change))
    return prices


def generate_mean_reverting_prices(
    mean_price: float = 100.0,
    num_bars: int = 100,
    reversion_speed: float = 0.1,
    volatility: float = 0.02,
) -> List[float]:
    """Generate mean-reverting price series."""
    random.seed(42)
    prices = [mean_price]
    for _ in range(num_bars - 1):
        deviation = (mean_price - prices[-1]) / mean_price
        change = reversion_speed * deviation + volatility * (random.random() - 0.5) * 2
        prices.append(prices[-1] * (1 + change))
    return prices


def generate_cyclic_prices(
    base_price: float = 100.0,
    num_bars: int = 100,
    amplitude: float = 10.0,
    period: int = 20,
) -> List[float]:
    """Generate cyclic price series."""
    import math
    prices = []
    for i in range(num_bars):
        prices.append(base_price + amplitude * math.sin(2 * math.pi * i / period))
    return prices


def generate_bars(
    symbol: str,
    prices: List[float],
    start_time: datetime = None,
    interval_seconds: int = 86400,  # Daily
) -> List[Bar]:
    """Generate Bar objects from price series."""
    start_time = start_time or datetime(2023, 1, 1)
    bars = []

    for i, close in enumerate(prices):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)

        # Generate OHLC from close
        volatility = 0.01
        high = close * (1 + volatility * random.random())
        low = close * (1 - volatility * random.random())
        open_price = (high + low) / 2

        bar = Bar(
            timestamp=timestamp,
            symbol=symbol,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=random.randint(10000, 100000),
        )
        bars.append(bar)

    return bars


def generate_order_flow(
    num_ticks: int = 100,
    bias: float = 0.0,  # Positive = more buying
) -> List[Tuple[datetime, float, str, float, float]]:
    """
    Generate order flow data.

    Returns list of (timestamp, price, aggressor, bid, ask)
    """
    random.seed(42)
    start_time = datetime(2023, 1, 1, 9, 30)
    price = 100.0
    spread = 0.02

    ticks = []
    for i in range(num_ticks):
        timestamp = start_time + timedelta(seconds=i)

        # Determine aggressor
        prob_buy = 0.5 + bias
        aggressor = "buy" if random.random() < prob_buy else "sell"

        # Update price based on aggressor
        if aggressor == "buy":
            price *= 1 + random.random() * 0.001
        else:
            price *= 1 - random.random() * 0.001

        bid = price - spread / 2
        ask = price + spread / 2
        volume = random.randint(100, 1000)

        ticks.append({
            "timestamp": timestamp,
            "price": price,
            "volume": volume,
            "aggressor": aggressor,
            "bid": bid,
            "ask": ask,
        })

    return ticks


def generate_news_events(
    symbol: str,
    num_events: int = 10,
    start_time: datetime = None,
) -> List[dict]:
    """Generate news events."""
    start_time = start_time or datetime(2023, 1, 1, 9, 30)
    events = []

    for i in range(num_events):
        timestamp = start_time + timedelta(minutes=i * 5)
        events.append({
            "symbol": symbol,
            "headline": f"News headline {i} for {symbol}",
            "timestamp": timestamp,
            "sentiment": random.uniform(-1, 1),
        })

    return events
