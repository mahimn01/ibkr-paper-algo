"""
Synthetic crypto data generator for backtesting.

Generates realistic BTC, ETH, SOL price series with:
  - Correlated returns (BTC leads, alts follow with lag)
  - Regime changes (bull/bear/sideways)
  - Realistic volatility clustering (GARCH-like)
  - Funding rates correlated with price momentum
  - Basis that mean-reverts around zero

This allows backtesting the Phantom Alpha Engine without
waiting for CCXT data downloads.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

from crypto_alpha.types import CryptoBar


def generate_gbm_with_regimes(
    n_days: int = 365 * 3,
    bars_per_day: int = 288,  # 5-min bars
    initial_price: float = 30000.0,
    annual_vol: float = 0.65,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate price series with regime changes.

    Returns:
        (prices, volatilities, regimes) where each element has
        n_days * bars_per_day entries.
    """
    rng = np.random.RandomState(seed)
    n_bars = n_days * bars_per_day
    dt = 1.0 / (365 * bars_per_day)  # Time step in years

    # Define regimes: (drift, vol_multiplier, duration_days)
    regime_defs = [
        ("bull",    0.50, 0.7, 90),   # Bull: positive drift, lower vol
        ("bear",   -0.30, 1.4, 60),   # Bear: negative drift, higher vol
        ("sideways", 0.02, 0.8, 120),  # Sideways: near-zero drift, lower vol
        ("crash",  -1.50, 3.0, 15),    # Crash: sharp negative, very high vol
        ("recovery", 0.80, 1.2, 45),   # Recovery: strong positive
    ]

    prices = np.zeros(n_bars)
    vols = np.zeros(n_bars)
    regimes = []

    prices[0] = initial_price
    bar_idx = 0

    while bar_idx < n_bars:
        # Pick a regime
        regime = regime_defs[rng.randint(0, len(regime_defs))]
        name, drift, vol_mult, duration_days = regime
        duration_bars = min(duration_days * bars_per_day, n_bars - bar_idx)

        bar_vol = annual_vol * vol_mult * math.sqrt(dt)

        for j in range(duration_bars):
            if bar_idx + j >= n_bars:
                break
            idx = bar_idx + j

            # GARCH-like volatility clustering
            if idx > 0:
                shock = abs(prices[idx - 1] / prices[max(0, idx - 2)] - 1) if idx > 1 else 0
                vol_scale = 1.0 + 2.0 * shock  # Vol increases after large moves
            else:
                vol_scale = 1.0

            effective_vol = bar_vol * vol_scale
            ret = drift * dt + effective_vol * rng.randn()

            if idx == 0:
                prices[idx] = initial_price
            else:
                prices[idx] = prices[idx - 1] * (1 + ret)
                prices[idx] = max(prices[idx], 1.0)  # Floor at $1

            vols[idx] = effective_vol * math.sqrt(1 / dt)  # Annualize
            regimes.append(name)

        bar_idx += duration_bars

    # Trim regimes to match
    regimes = regimes[:n_bars]
    while len(regimes) < n_bars:
        regimes.append("unknown")

    return prices, vols, regimes


def generate_correlated_alt(
    btc_prices: np.ndarray,
    correlation: float = 0.7,
    beta: float = 1.3,
    lag_bars: int = 3,
    initial_price: float = 2000.0,
    seed: int = 123,
) -> np.ndarray:
    """Generate an alt price series correlated with BTC with a lag."""
    rng = np.random.RandomState(seed)
    n = len(btc_prices)

    btc_returns = np.diff(btc_prices) / btc_prices[:-1]

    # Lag BTC returns
    lagged_returns = np.zeros(len(btc_returns))
    if lag_bars > 0:
        lagged_returns[lag_bars:] = btc_returns[:-lag_bars]
    else:
        lagged_returns = btc_returns

    # Generate alt returns = beta * lagged_btc + idiosyncratic noise
    idio_vol = np.std(btc_returns) * math.sqrt(1 - correlation ** 2) * beta
    alt_returns = beta * correlation * lagged_returns + idio_vol * rng.randn(len(btc_returns))

    # Build price series
    alt_prices = np.zeros(n)
    alt_prices[0] = initial_price
    for i in range(1, n):
        alt_prices[i] = alt_prices[i - 1] * (1 + alt_returns[i - 1])
        alt_prices[i] = max(alt_prices[i], 0.01)

    return alt_prices


def generate_funding_rates(
    prices: np.ndarray,
    bars_per_day: int = 288,
    funding_interval_bars: int = 96,  # Every 8 hours on 5-min bars
    base_rate: float = 0.0001,  # 0.01% per 8h
    momentum_sensitivity: float = 2.0,
    mean_reversion_speed: float = 0.15,
    seed: int = 456,
) -> Tuple[np.ndarray, List[datetime]]:
    """
    Generate realistic funding rates correlated with price momentum.

    Funding rates are:
    - Positive when price is trending up (longs pay shorts)
    - Negative when price is trending down (shorts pay longs)
    - Mean-reverting toward base_rate
    - Published every 8 hours
    """
    rng = np.random.RandomState(seed)
    n = len(prices)

    returns = np.diff(prices) / prices[:-1]
    funding = np.full(n, base_rate)

    # Only update at funding intervals
    rate = base_rate
    for i in range(funding_interval_bars, n, funding_interval_bars):
        # Momentum over last funding period
        window = returns[max(0, i - funding_interval_bars):i]
        momentum = np.sum(window) if len(window) > 0 else 0

        # Funding rate = base + momentum_sensitivity * momentum + noise
        noise = rng.randn() * base_rate * 0.5
        rate = (
            rate * (1 - mean_reversion_speed)
            + base_rate * mean_reversion_speed
            + momentum_sensitivity * momentum
            + noise
        )

        # Clip to realistic range
        rate = np.clip(rate, -0.003, 0.003)  # -0.3% to +0.3% per 8h

        funding[i:min(i + funding_interval_bars, n)] = rate

    return funding, []


def generate_basis(
    perp_prices: np.ndarray,
    funding_rates: np.ndarray,
    mean_reversion_speed: float = 0.05,
    basis_vol: float = 0.001,
    seed: int = 789,
) -> np.ndarray:
    """
    Generate spot prices from perp prices + basis.

    Basis = ln(perp) - ln(spot) mean-reverts around zero
    but is influenced by funding rate direction.
    """
    rng = np.random.RandomState(seed)
    n = len(perp_prices)

    basis = np.zeros(n)
    spot_prices = np.zeros(n)

    for i in range(n):
        if i == 0:
            basis[i] = 0.0001  # Small initial basis
        else:
            # Basis mean-reverts + is pushed by funding direction
            funding_push = funding_rates[i] * 0.5
            basis[i] = (
                basis[i - 1] * (1 - mean_reversion_speed)
                + funding_push
                + basis_vol * rng.randn()
            )
            basis[i] = np.clip(basis[i], -0.01, 0.01)

        # spot = perp * exp(-basis)
        spot_prices[i] = perp_prices[i] * math.exp(-basis[i])

    return spot_prices


def generate_ohlcv_from_close(
    close_prices: np.ndarray,
    seed: int = 101,
) -> np.ndarray:
    """Generate OHLCV from close prices."""
    rng = np.random.RandomState(seed)
    n = len(close_prices)
    ohlcv = np.zeros((n, 5))

    for i in range(n):
        c = close_prices[i]
        spread = abs(c * 0.002 * rng.randn())

        o = c * (1 + 0.001 * rng.randn()) if i == 0 else close_prices[i - 1]
        h = max(o, c) + spread
        l = min(o, c) - spread
        v = abs(rng.lognormal(15, 1.5))  # Log-normal volume

        ohlcv[i] = [o, h, l, c, v]

    return ohlcv


def generate_synthetic_universe(
    n_days: int = 365 * 3,
    bars_per_day: int = 288,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> Dict[str, List[CryptoBar]]:
    """
    Generate a full synthetic crypto universe for backtesting.

    Returns dict of symbol -> list of CryptoBar objects.
    Generates BTC, ETH, SOL with:
    - Correlated returns (BTC leads)
    - Realistic funding rates
    - Spot/perp basis
    """
    n_bars = n_days * bars_per_day
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Generate timestamps (5-min bars, 24/7)
    timestamps = [start_dt + timedelta(minutes=5 * i) for i in range(n_bars)]

    # BTC
    btc_perp, btc_vols, btc_regimes = generate_gbm_with_regimes(
        n_days=n_days, bars_per_day=bars_per_day,
        initial_price=42000.0, annual_vol=0.65, seed=seed,
    )
    btc_funding, _ = generate_funding_rates(btc_perp, bars_per_day, seed=seed + 10)
    btc_spot = generate_basis(btc_perp, btc_funding, seed=seed + 20)
    btc_ohlcv = generate_ohlcv_from_close(btc_perp, seed=seed + 30)

    # ETH (corr=0.8, beta=1.2, lag=2 bars)
    eth_perp = generate_correlated_alt(
        btc_perp, correlation=0.80, beta=1.2, lag_bars=2,
        initial_price=3000.0, seed=seed + 100,
    )
    eth_funding, _ = generate_funding_rates(eth_perp, bars_per_day, seed=seed + 110)
    eth_spot = generate_basis(eth_perp, eth_funding, seed=seed + 120)
    eth_ohlcv = generate_ohlcv_from_close(eth_perp, seed=seed + 130)

    # SOL (corr=0.65, beta=1.5, lag=4 bars)
    sol_perp = generate_correlated_alt(
        btc_perp, correlation=0.65, beta=1.5, lag_bars=4,
        initial_price=100.0, seed=seed + 200,
    )
    sol_funding, _ = generate_funding_rates(sol_perp, bars_per_day, seed=seed + 210)
    sol_spot = generate_basis(sol_perp, sol_funding, seed=seed + 220)
    sol_ohlcv = generate_ohlcv_from_close(sol_perp, seed=seed + 230)

    # Build CryptoBar lists
    def _build_bars(ohlcv, funding, spot, ts_list):
        bars = []
        for i in range(len(ts_list)):
            bars.append(CryptoBar(
                timestamp=ts_list[i],
                open=ohlcv[i, 0],
                high=ohlcv[i, 1],
                low=ohlcv[i, 2],
                close=ohlcv[i, 3],
                volume=ohlcv[i, 4],
                funding_rate=funding[i],
                spot_price=spot[i],
            ))
        return bars

    return {
        "BTC/USDT": _build_bars(btc_ohlcv, btc_funding, btc_spot, timestamps),
        "ETH/USDT": _build_bars(eth_ohlcv, eth_funding, eth_spot, timestamps),
        "SOL/USDT": _build_bars(sol_ohlcv, sol_funding, sol_spot, timestamps),
    }
