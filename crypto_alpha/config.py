"""
Configuration for the Phantom Alpha Engine.

Provides PhantomAlphaConfig (master config) and a pre-built
ControllerConfig for the 9-edge crypto system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from trading_algo.multi_strategy.controller import ControllerConfig, StrategyAllocation


@dataclass
class PhantomAlphaConfig:
    """Master configuration for the Phantom Alpha Engine."""

    # ── Trading universe ──
    trading_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT",
    ])
    reference_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT",  # BTC is the reference for regime detection
    ])

    # ── Venues ──
    spot_exchange: str = "kraken"
    perp_exchange: str = "hyperliquid"

    # ── Data ──
    bar_size: str = "5m"  # 5-minute bars
    funding_interval_hours: int = 8

    # ── Risk ──
    initial_capital: float = 30_000.0  # CAD
    max_leverage: float = 5.0
    max_drawdown: float = 0.25  # 25% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    max_exchange_concentration: float = 0.60  # Max 60% on one exchange

    # ── Backtest ──
    commission_bps_maker: float = 2.0  # Hyperliquid maker
    commission_bps_taker: float = 5.0  # Hyperliquid taker
    slippage_bps: float = 3.0
    maintenance_margin_ratio: float = 0.03  # 3%

    # ── Edge weights (sum to ~1.0) ──
    edge_weights: Dict[str, float] = field(default_factory=lambda: {
        "LiquidationCascade": 0.12,
        "FundingRateMomentum": 0.14,
        "CrossExchangePropagation": 0.08,
        "VolatilitySurface": 0.12,
        "WhaleFlow": 0.10,
        "SocialVelocity": 0.06,
        "PerpBasisMeanReversion": 0.14,
        "RegimeAdaptiveLeverage": 0.12,
        "IntermarketCascade": 0.12,
    })


def build_controller_config(pa_config: PhantomAlphaConfig) -> ControllerConfig:
    """Build a ControllerConfig from PhantomAlphaConfig."""
    allocations = {
        name: StrategyAllocation(weight=weight, max_positions=6)
        for name, weight in pa_config.edge_weights.items()
    }

    return ControllerConfig(
        allocations=allocations,
        max_gross_exposure=3.0,
        max_net_exposure=2.0,
        max_single_symbol_weight=0.40,
        max_portfolio_positions=20,
        conflict_resolution="weighted_confidence",
        enable_vol_management=True,
        vol_target=0.30,  # Crypto: higher vol target than equities
        vol_lookback=20,
        vol_scale_min=0.20,
        vol_scale_max=3.0,
        max_drawdown=pa_config.max_drawdown,
        daily_loss_limit=pa_config.daily_loss_limit,
        enable_entropy_filter=True,
    )
