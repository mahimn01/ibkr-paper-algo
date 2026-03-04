"""
Core types for the Phantom Alpha Engine.

CryptoBar extends the standard bar with crypto-specific fields.
CryptoEdgeVote mirrors the equity EdgeVote system for crypto edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional


class CryptoEdgeVote(IntEnum):
    """Vote from a crypto edge engine."""
    STRONG_SHORT = -2
    SHORT = -1
    NEUTRAL = 0
    LONG = 1
    STRONG_LONG = 2
    VETO_LONG = -99
    VETO_SHORT = 99


@dataclass
class CryptoBar:
    """A single bar of crypto market data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Crypto-specific fields
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    spot_price: Optional[float] = None  # For basis calculation on perp bars
    num_trades: Optional[int] = None


@dataclass
class EdgeSignal:
    """Signal emitted by a crypto edge engine."""
    edge_name: str
    vote: CryptoEdgeVote
    confidence: float  # 0.0 to 1.0
    reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FundingRateEntry:
    """A single funding rate observation."""
    timestamp: datetime
    symbol: str
    rate: float  # e.g., 0.0001 = 0.01% per 8h
    predicted_rate: Optional[float] = None


@dataclass
class LiquidationCluster:
    """A cluster of liquidation levels at a price range."""
    price_low: float
    price_high: float
    total_oi: float  # Total open interest in this cluster
    direction: str  # "long" or "short" — what gets liquidated
    distance_pct: float  # Distance from current price as percentage


@dataclass
class CascadeSignal:
    """Signal from the liquidation cascade predictor."""
    probability: float  # 0-1 probability of cascade
    direction: int  # 1 = buying cascade (short liqs), -1 = selling cascade (long liqs)
    magnitude: float  # Expected magnitude of cascade
    time_to_trigger_bars: int  # Estimated bars until trigger
    nearest_cluster: Optional[LiquidationCluster] = None


@dataclass
class CryptoAssetState:
    """Aggregated state for a crypto asset across all data sources."""
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float = 0.0
    # Price history for indicators
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    # Crypto-specific
    funding_rate: float = 0.0
    funding_history: List[float] = field(default_factory=list)
    open_interest: float = 0.0
    basis: float = 0.0  # ln(perp) - ln(spot)
    basis_history: List[float] = field(default_factory=list)
    # Regime
    regime: str = "unknown"
    regime_confidence: float = 0.0
