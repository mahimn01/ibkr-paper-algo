"""
Base class for all crypto edge engines.

Each edge produces an EdgeSignal (vote + confidence) per symbol.
Adapters wrap edges into the TradingStrategy protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from crypto_alpha.types import CryptoAssetState, EdgeSignal, CryptoEdgeVote


class CryptoEdge(ABC):
    """Abstract base class for crypto edge engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique edge name."""

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Minimum bars needed before producing signals."""

    @abstractmethod
    def update(self, symbol: str, timestamp: datetime,
               price: float, volume: float, **kwargs) -> None:
        """Feed a single bar of data to the edge."""

    @abstractmethod
    def get_vote(self, symbol: str, state: CryptoAssetState) -> EdgeSignal:
        """Produce a vote for a symbol given current state."""

    def reset(self) -> None:
        """Reset edge state."""
        pass
