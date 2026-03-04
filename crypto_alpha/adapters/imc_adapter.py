"""Adapter for Edge 9: Intermarket Momentum Cascade."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.intermarket_cascade import IntermarketCascade


class IMCAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.12, **edge_kwargs):
        edge = IntermarketCascade(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
