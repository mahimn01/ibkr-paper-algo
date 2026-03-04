"""Adapter for Edge 1: Liquidation Cascade Predictor."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.liquidation_cascade import LiquidationCascade


class LCPAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.14, **edge_kwargs):
        edge = LiquidationCascade(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
