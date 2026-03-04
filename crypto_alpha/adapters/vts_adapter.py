"""Adapter for Edge 4: Volatility Term Structure."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.volatility_term_structure import VolatilityTermStructure


class VTSAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.12, **edge_kwargs):
        edge = VolatilityTermStructure(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
