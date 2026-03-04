"""Adapter for Edge 3: Cross-Exchange Divergence."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.cross_exchange_divergence import CrossExchangeDivergence


class CEDAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.12, **edge_kwargs):
        edge = CrossExchangeDivergence(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
