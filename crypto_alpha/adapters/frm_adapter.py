"""Adapter for Edge 2: Funding Rate Momentum."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.funding_rate_momentum import FundingRateMomentum


class FRMAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.14, **edge_kwargs):
        edge = FundingRateMomentum(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
