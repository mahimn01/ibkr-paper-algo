"""Adapter for Edge 7: Perpetual Basis Mean Reversion."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.perpetual_basis import PerpetualBasisMeanReversion


class PBMRAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.14, **edge_kwargs):
        edge = PerpetualBasisMeanReversion(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
