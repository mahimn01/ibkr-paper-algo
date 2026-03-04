"""Adapter for Edge 5: Volume Flow Detector."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.volume_flow import VolumeFlowDetector


class VFAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.12, **edge_kwargs):
        edge = VolumeFlowDetector(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
