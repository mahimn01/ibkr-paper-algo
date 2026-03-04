"""Adapter for Edge 6: Volume Velocity Breakout."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.volume_velocity import VolumeVelocityBreakout


class VVAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.12, **edge_kwargs):
        edge = VolumeVelocityBreakout(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)
