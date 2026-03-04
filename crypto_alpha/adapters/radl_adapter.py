"""Adapter for Edge 8: Regime-Adaptive Dynamic Leverage."""

from crypto_alpha.adapters.edge_adapter import CryptoEdgeAdapter
from crypto_alpha.edges.regime_adaptive_leverage import RegimeAdaptiveLeverage


class RADLAdapter(CryptoEdgeAdapter):
    def __init__(self, base_weight: float = 0.12, **edge_kwargs):
        edge = RegimeAdaptiveLeverage(**edge_kwargs)
        super().__init__(edge=edge, base_weight=base_weight)

    def get_leverage_scalar(self) -> float:
        """Expose the regime leverage scalar for other adapters."""
        return self._edge.get_leverage_scalar()

    def get_regime_info(self):
        """Expose regime info."""
        return self._edge.get_regime_info()
