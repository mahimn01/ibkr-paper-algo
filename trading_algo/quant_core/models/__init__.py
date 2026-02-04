"""
Statistical and mathematical models for quantitative trading.

Models:
    - OrnsteinUhlenbeck: Mean reversion model based on Avellaneda & Lee (2010)
    - HiddenMarkovRegime: Regime detection using HMM
    - TimeSeriesMomentum: TSMOM based on Moskowitz, Ooi & Pedersen (2012)
    - VolatilityManagedMomentum: Daniel & Moskowitz (2016) volatility scaling
"""

from trading_algo.quant_core.models.ornstein_uhlenbeck import (
    OrnsteinUhlenbeck,
    OUParameters,
    OUSignal,
)
from trading_algo.quant_core.models.hmm_regime import (
    HiddenMarkovRegime,
    RegimeState,
)
from trading_algo.quant_core.models.tsmom import (
    TimeSeriesMomentum,
    TSMOMAssetSignal,
)
from trading_algo.quant_core.models.vol_managed_momentum import (
    VolatilityManagedMomentum,
    VolManagedSignal,
)

__all__ = [
    "OrnsteinUhlenbeck",
    "OUParameters",
    "OUSignal",
    "HiddenMarkovRegime",
    "RegimeState",
    "TimeSeriesMomentum",
    "TSMOMAssetSignal",
    "VolatilityManagedMomentum",
    "VolManagedSignal",
]
