"""
Machine Learning Signal Enhancement Module.

Provides:
    - FeatureEngine: Feature engineering for trading signals
    - SignalCombiner: ML-based signal combination
    - TimeSeriesCV: Time-series aware cross-validation
"""

from trading_algo.quant_core.ml.features import FeatureEngine, FeatureSet
from trading_algo.quant_core.ml.signal_combiner import SignalCombiner, CombinerConfig
from trading_algo.quant_core.ml.cross_validation import TimeSeriesCV, PurgedKFold

__all__ = [
    "FeatureEngine",
    "FeatureSet",
    "SignalCombiner",
    "CombinerConfig",
    "TimeSeriesCV",
    "PurgedKFold",
]
