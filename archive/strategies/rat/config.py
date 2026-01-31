"""
RAT Configuration: Comprehensive configuration for all RAT modules.

All configuration is via frozen dataclasses with defaults.
Environment variable loading supported via from_env().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from trading_algo.rat.combiner.combiner import WeightingMethod


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for Attention Flow module."""

    flow_window: int = 100
    news_weight: float = 0.4
    flow_weight: float = 0.35
    price_weight: float = 0.25
    news_decay_seconds: float = 3600.0
    min_attention_threshold: float = 0.3


@dataclass(frozen=True)
class ReflexivityConfig:
    """Configuration for Reflexivity Meter module."""

    lookback: int = 50
    lag_order: int = 5
    significance_level: float = 0.05
    min_data_points: int = 30


@dataclass(frozen=True)
class TopologyConfig:
    """Configuration for Topology Detector module."""

    embedding_dim: int = 3
    time_delay: int = 1
    max_dimension: int = 2
    max_edge_length: float = 2.0
    min_persistence: float = 0.1


@dataclass(frozen=True)
class AdversarialConfig:
    """Configuration for Adversarial Meta-Trader module."""

    flow_window: int = 500
    detection_threshold: float = 0.65
    round_number_tolerance: float = 0.001


@dataclass(frozen=True)
class AlphaConfig:
    """Configuration for Self-Cannibalizing Alpha module."""

    sharpe_window: int = 20
    ic_window: int = 20
    decay_threshold: float = 0.5
    crowding_threshold: float = 0.7
    initial_factors: int = 5
    enable_llm_mutation: bool = False
    llm_cooldown_hours: float = 24.0


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for signal combination and filtering."""

    weighting_method: WeightingMethod = WeightingMethod.SHARPE_WEIGHTED
    min_signals_required: int = 2
    agreement_threshold: float = 0.6
    confidence_threshold: float = 0.5
    max_position_pct: float = 0.25
    max_signals_per_hour: int = 10


@dataclass(frozen=True)
class RATBacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100000.0
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    warmup_bars: int = 50


@dataclass(frozen=True)
class RATConfig:
    """Master configuration for RAT framework."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    reflexivity: ReflexivityConfig = field(default_factory=ReflexivityConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    backtest: RATBacktestConfig = field(default_factory=RATBacktestConfig)

    @classmethod
    def from_env(cls) -> "RATConfig":
        """Load configuration from environment variables."""

        def get_float(key: str, default: float) -> float:
            return float(os.environ.get(key, default))

        def get_int(key: str, default: int) -> int:
            return int(os.environ.get(key, default))

        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, str(default)).lower()
            return val in ("true", "1", "yes")

        attention = AttentionConfig(
            flow_window=get_int("RAT_ATTENTION_FLOW_WINDOW", 100),
            news_weight=get_float("RAT_ATTENTION_NEWS_WEIGHT", 0.4),
            flow_weight=get_float("RAT_ATTENTION_FLOW_WEIGHT", 0.35),
            price_weight=get_float("RAT_ATTENTION_PRICE_WEIGHT", 0.25),
        )

        reflexivity = ReflexivityConfig(
            lookback=get_int("RAT_REFLEXIVITY_LOOKBACK", 50),
            lag_order=get_int("RAT_REFLEXIVITY_LAG_ORDER", 5),
            significance_level=get_float("RAT_REFLEXIVITY_SIGNIFICANCE", 0.05),
        )

        topology = TopologyConfig(
            embedding_dim=get_int("RAT_TOPOLOGY_EMBEDDING_DIM", 3),
            time_delay=get_int("RAT_TOPOLOGY_TIME_DELAY", 1),
            max_dimension=get_int("RAT_TOPOLOGY_MAX_DIM", 2),
        )

        adversarial = AdversarialConfig(
            flow_window=get_int("RAT_ADVERSARIAL_FLOW_WINDOW", 500),
            detection_threshold=get_float("RAT_ADVERSARIAL_THRESHOLD", 0.65),
        )

        alpha = AlphaConfig(
            sharpe_window=get_int("RAT_ALPHA_SHARPE_WINDOW", 20),
            ic_window=get_int("RAT_ALPHA_IC_WINDOW", 20),
            decay_threshold=get_float("RAT_ALPHA_DECAY_THRESHOLD", 0.5),
            enable_llm_mutation=get_bool("RAT_ALPHA_ENABLE_LLM", False),
        )

        signal = SignalConfig(
            confidence_threshold=get_float("RAT_SIGNAL_CONFIDENCE_THRESHOLD", 0.5),
            max_position_pct=get_float("RAT_SIGNAL_MAX_POSITION_PCT", 0.25),
            max_signals_per_hour=get_int("RAT_SIGNAL_MAX_PER_HOUR", 10),
        )

        backtest = RATBacktestConfig(
            initial_capital=get_float("RAT_BACKTEST_CAPITAL", 100000.0),
            commission_per_share=get_float("RAT_BACKTEST_COMMISSION", 0.005),
            slippage_bps=get_float("RAT_BACKTEST_SLIPPAGE_BPS", 5.0),
        )

        return cls(
            attention=attention,
            reflexivity=reflexivity,
            topology=topology,
            adversarial=adversarial,
            alpha=alpha,
            signal=signal,
            backtest=backtest,
        )
