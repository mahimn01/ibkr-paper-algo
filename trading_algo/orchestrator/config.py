"""
Orchestrator Configuration

Replaces hardcoded magic numbers with a single, tunable configuration
dataclass. Every parameter that affects trading decisions is exposed here
so it can be varied in sensitivity analysis and parameter sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SizingConfig:
    """Position sizing parameters."""

    base_size: float = 0.008
    """Base position as fraction of account equity (0.8%)."""

    consensus_weight: float = 1.25
    """Multiplier applied to consensus_score in quality_boost."""

    agreement_weight: float = 0.5
    """Multiplier applied to agreement_ratio in quality_boost."""

    regime_weight: float = 0.75
    """Multiplier applied to excess regime confidence (above 0.5)."""

    vol_target_atr_pct: float = 0.01
    """ATR% that receives a 1.0 volatility scalar (ideal range)."""

    vol_scalar_min: float = 0.35
    """Floor on the volatility scalar (high-vol environment)."""

    vol_scalar_max: float = 1.35
    """Cap on the volatility scalar (low-vol environment)."""


@dataclass
class ExitConfig:
    """Exit / trailing-stop parameters."""

    trailing_activation_atr: float = 2.0
    """ATR multiples of profit before trailing stop activates."""

    trailing_breakeven_offset_atr: float = 1.0
    """After activation, move stop to entry + this many ATR."""

    trailing_distance_atr: float = 1.5
    """Trail distance once trailing stop is active (ATR multiples)."""

    eod_exit_time_hour: int = 15
    """Hour component of the end-of-day forced exit (24-h)."""

    eod_exit_time_minute: int = 55
    """Minute component of the end-of-day forced exit."""


@dataclass
class OrchestratorConfig:
    """
    Complete configuration for the Orchestrator strategy.

    All fields have sensible defaults matching the current production values
    so the Orchestrator remains backward-compatible when no config is passed.
    """

    # ── Consensus thresholds ──────────────────────────────────────────
    min_consensus_edges: int = 4
    """Minimum number of edges that must agree with the trade direction."""

    min_consensus_score: float = 0.5
    """Minimum weighted consensus score to enter a trade."""

    max_opposition_score: float = 0.35
    """Maximum opposition score allowed. Blocks entry if exceeded."""

    min_directional_quality: float = 0.6
    """Minimum support / (support + opposition) ratio for entry."""

    # ── Position limits ───────────────────────────────────────────────
    max_position_pct: float = 0.03
    """Maximum position size as fraction of account equity (3%)."""

    # ── Cooldown / churn reduction ────────────────────────────────────
    min_reentry_bars: int = 12
    """Minimum bars after an exit before re-entering the same symbol."""

    # ── Regime filters ────────────────────────────────────────────────
    min_regime_confidence: float = 0.5
    """Minimum market-regime confidence to consider trading."""

    # ── ATR volatility band ───────────────────────────────────────────
    min_atr_pct: float = 0.0015
    """Skip symbols whose ATR% is below this (choppy, low-range)."""

    max_atr_pct: float = 0.03
    """Skip symbols whose ATR% is above this (panic volatility)."""

    # ── Stop / target multiples ───────────────────────────────────────
    atr_stop_mult: float = 2.5
    """Stop-loss distance in ATR multiples from entry."""

    atr_target_mult: float = 4.0
    """Take-profit distance in ATR multiples from entry."""

    # ── Mean-reversion z-score threshold ──────────────────────────────
    mean_reversion_zscore: float = 1.5
    """Z-score threshold for mean-reversion entry in range-bound regimes."""

    # ── Warmup ────────────────────────────────────────────────────────
    warmup_bars: int = 30
    """Minimum bars required before the Orchestrator generates signals."""

    # ── Sub-configs ───────────────────────────────────────────────────
    sizing: SizingConfig = field(default_factory=SizingConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)

    # ── Quant edge integration ────────────────────────────────────────
    enable_quant_edge: bool = True
    """When True, adds the QuantEdge (quant_core signals) as a 7th voting edge."""

    quant_edge_min_bars: int = 60
    """Minimum price history bars before the QuantEdge produces a vote."""
