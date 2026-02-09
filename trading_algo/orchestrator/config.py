"""
Orchestrator Configuration

Replaces hardcoded magic numbers with a single, tunable configuration
dataclass. Every parameter that affects trading decisions is exposed here
so it can be varied in sensitivity analysis and parameter sweeps.

Includes multiple config profiles:
- Default (OrchestratorConfig): Conservative, original production values.
- Aggressive: Targets 25-50% annual returns with higher position sizing
  and optional intraday leverage.
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

    # ── Kelly-criterion sizing (optional) ──────────────────────────────
    use_kelly: bool = False
    """When True, use dynamic Kelly-criterion sizing based on trade history."""

    kelly_fraction: float = 0.25
    """Fraction of full Kelly to use (0.25 = quarter-Kelly, safest)."""

    kelly_min_trades: int = 30
    """Minimum completed trades before Kelly sizing activates."""


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

    # ── Leverage ────────────────────────────────────────────────────────
    max_gross_exposure: float = 1.0
    """Maximum gross portfolio exposure as fraction of equity (1.0 = 100%)."""

    intraday_leverage_mult: float = 1.0
    """Intraday leverage multiplier applied to high-conviction trades.
    IBKR paper accounts allow up to 4.0 for PDT accounts.
    Set to 1.0 (no leverage) by default for safety."""

    leverage_min_regime_confidence: float = 0.65
    """Minimum regime confidence required before leverage is applied."""


def _aggressive_sizing() -> SizingConfig:
    """Factory for aggressive sizing config."""
    return SizingConfig(
        base_size=0.05,
        consensus_weight=1.5,
        agreement_weight=0.75,
        regime_weight=1.0,
        vol_target_atr_pct=0.01,
        vol_scalar_min=0.5,
        vol_scalar_max=2.0,
        use_kelly=True,
        kelly_fraction=0.25,
        kelly_min_trades=30,
    )


def _aggressive_exit() -> ExitConfig:
    """Factory for aggressive exit config."""
    return ExitConfig(
        trailing_activation_atr=1.5,
        trailing_breakeven_offset_atr=0.5,
        trailing_distance_atr=1.0,
        eod_exit_time_hour=15,
        eod_exit_time_minute=55,
    )


def create_aggressive_config(
    leverage: float = 1.5,
) -> OrchestratorConfig:
    """
    Create an aggressive Orchestrator config targeting 25-50% annual returns.

    Key differences from default:
    - 5% base position size (vs 0.8%)
    - 15% max position (vs 3%)
    - Quarter-Kelly dynamic sizing enabled
    - Tighter stops (2.0 ATR vs 2.5) for better R:R
    - Faster re-entry (8 bars vs 12)
    - Optional intraday leverage (default 1.5x)
    - Slightly lower consensus threshold (0.45 vs 0.5) to catch more trades

    Args:
        leverage: Intraday leverage multiplier (1.0-4.0, default 1.5).

    Returns:
        OrchestratorConfig tuned for aggressive returns.
    """
    return OrchestratorConfig(
        # Consensus -- still require 4+ edges, but slightly lower score bar
        min_consensus_edges=4,
        min_consensus_score=0.45,
        max_opposition_score=0.35,
        min_directional_quality=0.55,
        # Position limits
        max_position_pct=0.15,
        # Cooldown
        min_reentry_bars=8,
        # Regime
        min_regime_confidence=0.50,
        # ATR band (slightly wider to capture more setups)
        min_atr_pct=0.0012,
        max_atr_pct=0.04,
        # Tighter stops, still favorable R:R
        atr_stop_mult=2.0,
        atr_target_mult=3.5,
        # Mean reversion
        mean_reversion_zscore=1.5,
        # Warmup
        warmup_bars=30,
        # Sub-configs
        sizing=_aggressive_sizing(),
        exit=_aggressive_exit(),
        # Quant edge
        enable_quant_edge=True,
        quant_edge_min_bars=60,
        # Leverage
        max_gross_exposure=min(leverage, 4.0),
        intraday_leverage_mult=min(leverage, 4.0),
        leverage_min_regime_confidence=0.65,
    )
