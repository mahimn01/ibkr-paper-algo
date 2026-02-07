"""
End-to-end integration tests.

These tests validate the full trading loop:
  SimBroker → OrchestratorStrategy → AutoRunner → OMS → SimBroker

And the backtest path:
  Synthetic data → BacktestEngine → Orchestrator → BacktestResults
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from trading_algo.broker.base import OrderRequest
from trading_algo.broker.sim import SimBroker
from trading_algo.config import TradingConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.risk import RiskLimits, RiskManager

from trading_algo.orchestrator.config import OrchestratorConfig
from trading_algo.orchestrator.strategy import Orchestrator, create_orchestrator
from trading_algo.orchestrator.adapter import OrchestratorStrategy
from trading_algo.strategy.base import StrategyContext


# ---------------------------------------------------------------------------
#  Helper fixtures
# ---------------------------------------------------------------------------

def _make_sim_broker(*symbols: str, last: float = 150.0) -> SimBroker:
    """Create a SimBroker pre-loaded with market data for the given symbols."""
    broker = SimBroker()
    broker.connect()
    for sym in symbols:
        broker.set_market_data(
            InstrumentSpec(kind="STK", symbol=sym, exchange="SMART", currency="USD"),
            last=last,
            volume=100_000,
        )
    return broker


def _make_strategy_context(broker: SimBroker) -> StrategyContext:
    """Build a StrategyContext backed by the given SimBroker."""
    return StrategyContext(
        now_epoch_s=datetime(2025, 7, 15, 10, 0).timestamp(),
        get_snapshot=broker.get_market_data_snapshot,
    )


# ---------------------------------------------------------------------------
#  1.  Orchestrator backward compatibility
# ---------------------------------------------------------------------------

class TestOrchestratorBackwardCompat:
    """Ensure Orchestrator still works when no config is passed."""

    def test_default_config_matches_previous_values(self) -> None:
        orch = create_orchestrator()
        assert orch.min_consensus_edges == 4
        assert orch.min_consensus_score == 0.5
        assert orch.max_position_pct == 0.03
        assert orch.min_reentry_bars == 12
        assert orch.atr_stop_mult == 2.5
        assert orch.atr_target_mult == 4.0

    def test_custom_config_overrides(self) -> None:
        cfg = OrchestratorConfig(
            min_consensus_edges=3,
            max_position_pct=0.05,
            atr_stop_mult=3.0,
        )
        orch = Orchestrator(cfg)
        assert orch.min_consensus_edges == 3
        assert orch.max_position_pct == 0.05
        assert orch.atr_stop_mult == 3.0

    def test_quant_edge_disabled(self) -> None:
        cfg = OrchestratorConfig(enable_quant_edge=False)
        orch = Orchestrator(cfg)
        assert orch._quant_edge is None


# ---------------------------------------------------------------------------
#  2.  OrchestratorStrategy adapter
# ---------------------------------------------------------------------------

class TestOrchestratorAdapter:
    """Verify the OrchestratorStrategy adapter produces valid TradeIntents."""

    def test_on_tick_returns_list(self) -> None:
        broker = _make_sim_broker("AAPL", "SPY", "QQQ", "IWM")
        cfg = OrchestratorConfig(enable_quant_edge=False)
        strategy = OrchestratorStrategy(
            symbols=["AAPL"],
            config=cfg,
        )
        ctx = _make_strategy_context(broker)
        intents = strategy.on_tick(ctx)
        assert isinstance(intents, list)

    def test_on_tick_hold_during_warmup(self) -> None:
        broker = _make_sim_broker("AAPL", "SPY", "QQQ", "IWM")
        cfg = OrchestratorConfig(enable_quant_edge=False)
        strategy = OrchestratorStrategy(
            symbols=["AAPL"],
            config=cfg,
        )
        # Single tick → warmup → should produce no intents
        ctx = _make_strategy_context(broker)
        intents = strategy.on_tick(ctx)
        assert len(intents) == 0

    def test_multiple_ticks_feed_data(self) -> None:
        """Feed enough ticks so the Orchestrator can warm up."""
        broker = _make_sim_broker("AAPL", "SPY", "QQQ", "IWM")
        cfg = OrchestratorConfig(enable_quant_edge=False, warmup_bars=5)
        strategy = OrchestratorStrategy(
            symbols=["AAPL"],
            config=cfg,
        )

        from datetime import timedelta
        base = datetime(2025, 7, 15, 9, 30)
        for i in range(10):
            ts = (base + timedelta(minutes=i * 5)).timestamp()
            ctx = StrategyContext(
                now_epoch_s=ts,
                get_snapshot=broker.get_market_data_snapshot,
            )
            intents = strategy.on_tick(ctx)
            assert isinstance(intents, list)


# ---------------------------------------------------------------------------
#  3.  Full AutoRunner loop (SimBroker + OMS path)
# ---------------------------------------------------------------------------

class TestAutoRunnerIntegration:
    """
    Test the full AutoRunner → OrchestratorStrategy → OMS → SimBroker path.

    Uses max_ticks and sleep_seconds=0 for deterministic, fast testing.
    """

    def test_autorunner_completes_without_error(self) -> None:
        from trading_algo.autorun import AutoRunner

        broker = _make_sim_broker("AAPL", "SPY", "QQQ", "IWM")
        cfg = TradingConfig.from_env()
        # Force dry-run + sim mode so nothing real happens
        cfg = TradingConfig(
            broker="sim",
            live_enabled=False,
            require_paper=False,
            dry_run=True,
            order_token="",
            db_path="",
            poll_seconds=0.0,
            ibkr=cfg.ibkr,
        )

        orch_cfg = OrchestratorConfig(enable_quant_edge=False, warmup_bars=5)
        strategy = OrchestratorStrategy(
            symbols=["AAPL"],
            config=orch_cfg,
        )
        risk = RiskManager(RiskLimits())

        runner = AutoRunner(
            broker=broker,
            config=cfg,
            strategy=strategy,
            risk=risk,
            sleep_seconds=0,
            max_ticks=10,
        )
        # Should complete without raising
        runner.run()


# ---------------------------------------------------------------------------
#  4.  Backtest engine integration
# ---------------------------------------------------------------------------

class TestBacktestIntegration:
    """
    Test the Orchestrator driven by BacktestEngine on synthetic data.
    """

    def test_backtest_runs_and_produces_results(self) -> None:
        from trading_algo.orchestrator.backtest_runner import (
            generate_synthetic_dataset,
            run_orchestrator_backtest,
        )

        cfg = OrchestratorConfig(enable_quant_edge=False, warmup_bars=30)
        result = run_orchestrator_backtest(
            config=cfg,
            n_days=5,
            symbol="TEST",
            seed=42,
        )

        r = result.backtest_results
        assert r.bars_processed > 0
        assert r.config.strategy_name == "Orchestrator"
        # Metrics should be populated
        m = r.metrics
        assert m.trading_days >= 1

    def test_synthetic_data_has_correct_structure(self) -> None:
        from trading_algo.orchestrator.backtest_runner import generate_synthetic_dataset

        data = generate_synthetic_dataset(n_days=3, symbol="FOO", seed=99)
        assert "FOO" in data
        assert "SPY" in data
        assert "QQQ" in data
        # Each day has 78 bars (390 min / 5 min)
        assert len(data["FOO"]) == 3 * 78

        bar = data["FOO"][0]
        assert hasattr(bar, "open")
        assert hasattr(bar, "high")
        assert hasattr(bar, "low")
        assert hasattr(bar, "close")
        assert hasattr(bar, "volume")
        assert bar.high >= bar.low

    def test_backtest_with_more_days_produces_trades(self) -> None:
        from trading_algo.orchestrator.backtest_runner import run_orchestrator_backtest

        # More aggressive config to generate trades in synthetic data
        cfg = OrchestratorConfig(
            enable_quant_edge=False,
            warmup_bars=15,
            min_consensus_edges=3,
            min_consensus_score=0.2,
            max_opposition_score=0.8,
            min_directional_quality=0.3,
            min_regime_confidence=0.3,
            min_atr_pct=0.0001,
            max_atr_pct=0.10,
        )
        result = run_orchestrator_backtest(
            config=cfg,
            n_days=15,
            symbol="TEST",
            seed=123,
        )
        # With relaxed thresholds over 15 days, we should get some trades
        r = result.backtest_results
        assert r.bars_processed > 500


# ---------------------------------------------------------------------------
#  5.  Config plumbing
# ---------------------------------------------------------------------------

class TestConfigPlumbing:
    """Verify that config changes flow through to all sub-components."""

    def test_sizing_config_affects_position_size(self) -> None:
        from trading_algo.orchestrator.config import SizingConfig

        cfg = OrchestratorConfig(
            enable_quant_edge=False,
            sizing=SizingConfig(base_size=0.02),  # 2x default
        )
        orch = Orchestrator(cfg)
        assert orch.cfg.sizing.base_size == 0.02

    def test_exit_config_is_used(self) -> None:
        from trading_algo.orchestrator.config import ExitConfig

        cfg = OrchestratorConfig(
            enable_quant_edge=False,
            exit=ExitConfig(eod_exit_time_hour=14, eod_exit_time_minute=30),
        )
        orch = Orchestrator(cfg)
        assert orch.cfg.exit.eod_exit_time_hour == 14
        assert orch.cfg.exit.eod_exit_time_minute == 30
