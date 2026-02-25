"""Tests for the multi-strategy controller and adapters."""

from datetime import datetime, time
from typing import Dict, List, Optional

import numpy as np
import pytest

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)


# ────────────────────────────────────────────────────────────────────────
# Stub strategy for isolated controller tests
# ────────────────────────────────────────────────────────────────────────

class StubStrategy(TradingStrategy):
    """Configurable stub for testing the controller in isolation."""

    def __init__(
        self,
        name: str = "Stub",
        signals: Optional[List[StrategySignal]] = None,
        warmup_bars: int = 0,
    ):
        self._name = name
        self._signals = signals or []
        self._state = StrategyState.ACTIVE if warmup_bars == 0 else StrategyState.WARMING_UP
        self._warmup_bars = warmup_bars
        self._bars = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> StrategyState:
        return self._state

    def update(self, symbol, timestamp, open_price, high, low, close, volume):
        self._bars += 1
        if self._state == StrategyState.WARMING_UP and self._bars >= self._warmup_bars:
            self._state = StrategyState.ACTIVE

    def generate_signals(self, symbols, timestamp):
        return list(self._signals)

    def set_signals(self, signals: List[StrategySignal]):
        self._signals = signals


# ────────────────────────────────────────────────────────────────────────
# Protocol / StrategySignal tests
# ────────────────────────────────────────────────────────────────────────

class TestStrategySignal:
    def test_entry_signal(self):
        sig = StrategySignal(
            strategy_name="test", symbol="AAPL", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        assert sig.is_entry is True
        assert sig.is_exit is False

    def test_exit_signal(self):
        sig = StrategySignal(
            strategy_name="test", symbol="AAPL", direction=0,
            target_weight=0.0, confidence=0.5,
        )
        assert sig.is_entry is False
        assert sig.is_exit is True

    def test_short_is_entry(self):
        sig = StrategySignal(
            strategy_name="test", symbol="AAPL", direction=-1,
            target_weight=0.05, confidence=0.7,
        )
        assert sig.is_entry is True
        assert sig.is_exit is False


# ────────────────────────────────────────────────────────────────────────
# Controller registration tests
# ────────────────────────────────────────────────────────────────────────

class TestControllerRegistration:
    def test_register_strategy(self):
        ctrl = MultiStrategyController()
        stub = StubStrategy("TestStrat")
        ctrl.register(stub)
        assert "TestStrat" in ctrl.strategies

    def test_register_replaces_duplicate(self):
        ctrl = MultiStrategyController()
        s1 = StubStrategy("Same")
        s2 = StubStrategy("Same")
        ctrl.register(s1)
        ctrl.register(s2)
        assert len(ctrl.strategies) == 1
        assert ctrl.strategies["Same"] is s2

    def test_unregister(self):
        ctrl = MultiStrategyController()
        ctrl.register(StubStrategy("A"))
        ctrl.register(StubStrategy("B"))
        ctrl.unregister("A")
        assert "A" not in ctrl.strategies
        assert "B" in ctrl.strategies

    def test_active_strategies_respects_state(self):
        ctrl = MultiStrategyController()
        active = StubStrategy("Active", warmup_bars=0)
        warming = StubStrategy("Warming", warmup_bars=999)
        ctrl.register(active)
        ctrl.register(warming)
        assert len(ctrl.active_strategies) == 1
        assert ctrl.active_strategies[0].name == "Active"

    def test_active_strategies_respects_enabled(self):
        cfg = ControllerConfig(allocations={
            "Enabled": StrategyAllocation(weight=0.5, enabled=True),
            "Disabled": StrategyAllocation(weight=0.5, enabled=False),
        })
        ctrl = MultiStrategyController(cfg)
        ctrl.register(StubStrategy("Enabled"))
        ctrl.register(StubStrategy("Disabled"))
        assert len(ctrl.active_strategies) == 1
        assert ctrl.active_strategies[0].name == "Enabled"


# ────────────────────────────────────────────────────────────────────────
# Controller data feed tests
# ────────────────────────────────────────────────────────────────────────

class TestControllerUpdate:
    def test_update_feeds_all_strategies(self):
        ctrl = MultiStrategyController()
        s1 = StubStrategy("A", warmup_bars=2)
        s2 = StubStrategy("B", warmup_bars=3)
        ctrl.register(s1)
        ctrl.register(s2)

        ts = datetime(2025, 6, 15, 10, 0)
        ctrl.update("AAPL", ts, 150.0, 151.0, 149.0, 150.5, 1000)
        ctrl.update("AAPL", ts, 150.0, 151.0, 149.0, 150.5, 1000)

        assert s1.state == StrategyState.ACTIVE
        assert s2.state == StrategyState.WARMING_UP

        ctrl.update("AAPL", ts, 150.0, 151.0, 149.0, 150.5, 1000)
        assert s2.state == StrategyState.ACTIVE


# ────────────────────────────────────────────────────────────────────────
# Signal generation pipeline tests
# ────────────────────────────────────────────────────────────────────────

class TestSignalGeneration:
    def _make_controller(self, *strategies: StubStrategy, config=None):
        cfg = config or ControllerConfig(allocations={
            s.name: StrategyAllocation(weight=0.25, max_positions=10)
            for s in strategies
        })
        ctrl = MultiStrategyController(cfg)
        for s in strategies:
            ctrl.register(s)
        return ctrl

    def test_no_signals_when_empty(self):
        ctrl = self._make_controller()
        sigs = ctrl.generate_signals(["AAPL"], datetime.now())
        assert sigs == []

    def test_single_strategy_single_signal(self):
        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8, trade_type="test",
        )
        s = StubStrategy("A", signals=[sig])
        ctrl = self._make_controller(s)
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        assert result[0].symbol == "AAPL"
        assert result[0].direction == 1
        # Weight should be scaled by allocation (0.10 * 0.25 = 0.025)
        assert result[0].target_weight == pytest.approx(0.025)

    def test_allocation_scaling(self):
        sig = StrategySignal(
            strategy_name="Big", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        cfg = ControllerConfig(allocations={
            "Big": StrategyAllocation(weight=0.60),
        })
        s = StubStrategy("Big", signals=[sig])
        ctrl = self._make_controller(s, config=cfg)
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        assert result[0].target_weight == pytest.approx(0.06)

    def test_exit_signals_pass_through(self):
        exit_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=0,
            target_weight=0.0, confidence=0.5,
        )
        s = StubStrategy("A", signals=[exit_sig])
        ctrl = self._make_controller(s)
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        assert result[0].is_exit is True


# ────────────────────────────────────────────────────────────────────────
# Conflict resolution tests
# ────────────────────────────────────────────────────────────────────────

class TestConflictResolution:
    def _make_controller_with_signals(
        self, signals_a, signals_b, method="weighted_confidence"
    ):
        cfg = ControllerConfig(
            allocations={
                "A": StrategyAllocation(weight=0.50),
                "B": StrategyAllocation(weight=0.50),
            },
            conflict_resolution=method,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.register(StubStrategy("A", signals=signals_a))
        ctrl.register(StubStrategy("B", signals=signals_b))
        return ctrl

    def test_no_conflict_same_direction(self):
        long_a = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        long_b = StrategySignal(
            strategy_name="B", symbol="AAPL", direction=1,
            target_weight=0.08, confidence=0.6,
        )
        ctrl = self._make_controller_with_signals([long_a], [long_b])
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        # Should merge into one signal
        assert len(result) == 1
        assert result[0].direction == 1
        # Merged weight = 0.10*0.50 + 0.08*0.50 = 0.09
        assert result[0].target_weight == pytest.approx(0.09)

    def test_conflict_weighted_confidence_long_wins(self):
        long_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.9,
        )
        short_sig = StrategySignal(
            strategy_name="B", symbol="AAPL", direction=-1,
            target_weight=0.05, confidence=0.3,
        )
        ctrl = self._make_controller_with_signals([long_sig], [short_sig])
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        # Long should win: 0.10*0.50*0.9=0.045 > 0.05*0.50*0.3=0.0075
        assert result[0].direction == 1

    def test_conflict_weighted_confidence_short_wins(self):
        long_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.02, confidence=0.3,
        )
        short_sig = StrategySignal(
            strategy_name="B", symbol="AAPL", direction=-1,
            target_weight=0.10, confidence=0.9,
        )
        ctrl = self._make_controller_with_signals([long_sig], [short_sig])
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        assert result[0].direction == -1

    def test_conflict_veto_mode(self):
        long_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.9,
        )
        short_sig = StrategySignal(
            strategy_name="B", symbol="AAPL", direction=-1,
            target_weight=0.02, confidence=0.3,
        )
        ctrl = self._make_controller_with_signals(
            [long_sig], [short_sig], method="veto"
        )
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        # In veto mode, shorts block longs
        assert len(result) == 1
        assert result[0].direction == -1

    def test_conflict_net_mode(self):
        long_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        short_sig = StrategySignal(
            strategy_name="B", symbol="AAPL", direction=-1,
            target_weight=0.05, confidence=0.4,
        )
        ctrl = self._make_controller_with_signals(
            [long_sig], [short_sig], method="net"
        )
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        # Long score: 0.10*0.50*0.8=0.04 > Short score: 0.05*0.50*0.4=0.01
        assert result[0].direction == 1

    def test_different_symbols_no_conflict(self):
        long_aapl = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        short_msft = StrategySignal(
            strategy_name="B", symbol="MSFT", direction=-1,
            target_weight=0.08, confidence=0.7,
        )
        ctrl = self._make_controller_with_signals([long_aapl], [short_msft])
        result = ctrl.generate_signals(["AAPL", "MSFT"], datetime.now())
        assert len(result) == 2
        symbols = {s.symbol for s in result}
        assert symbols == {"AAPL", "MSFT"}


# ────────────────────────────────────────────────────────────────────────
# Portfolio limits tests
# ────────────────────────────────────────────────────────────────────────

class TestPortfolioLimits:
    def test_single_symbol_cap(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            max_single_symbol_weight=0.10,
        )
        ctrl = MultiStrategyController(cfg)
        # Simulate existing position
        ctrl._current_positions["AAPL"] = 0.08

        big_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[big_sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        if result:
            # Weight should be capped to headroom: 0.10 - 0.08 = 0.02
            assert result[0].target_weight <= 0.02 + 1e-9

    def test_max_positions_limit(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            max_portfolio_positions=2,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._current_positions = {"AAPL": 0.05, "MSFT": 0.05}

        new_sig = StrategySignal(
            strategy_name="A", symbol="GOOG", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[new_sig]))
        result = ctrl.generate_signals(["GOOG"], datetime.now())
        # Should be empty — already at max positions
        entry_sigs = [s for s in result if s.is_entry]
        assert len(entry_sigs) == 0

    def test_gross_exposure_scaling(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            max_gross_exposure=0.50,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._current_positions = {"AAPL": 0.30}

        sig = StrategySignal(
            strategy_name="A", symbol="MSFT", direction=1,
            target_weight=0.40, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["MSFT"], datetime.now())

        if result:
            entry_sigs = [s for s in result if s.is_entry]
            for s in entry_sigs:
                # Should be scaled down: headroom = 0.50 - 0.30 = 0.20
                assert s.target_weight <= 0.20 + 1e-9


# ────────────────────────────────────────────────────────────────────────
# Risk checks tests
# ────────────────────────────────────────────────────────────────────────

class TestRiskChecks:
    def test_max_drawdown_blocks_entries(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            max_drawdown=0.15,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._peak_equity = 100_000
        ctrl._equity = 84_000  # 16% drawdown > 15% limit

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        entries = [s for s in result if s.is_entry]
        assert len(entries) == 0

    def test_drawdown_scales_down_near_limit(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            max_drawdown=0.20,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._peak_equity = 100_000
        ctrl._equity = 84_000  # 16% drawdown, 75% of 20% limit = 15%

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        entries = [s for s in result if s.is_entry]
        if entries:
            # Weight should be scaled down
            assert entries[0].target_weight < 0.10

    def test_daily_loss_limit_halts(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            daily_loss_limit=0.03,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._daily_pnl = -0.04  # 4% loss > 3% limit

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        entries = [s for s in result if s.is_entry]
        assert len(entries) == 0
        assert ctrl._halted is True

    def test_halted_blocks_subsequent_calls(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._halted = True

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert result == []

    def test_new_trading_day_resets_halt(self):
        ctrl = MultiStrategyController()
        ctrl._halted = True
        ctrl._daily_pnl = -0.05
        ctrl.new_trading_day()
        assert ctrl._halted is False
        assert ctrl._daily_pnl == 0.0

    def test_risk_disabled_passes_through(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            enable_risk_controller=False,
            max_drawdown=0.15,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._peak_equity = 100_000
        ctrl._equity = 80_000  # 20% drawdown, but risk is disabled

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        entries = [s for s in result if s.is_entry]
        assert len(entries) == 1


# ────────────────────────────────────────────────────────────────────────
# State management tests
# ────────────────────────────────────────────────────────────────────────

class TestStateManagement:
    def test_update_portfolio_state(self):
        ctrl = MultiStrategyController()
        ctrl.update_portfolio_state(
            equity=120_000,
            positions={"AAPL": 0.05, "MSFT": -0.03},
            daily_pnl=0.02,
        )
        assert ctrl._equity == 120_000
        assert ctrl._peak_equity == 120_000
        assert ctrl._daily_pnl == 0.02
        assert len(ctrl._current_positions) == 2

    def test_peak_equity_tracking(self):
        ctrl = MultiStrategyController()
        ctrl.update_portfolio_state(110_000, {})
        ctrl.update_portfolio_state(115_000, {})
        ctrl.update_portfolio_state(108_000, {})
        assert ctrl._peak_equity == 115_000

    def test_reset(self):
        ctrl = MultiStrategyController()
        ctrl.register(StubStrategy("A"))
        ctrl._current_positions = {"AAPL": 0.1}
        ctrl._halted = True
        ctrl.reset()
        assert ctrl._current_positions == {}
        assert ctrl._halted is False

    def test_get_status(self):
        ctrl = MultiStrategyController()
        ctrl.register(StubStrategy("Alpha"))
        ctrl._equity = 105_000
        ctrl._peak_equity = 110_000
        status = ctrl.get_status()
        assert status["equity"] == 105_000
        assert "Alpha" in status["strategies"]
        assert status["strategies"]["Alpha"]["state"] == "ACTIVE"


# ────────────────────────────────────────────────────────────────────────
# Adapter import tests (verify adapters load without error)
# ────────────────────────────────────────────────────────────────────────

class TestAdapterImports:
    def test_import_orchestrator_adapter(self):
        from trading_algo.multi_strategy.adapters.orchestrator_adapter import (
            OrchestratorStrategyAdapter,
        )
        assert OrchestratorStrategyAdapter is not None

    def test_import_orb_adapter(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import (
            ORBStrategyAdapter,
        )
        assert ORBStrategyAdapter is not None

    def test_import_pairs_adapter(self):
        from trading_algo.multi_strategy.adapters.pairs_adapter import (
            PairsStrategyAdapter,
        )
        assert PairsStrategyAdapter is not None

    def test_import_momentum_adapter(self):
        from trading_algo.multi_strategy.adapters.momentum_adapter import (
            MomentumStrategyAdapter,
        )
        assert MomentumStrategyAdapter is not None

    def test_import_controller(self):
        from trading_algo.multi_strategy.controller import (
            MultiStrategyController,
            ControllerConfig,
        )
        assert MultiStrategyController is not None
        assert ControllerConfig is not None

    def test_package_init_imports(self):
        from trading_algo.multi_strategy import (
            StrategySignal,
            StrategyState,
            TradingStrategy,
            MultiStrategyController,
            ControllerConfig,
        )
        assert StrategySignal is not None


# ────────────────────────────────────────────────────────────────────────
# Momentum adapter unit tests
# ────────────────────────────────────────────────────────────────────────

class TestMomentumAdapter:
    def test_warmup_state(self):
        from trading_algo.multi_strategy.adapters.momentum_adapter import (
            MomentumStrategyAdapter,
        )
        from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig

        config = MomentumConfig(trend_ma=10, fast_ma=3, slow_ma=5, momentum_lookback=5)
        adapter = MomentumStrategyAdapter(config)

        assert adapter.name == "PureMomentum"
        assert adapter.state == StrategyState.WARMING_UP

        # Feed 9 bars — still warming up
        ts = datetime(2025, 6, 15, 10, 0)
        for i in range(9):
            adapter.update("AAPL", ts, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000)
        assert adapter.state == StrategyState.WARMING_UP

        # 10th bar — should activate
        adapter.update("AAPL", ts, 110, 111, 109, 110, 1000)
        assert adapter.state == StrategyState.ACTIVE

    def test_no_signals_during_warmup(self):
        from trading_algo.multi_strategy.adapters.momentum_adapter import (
            MomentumStrategyAdapter,
        )
        from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig

        config = MomentumConfig(trend_ma=50)
        adapter = MomentumStrategyAdapter(config)
        ts = datetime(2025, 6, 15, 10, 0)

        # Only 5 bars
        for i in range(5):
            adapter.update("AAPL", ts, 100 + i, 101 + i, 99 + i, 100 + i, 1000)

        signals = adapter.generate_signals(["AAPL"], ts)
        assert signals == []

    def test_reset_clears_state(self):
        from trading_algo.multi_strategy.adapters.momentum_adapter import (
            MomentumStrategyAdapter,
        )
        from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig

        config = MomentumConfig(trend_ma=5, fast_ma=2, slow_ma=3, momentum_lookback=3)
        adapter = MomentumStrategyAdapter(config)
        ts = datetime(2025, 6, 15, 10, 0)

        for i in range(10):
            adapter.update("AAPL", ts, 100 + i, 101 + i, 99 + i, 100 + i, 1000)
        assert adapter.state == StrategyState.ACTIVE

        adapter.reset()
        assert adapter.state == StrategyState.WARMING_UP


# ────────────────────────────────────────────────────────────────────────
# Pairs adapter unit tests
# ────────────────────────────────────────────────────────────────────────

class TestPairsAdapter:
    def test_warmup_requires_two_symbols(self):
        from trading_algo.multi_strategy.adapters.pairs_adapter import (
            PairsStrategyAdapter,
        )
        adapter = PairsStrategyAdapter()
        assert adapter.name == "PairsTrading"
        assert adapter.state == StrategyState.WARMING_UP

    def test_no_signals_during_warmup(self):
        from trading_algo.multi_strategy.adapters.pairs_adapter import (
            PairsStrategyAdapter,
        )
        adapter = PairsStrategyAdapter()
        ts = datetime(2025, 6, 15, 10, 0)

        # Only 5 bars for one symbol
        for i in range(5):
            adapter.update("AAPL", ts, 100 + i, 101 + i, 99 + i, 100 + i, 1000)

        signals = adapter.generate_signals(["AAPL"], ts)
        assert signals == []


# ────────────────────────────────────────────────────────────────────────
# ORB adapter unit tests
# ────────────────────────────────────────────────────────────────────────

class TestORBAdapter:
    def test_name_and_initial_state(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import (
            ORBStrategyAdapter,
        )
        adapter = ORBStrategyAdapter()
        assert adapter.name == "ORB"
        assert adapter.state == StrategyState.WARMING_UP

    def test_activates_after_3_bars(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import (
            ORBStrategyAdapter,
        )
        adapter = ORBStrategyAdapter()
        ts = datetime(2025, 6, 15, 9, 30)

        for i in range(3):
            adapter.update("AAPL", ts, 100, 101, 99, 100, 1000)
        assert adapter.state == StrategyState.ACTIVE

    def test_no_signals_outside_window(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import (
            ORBStrategyAdapter,
        )
        adapter = ORBStrategyAdapter()

        # Feed bars at 9:30 (before window)
        ts_early = datetime(2025, 6, 15, 9, 30)
        for i in range(5):
            adapter.update("AAPL", ts_early, 100, 101, 99, 100, 1000)

        # Generate signals at 9:45 — before ORB window opens (10:00)
        signals = adapter.generate_signals(["AAPL"], datetime(2025, 6, 15, 9, 45))
        assert signals == []


# ────────────────────────────────────────────────────────────────────────
# Integration: multiple adapters through controller
# ────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_multi_strategy_signals_combined(self):
        """Two strategies emit signals on different symbols — both pass through."""
        sig_a = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        sig_b = StrategySignal(
            strategy_name="B", symbol="MSFT", direction=-1,
            target_weight=0.08, confidence=0.7,
        )
        cfg = ControllerConfig(allocations={
            "A": StrategyAllocation(weight=0.50),
            "B": StrategyAllocation(weight=0.50),
        })
        ctrl = MultiStrategyController(cfg)
        ctrl.register(StubStrategy("A", signals=[sig_a]))
        ctrl.register(StubStrategy("B", signals=[sig_b]))

        result = ctrl.generate_signals(["AAPL", "MSFT"], datetime.now())
        assert len(result) == 2
        syms = {s.symbol for s in result}
        assert syms == {"AAPL", "MSFT"}

    def test_multi_strategy_agreement_boosts_weight(self):
        """When two strategies agree on same symbol, weights are summed."""
        sig_a = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        sig_b = StrategySignal(
            strategy_name="B", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.7,
        )
        cfg = ControllerConfig(allocations={
            "A": StrategyAllocation(weight=0.50),
            "B": StrategyAllocation(weight=0.50),
        })
        ctrl = MultiStrategyController(cfg)
        ctrl.register(StubStrategy("A", signals=[sig_a]))
        ctrl.register(StubStrategy("B", signals=[sig_b]))

        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        # Sum: 0.10*0.50 + 0.10*0.50 = 0.10
        assert result[0].target_weight == pytest.approx(0.10)
        assert result[0].direction == 1

    def test_full_pipeline_with_risk(self):
        """End-to-end: signal → allocation → conflict → limits → risk."""
        cfg = ControllerConfig(
            allocations={
                "Long": StrategyAllocation(weight=0.60),
                "Short": StrategyAllocation(weight=0.40),
            },
            max_gross_exposure=1.0,
            max_drawdown=0.20,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl._equity = 100_000
        ctrl._peak_equity = 100_000

        long_sig = StrategySignal(
            strategy_name="Long", symbol="AAPL", direction=1,
            target_weight=0.15, confidence=0.9,
        )
        short_sig = StrategySignal(
            strategy_name="Short", symbol="MSFT", direction=-1,
            target_weight=0.10, confidence=0.7,
        )
        ctrl.register(StubStrategy("Long", signals=[long_sig]))
        ctrl.register(StubStrategy("Short", signals=[short_sig]))

        result = ctrl.generate_signals(["AAPL", "MSFT"], datetime.now())
        assert len(result) == 2

        # Verify both directions present
        directions = {s.symbol: s.direction for s in result}
        assert directions["AAPL"] == 1
        assert directions["MSFT"] == -1


# ────────────────────────────────────────────────────────────────────────
# Entropy filter integration tests
# ────────────────────────────────────────────────────────────────────────

class TestEntropyFilter:
    def test_entropy_filter_disabled_by_default(self):
        ctrl = MultiStrategyController()
        assert ctrl._entropy_filter is None

    def test_entropy_filter_enabled(self):
        cfg = ControllerConfig(enable_entropy_filter=True)
        ctrl = MultiStrategyController(cfg)
        assert ctrl._entropy_filter is not None
        assert ctrl._entropy_ref_symbol == "SPY"

    def test_entropy_passthrough_when_disabled(self):
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            enable_entropy_filter=False,
        )
        ctrl = MultiStrategyController(cfg)
        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        entries = [s for s in result if s.is_entry]
        assert len(entries) == 1
        # Weight only scaled by allocation (1.0), not entropy
        assert entries[0].target_weight == pytest.approx(0.10)

    def test_entropy_filter_scales_entries(self):
        """When entropy filter is enabled and warmed up, entries get scaled."""
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            enable_entropy_filter=True,
            enable_vol_management=False,
        )
        ctrl = MultiStrategyController(cfg)

        # Manually set the filter to high-entropy regime (scale=0.10)
        ctrl._entropy_filter._current_scale = 0.10
        ctrl._entropy_filter._current_regime = "HIGH"
        ctrl._entropy_filter._n_updates = 100  # Past warmup

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        entries = [s for s in result if s.is_entry]
        assert len(entries) == 1
        # Weight should be scaled: 0.10 * 1.0 (alloc) * 0.10 (entropy) = 0.01
        assert entries[0].target_weight == pytest.approx(0.01)
        assert entries[0].metadata.get("entropy_regime") == "HIGH"

    def test_entropy_filter_preserves_exits(self):
        """Exit signals should not be scaled by entropy filter."""
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            enable_entropy_filter=True,
            enable_vol_management=False,
        )
        ctrl = MultiStrategyController(cfg)

        # Set high-entropy regime
        ctrl._entropy_filter._current_scale = 0.10
        ctrl._entropy_filter._n_updates = 100

        exit_sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=0,
            target_weight=0.0, confidence=0.5,
        )
        ctrl.register(StubStrategy("A", signals=[exit_sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        assert len(result) == 1
        assert result[0].is_exit
        assert result[0].target_weight == 0.0

    def test_entropy_feed_tracks_daily_returns(self):
        """Controller should feed daily returns to entropy filter from SPY bars."""
        cfg = ControllerConfig(enable_entropy_filter=True)
        ctrl = MultiStrategyController(cfg)

        # Day 1 bars
        ctrl.update("SPY", datetime(2024, 1, 2, 9, 30), 470.0, 471.0, 469.0, 470.5, 1e6)
        ctrl.update("SPY", datetime(2024, 1, 2, 15, 55), 470.5, 472.0, 469.0, 471.0, 1e6)

        # Day 2 first bar triggers daily return computation for Day 1
        ctrl.update("SPY", datetime(2024, 1, 3, 9, 30), 471.0, 472.0, 470.0, 471.5, 1e6)

        assert ctrl._entropy_filter._n_updates == 1
        assert ctrl._entropy_last_day == "2024-01-03"

    def test_new_strategy_allocations(self):
        """Verify new strategies have allocations in default config."""
        cfg = ControllerConfig()
        assert "LeadLagArbitrage" in cfg.allocations
        assert "HurstAdaptive" in cfg.allocations
        assert "TimeAdaptive" in cfg.allocations
        assert cfg.allocations["LeadLagArbitrage"].weight == 0.08
        assert cfg.allocations["HurstAdaptive"].weight == 0.08
        assert cfg.allocations["TimeAdaptive"].weight == 0.06

    def test_new_adapter_imports(self):
        """Verify new adapters are importable from the package."""
        from trading_algo.multi_strategy.adapters import (
            HurstAdaptiveAdapter,
            LeadLagAdapter,
            TimeAdaptiveAdapter,
        )
        assert HurstAdaptiveAdapter is not None
        assert LeadLagAdapter is not None
        assert TimeAdaptiveAdapter is not None
