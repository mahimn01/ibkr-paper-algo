"""Tests for Phase 3 alpha sources and enhancements."""

from datetime import datetime, time, timedelta

import numpy as np
import pytest

from trading_algo.multi_strategy.protocol import StrategySignal, StrategyState


# ────────────────────────────────────────────────────────────────────────
# Intraday Momentum Strategy
# ────────────────────────────────────────────────────────────────────────

class TestIntradayMomentumStrategy:
    def test_basic_import(self):
        from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
            IntradayMomentumStrategy,
            IntradayMomentumConfig,
        )
        s = IntradayMomentumStrategy()
        assert s.config.min_opening_return == 0.003

    def test_no_signals_before_entry_time(self):
        from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
            IntradayMomentumStrategy,
        )
        s = IntradayMomentumStrategy()

        # Feed bars in opening window
        ts_open = datetime(2025, 6, 15, 9, 30)
        s.update("AAPL", ts_open, 150.0, 100)
        ts_end = datetime(2025, 6, 15, 10, 0)
        s.update("AAPL", ts_end, 152.0, 200)  # +1.3% opening return

        # Try to generate at noon — should be empty (before entry_time)
        ts_noon = datetime(2025, 6, 15, 12, 0)
        signals = s.generate_signals(["AAPL"], ts_noon)
        assert signals == []

    def test_signal_at_entry_time(self):
        from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
            IntradayMomentumStrategy,
            IntradayMomentumConfig,
        )
        config = IntradayMomentumConfig(min_opening_return=0.002)
        s = IntradayMomentumStrategy(config)

        # Opening: price goes up
        ts_open = datetime(2025, 6, 15, 9, 30)
        s.update("AAPL", ts_open, 150.0, 100)
        ts_close_window = datetime(2025, 6, 15, 10, 0)
        s.update("AAPL", ts_close_window, 152.0, 200)

        # Afternoon price still elevated
        ts_afternoon = datetime(2025, 6, 15, 15, 30)
        s.update("AAPL", ts_afternoon, 153.0, 300)

        signals = s.generate_signals(["AAPL"], ts_afternoon)
        assert len(signals) == 1
        assert signals[0].direction == 1  # Long (follows opening return)
        assert signals[0].opening_return > 0

    def test_negative_opening_return_gives_short(self):
        from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
            IntradayMomentumStrategy,
            IntradayMomentumConfig,
        )
        config = IntradayMomentumConfig(min_opening_return=0.002)
        s = IntradayMomentumStrategy(config)

        # Opening: price goes down
        ts_open = datetime(2025, 6, 15, 9, 30)
        s.update("AAPL", ts_open, 150.0, 100)
        ts_close_window = datetime(2025, 6, 15, 10, 0)
        s.update("AAPL", ts_close_window, 148.0, 200)

        ts_afternoon = datetime(2025, 6, 15, 15, 30)
        s.update("AAPL", ts_afternoon, 147.0, 300)

        signals = s.generate_signals(["AAPL"], ts_afternoon)
        assert len(signals) == 1
        assert signals[0].direction == -1

    def test_small_opening_return_no_signal(self):
        from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
            IntradayMomentumStrategy,
            IntradayMomentumConfig,
        )
        config = IntradayMomentumConfig(min_opening_return=0.005)
        s = IntradayMomentumStrategy(config)

        ts_open = datetime(2025, 6, 15, 9, 30)
        s.update("AAPL", ts_open, 150.0, 100)
        ts_close = datetime(2025, 6, 15, 10, 0)
        s.update("AAPL", ts_close, 150.2, 200)  # Only 0.13%

        ts_entry = datetime(2025, 6, 15, 15, 30)
        s.update("AAPL", ts_entry, 150.3, 300)

        signals = s.generate_signals(["AAPL"], ts_entry)
        assert signals == []

    def test_reset(self):
        from trading_algo.quant_core.strategies.intraday.intraday_momentum import (
            IntradayMomentumStrategy,
        )
        s = IntradayMomentumStrategy()
        ts = datetime(2025, 6, 15, 9, 30)
        s.update("AAPL", ts, 150.0, 100)
        s.reset()
        assert len(s._day_open) == 0


# ────────────────────────────────────────────────────────────────────────
# Short-Term Reversal Strategy
# ────────────────────────────────────────────────────────────────────────

class TestShortTermReversalStrategy:
    def test_basic_import(self):
        from trading_algo.quant_core.strategies.short_term_reversal import (
            ShortTermReversalStrategy,
            ReversalConfig,
        )
        s = ShortTermReversalStrategy()
        assert s.config.lookback_days == 5

    def test_needs_minimum_data(self):
        from trading_algo.quant_core.strategies.short_term_reversal import (
            ShortTermReversalStrategy,
            ReversalConfig,
        )
        config = ReversalConfig(lookback_days=3, vol_lookback=5)
        s = ShortTermReversalStrategy(config)

        # Only 2 bars
        s.update("AAPL", 100.0)
        s.update("AAPL", 101.0)
        signals = s.generate_signals(["AAPL"])
        assert signals == []

    def test_long_losers_short_winners(self):
        from trading_algo.quant_core.strategies.short_term_reversal import (
            ShortTermReversalStrategy,
            ReversalConfig,
        )
        config = ReversalConfig(
            lookback_days=3, vol_lookback=5, vol_scale=False,
            long_quantile=0.34, short_quantile=0.34, min_price=1.0,
        )
        s = ShortTermReversalStrategy(config)

        # Create diverging histories
        # LOSER: drops from 100 to 90
        for i in range(10):
            s.update("LOSER", 100 - i)
        # MIDDLE: flat
        for i in range(10):
            s.update("MIDDLE", 100.0)
        # WINNER: rises from 100 to 110
        for i in range(10):
            s.update("WINNER", 100 + i)

        signals = s.generate_signals(["LOSER", "MIDDLE", "WINNER"])

        # Should long LOSER and short WINNER
        long_syms = [s.symbol for s in signals if s.direction == 1]
        short_syms = [s.symbol for s in signals if s.direction == -1]
        assert "LOSER" in long_syms
        assert "WINNER" in short_syms

    def test_rebalance_frequency(self):
        from trading_algo.quant_core.strategies.short_term_reversal import (
            ShortTermReversalStrategy,
            ReversalConfig,
        )
        config = ReversalConfig(rebalance_frequency=5)
        s = ShortTermReversalStrategy(config)
        assert s.should_rebalance() is False  # bar 1
        assert s.should_rebalance() is False  # bar 2
        assert s.should_rebalance() is False  # bar 3
        assert s.should_rebalance() is False  # bar 4
        assert s.should_rebalance() is True   # bar 5


# ────────────────────────────────────────────────────────────────────────
# Overnight Returns Strategy
# ────────────────────────────────────────────────────────────────────────

class TestOvernightReturnsStrategy:
    def test_basic_import(self):
        from trading_algo.quant_core.strategies.overnight_returns import (
            OvernightReturnsStrategy,
            OvernightConfig,
        )
        s = OvernightReturnsStrategy()
        assert s.config.top_n == 5

    def test_no_signals_before_entry_time(self):
        from trading_algo.quant_core.strategies.overnight_returns import (
            OvernightReturnsStrategy,
        )
        s = OvernightReturnsStrategy()
        ts = datetime(2025, 6, 15, 12, 0)
        signals = s.generate_signals(["AAPL"], ts)
        assert signals == []

    def test_reset(self):
        from trading_algo.quant_core.strategies.overnight_returns import (
            OvernightReturnsStrategy,
        )
        s = OvernightReturnsStrategy()
        ts = datetime(2025, 6, 15, 9, 30)
        s.update("AAPL", ts, 150.0, 151.0)
        s.reset()
        assert len(s._closes) == 0
        assert len(s._opens) == 0


# ────────────────────────────────────────────────────────────────────────
# ORB + VWAP confirmation
# ────────────────────────────────────────────────────────────────────────

class TestORBVWAPFilter:
    def test_vwap_tracking(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import ORBStrategyAdapter
        adapter = ORBStrategyAdapter(vwap_filter=True)

        ts = datetime(2025, 6, 15, 9, 35)
        # Bar: H=102, L=98, C=100, V=1000
        adapter.update("AAPL", ts, 99.0, 102.0, 98.0, 100.0, 1000)
        vwap = adapter._get_vwap("AAPL")
        # Typical price = (102+98+100)/3 = 100.0
        assert vwap == pytest.approx(100.0)

    def test_vwap_resets_daily(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import ORBStrategyAdapter
        adapter = ORBStrategyAdapter(vwap_filter=True)

        ts1 = datetime(2025, 6, 15, 9, 35)
        adapter.update("AAPL", ts1, 100.0, 110.0, 90.0, 100.0, 1000)

        ts2 = datetime(2025, 6, 16, 9, 35)  # Next day
        adapter.update("AAPL", ts2, 200.0, 210.0, 190.0, 200.0, 1000)

        vwap = adapter._get_vwap("AAPL")
        # Should only reflect day 2: (210+190+200)/3 = 200.0
        assert vwap == pytest.approx(200.0)

    def test_vwap_filter_disabled(self):
        from trading_algo.multi_strategy.adapters.orb_adapter import ORBStrategyAdapter
        adapter = ORBStrategyAdapter(vwap_filter=False)
        assert adapter._vwap_filter is False


# ────────────────────────────────────────────────────────────────────────
# Adapter import tests for new strategies
# ────────────────────────────────────────────────────────────────────────

class TestPhase3AdapterImports:
    def test_import_intraday_momentum_adapter(self):
        from trading_algo.multi_strategy.adapters.intraday_momentum_adapter import (
            IntradayMomentumAdapter,
        )
        adapter = IntradayMomentumAdapter()
        assert adapter.name == "IntradayMomentum"
        assert adapter.state == StrategyState.WARMING_UP

    def test_import_reversal_adapter(self):
        from trading_algo.multi_strategy.adapters.reversal_adapter import (
            ReversalStrategyAdapter,
        )
        adapter = ReversalStrategyAdapter()
        assert adapter.name == "ShortTermReversal"
        assert adapter.state == StrategyState.WARMING_UP

    def test_import_overnight_adapter(self):
        from trading_algo.multi_strategy.adapters.overnight_adapter import (
            OvernightReturnsAdapter,
        )
        adapter = OvernightReturnsAdapter()
        assert adapter.name == "OvernightReturns"
        assert adapter.state == StrategyState.WARMING_UP

    def test_all_adapters_from_package(self):
        from trading_algo.multi_strategy.adapters import (
            IntradayMomentumAdapter,
            ReversalStrategyAdapter,
            OvernightReturnsAdapter,
        )
        assert IntradayMomentumAdapter is not None
        assert ReversalStrategyAdapter is not None
        assert OvernightReturnsAdapter is not None


# ────────────────────────────────────────────────────────────────────────
# Intraday Momentum Adapter tests
# ────────────────────────────────────────────────────────────────────────

class TestIntradayMomentumAdapter:
    def test_warmup(self):
        from trading_algo.multi_strategy.adapters.intraday_momentum_adapter import (
            IntradayMomentumAdapter,
        )
        adapter = IntradayMomentumAdapter()
        ts = datetime(2025, 6, 15, 9, 30)

        for i in range(9):
            adapter.update("AAPL", ts, 100, 101, 99, 100, 1000)
        assert adapter.state == StrategyState.WARMING_UP

        adapter.update("AAPL", ts, 100, 101, 99, 100, 1000)
        assert adapter.state == StrategyState.ACTIVE

    def test_no_signals_during_warmup(self):
        from trading_algo.multi_strategy.adapters.intraday_momentum_adapter import (
            IntradayMomentumAdapter,
        )
        adapter = IntradayMomentumAdapter()
        ts = datetime(2025, 6, 15, 15, 30)
        for i in range(5):
            adapter.update("AAPL", ts, 100, 101, 99, 100, 1000)
        signals = adapter.generate_signals(["AAPL"], ts)
        assert signals == []


# ────────────────────────────────────────────────────────────────────────
# Vol management tests
# ────────────────────────────────────────────────────────────────────────

class TestVolManagement:
    def test_vol_management_scales_signals(self):
        from trading_algo.multi_strategy.controller import (
            MultiStrategyController,
            ControllerConfig,
            StrategyAllocation,
        )

        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            enable_vol_management=True,
            vol_target=0.15,
            vol_lookback=5,
        )
        ctrl = MultiStrategyController(cfg)

        # Feed high-vol returns (annualized ~30%)
        high_vol_returns = [0.02, -0.02, 0.02, -0.02, 0.02]
        for r in high_vol_returns:
            ctrl.add_return(r)

        # Create a signal that would go through the pipeline
        from tests.test_multi_strategy import StubStrategy
        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        # With high vol, signals should be scaled down
        if result:
            assert result[0].target_weight < 0.10

    def test_vol_management_disabled(self):
        from trading_algo.multi_strategy.controller import (
            MultiStrategyController,
            ControllerConfig,
            StrategyAllocation,
        )

        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
            enable_vol_management=False,
        )
        ctrl = MultiStrategyController(cfg)

        # Even with high vol returns, signals not scaled
        for r in [0.05, -0.05, 0.05, -0.05, 0.05]:
            ctrl.add_return(r)

        from tests.test_multi_strategy import StubStrategy
        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())
        if result:
            assert result[0].target_weight == pytest.approx(0.10)

    def test_add_return_tracks_history(self):
        from trading_algo.multi_strategy.controller import MultiStrategyController
        ctrl = MultiStrategyController()
        ctrl.add_return(0.01)
        ctrl.add_return(-0.005)
        ctrl.add_return(0.002)
        assert len(ctrl._returns) == 3
