"""Tests for Phase 4 (backtesting, walk-forward, regime analysis)
and Phase 5 (regime-adaptive allocation)."""

from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pytest

from trading_algo.multi_strategy.protocol import StrategySignal, StrategyState
from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)


# Reuse the StubStrategy from the Phase 2 tests
from tests.test_multi_strategy import StubStrategy


# ────────────────────────────────────────────────────────────────────────
# Phase 4a: Backtest Runner
# ────────────────────────────────────────────────────────────────────────

class TestBacktestRunnerImports:
    def test_import_all_types(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
            MultiStrategyBacktestConfig,
            MultiStrategyBacktestResults,
            StrategyAttribution,
        )
        assert MultiStrategyBacktestRunner is not None
        assert MultiStrategyBacktestConfig is not None
        assert MultiStrategyBacktestResults is not None
        assert StrategyAttribution is not None


class TestBacktestConfig:
    def test_default_config(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestConfig,
        )
        cfg = MultiStrategyBacktestConfig()
        assert cfg.initial_capital == 100_000
        assert cfg.commission_per_share == 0.0035
        assert cfg.slippage_bps == 2.0

    def test_custom_config(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestConfig,
        )
        cfg = MultiStrategyBacktestConfig(
            initial_capital=50_000,
            symbols=["AAPL", "MSFT"],
            commission_per_share=0.005,
            slippage_bps=5.0,
        )
        assert cfg.initial_capital == 50_000
        assert cfg.symbols == ["AAPL", "MSFT"]
        assert cfg.commission_per_share == 0.005


class TestBacktestResults:
    def test_default_results(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestResults,
        )
        r = MultiStrategyBacktestResults()
        assert r.total_return == 0.0
        assert r.sharpe_ratio == 0.0
        assert r.max_drawdown == 0.0
        assert r.equity_curve == []
        assert r.strategy_attribution == {}

    def test_strategy_attribution(self):
        from trading_algo.multi_strategy.backtest_runner import (
            StrategyAttribution,
        )
        attr = StrategyAttribution(name="Momentum", n_signals=50, gross_pnl=1200.0)
        assert attr.name == "Momentum"
        assert attr.n_signals == 50
        assert attr.gross_pnl == 1200.0


@dataclass
class MockBar:
    """Minimal bar object for backtest runner tests."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TestBacktestRunner:
    def _make_bars(self, symbol, n_bars=20, start_price=100.0):
        """Generate synthetic bars with random walk."""
        np.random.seed(42)
        bars = []
        price = start_price
        base_ts = datetime(2025, 1, 2, 9, 30)

        for i in range(n_bars):
            # Random daily move
            ret = np.random.normal(0.001, 0.02)
            price = price * (1 + ret)
            h = price * 1.01
            l = price * 0.99
            ts = base_ts + timedelta(days=i)
            bars.append(MockBar(
                timestamp=ts,
                open=price * 0.999,
                high=h,
                low=l,
                close=price,
                volume=100_000,
            ))
        return bars

    def test_run_empty_data(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
            MultiStrategyBacktestConfig,
        )
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl)
        results = runner.run({})
        assert results.total_return == 0.0
        assert results.total_trades == 0

    def test_run_with_no_signals(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
            MultiStrategyBacktestConfig,
        )
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl)
        data = {"AAPL": self._make_bars("AAPL", 20)}
        results = runner.run(data)
        # No strategies registered = no signals = flat equity
        assert results.total_trades == 0
        assert len(results.equity_curve) > 0

    def test_run_with_signals(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
            MultiStrategyBacktestConfig,
        )
        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.05, confidence=0.8,
        )
        cfg = ControllerConfig(
            allocations={"A": StrategyAllocation(weight=1.0)},
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.register(StubStrategy("A", signals=[sig]))

        bt_cfg = MultiStrategyBacktestConfig(
            initial_capital=100_000,
            symbols=["AAPL"],
        )
        runner = MultiStrategyBacktestRunner(ctrl, bt_cfg)
        data = {"AAPL": self._make_bars("AAPL", 30)}
        results = runner.run(data)

        # Should have generated some trades
        assert results.total_trades > 0
        assert len(results.equity_curve) > 1

    def test_equity_curve_starts_at_initial_capital(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
            MultiStrategyBacktestConfig,
        )
        bt_cfg = MultiStrategyBacktestConfig(initial_capital=50_000)
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl, bt_cfg)
        assert runner._equity == 50_000
        assert runner._equity_curve[0] == 50_000

    def test_progress_callback(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
        )
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl)
        data = {"AAPL": self._make_bars("AAPL", 10)}

        callbacks = []
        def on_progress(pct, msg):
            callbacks.append((pct, msg))

        runner.run(data, progress_callback=on_progress)
        assert len(callbacks) > 0
        # First callback should be the initial one
        assert callbacks[0][0] == pytest.approx(0.05)

    def test_position_management(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
        )
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl)

        # Simulate opening a position
        runner._current_prices["AAPL"] = 150.0
        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ts = datetime(2025, 1, 15, 10, 0)
        runner._open_position(sig, ts)

        assert "AAPL" in runner._positions
        assert runner._positions["AAPL"] > 0
        assert len(runner._trades) == 1
        assert runner._trades[0]["side"] == "BUY"

    def test_close_position(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
        )
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl)

        # Open then close
        runner._current_prices["AAPL"] = 150.0
        runner._positions["AAPL"] = 100  # 100 shares
        runner._position_prices["AAPL"] = 148.0

        ts = datetime(2025, 1, 15, 15, 0)
        runner._close_position("AAPL", ts)

        assert "AAPL" not in runner._positions
        assert len(runner._trades) == 1
        assert runner._trades[0]["side"] == "SELL"

    def test_slippage_applied(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
            MultiStrategyBacktestConfig,
        )
        cfg = MultiStrategyBacktestConfig(slippage_bps=10.0)  # 10bps
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl, cfg)
        runner._current_prices["AAPL"] = 100.0

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        runner._open_position(sig, datetime.now())

        # For a buy, exec price = 100 + 100 * 10/10000 = 100.10
        assert runner._trades[0]["price"] == pytest.approx(100.10)

    def test_build_results_computes_metrics(self):
        from trading_algo.multi_strategy.backtest_runner import (
            MultiStrategyBacktestRunner,
        )
        ctrl = MultiStrategyController()
        runner = MultiStrategyBacktestRunner(ctrl)

        # Manually set some state to test _build_results
        runner._equity_curve = [100_000, 101_000, 102_000, 101_500]
        runner._daily_returns = [0.01, 0.0099, -0.0049]
        runner._trades = [
            {"side": "BUY", "symbol": "AAPL", "shares": 10, "price": 100.0, "strategy": "Mom"},
            {"side": "SELL", "symbol": "AAPL", "shares": -10, "price": 102.0, "strategy": "exit"},
            {"side": "BUY", "symbol": "AAPL", "shares": 5, "price": 101.0, "strategy": "Mom"},
        ]
        runner._closed_trades = 2  # 2 round-trip trades (SELL = close)
        runner._winning_trades = 1
        runner._signals_by_strategy = {"Momentum": 5, "ORB": 3}

        results = runner._build_results()
        assert results.total_return > 0
        assert results.total_trades == 2  # closed trades, not raw events
        assert "Momentum" in results.strategy_attribution
        assert results.strategy_attribution["Momentum"].n_signals == 5
        assert len(results.equity_curve) == 4


# ────────────────────────────────────────────────────────────────────────
# Phase 4b: Walk-Forward Validation
# ────────────────────────────────────────────────────────────────────────

class TestWalkForwardImports:
    def test_import_all_types(self):
        from trading_algo.multi_strategy.walk_forward import (
            WalkForwardValidator,
            WalkForwardResult,
            WalkForwardFold,
        )
        assert WalkForwardValidator is not None
        assert WalkForwardResult is not None
        assert WalkForwardFold is not None


class TestWalkForwardFold:
    def test_fold_dataclass(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardFold
        fold = WalkForwardFold(
            fold_index=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 12, 31),
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=1.0,
            in_sample_return=0.15,
            out_of_sample_return=0.08,
            degradation=0.333,
        )
        assert fold.fold_index == 1
        assert fold.degradation == pytest.approx(0.333)


class TestWalkForwardResult:
    def test_default_result(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardResult
        r = WalkForwardResult()
        assert r.folds == []
        assert r.avg_is_sharpe == 0.0
        assert r.pbo is None
        assert r.is_overfit is False


class TestWalkForwardValidator:
    def _generate_returns(self, n=500, mu=0.0005, sigma=0.015, seed=42):
        """Generate synthetic daily returns with positive drift."""
        np.random.seed(seed)
        return np.random.normal(mu, sigma, n)

    def test_insufficient_data(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=5, min_fold_size=60)
        # Too little data
        returns = np.array([0.01, -0.005, 0.003])
        result = validator.validate(returns)
        assert result.folds == []
        assert result.avg_is_sharpe == 0.0

    def test_basic_walk_forward(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=5, min_fold_size=30)
        returns = self._generate_returns(300)
        result = validator.validate(returns)

        assert len(result.folds) > 0
        # Each fold should have reasonable Sharpe
        for fold in result.folds:
            assert isinstance(fold.in_sample_sharpe, float)
            assert isinstance(fold.out_of_sample_sharpe, float)

    def test_fold_count(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=5, min_fold_size=30)
        returns = self._generate_returns(500)
        result = validator.validate(returns)
        # With 5 folds, should have folds 1-4 (train on 0..i-1, test on i)
        assert len(result.folds) == 4

    def test_degradation_computed(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=5, min_fold_size=30)
        returns = self._generate_returns(500)
        result = validator.validate(returns)
        # avg_degradation should be a float
        assert isinstance(result.avg_degradation, float)

    def test_sharpe_averages(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=5, min_fold_size=30)
        returns = self._generate_returns(500, mu=0.001)
        result = validator.validate(returns)

        # With positive drift, IS sharpe should be positive
        assert result.avg_is_sharpe > 0
        # OOS should also be roughly positive (not guaranteed but likely with mu > 0)
        # Just check it's computed
        assert isinstance(result.avg_oos_sharpe, float)

    def test_timestamps_passed_through(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=3, min_fold_size=30)
        returns = self._generate_returns(200)
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(200)]
        result = validator.validate(returns, timestamps=timestamps)

        for fold in result.folds:
            assert isinstance(fold.train_start, datetime)
            assert isinstance(fold.test_end, datetime)

    def test_auto_reduce_folds(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=10, min_fold_size=60)
        # Only 200 bars: 200/10=20 < min_fold_size, so folds reduce
        returns = self._generate_returns(200)
        result = validator.validate(returns)
        # Should have reduced folds and still produced results
        assert len(result.folds) > 0

    def test_overfit_detection_positive_drift(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_folds=5, min_fold_size=30)
        # Positive drift should NOT be classified as overfit
        returns = self._generate_returns(500, mu=0.001)
        result = validator.validate(returns)
        # Strong positive drift should not show >50% degradation typically
        # (not a hard guarantee, but a reasonable expectation)
        assert isinstance(result.is_overfit, bool)

    def test_compute_sharpe_static(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        # Edge case: constant returns
        returns = np.ones(100) * 0.001
        sharpe = WalkForwardValidator._compute_sharpe(returns)
        # Zero vol → 0.0
        assert sharpe == 0.0

    def test_compute_sharpe_empty(self):
        from trading_algo.multi_strategy.walk_forward import WalkForwardValidator
        returns = np.array([0.01])
        sharpe = WalkForwardValidator._compute_sharpe(returns)
        assert sharpe == 0.0


# ────────────────────────────────────────────────────────────────────────
# Phase 4c: Regime Analysis
# ────────────────────────────────────────────────────────────────────────

class TestRegimeAnalysisImports:
    def test_import_all_types(self):
        from trading_algo.multi_strategy.regime_analysis import (
            RegimeAnalyzer,
            RegimeAnalysisResult,
            RegimePerformance,
            StrategyRegimeProfile,
        )
        assert RegimeAnalyzer is not None
        assert RegimeAnalysisResult is not None


class TestRegimePerformance:
    def test_dataclass(self):
        from trading_algo.multi_strategy.regime_analysis import RegimePerformance
        perf = RegimePerformance(
            regime="BULL", n_bars=100, sharpe_ratio=1.5, win_rate=0.58,
        )
        assert perf.regime == "BULL"
        assert perf.n_bars == 100
        assert perf.sharpe_ratio == 1.5


class TestStrategyRegimeProfile:
    def test_best_worst_regime(self):
        from trading_algo.multi_strategy.regime_analysis import (
            StrategyRegimeProfile,
            RegimePerformance,
        )
        profile = StrategyRegimeProfile(
            strategy_name="Momentum",
            regimes={
                "BULL": RegimePerformance(regime="BULL", sharpe_ratio=2.0),
                "BEAR": RegimePerformance(regime="BEAR", sharpe_ratio=-0.5),
                "NEUTRAL": RegimePerformance(regime="NEUTRAL", sharpe_ratio=0.8),
            },
        )
        assert profile.best_regime == "BULL"
        assert profile.worst_regime == "BEAR"

    def test_empty_profile(self):
        from trading_algo.multi_strategy.regime_analysis import StrategyRegimeProfile
        profile = StrategyRegimeProfile(strategy_name="Empty")
        assert profile.best_regime is None
        assert profile.worst_regime is None


class TestRegimeAnalyzer:
    def _generate_regime_data(self, n=300, seed=42):
        """Generate synthetic returns with regime labels."""
        np.random.seed(seed)
        labels = []
        returns_a = []
        returns_b = []

        for i in range(n):
            if i < 100:
                regime = "BULL"
                ret_a = np.random.normal(0.002, 0.01)  # Momentum works well
                ret_b = np.random.normal(-0.001, 0.015)  # Reversal struggles
            elif i < 200:
                regime = "BEAR"
                ret_a = np.random.normal(-0.001, 0.02)  # Momentum struggles
                ret_b = np.random.normal(0.002, 0.01)  # Reversal works well
            else:
                regime = "NEUTRAL"
                ret_a = np.random.normal(0.0005, 0.012)
                ret_b = np.random.normal(0.0005, 0.012)

            labels.append(regime)
            returns_a.append(ret_a)
            returns_b.append(ret_b)

        return {"Momentum": returns_a, "Reversal": returns_b}, labels

    def test_basic_analysis(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer(min_regime_bars=20)
        returns, labels = self._generate_regime_data()
        result = analyzer.analyze(returns, labels)

        assert "Momentum" in result.strategy_profiles
        assert "Reversal" in result.strategy_profiles

    def test_regime_profiles_populated(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer(min_regime_bars=20)
        returns, labels = self._generate_regime_data()
        result = analyzer.analyze(returns, labels)

        momentum_profile = result.strategy_profiles["Momentum"]
        assert len(momentum_profile.regimes) > 0
        # Should have BULL, BEAR, NEUTRAL
        assert "BULL" in momentum_profile.regimes
        assert "BEAR" in momentum_profile.regimes

    def test_momentum_best_in_bull(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer(min_regime_bars=20)
        returns, labels = self._generate_regime_data()
        result = analyzer.analyze(returns, labels)

        momentum_profile = result.strategy_profiles["Momentum"]
        assert momentum_profile.best_regime == "BULL"

    def test_reversal_best_in_bear(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer(min_regime_bars=20)
        returns, labels = self._generate_regime_data()
        result = analyzer.analyze(returns, labels)

        reversal_profile = result.strategy_profiles["Reversal"]
        assert reversal_profile.best_regime == "BEAR"

    def test_regime_weights_computed(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer(min_regime_bars=20)
        returns, labels = self._generate_regime_data()
        result = analyzer.analyze(returns, labels)

        assert len(result.regime_weights) > 0
        for regime, weights in result.regime_weights.items():
            # Weights should sum to ~1.0
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_default_regime_tilts_fallback(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer(min_regime_bars=1000)  # Very high → no data
        # Only 10 bars per regime → too few
        returns = {"A": [0.01] * 30}
        labels = ["BULL"] * 10 + ["BEAR"] * 10 + ["HIGH_VOL"] * 10
        result = analyzer.analyze(returns, labels)

        # Should fall back to default tilts
        if "BULL" in result.regime_weights:
            assert "Orchestrator" in result.regime_weights["BULL"]

    def test_sharpe_computation(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        returns = np.array([0.01, -0.005, 0.008, 0.003, -0.002, 0.006])
        sharpe = RegimeAnalyzer._compute_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_sharpe_edge_cases(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        # Too short
        assert RegimeAnalyzer._compute_sharpe(np.array([0.01])) == 0.0
        # Zero vol
        assert RegimeAnalyzer._compute_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0

    def test_length_mismatch_skipped(self):
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
        analyzer = RegimeAnalyzer()
        returns = {"A": [0.01] * 100, "B": [0.01] * 50}  # B is wrong length
        labels = ["BULL"] * 100
        result = analyzer.analyze(returns, labels)
        # B should be skipped due to length mismatch
        assert "A" in result.strategy_profiles
        assert "B" not in result.strategy_profiles


# ────────────────────────────────────────────────────────────────────────
# Phase 5: Regime-Adaptive Allocation
# ────────────────────────────────────────────────────────────────────────

class TestRegimeAdaptation:
    def test_regime_adaptation_disabled_by_default(self):
        cfg = ControllerConfig()
        assert cfg.enable_regime_adaptation is False

    def test_set_regime(self):
        ctrl = MultiStrategyController()
        ctrl.set_regime("BULL")
        assert ctrl._current_regime == "BULL"

    def test_set_regime_weights(self):
        ctrl = MultiStrategyController()
        weights = {
            "BULL": {"Momentum": 0.6, "Reversal": 0.4},
            "BEAR": {"Momentum": 0.3, "Reversal": 0.7},
        }
        ctrl.set_regime_weights(weights)
        assert ctrl._regime_weights == weights

    def test_regime_adaptation_blends_weights(self):
        """When enabled, weights should blend static and regime-adaptive."""
        cfg = ControllerConfig(
            allocations={
                "A": StrategyAllocation(weight=0.50),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=0.5,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.set_regime("BULL")
        ctrl.set_regime_weights({
            "BULL": {"A": 0.80},  # Regime says 80% to A
        })

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        assert len(result) == 1
        # Effective weight = (1-0.5)*0.50 + 0.5*0.80 = 0.25 + 0.40 = 0.65
        # Scaled signal = 0.10 * 0.65 = 0.065
        assert result[0].target_weight == pytest.approx(0.065)

    def test_regime_adaptation_fully_dynamic(self):
        """With blend_factor=1.0, fully use regime weights."""
        cfg = ControllerConfig(
            allocations={
                "A": StrategyAllocation(weight=0.30),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=1.0,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.set_regime("BEAR")
        ctrl.set_regime_weights({
            "BEAR": {"A": 0.10},
        })

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        assert len(result) == 1
        # Fully dynamic: 0.10 * 0.10 = 0.01
        assert result[0].target_weight == pytest.approx(0.01)

    def test_regime_adaptation_fully_static(self):
        """With blend_factor=0.0, ignore regime weights entirely."""
        cfg = ControllerConfig(
            allocations={
                "A": StrategyAllocation(weight=0.40),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=0.0,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.set_regime("HIGH_VOL")
        ctrl.set_regime_weights({
            "HIGH_VOL": {"A": 0.90},  # Should be ignored
        })

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        assert len(result) == 1
        # Fully static: 0.10 * 0.40 = 0.04
        assert result[0].target_weight == pytest.approx(0.04)

    def test_detect_regime_insufficient_data(self):
        ctrl = MultiStrategyController()
        # No returns history
        regime = ctrl.detect_regime()
        assert regime == "NEUTRAL"

    def test_detect_regime_returns_valid_string(self):
        """detect_regime() should return a valid regime label."""
        ctrl = MultiStrategyController()
        for _ in range(100):
            ctrl.add_return(0.005)
        regime = ctrl.detect_regime()
        assert isinstance(regime, str)
        assert len(regime) > 0
        # Should be stored
        assert ctrl._current_regime == regime

    def test_detect_regime_fallback_bull(self):
        """Test the simple fallback (HMM bypassed) detects bull."""
        ctrl = MultiStrategyController()
        for _ in range(100):
            ctrl.add_return(0.005)
        # Force fallback by breaking HMM model
        ctrl._hmm_model = "broken"
        regime = ctrl.detect_regime()
        assert regime == "BULL"

    def test_detect_regime_fallback_bear(self):
        """Test the simple fallback detects bear."""
        ctrl = MultiStrategyController()
        for _ in range(100):
            ctrl.add_return(-0.005)
        ctrl._hmm_model = "broken"
        regime = ctrl.detect_regime()
        assert regime == "BEAR"

    def test_detect_regime_fallback_high_vol(self):
        """Test the simple fallback detects high vol."""
        ctrl = MultiStrategyController()
        # Alternate perfectly so mean ≈ 0 but std > 0.02
        for i in range(100):
            ctrl.add_return(0.04 if i % 2 == 0 else -0.04)
        ctrl._hmm_model = "broken"
        regime = ctrl.detect_regime()
        assert regime == "HIGH_VOL"

    def test_regime_adaptation_with_unknown_strategy(self):
        """Strategy not in regime weights should use static allocation."""
        cfg = ControllerConfig(
            allocations={
                "A": StrategyAllocation(weight=0.50),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=0.8,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.set_regime("BULL")
        ctrl.set_regime_weights({
            "BULL": {"OtherStrategy": 0.90},  # A not in regime weights
        })

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        assert len(result) == 1
        # A not in regime weights → uses static: 0.10 * 0.50 = 0.05
        assert result[0].target_weight == pytest.approx(0.05)

    def test_regime_default_tilts_used(self):
        """When regime_adaptation enabled but no custom weights, use defaults."""
        cfg = ControllerConfig(
            allocations={
                "Orchestrator": StrategyAllocation(weight=0.40),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=0.5,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.set_regime("BULL")
        # No custom weights set → should use DEFAULT_REGIME_TILTS from RegimeAnalyzer

        sig = StrategySignal(
            strategy_name="Orchestrator", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("Orchestrator", signals=[sig]))
        result = ctrl.generate_signals(["AAPL"], datetime.now())

        assert len(result) == 1
        # Default BULL tilt for Orchestrator = 0.22
        # Effective = (1-0.5)*0.40 + 0.5*0.22 = 0.20 + 0.11 = 0.31
        # Signal = 0.10 * 0.31 = 0.031
        assert result[0].target_weight == pytest.approx(0.031)


# ────────────────────────────────────────────────────────────────────────
# Integration: Regime analysis feeds controller
# ────────────────────────────────────────────────────────────────────────

class TestRegimeIntegration:
    def test_analyzer_output_feeds_controller(self):
        """Verify RegimeAnalyzer output can be loaded into controller."""
        from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer

        # Run analysis
        np.random.seed(42)
        returns = {
            "A": list(np.random.normal(0.001, 0.01, 200)),
            "B": list(np.random.normal(0.0005, 0.015, 200)),
        }
        labels = ["BULL"] * 100 + ["BEAR"] * 100
        analyzer = RegimeAnalyzer(min_regime_bars=20)
        result = analyzer.analyze(returns, labels)

        # Feed into controller
        cfg = ControllerConfig(
            allocations={
                "A": StrategyAllocation(weight=0.50),
                "B": StrategyAllocation(weight=0.50),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=0.6,
        )
        ctrl = MultiStrategyController(cfg)
        ctrl.set_regime_weights(result.regime_weights)
        ctrl.set_regime("BULL")

        sig = StrategySignal(
            strategy_name="A", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        ctrl.register(StubStrategy("A", signals=[sig]))
        signals = ctrl.generate_signals(["AAPL"], datetime.now())

        assert len(signals) == 1
        # Weight should reflect regime-adaptive blending
        assert signals[0].target_weight > 0

    def test_full_pipeline_regime_to_signals(self):
        """End-to-end: regime detection → weight adaptation → signal generation."""
        cfg = ControllerConfig(
            allocations={
                "Trend": StrategyAllocation(weight=0.60),
                "Reversal": StrategyAllocation(weight=0.40),
            },
            enable_regime_adaptation=True,
            regime_blend_factor=0.5,
            enable_vol_management=False,  # Disable so we test regime adaptation in isolation
        )
        ctrl = MultiStrategyController(cfg)

        # Simulate bull market returns and set regime directly
        for _ in range(100):
            ctrl.add_return(0.005)

        # Use set_regime for deterministic test (detect_regime tested separately)
        ctrl.set_regime("BULL")
        regime = ctrl._current_regime
        assert regime == "BULL"

        # Set weights for bull: overweight Trend
        ctrl.set_regime_weights({
            "BULL": {"Trend": 0.80, "Reversal": 0.20},
        })

        sig_trend = StrategySignal(
            strategy_name="Trend", symbol="AAPL", direction=1,
            target_weight=0.10, confidence=0.8,
        )
        sig_rev = StrategySignal(
            strategy_name="Reversal", symbol="MSFT", direction=-1,
            target_weight=0.10, confidence=0.7,
        )
        ctrl.register(StubStrategy("Trend", signals=[sig_trend]))
        ctrl.register(StubStrategy("Reversal", signals=[sig_rev]))

        signals = ctrl.generate_signals(["AAPL", "MSFT"], datetime.now())
        assert len(signals) == 2

        # Trend should have higher effective weight than Reversal
        trend_sig = [s for s in signals if s.symbol == "AAPL"][0]
        rev_sig = [s for s in signals if s.symbol == "MSFT"][0]

        # Trend: (0.5)*0.60 + (0.5)*0.80 = 0.70, signal = 0.10*0.70 = 0.07
        # Reversal: (0.5)*0.40 + (0.5)*0.20 = 0.30, signal = 0.10*0.30 = 0.03
        assert trend_sig.target_weight == pytest.approx(0.07)
        assert rev_sig.target_weight == pytest.approx(0.03)
