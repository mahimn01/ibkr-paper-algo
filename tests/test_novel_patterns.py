"""Tests for novel pattern discovery modules and advanced trading strategies.

Covers 8 new modules:
    1. AdvancedFeatureEngine
    2. RegimeTransitionStrategy
    3. CrossAssetDivergenceStrategy
    4. FlowPressureStrategy
    5. LiquidityCycleStrategy
    6. PatternScanner
    7. NonlinearSignalCombiner
    8. AlphaDecayMonitor
"""

import datetime
from datetime import date, timedelta

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Deterministic random state for reproducible synthetic data
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)


def _synthetic_prices(n: int = 500, start: float = 100.0) -> np.ndarray:
    """Generate a synthetic price series with realistic drift."""
    returns = _RNG.randn(n) * 0.01 + 0.0002
    prices = start * np.exp(np.cumsum(returns))
    return prices


def _synthetic_volumes(n: int = 500) -> np.ndarray:
    """Generate synthetic volume data."""
    return np.abs(_RNG.randn(n) * 1e6 + 5e6)


# =========================================================================
# 1. AdvancedFeatureEngine
# =========================================================================


class TestAdvancedFeatureEngine:
    """Tests for trading_algo.quant_core.ml.advanced_features."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine()
        assert engine.base_engine is not None
        assert engine.lead_lag_lags == [1, 2, 5, 10]
        assert engine.stat_return_window == 20

    def test_creation_with_custom_params(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine(
            lead_lag_lags=[1, 5],
            cross_corr_window=30,
            interaction_top_n=10,
            stat_return_window=15,
            hurst_window=40,
            normalize=False,
        )
        assert engine.lead_lag_lags == [1, 5]
        assert engine.cross_corr_window == 30
        assert engine.normalize is False

    def test_compute_advanced_features_basic(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine(normalize=False)
        prices = _synthetic_prices(300)
        volumes = _synthetic_volumes(300)

        fs = engine.compute_advanced_features(
            prices=prices,
            volumes=volumes,
        )
        assert fs.features.shape[0] == 300
        assert fs.features.shape[1] > 0
        assert len(fs.feature_names) == fs.features.shape[1]

    def test_compute_with_timestamps_produces_calendar_features(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine(normalize=False)
        n = 300
        prices = _synthetic_prices(n)
        volumes = _synthetic_volumes(n)
        base_date = datetime.date(2023, 1, 2)
        timestamps = np.array(
            [base_date + datetime.timedelta(days=i) for i in range(n)]
        )

        fs = engine.compute_advanced_features(
            prices=prices,
            volumes=volumes,
            timestamps=timestamps,
        )
        # Calendar features should be present
        calendar_names = [n for n in fs.feature_names if "dow" in n or "moy" in n or "fomc" in n or "eom" in n]
        assert len(calendar_names) > 0

    def test_compute_with_cross_asset_prices(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine(normalize=False, cross_corr_window=30)
        n = 300
        prices = _synthetic_prices(n)
        volumes = _synthetic_volumes(n)
        cross_prices = {
            "SPY": _synthetic_prices(n, start=400),
            "TLT": _synthetic_prices(n, start=120),
        }

        fs = engine.compute_advanced_features(
            prices=prices,
            volumes=volumes,
            cross_asset_prices=cross_prices,
        )
        assert fs.features.shape[0] == n
        assert fs.features.shape[1] > 0

    def test_statistical_features_are_computed(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine(normalize=False, stat_return_window=20, hurst_window=60)
        prices = _synthetic_prices(300)

        fs = engine.compute_advanced_features(prices=prices)
        stat_names = [n for n in fs.feature_names if "skew" in n or "kurt" in n or "hurst" in n or "autocorr" in n]
        assert len(stat_names) > 0, "Expected statistical features in the output"

    def test_add_bar_incremental(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine(normalize=False)
        prices = _synthetic_prices(100)
        for p in prices:
            fs = engine.add_bar(close=float(p))
        assert fs.features.shape[0] == 100

    def test_reset_buffers(self):
        from trading_algo.quant_core.ml.advanced_features import (
            AdvancedFeatureEngine,
        )
        engine = AdvancedFeatureEngine()
        engine.add_bar(close=100.0)
        engine.add_bar(close=101.0)
        engine.reset_buffers()
        assert len(engine._bar_buffer["close"]) == 0
        assert engine._last_feature_set is None


# =========================================================================
# 2. RegimeTransitionStrategy
# =========================================================================


class TestRegimeTransitionStrategy:
    """Tests for trading_algo.quant_core.strategies.regime_transition."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.strategies.regime_transition import (
            RegimeTransitionStrategy,
            TransitionConfig,
        )
        strategy = RegimeTransitionStrategy()
        assert strategy.config.transition_threshold == 0.25
        assert strategy.config.velocity_window == 5

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.strategies.regime_transition import (
            RegimeTransitionStrategy,
            TransitionConfig,
        )
        config = TransitionConfig(
            transition_threshold=0.30,
            max_holding_days=15,
            kelly_fraction=0.25,
        )
        strategy = RegimeTransitionStrategy(config)
        assert strategy.config.transition_threshold == 0.30
        assert strategy.config.max_holding_days == 15
        assert strategy.config.kelly_fraction == 0.25

    def test_reset_clears_state(self):
        from trading_algo.quant_core.strategies.regime_transition import (
            RegimeTransitionStrategy,
        )
        strategy = RegimeTransitionStrategy()
        # Manually add something to internal state
        strategy._transition_history.append(
            (datetime.datetime.now(), np.eye(3))
        )
        strategy.reset()
        assert len(strategy._transition_history) == 0
        assert len(strategy._regime_state_history) == 0
        assert len(strategy._active_positions) == 0

    def test_generate_signal_returns_none_with_short_history(self):
        from trading_algo.quant_core.strategies.regime_transition import (
            RegimeTransitionStrategy,
            TransitionConfig,
        )
        from trading_algo.quant_core.models.hmm_regime import HiddenMarkovRegime

        config = TransitionConfig(min_regime_history=60)
        strategy = RegimeTransitionStrategy(config)
        hmm = HiddenMarkovRegime(n_states=3)

        prices = _synthetic_prices(30)
        returns = np.diff(prices) / prices[:-1]

        signal = strategy.generate_signal(
            symbol="SPY",
            prices=prices,
            returns=returns,
            hmm_model=hmm,
            timestamp=datetime.datetime.now(),
        )
        assert signal is None, "Should return None with insufficient history"

    def test_estimate_current_volatility(self):
        from trading_algo.quant_core.strategies.regime_transition import (
            RegimeTransitionStrategy,
        )
        strategy = RegimeTransitionStrategy()
        returns = np.array(_RNG.randn(100) * 0.01, dtype=np.float64)
        vol = strategy._estimate_current_volatility(returns)
        assert vol > 0
        assert vol < 2.0

    def test_compute_confidence(self):
        from trading_algo.quant_core.strategies.regime_transition import (
            RegimeTransitionStrategy,
        )
        strategy = RegimeTransitionStrategy()
        confidence = strategy._compute_confidence(
            trans_prob=0.4,
            trans_velocity=0.1,
            current_prob=0.6,
        )
        assert 0.0 <= confidence <= 1.0


# =========================================================================
# 3. CrossAssetDivergenceStrategy
# =========================================================================


class TestCrossAssetDivergenceStrategy:
    """Tests for trading_algo.quant_core.strategies.cross_asset_divergence."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
            DivergenceConfig,
        )
        strategy = CrossAssetDivergenceStrategy()
        assert strategy.config.lookback_windows == [5, 10, 20]
        assert strategy.config.entry_threshold == 2.0

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
            DivergenceConfig,
        )
        config = DivergenceConfig(
            entry_threshold=1.5,
            max_holding_days=5,
            max_position=0.20,
        )
        strategy = CrossAssetDivergenceStrategy(config)
        assert strategy.config.entry_threshold == 1.5
        assert strategy.config.max_holding_days == 5

    def test_update_stores_price_history(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
        )
        strategy = CrossAssetDivergenceStrategy()
        for i in range(10):
            strategy.update("SPY", 400.0 + i * 0.5)
        assert len(strategy._price_history["SPY"]) == 10

    def test_generate_signals_with_insufficient_data(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
            DivergenceConfig,
        )
        config = DivergenceConfig(min_history=60)
        strategy = CrossAssetDivergenceStrategy(config)
        # Only 30 data points -- not enough
        prices = {
            "SPY": _synthetic_prices(30),
            "HYG": _synthetic_prices(30),
        }
        signals = strategy.generate_signals(prices)
        assert signals == []

    def test_generate_signals_with_divergent_data(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
            DivergenceConfig,
        )
        # Create highly divergent assets to trigger a signal
        config = DivergenceConfig(
            min_history=60,
            entry_threshold=1.0,  # Lower threshold to make triggering easier
            lookback_windows=[5, 10],
            zscore_lookback=30,
            asset_pairs=[("SPY", "HYG")],
        )
        strategy = CrossAssetDivergenceStrategy(config)

        n = 200
        # SPY rallies hard
        spy_prices = 400.0 + np.cumsum(np.abs(_RNG.randn(n)) * 0.02)
        # HYG crashes
        hyg_prices = 80.0 - np.cumsum(np.abs(_RNG.randn(n)) * 0.01)
        hyg_prices = np.maximum(hyg_prices, 50.0)

        prices = {
            "SPY": spy_prices.astype(np.float64),
            "HYG": hyg_prices.astype(np.float64),
        }
        signals = strategy.generate_signals(prices)
        # With extreme divergence and low threshold, we expect signals
        # (though not guaranteed depending on the randomness)
        assert isinstance(signals, list)

    def test_reset_clears_state(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
        )
        strategy = CrossAssetDivergenceStrategy()
        strategy.update("SPY", 400.0)
        strategy.update("SPY", 401.0)
        strategy.reset()
        assert len(strategy._price_history) == 0

    def test_get_target_weights_returns_dict(self):
        from trading_algo.quant_core.strategies.cross_asset_divergence import (
            CrossAssetDivergenceStrategy,
        )
        strategy = CrossAssetDivergenceStrategy()
        prices = {
            "SPY": _synthetic_prices(30),
        }
        weights = strategy.get_target_weights(prices)
        assert isinstance(weights, dict)


# =========================================================================
# 4. FlowPressureStrategy
# =========================================================================


class TestFlowPressureStrategy:
    """Tests for trading_algo.quant_core.strategies.flow_pressure."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
            FlowPressureConfig,
        )
        strategy = FlowPressureStrategy()
        assert strategy.config.tom_entry_days_before_eom == 2
        assert strategy.config.max_position == 0.10

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
            FlowPressureConfig,
        )
        config = FlowPressureConfig(
            tom_entry_days_before_eom=3,
            max_position=0.15,
            vol_target=0.20,
        )
        strategy = FlowPressureStrategy(config)
        assert strategy.config.tom_entry_days_before_eom == 3
        assert strategy.config.vol_target == 0.20

    def test_update_stores_prices(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
        )
        strategy = FlowPressureStrategy()
        dt = datetime.datetime(2025, 6, 15, 16, 0)
        strategy.update(dt, {"SPY": 450.0, "AAPL": 180.0})
        assert len(strategy._price_history["SPY"]) == 1
        assert len(strategy._price_history["AAPL"]) == 1

    def test_turn_of_month_signal_at_month_end(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
            FlowPressureConfig,
        )
        config = FlowPressureConfig(tom_lookback=20)
        strategy = FlowPressureStrategy(config)

        # Seed enough price history
        n = 100
        spy_prices = _synthetic_prices(n, start=450.0)
        price_arrays = {"SPY": spy_prices.astype(np.float64)}

        # Pick a date near the end of a month (e.g. Jan 29, 2025 is near month end)
        bar_date = datetime.datetime(2025, 1, 29, 16, 0)

        signals = strategy.generate_signals(bar_date, price_arrays)
        # Signals is a list -- may or may not trigger depending on exact date position
        assert isinstance(signals, list)

    def test_options_expiry_date_calculation(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
        )
        # 3rd Friday of Jan 2025 is Jan 17
        d = date(2025, 1, 10)
        expiry = FlowPressureStrategy._next_options_expiry(d)
        assert expiry == date(2025, 1, 17)

    def test_is_trading_day(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
        )
        # A Wednesday
        assert FlowPressureStrategy._is_trading_day(datetime.datetime(2025, 6, 11)) is True
        # A Saturday
        assert FlowPressureStrategy._is_trading_day(datetime.datetime(2025, 6, 14)) is False

    def test_days_to_month_end(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
        )
        # June 25 2025 is a Wednesday. Last business day of June 2025 is June 30 (Monday).
        dt = datetime.datetime(2025, 6, 25)
        days = FlowPressureStrategy._days_to_month_end(dt)
        assert days >= 0

    def test_reset_clears_state(self):
        from trading_algo.quant_core.strategies.flow_pressure import (
            FlowPressureStrategy,
        )
        strategy = FlowPressureStrategy()
        strategy.update(
            datetime.datetime(2025, 6, 15, 16, 0),
            {"SPY": 450.0},
        )
        strategy.reset()
        assert len(strategy._price_history) == 0
        assert len(strategy._dates) == 0


# =========================================================================
# 5. LiquidityCycleStrategy
# =========================================================================


class TestLiquidityCycleStrategy:
    """Tests for trading_algo.quant_core.strategies.intraday.liquidity_cycles."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            LiquidityCycleConfig,
        )
        strategy = LiquidityCycleStrategy()
        assert strategy.config.opening_end == 30
        assert strategy.config.vol_target == 0.15

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            LiquidityCycleConfig,
        )
        config = LiquidityCycleConfig(
            opening_min_move=0.005,
            max_daily_trades=4,
            max_position=0.10,
        )
        strategy = LiquidityCycleStrategy(config)
        assert strategy.config.opening_min_move == 0.005
        assert strategy.config.max_daily_trades == 4

    def test_classify_regime_opening(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            IntradayRegime,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 9, 45)
        regime = strategy.classify_regime(ts)
        assert regime == IntradayRegime.OPENING_AUCTION

    def test_classify_regime_morning(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            IntradayRegime,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 10, 30)
        regime = strategy.classify_regime(ts)
        assert regime == IntradayRegime.MORNING_TREND

    def test_classify_regime_lunch(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            IntradayRegime,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 12, 0)
        regime = strategy.classify_regime(ts)
        assert regime == IntradayRegime.LUNCH_LULL

    def test_classify_regime_after_hours(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            IntradayRegime,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 16, 30)
        regime = strategy.classify_regime(ts)
        assert regime == IntradayRegime.AFTER_HOURS

    def test_update_bar_records_state(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 9, 35)
        strategy.update_bar("AAPL", ts, 150.0, 151.0, 149.5, 150.5, 10000)
        assert "AAPL" in strategy._state
        state = strategy._state["AAPL"]
        assert state.day_open == 150.0
        assert len(state.closes) == 1

    def test_opening_reversion_signal(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
            LiquidityCycleConfig,
        )
        config = LiquidityCycleConfig(opening_min_move=0.003)
        strategy = LiquidityCycleStrategy(config)

        # Simulate opening bars with a strong upward move
        base_time = datetime.datetime(2025, 6, 15, 9, 30)
        # Opening bar: price jumps from 100 to 101 (+1%)
        for i in range(6):
            ts = base_time + datetime.timedelta(minutes=i * 5)
            price = 100.0 + i * 0.2
            strategy.update_bar("AAPL", ts, price, price + 0.1, price - 0.1, price, 5000)

        # First morning-trend bar at 10:00 -- reversion signal should fire
        ts_morning = datetime.datetime(2025, 6, 15, 10, 0)
        strategy.update_bar("AAPL", ts_morning, 101.0, 101.2, 100.8, 101.0, 5000)
        signal = strategy.generate_signal("AAPL", ts_morning)
        # The signal may or may not fire depending on exact conditions, but the method should return cleanly
        assert signal is None or hasattr(signal, "direction")

    def test_reset_day_clears_state(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 9, 35)
        strategy.update_bar("AAPL", ts, 150.0, 151.0, 149.5, 150.5, 10000)
        strategy.reset_day()
        assert len(strategy._state) == 0
        assert strategy._current_date is None

    def test_get_session_stats(self):
        from trading_algo.quant_core.strategies.intraday.liquidity_cycles import (
            LiquidityCycleStrategy,
        )
        strategy = LiquidityCycleStrategy()
        ts = datetime.datetime(2025, 6, 15, 9, 35)
        strategy.update_bar("AAPL", ts, 150.0, 151.0, 149.5, 150.5, 10000)
        stats = strategy.get_session_stats()
        assert "total_signals" in stats
        assert "symbols_active" in stats
        assert stats["symbols_active"] == 1


# =========================================================================
# 6. PatternScanner
# =========================================================================


class TestPatternScanner:
    """Tests for trading_algo.quant_core.discovery.pattern_scanner."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.discovery.pattern_scanner import (
            PatternScanner,
            ScannerConfig,
        )
        scanner = PatternScanner()
        assert scanner.config.min_ic == 0.02
        assert scanner.config.n_folds == 5

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.discovery.pattern_scanner import (
            PatternScanner,
            ScannerConfig,
        )
        config = ScannerConfig(
            min_ic=0.03,
            n_folds=3,
            min_observations=100,
            max_candidates_per_run=100,
        )
        scanner = PatternScanner(config)
        assert scanner.config.min_ic == 0.03
        assert scanner.config.max_candidates_per_run == 100

    def test_scan_raises_on_insufficient_observations(self):
        from trading_algo.quant_core.discovery.pattern_scanner import (
            PatternScanner,
            ScannerConfig,
        )
        config = ScannerConfig(min_observations=252)
        scanner = PatternScanner(config)

        n = 100  # too short
        features = _RNG.randn(n, 3).astype(np.float64)
        feature_names = ["f1", "f2", "f3"]
        forward_returns = _RNG.randn(n).astype(np.float64) * 0.01

        with pytest.raises(ValueError, match="Insufficient observations"):
            scanner.scan(features, feature_names, forward_returns)

    def test_scan_raises_on_dimension_mismatch(self):
        from trading_algo.quant_core.discovery.pattern_scanner import (
            PatternScanner,
            ScannerConfig,
        )
        config = ScannerConfig(min_observations=50)
        scanner = PatternScanner(config)

        features = _RNG.randn(100, 3).astype(np.float64)
        feature_names = ["f1", "f2"]  # wrong number
        forward_returns = _RNG.randn(100).astype(np.float64) * 0.01

        with pytest.raises(ValueError, match="feature_names length"):
            scanner.scan(features, feature_names, forward_returns)

    def test_scan_with_synthetic_data(self):
        from trading_algo.quant_core.discovery.pattern_scanner import (
            PatternScanner,
            ScannerConfig,
        )
        config = ScannerConfig(
            min_observations=100,
            lookback_windows=[10, 20],
            holding_periods=[1, 5],
            combination_types=["single"],
            max_candidates_per_run=200,
            min_ic=0.01,
            min_ic_ir=0.1,
            min_oos_ic=0.005,
            max_adjusted_pvalue=0.5,
            max_decay_rate=0.99,
            n_folds=3,
        )
        scanner = PatternScanner(config)

        n = 500
        # Create a feature that is correlated with forward returns
        signal = _RNG.randn(n).astype(np.float64)
        forward_returns = signal * 0.01 + _RNG.randn(n).astype(np.float64) * 0.005

        features = np.column_stack([
            signal,
            _RNG.randn(n).astype(np.float64),
        ])
        feature_names = ["alpha_signal", "noise"]

        results = scanner.scan(features, feature_names, forward_returns)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_scan_summary(self):
        from trading_algo.quant_core.discovery.pattern_scanner import (
            PatternScanner,
            ScannerConfig,
        )
        config = ScannerConfig(
            min_observations=100,
            lookback_windows=[10],
            holding_periods=[1],
            combination_types=["single"],
            max_candidates_per_run=50,
        )
        scanner = PatternScanner(config)

        n = 300
        features = _RNG.randn(n, 2).astype(np.float64)
        feature_names = ["f1", "f2"]
        forward_returns = _RNG.randn(n).astype(np.float64) * 0.01

        scanner.scan(features, feature_names, forward_returns)
        summary = scanner.get_scan_summary()
        assert "total_candidates" in summary
        assert "stage_counts" in summary
        assert summary["total_candidates"] > 0


# =========================================================================
# 7. NonlinearSignalCombiner
# =========================================================================


class TestNonlinearSignalCombiner:
    """Tests for trading_algo.quant_core.ml.nonlinear_combiner."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
            NonlinearCombinerConfig,
        )
        combiner = NonlinearSignalCombiner()
        assert combiner.config.model_type == "gradient_boosting"
        assert combiner.config.n_estimators == 200
        assert combiner._fitted is False

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
            NonlinearCombinerConfig,
        )
        config = NonlinearCombinerConfig(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            include_interaction_features=False,
        )
        combiner = NonlinearSignalCombiner(config)
        assert combiner.config.n_estimators == 50
        assert combiner.config.include_interaction_features is False

    def test_predict_before_fit_returns_zeros(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
        )
        combiner = NonlinearSignalCombiner()
        X = _RNG.randn(10, 3).astype(np.float64)
        pred = combiner.predict(X)
        assert pred.shape == (10,)
        np.testing.assert_array_equal(pred, np.zeros(10))

    def test_fit_and_predict(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
            NonlinearCombinerConfig,
        )
        config = NonlinearCombinerConfig(
            n_estimators=10,
            max_depth=2,
            include_interaction_features=False,
            include_regime_features=False,
            include_time_features=False,
            fallback_to_linear=True,
        )
        combiner = NonlinearSignalCombiner(config)

        n = 200
        X = _RNG.randn(n, 5).astype(np.float64)
        y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + _RNG.randn(n) * 0.1).astype(np.float64)

        combiner.fit(X, y)
        assert combiner._fitted is True

        pred = combiner.predict(X[:10])
        assert pred.shape == (10,)
        # Predictions should be clipped to [-1, 1] in "direction" mode
        assert np.all(pred >= -1.0)
        assert np.all(pred <= 1.0)

    def test_fit_raises_on_dimension_mismatch(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
        )
        combiner = NonlinearSignalCombiner()
        X = _RNG.randn(100, 3).astype(np.float64)
        y = _RNG.randn(50).astype(np.float64)  # wrong length

        with pytest.raises(ValueError, match="rows"):
            combiner.fit(X, y)

    def test_evaluate_returns_metrics(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
            NonlinearCombinerConfig,
        )
        config = NonlinearCombinerConfig(
            n_estimators=10,
            max_depth=2,
            include_interaction_features=False,
            include_regime_features=False,
            include_time_features=False,
            fallback_to_linear=True,
        )
        combiner = NonlinearSignalCombiner(config)

        n = 200
        X = _RNG.randn(n, 3).astype(np.float64)
        y = (X[:, 0] * 0.5 + _RNG.randn(n) * 0.1).astype(np.float64)
        combiner.fit(X, y)

        metrics = combiner.evaluate(X[:50], y[:50])
        assert "IC" in metrics
        assert "hit_rate" in metrics
        assert "MSE" in metrics

    def test_ridge_fallback_model(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            _RidgeFallback,
        )
        model = _RidgeFallback(alpha=1.0)
        X = _RNG.randn(100, 3).astype(np.float64)
        y = (X[:, 0] + _RNG.randn(100) * 0.1).astype(np.float64)
        model.fit(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5,)
        assert model.feature_importances_ is not None

    def test_get_feature_importance(self):
        from trading_algo.quant_core.ml.nonlinear_combiner import (
            NonlinearSignalCombiner,
            NonlinearCombinerConfig,
        )
        config = NonlinearCombinerConfig(
            n_estimators=10,
            max_depth=2,
            include_interaction_features=False,
            include_regime_features=False,
            include_time_features=False,
            fallback_to_linear=True,
        )
        combiner = NonlinearSignalCombiner(config)

        n = 200
        X = _RNG.randn(n, 3).astype(np.float64)
        y = (X[:, 0] * 0.5 + _RNG.randn(n) * 0.1).astype(np.float64)
        combiner.fit(X, y)

        importance = combiner.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0


# =========================================================================
# 8. AlphaDecayMonitor
# =========================================================================


class TestAlphaDecayMonitor:
    """Tests for trading_algo.quant_core.discovery.alpha_monitor."""

    def test_import_and_creation(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            MonitorConfig,
        )
        monitor = AlphaDecayMonitor()
        assert monitor._config.ic_window_short == 30
        assert monitor._config.min_observations == 60

    def test_creation_with_custom_config(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            MonitorConfig,
        )
        config = MonitorConfig(
            ic_window_short=20,
            warning_ic_threshold=0.01,
            min_observations=30,
        )
        monitor = AlphaDecayMonitor(config)
        assert monitor._config.ic_window_short == 20
        assert monitor._config.warning_ic_threshold == 0.01

    def test_register_signal(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("momentum_20d", initial_ic=0.05)
        assert "momentum_20d" in monitor._signals

    def test_register_duplicate_raises(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("sig_a")
        with pytest.raises(ValueError, match="already registered"):
            monitor.register_signal("sig_a")

    def test_update_returns_health(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            MonitorConfig,
            STATUS_HEALTHY,
        )
        config = MonitorConfig(min_observations=5)
        monitor = AlphaDecayMonitor(config)
        monitor.register_signal("test_signal", initial_ic=0.05)

        ts = datetime.datetime(2025, 1, 1)
        for i in range(10):
            health = monitor.update(
                "test_signal",
                signal_value=float(_RNG.randn()),
                realized_return=float(_RNG.randn() * 0.01),
                timestamp=ts + datetime.timedelta(days=i),
            )

        assert hasattr(health, "status")
        assert hasattr(health, "current_ic")
        assert hasattr(health, "rolling_ic_30d")

    def test_health_status_starts_healthy(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            STATUS_HEALTHY,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("test_signal")
        health = monitor.get_health("test_signal")
        assert health.status == STATUS_HEALTHY

    def test_health_transitions_to_degraded_with_bad_ic(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            MonitorConfig,
            STATUS_DEGRADED,
        )
        config = MonitorConfig(
            min_observations=10,
            ic_window_short=10,
            warning_ic_threshold=0.005,
            degraded_ic_threshold=0.0,
            min_sharpe_90d=-999.0,  # disable sharpe retirement
            max_drawdown_pct=999.0,  # disable dd retirement
            max_signal_age_days=99999,  # disable age retirement
        )
        monitor = AlphaDecayMonitor(config)
        monitor.register_signal("bad_signal", initial_ic=0.05)

        ts = datetime.datetime(2025, 1, 1)
        # Feed strongly anti-correlated data: negative signal, positive return
        for i in range(100):
            monitor.update(
                "bad_signal",
                signal_value=1.0,
                realized_return=-0.01,
                timestamp=ts + datetime.timedelta(days=i),
            )

        health = monitor.get_health("bad_signal")
        # With consistently negative IC, status should be WARNING or DEGRADED
        assert health.status in ("warning", "degraded", "retired")

    def test_retire_signal(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            STATUS_RETIRED,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("old_signal")
        monitor.retire_signal("old_signal", reason="Manual retirement")
        health = monitor.get_health("old_signal")
        assert health.status == STATUS_RETIRED

    def test_get_portfolio_health_score(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("sig1")
        monitor.register_signal("sig2")
        score = monitor.get_portfolio_health_score()
        # Both healthy = 100.0
        assert score == pytest.approx(100.0)

    def test_get_portfolio_health_score_mixed(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("healthy_sig")
        monitor.register_signal("retired_sig")
        monitor.retire_signal("retired_sig", reason="test")
        score = monitor.get_portfolio_health_score()
        # (100 + 0) / 2 = 50.0
        assert score == pytest.approx(50.0)

    def test_estimate_decay_with_insufficient_data(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            MonitorConfig,
        )
        config = MonitorConfig(min_observations=60)
        monitor = AlphaDecayMonitor(config)
        monitor.register_signal("new_signal", initial_ic=0.05)

        # Only 5 observations -- insufficient
        ts = datetime.datetime(2025, 1, 1)
        for i in range(5):
            monitor.update(
                "new_signal",
                signal_value=float(_RNG.randn()),
                realized_return=float(_RNG.randn() * 0.01),
                timestamp=ts + datetime.timedelta(days=i),
            )

        metrics = monitor.estimate_decay("new_signal")
        assert metrics.remaining_alpha_pct == 100.0
        assert metrics.half_life_days == float("inf")

    def test_batch_update(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
        )
        monitor = AlphaDecayMonitor()
        monitor.register_signal("sig_a")
        monitor.register_signal("sig_b")

        ts = datetime.datetime(2025, 1, 1)
        results = monitor.batch_update(
            {"sig_a": 0.5, "sig_b": -0.3},
            realized_return=0.01,
            timestamp=ts,
        )
        assert "sig_a" in results
        assert "sig_b" in results

    def test_get_decay_report(self):
        from trading_algo.quant_core.discovery.alpha_monitor import (
            AlphaDecayMonitor,
            MonitorConfig,
        )
        config = MonitorConfig(min_observations=5)
        monitor = AlphaDecayMonitor(config)
        monitor.register_signal("test_sig", initial_ic=0.05)

        ts = datetime.datetime(2025, 1, 1)
        for i in range(10):
            monitor.update(
                "test_sig",
                signal_value=float(_RNG.randn()),
                realized_return=float(_RNG.randn() * 0.01),
                timestamp=ts + datetime.timedelta(days=i),
            )

        report = monitor.get_decay_report()
        assert "signals" in report
        assert "portfolio" in report
        assert "recommendations" in report
        assert "test_sig" in report["signals"]
