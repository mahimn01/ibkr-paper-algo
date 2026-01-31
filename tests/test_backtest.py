"""
Tests for the backtest system.

Run with: python -m pytest tests/test_backtest.py -v
"""

import pytest
from datetime import date, datetime, timedelta
from typing import Dict, List, Any

from trading_algo.backtest_v2 import (
    BacktestEngine,
    BacktestConfig,
    BacktestExporter,
    DataProvider,
    DataRequest,
    Bar,
    BacktestResults,
)


class MockStrategy:
    """Mock strategy for testing the backtest engine."""

    def __init__(self):
        self.asset_states: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self._bar_count = 0
        self._signal_every_n = 20  # Generate signal every N bars

    def update_asset(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Update the strategy with a new bar."""
        if symbol not in self.asset_states:
            self.asset_states[symbol] = {
                'prices': [],
                'atr': 1.0,
                'vwap': close,
            }

        state = self.asset_states[symbol]
        state['prices'].append(close)
        state['vwap'] = close
        self._bar_count += 1

    def generate_signal(self, symbol: str, timestamp: datetime) -> Any:
        """Generate a trading signal."""
        state = self.asset_states.get(symbol)
        if not state or len(state['prices']) < 10:
            return None

        # Simple alternating buy/sell strategy for testing
        if self._bar_count % self._signal_every_n == 0:
            if symbol not in self.positions:
                return MockSignal(
                    action='buy',
                    symbol=symbol,
                    entry_price=state['prices'][-1],
                    stop_loss=state['prices'][-1] * 0.98,
                    take_profit=state['prices'][-1] * 1.04,
                    confidence=0.7,
                    reason="Test buy signal",
                    edge_votes={},
                    market_regime=MockRegime(),
                )
            else:
                return MockSignal(
                    action='sell',
                    symbol=symbol,
                    entry_price=state['prices'][-1],
                    confidence=0.7,
                    reason="Test sell signal",
                )

        return None

    def clear_positions(self) -> None:
        """Clear all positions."""
        self.positions.clear()


class MockSignal:
    """Mock signal for testing."""

    def __init__(
        self,
        action: str,
        symbol: str,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        confidence: float = 0.5,
        reason: str = "",
        edge_votes: dict = None,
        market_regime: Any = None,
    ):
        self.action = action
        self.symbol = symbol
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.reason = reason
        self.edge_votes = edge_votes or {}
        self.market_regime = market_regime


class MockRegime:
    """Mock market regime."""
    name = "TRENDING"


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_config_creation(self):
        """Test that BacktestConfig can be created with required fields."""
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY", "AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert config.strategy_name == "TestStrategy"
        assert config.symbols == ["SPY", "AAPL"]
        assert config.initial_capital == 100_000  # Default
        assert config.commission_per_share == 0.005  # Default

    def test_config_with_custom_values(self):
        """Test BacktestConfig with custom capital and costs."""
        config = BacktestConfig(
            strategy_name="CustomStrategy",
            symbols=["GOOGL"],
            initial_capital=50_000,
            commission_per_share=0.01,
            slippage_pct=0.001,
        )

        assert config.initial_capital == 50_000
        assert config.commission_per_share == 0.01
        assert config.slippage_pct == 0.001


class TestDataProvider:
    """Tests for DataProvider."""

    def test_generate_sample_data(self):
        """Test sample data generation."""
        provider = DataProvider()

        bars = provider.generate_sample_data(
            symbol="TEST",
            start_date=date(2024, 1, 2),  # Skip weekend
            end_date=date(2024, 1, 5),
            bar_size="5 mins",
        )

        assert len(bars) > 0
        assert all(isinstance(b, Bar) for b in bars)
        assert all(b.high >= b.low for b in bars)
        assert all(b.high >= b.open for b in bars)
        assert all(b.high >= b.close for b in bars)
        assert all(b.low <= b.open for b in bars)
        assert all(b.low <= b.close for b in bars)

    def test_bar_properties(self):
        """Test Bar dataclass properties."""
        bar = Bar(
            timestamp=datetime(2024, 1, 2, 10, 0),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=10000,
        )

        assert bar.typical_price == (102.0 + 99.0 + 101.0) / 3
        assert bar.range == 3.0  # 102 - 99
        assert bar.body == 1.0  # |101 - 100|
        assert bar.is_bullish == True  # close > open


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_engine_creation(self):
        """Test that BacktestEngine can be created."""
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        engine = BacktestEngine(config)

        assert engine.config == config
        assert engine.equity == config.initial_capital
        assert engine.cash == config.initial_capital
        assert len(engine.positions) == 0

    def test_engine_run_with_sample_data(self):
        """Test running backtest with sample data."""
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 10),
            initial_capital=100_000,
        )

        # Generate sample data
        provider = DataProvider()
        data = {
            "SPY": provider.generate_sample_data(
                "SPY",
                config.start_date,
                config.end_date,
                config.bar_size,
            )
        }

        # Create strategy and engine
        strategy = MockStrategy()
        engine = BacktestEngine(config)

        # Run backtest
        results = engine.run(strategy, data)

        # Verify results
        assert isinstance(results, BacktestResults)
        assert results.config == config
        assert results.metrics is not None
        assert results.metrics.initial_capital == 100_000
        assert len(results.equity_curve) > 0

    def test_engine_tracks_equity(self):
        """Test that engine tracks equity over time."""
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            initial_capital=50_000,
        )

        provider = DataProvider()
        data = {
            "SPY": provider.generate_sample_data(
                "SPY",
                config.start_date,
                config.end_date,
            )
        }

        strategy = MockStrategy()
        engine = BacktestEngine(config)
        results = engine.run(strategy, data)

        # Check equity curve
        assert len(results.equity_curve) > 0
        assert results.equity_curve[0].equity > 0

        # Final equity should be tracked
        assert results.metrics.final_capital > 0

    def test_engine_with_progress_callback(self):
        """Test that progress callback is called."""
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        provider = DataProvider()
        data = {
            "SPY": provider.generate_sample_data(
                "SPY",
                config.start_date,
                config.end_date,
            )
        }

        progress_calls = []

        def progress_callback(pct, msg):
            progress_calls.append((pct, msg))

        strategy = MockStrategy()
        engine = BacktestEngine(config)
        engine.run(strategy, data, progress_callback)

        # Should have received progress calls
        assert len(progress_calls) > 0
        # Last call should be 100%
        assert progress_calls[-1][0] == 1.0


class TestBacktestMetrics:
    """Tests for backtest metrics calculation."""

    def test_metrics_calculated(self):
        """Test that metrics are calculated correctly."""
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 31),
            initial_capital=100_000,
        )

        provider = DataProvider()
        data = {
            "SPY": provider.generate_sample_data(
                "SPY",
                config.start_date,
                config.end_date,
            )
        }

        strategy = MockStrategy()
        strategy._signal_every_n = 10  # More frequent signals for testing

        engine = BacktestEngine(config)
        results = engine.run(strategy, data)

        m = results.metrics

        # Basic metrics should exist
        assert m.initial_capital == 100_000
        assert m.trading_days >= 0
        assert m.total_trades >= 0

        # If trades occurred, check trade stats
        if m.total_trades > 0:
            assert m.winning_trades + m.losing_trades + m.break_even_trades == m.total_trades
            assert 0 <= m.win_rate <= 100


class TestBacktestExporter:
    """Tests for BacktestExporter."""

    def test_exporter_creation(self):
        """Test that BacktestExporter can be created."""
        exporter = BacktestExporter()
        assert exporter.output_dir.name == "backtest_results"

    def test_export_creates_files(self, tmp_path):
        """Test that export creates expected files."""
        # Run a quick backtest
        config = BacktestConfig(
            strategy_name="TestStrategy",
            symbols=["SPY"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            initial_capital=100_000,
        )

        provider = DataProvider()
        data = {
            "SPY": provider.generate_sample_data(
                "SPY",
                config.start_date,
                config.end_date,
            )
        }

        strategy = MockStrategy()
        engine = BacktestEngine(config)
        results = engine.run(strategy, data)

        # Export to temp directory
        exporter = BacktestExporter(output_dir=tmp_path)
        export_path = exporter.export(results, folder_name="test_export")

        # Check files exist
        assert (export_path / "summary.json").exists()
        assert (export_path / "config.json").exists()
        assert (export_path / "metrics.json").exists()
        assert (export_path / "equity_curve.csv").exists()
        assert (export_path / "report.html").exists()


class TestIntegration:
    """Integration tests for the full backtest workflow."""

    def test_full_backtest_workflow(self):
        """Test complete backtest workflow from config to results."""
        # 1. Create config
        config = BacktestConfig(
            strategy_name="IntegrationTestStrategy",
            symbols=["AAPL", "GOOGL"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 15),
            initial_capital=100_000,
            commission_per_share=0.005,
            slippage_pct=0.0005,
        )

        # 2. Generate data
        provider = DataProvider()
        data = {}
        for symbol in config.symbols:
            data[symbol] = provider.generate_sample_data(
                symbol,
                config.start_date,
                config.end_date,
                config.bar_size,
            )

        # 3. Create strategy
        strategy = MockStrategy()
        strategy._signal_every_n = 15

        # 4. Run backtest
        engine = BacktestEngine(config)
        results = engine.run(strategy, data)

        # 5. Verify results structure
        assert results.run_id.startswith("BT-")
        assert results.config == config
        assert results.metrics is not None
        assert len(results.equity_curve) > 0
        assert len(results.daily_results) > 0

        # 6. Verify metrics
        m = results.metrics
        assert m.start_date == config.start_date
        assert m.end_date == config.end_date
        assert m.initial_capital == config.initial_capital

        print(f"\nIntegration Test Results:")
        print(f"  Total Return: ${m.total_return:,.2f} ({m.total_return_pct:.2f}%)")
        print(f"  Total Trades: {m.total_trades}")
        print(f"  Win Rate: {m.win_rate:.1f}%")
        print(f"  Sharpe Ratio: {m.sharpe_ratio:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
