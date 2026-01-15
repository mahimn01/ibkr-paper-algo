"""Tests for RAT Backtesting infrastructure."""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from trading_algo.rat.config import RATConfig
from trading_algo.rat.backtest.backtester import (
    RATBacktester,
    BacktestConfig,
    BacktestResult,
    SimulatedBroker,
    SimulatedPosition,
)
from trading_algo.rat.backtest.data_loader import (
    Bar,
    CSVLoader,
    MultiSymbolLoader,
)
from trading_algo.rat.backtest.analytics import (
    PerformanceAnalytics,
    PerformanceMetrics,
    Trade,
)


class TestBar(unittest.TestCase):
    """Test Bar dataclass."""

    def test_bar_creation(self):
        """Test creating a Bar."""
        bar = Bar(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=149.0,
            high=151.0,
            low=148.5,
            close=150.0,
            volume=100000,
        )

        self.assertEqual(bar.symbol, "AAPL")
        self.assertAlmostEqual(bar.close, 150.0)

    def test_bar_typical_price(self):
        """Test typical price calculation."""
        bar = Bar(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=100,
            high=105,
            low=95,
            close=100,
            volume=1000,
        )

        self.assertAlmostEqual(bar.typical_price, 100.0)  # (105+95+100)/3

    def test_bar_range(self):
        """Test range calculation."""
        bar = Bar(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=100,
            high=110,
            low=90,
            close=105,
            volume=1000,
        )

        self.assertAlmostEqual(bar.range, 20.0)

    def test_bar_body(self):
        """Test body calculation."""
        bar = Bar(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=100,
            high=110,
            low=90,
            close=108,
            volume=1000,
        )

        self.assertAlmostEqual(bar.body, 8.0)  # close - open

    def test_bar_is_bullish(self):
        """Test bullish detection."""
        bullish_bar = Bar(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=100,
            high=110,
            low=99,
            close=108,
            volume=1000,
        )
        self.assertTrue(bullish_bar.is_bullish)

        bearish_bar = Bar(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=100,
            high=101,
            low=90,
            close=92,
            volume=1000,
        )
        self.assertFalse(bearish_bar.is_bullish)


class TestCSVLoader(unittest.TestCase):
    """Test CSVLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = CSVLoader(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_csv_file(self):
        """Test loading data from CSV."""
        # Create test CSV
        csv_content = """date,open,high,low,close,volume
2023-01-01,100,105,95,102,10000
2023-01-02,102,108,100,106,12000
2023-01-03,106,110,105,108,11000
"""
        csv_path = Path(self.temp_dir) / "AAPL.csv"
        csv_path.write_text(csv_content)

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)

        bars = self.loader.load("AAPL", start, end)

        self.assertEqual(len(bars), 3)
        self.assertAlmostEqual(bars[0].open, 100)
        self.assertAlmostEqual(bars[0].close, 102)

    def test_load_csv_date_filtering(self):
        """Test CSV loading with date filtering."""
        csv_content = """date,open,high,low,close,volume
2023-01-01,100,105,95,102,10000
2023-01-02,102,108,100,106,12000
2023-01-03,106,110,105,108,11000
2023-01-04,108,112,107,110,10500
"""
        csv_path = Path(self.temp_dir) / "AAPL.csv"
        csv_path.write_text(csv_content)

        # Only request 2 days
        start = datetime(2023, 1, 2)
        end = datetime(2023, 1, 3)

        bars = self.loader.load("AAPL", start, end)

        self.assertEqual(len(bars), 2)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load(
                "NONEXISTENT",
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
            )

    def test_stream_csv_file(self):
        """Test streaming data from CSV."""
        csv_content = """date,open,high,low,close,volume
2023-01-01,100,105,95,102,10000
2023-01-02,102,108,100,106,12000
"""
        csv_path = Path(self.temp_dir) / "AAPL.csv"
        csv_path.write_text(csv_content)

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)

        bars = list(self.loader.stream("AAPL", start, end))

        self.assertEqual(len(bars), 2)


class TestSimulatedBroker(unittest.TestCase):
    """Test SimulatedBroker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            initial_capital=100000,
            commission_per_share=0.01,
            slippage_pct=0.0005,
        )
        self.broker = SimulatedBroker(self.config)
        self.base_time = datetime(2023, 1, 1, 10, 0)

    def test_initial_state(self):
        """Test initial broker state."""
        self.assertAlmostEqual(self.broker.cash, 100000)
        self.assertAlmostEqual(self.broker.equity, 100000)
        self.assertEqual(len(self.broker.positions), 0)

    def test_update_prices(self):
        """Test price update."""
        self.broker.positions["AAPL"] = SimulatedPosition(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            entry_time=self.base_time,
            side="long",
        )

        self.broker.update_prices({"AAPL": 155.0})

        # Should have unrealized profit
        self.assertGreater(
            self.broker.positions["AAPL"].unrealized_pnl, 0
        )

    def test_place_buy_order(self):
        """Test placing a buy order."""
        trade = self.broker.place_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            timestamp=self.base_time,
        )

        # Order opens position, no trade returned
        self.assertIsNone(trade)
        self.assertIn("AAPL", self.broker.positions)
        self.assertEqual(self.broker.positions["AAPL"].quantity, 100)

    def test_place_sell_order_close_position(self):
        """Test selling to close position."""
        # First buy
        self.broker.place_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            timestamp=self.base_time,
        )

        # Then sell
        trade = self.broker.place_order(
            symbol="AAPL",
            side="SELL",
            quantity=100,
            price=155.0,
            timestamp=self.base_time + timedelta(hours=1),
        )

        self.assertIsNotNone(trade)
        self.assertGreater(trade.pnl, 0)  # Should be profitable
        self.assertNotIn("AAPL", self.broker.positions)

    def test_insufficient_cash(self):
        """Test order rejected for insufficient cash."""
        # Try to buy way too much
        trade = self.broker.place_order(
            symbol="AAPL",
            side="BUY",
            quantity=10000,  # Would cost ~1.5M
            price=150.0,
            timestamp=self.base_time,
        )

        self.assertIsNone(trade)

    def test_commission_applied(self):
        """Test that commission is applied."""
        initial_cash = self.broker.cash

        self.broker.place_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=100.0,
            timestamp=self.base_time,
        )

        # Cash should be reduced by cost + commission
        expected_cost = 100 * 100 * (1 + self.config.slippage_pct)
        expected_commission = max(self.config.commission_minimum, 100 * self.config.commission_per_share)

        self.assertLess(self.broker.cash, initial_cash - expected_cost)

    def test_close_all_positions(self):
        """Test closing all positions."""
        # Open positions
        self.broker.place_order("AAPL", "BUY", 50, 150.0, self.base_time)
        self.broker.place_order("MSFT", "BUY", 30, 300.0, self.base_time)

        trades = self.broker.close_all_positions(
            {"AAPL": 155.0, "MSFT": 310.0},
            self.base_time + timedelta(hours=1),
        )

        self.assertEqual(len(trades), 2)
        self.assertEqual(len(self.broker.positions), 0)


class TestPerformanceAnalytics(unittest.TestCase):
    """Test PerformanceAnalytics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analytics = PerformanceAnalytics(
            initial_capital=100000,
            risk_free_rate=0.02,
        )
        self.base_time = datetime(2023, 1, 1)

    def test_record_equity(self):
        """Test recording equity points."""
        equities = [100000, 101000, 100500, 102000, 101500]

        for i, equity in enumerate(equities):
            self.analytics.record_equity(
                self.base_time + timedelta(days=i),
                equity,
            )

        self.assertEqual(len(self.analytics._equity_curve), 5)

    def test_record_trade(self):
        """Test recording trades."""
        trade = Trade(
            symbol="AAPL",
            side="long",
            entry_time=self.base_time,
            exit_time=self.base_time + timedelta(hours=4),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500,
            pnl_pct=0.033,
        )

        self.analytics.record_trade(trade)
        self.assertEqual(len(self.analytics._trades), 1)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Record some equity and trades
        equities = [100000 + i * 100 for i in range(100)]

        for i, equity in enumerate(equities):
            self.analytics.record_equity(
                self.base_time + timedelta(days=i),
                equity,
            )

        for i in range(10):
            trade = Trade(
                symbol="AAPL",
                side="long",
                entry_time=self.base_time + timedelta(days=i*10),
                exit_time=self.base_time + timedelta(days=i*10+5),
                entry_price=150.0,
                exit_price=152.0 if i % 3 != 0 else 148.0,
                quantity=100,
                pnl=200 if i % 3 != 0 else -200,
                pnl_pct=0.013 if i % 3 != 0 else -0.013,
            )
            self.analytics.record_trade(trade)

        metrics = self.analytics.calculate_metrics()

        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.total_return, 0)
        self.assertEqual(metrics.total_trades, 10)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Add positive returns
        for i in range(100):
            self.analytics.record_equity(
                self.base_time + timedelta(days=i),
                100000 * (1.001 ** i),
            )

        metrics = self.analytics.calculate_metrics()

        # With consistent positive returns, Sharpe should be positive
        self.assertGreater(metrics.sharpe_ratio, 0)

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create a drawdown scenario
        equities = [
            100000, 105000, 110000,  # Rising
            105000, 100000, 95000,   # Drawdown
            100000, 105000,          # Recovery
        ]

        for i, equity in enumerate(equities):
            self.analytics.record_equity(
                self.base_time + timedelta(days=i),
                equity,
            )

        metrics = self.analytics.calculate_metrics()

        # Max drawdown should be (110000 - 95000) / 110000 = ~13.6%
        self.assertGreater(metrics.max_drawdown, 0.1)
        self.assertLess(metrics.max_drawdown, 0.2)

    def test_generate_report(self):
        """Test report generation."""
        # Add some data
        for i in range(50):
            self.analytics.record_equity(
                self.base_time + timedelta(days=i),
                100000 + i * 100,
            )

        report = self.analytics.generate_report()

        self.assertIsInstance(report, str)
        self.assertIn("RETURNS", report)
        self.assertIn("RISK", report)

    def test_reset(self):
        """Test analytics reset."""
        # Add some data
        self.analytics.record_equity(datetime.now(), 100000)

        self.analytics.reset()

        self.assertEqual(len(self.analytics._equity_curve), 0)
        self.assertEqual(len(self.analytics._trades), 0)


class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()

        self.assertAlmostEqual(config.initial_capital, 100000)
        self.assertAlmostEqual(config.max_position_pct, 0.25)

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=50000,
            commission_per_share=0.001,
            slippage_pct=0.001,
            max_position_pct=0.1,
        )

        self.assertAlmostEqual(config.initial_capital, 50000)
        self.assertAlmostEqual(config.max_position_pct, 0.1)


class TestRATBacktester(unittest.TestCase):
    """Test RATBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.rat_config = RATConfig.from_env()
        self.bt_config = BacktestConfig(initial_capital=100000)
        self.loader = CSVLoader(data_dir=self.temp_dir)

        # Create test data
        csv_content = "date,open,high,low,close,volume\n"
        for i in range(100):
            date = (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            price = 100 + i * 0.5
            csv_content += f"{date},{price-1},{price+1},{price-2},{price},{10000}\n"

        csv_path = Path(self.temp_dir) / "AAPL.csv"
        csv_path.write_text(csv_content)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        backtester = RATBacktester(
            rat_config=self.rat_config,
            backtest_config=self.bt_config,
            data_loader=self.loader,
        )

        self.assertIsNotNone(backtester.broker)
        self.assertIsNotNone(backtester.analytics)

    def test_run_backtest(self):
        """Test running a backtest."""
        backtester = RATBacktester(
            rat_config=self.rat_config,
            backtest_config=self.bt_config,
            data_loader=self.loader,
        )

        result = backtester.run(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 4, 10),
        )

        self.assertIsInstance(result, BacktestResult)
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.report)

    def test_backtest_result_contents(self):
        """Test backtest result contains expected data."""
        backtester = RATBacktester(
            rat_config=self.rat_config,
            backtest_config=self.bt_config,
            data_loader=self.loader,
        )

        result = backtester.run(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 1),
        )

        self.assertGreater(len(result.equity_curve), 0)
        self.assertIn("alpha_health", result.engine_stats)


class TestTrade(unittest.TestCase):
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a Trade."""
        entry = datetime(2023, 1, 1, 10, 0)
        exit_ = datetime(2023, 1, 1, 14, 0)

        trade = Trade(
            symbol="AAPL",
            side="long",
            entry_time=entry,
            exit_time=exit_,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500,
            pnl_pct=0.033,
        )

        self.assertEqual(trade.symbol, "AAPL")
        self.assertAlmostEqual(trade.pnl, 500)
        self.assertEqual(trade.holding_period.total_seconds(), 4 * 3600)


if __name__ == "__main__":
    unittest.main()
