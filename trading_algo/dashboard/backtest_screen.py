"""
Backtest dashboard screen.

Provides an interactive TUI for running and analyzing backtests.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    Select,
    ProgressBar,
)
from textual.reactive import reactive

if TYPE_CHECKING:
    from ..backtest_v2 import BacktestResults


class BacktestConfigPanel(Static):
    """Panel for configuring backtest parameters."""

    DEFAULT_CSS = """
    BacktestConfigPanel {
        width: 100%;
        height: auto;
        border: solid #00d4ff;
        padding: 1;
        margin-bottom: 1;
    }

    BacktestConfigPanel .config-row {
        height: 3;
        margin-bottom: 1;
    }

    BacktestConfigPanel Label {
        width: 20;
        content-align: right middle;
        padding-right: 1;
    }

    BacktestConfigPanel Input {
        width: 30;
    }

    BacktestConfigPanel Select {
        width: 30;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("[bold cyan]Backtest Configuration[/]", classes="title")

        with Horizontal(classes="config-row"):
            yield Label("Symbols:")
            yield Input(
                placeholder="AAPL, MSFT, GOOGL",
                id="symbols-input",
                value="SPY",
            )

        with Horizontal(classes="config-row"):
            yield Label("Start Date:")
            yield Input(
                placeholder="YYYY-MM-DD",
                id="start-date-input",
                value=(date.today() - timedelta(days=365)).isoformat(),
            )

        with Horizontal(classes="config-row"):
            yield Label("End Date:")
            yield Input(
                placeholder="YYYY-MM-DD",
                id="end-date-input",
                value=date.today().isoformat(),
            )

        with Horizontal(classes="config-row"):
            yield Label("Initial Capital:")
            yield Input(
                placeholder="100000",
                id="capital-input",
                value="100000",
            )

        with Horizontal(classes="config-row"):
            yield Label("Bar Size:")
            yield Select(
                [
                    ("5 mins", "5 mins"),
                    ("1 min", "1 min"),
                    ("15 mins", "15 mins"),
                    ("30 mins", "30 mins"),
                    ("1 hour", "1 hour"),
                ],
                id="bar-size-select",
                value="5 mins",
            )

        with Horizontal(classes="config-row"):
            yield Button("Run Backtest", id="run-backtest-btn", variant="primary")
            yield Button("Export Results", id="export-btn", variant="default")
            yield Button("Clear", id="clear-btn", variant="warning")


class BacktestProgressPanel(Static):
    """Panel showing backtest progress."""

    DEFAULT_CSS = """
    BacktestProgressPanel {
        width: 100%;
        height: 5;
        border: solid #444;
        padding: 1;
        margin-bottom: 1;
    }

    BacktestProgressPanel ProgressBar {
        width: 100%;
    }
    """

    progress_pct = reactive(0.0)
    status_text = reactive("Ready")

    def compose(self) -> ComposeResult:
        yield Static(id="progress-status")
        yield ProgressBar(id="progress-bar", total=100, show_eta=False)

    def watch_progress_pct(self, value: float) -> None:
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.progress = value * 100

    def watch_status_text(self, value: str) -> None:
        status = self.query_one("#progress-status", Static)
        status.update(f"[cyan]{value}[/]")


class MetricsSummaryPanel(Static):
    """Panel showing key metrics summary."""

    DEFAULT_CSS = """
    MetricsSummaryPanel {
        width: 100%;
        height: auto;
        min-height: 12;
        border: solid #00d4ff;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: Optional[BacktestResults] = None

    def update_results(self, results: BacktestResults) -> None:
        self._results = results
        self.refresh()

    def render(self) -> RenderableType:
        if not self._results:
            return Panel(
                "[dim]Run a backtest to see metrics[/]",
                title="[bold cyan]Performance Summary[/]",
                border_style="cyan",
            )

        m = self._results.metrics

        # Create a table with metrics
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        pnl_style = "green" if m.net_profit >= 0 else "red"
        ret_style = "green" if m.total_return_pct >= 0 else "red"

        table.add_row(
            "Total P&L",
            f"[{pnl_style}]${m.net_profit:,.2f}[/]",
            "Total Return",
            f"[{ret_style}]{m.total_return_pct:.2f}%[/]",
        )
        table.add_row(
            "Sharpe Ratio",
            f"{m.sharpe_ratio:.2f}",
            "Sortino Ratio",
            f"{m.sortino_ratio:.2f}",
        )
        table.add_row(
            "Max Drawdown",
            f"[red]-{m.max_drawdown_pct:.2f}%[/]",
            "Win Rate",
            f"{m.win_rate:.1f}%",
        )
        table.add_row(
            "Total Trades",
            f"{m.total_trades}",
            "Profit Factor",
            f"{m.profit_factor:.2f}",
        )
        table.add_row(
            "Avg Daily P&L",
            f"[{pnl_style}]${m.avg_daily_pnl:,.2f}[/]",
            "Expectancy",
            f"${m.expectancy:.2f}",
        )
        table.add_row(
            "Best Day",
            f"[green]${m.best_day:,.2f}[/]",
            "Worst Day",
            f"[red]${m.worst_day:,.2f}[/]",
        )

        return Panel(
            table,
            title="[bold cyan]Performance Summary[/]",
            border_style="cyan",
        )


class DailyPnLPanel(Static):
    """Panel showing daily P&L breakdown."""

    DEFAULT_CSS = """
    DailyPnLPanel {
        width: 100%;
        height: 100%;
        border: solid #00d4ff;
    }

    DailyPnLPanel DataTable {
        height: 100%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: Optional[BacktestResults] = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="daily-pnl-table")

    def on_mount(self) -> None:
        table = self.query_one("#daily-pnl-table", DataTable)
        table.add_columns("Date", "P&L", "Return", "Trades", "Win Rate", "Drawdown")

    def update_results(self, results: BacktestResults) -> None:
        self._results = results
        table = self.query_one("#daily-pnl-table", DataTable)
        table.clear()

        for day in reversed(results.daily_results[-50:]):  # Last 50 days
            total_trades = day.trades_won + day.trades_lost
            win_rate = day.trades_won / total_trades * 100 if total_trades > 0 else 0
            pnl_style = "green" if day.net_pnl >= 0 else "red"

            table.add_row(
                day.date.isoformat(),
                Text(f"${day.net_pnl:,.2f}", style=pnl_style),
                Text(f"{day.return_pct:.2f}%", style=pnl_style),
                str(total_trades),
                f"{win_rate:.1f}%",
                f"-{day.max_drawdown_pct:.2f}%",
            )


class TradesPanel(Static):
    """Panel showing trade history."""

    DEFAULT_CSS = """
    TradesPanel {
        width: 100%;
        height: 100%;
        border: solid #00d4ff;
    }

    TradesPanel DataTable {
        height: 100%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: Optional[BacktestResults] = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="trades-table")

    def on_mount(self) -> None:
        table = self.query_one("#trades-table", DataTable)
        table.add_columns(
            "Symbol", "Dir", "Entry", "Exit", "Entry$", "Exit$", "P&L", "P&L%"
        )

    def update_results(self, results: BacktestResults) -> None:
        self._results = results
        table = self.query_one("#trades-table", DataTable)
        table.clear()

        for trade in reversed(results.trades[-100:]):  # Last 100 trades
            pnl = trade.net_pnl
            pnl_pct = trade.pnl_percent
            pnl_style = "green" if pnl >= 0 else "red"
            dir_style = "green" if trade.direction == "LONG" else "red"

            entry_time = trade.entry_time.strftime("%m/%d %H:%M") if trade.entry_time else ""
            exit_time = trade.exit_time.strftime("%m/%d %H:%M") if trade.exit_time else ""

            table.add_row(
                trade.symbol,
                Text(trade.direction[:1], style=dir_style),
                entry_time,
                exit_time,
                f"${trade.entry_price:.2f}",
                f"${trade.exit_price:.2f}" if trade.exit_price else "-",
                Text(f"${pnl:.2f}", style=pnl_style),
                Text(f"{pnl_pct:.2f}%", style=pnl_style),
            )


class EquityCurvePanel(Static):
    """Panel showing text-based equity curve."""

    DEFAULT_CSS = """
    EquityCurvePanel {
        width: 100%;
        height: 12;
        border: solid #00d4ff;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: Optional[BacktestResults] = None

    def update_results(self, results: BacktestResults) -> None:
        self._results = results
        self.refresh()

    def render(self) -> RenderableType:
        if not self._results or not self._results.equity_curve:
            return Panel(
                "[dim]Run a backtest to see equity curve[/]",
                title="[bold cyan]Equity Curve[/]",
                border_style="cyan",
            )

        # Create ASCII chart
        points = self._results.equity_curve
        if len(points) > 60:
            # Sample points
            step = len(points) // 60
            sampled = [points[i] for i in range(0, len(points), step)]
        else:
            sampled = points

        equity_values = [p.equity for p in sampled]
        min_eq = min(equity_values)
        max_eq = max(equity_values)
        range_eq = max_eq - min_eq if max_eq != min_eq else 1

        # Create 8 rows of chart
        height = 6
        chart_lines = []
        for row in range(height):
            threshold = max_eq - (row / (height - 1)) * range_eq
            line = ""
            for eq in equity_values:
                if eq >= threshold:
                    line += "█"
                else:
                    line += " "
            chart_lines.append(line)

        # Add axis labels
        chart = f"${max_eq:,.0f} ┤{chart_lines[0]}\n"
        for i, line in enumerate(chart_lines[1:-1], 1):
            chart += f"         │{line}\n"
        chart += f"${min_eq:,.0f} ┤{chart_lines[-1]}"

        initial = self._results.config.initial_capital
        final = equity_values[-1]
        pnl = final - initial
        pnl_style = "green" if pnl >= 0 else "red"

        return Panel(
            f"[{pnl_style}]{chart}[/]\n"
            f"[dim]Start: ${initial:,.0f} → End: ${final:,.0f} ({pnl/initial*100:+.2f}%)[/]",
            title="[bold cyan]Equity Curve[/]",
            border_style="cyan",
        )


class BacktestScreen(Screen):
    """
    Full-screen backtest analysis dashboard.

    Provides:
    - Configuration panel for setting backtest parameters
    - Progress indicator during backtest
    - Metrics summary
    - Daily P&L breakdown
    - Trade history
    - Equity curve visualization
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("r", "run_backtest", "Run Backtest"),
        Binding("e", "export_results", "Export"),
        Binding("c", "clear_results", "Clear"),
    ]

    DEFAULT_CSS = """
    BacktestScreen {
        background: #1a1a2e;
    }

    BacktestScreen #main-container {
        padding: 1;
    }

    BacktestScreen #left-panel {
        width: 50%;
        height: 100%;
        padding-right: 1;
    }

    BacktestScreen #right-panel {
        width: 50%;
        height: 100%;
    }

    BacktestScreen .title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        run_backtest_callback: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._run_backtest = run_backtest_callback
        self._results: Optional[BacktestResults] = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield BacktestConfigPanel(id="config-panel")
                yield BacktestProgressPanel(id="progress-panel")
                yield MetricsSummaryPanel(id="metrics-panel")
                yield EquityCurvePanel(id="equity-panel")

            with Vertical(id="right-panel"):
                with Container(id="daily-container"):
                    yield Static("[bold cyan]Daily P&L[/]", classes="title")
                    yield DailyPnLPanel(id="daily-panel")
                with Container(id="trades-container"):
                    yield Static("[bold cyan]Trades[/]", classes="title")
                    yield TradesPanel(id="trades-panel")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-backtest-btn":
            self.action_run_backtest()
        elif event.button.id == "export-btn":
            self.action_export_results()
        elif event.button.id == "clear-btn":
            self.action_clear_results()

    def action_run_backtest(self) -> None:
        """Run the backtest with current configuration."""
        if not self._run_backtest:
            self.notify("No backtest callback configured", severity="error")
            return

        # Get configuration from inputs
        try:
            symbols_input = self.query_one("#symbols-input", Input)
            start_input = self.query_one("#start-date-input", Input)
            end_input = self.query_one("#end-date-input", Input)
            capital_input = self.query_one("#capital-input", Input)
            bar_select = self.query_one("#bar-size-select", Select)

            symbols = [s.strip() for s in symbols_input.value.split(",")]
            start_date = date.fromisoformat(start_input.value)
            end_date = date.fromisoformat(end_input.value)
            initial_capital = float(capital_input.value)
            bar_size = bar_select.value

            config = {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "bar_size": bar_size,
            }

            # Update progress panel
            progress = self.query_one("#progress-panel", BacktestProgressPanel)
            progress.status_text = "Starting backtest..."
            progress.progress_pct = 0

            # Run backtest in background
            self.run_worker(
                self._run_backtest_async(config),
                name="backtest",
                exclusive=True,
            )

        except Exception as e:
            self.notify(f"Configuration error: {e}", severity="error")

    async def _run_backtest_async(self, config: dict) -> None:
        """Run backtest asynchronously."""
        progress = self.query_one("#progress-panel", BacktestProgressPanel)

        def update_progress(pct: float, status: str) -> None:
            progress.progress_pct = pct
            progress.status_text = status

        try:
            results = await self._run_backtest(config, update_progress)
            self._results = results
            self._update_displays()
            progress.status_text = "Backtest complete!"
            progress.progress_pct = 1.0
            self.notify("Backtest completed successfully!", severity="information")
        except Exception as e:
            progress.status_text = f"Error: {e}"
            self.notify(f"Backtest failed: {e}", severity="error")

    def _update_displays(self) -> None:
        """Update all display panels with results."""
        if not self._results:
            return

        metrics = self.query_one("#metrics-panel", MetricsSummaryPanel)
        metrics.update_results(self._results)

        equity = self.query_one("#equity-panel", EquityCurvePanel)
        equity.update_results(self._results)

        daily = self.query_one("#daily-panel", DailyPnLPanel)
        daily.update_results(self._results)

        trades = self.query_one("#trades-panel", TradesPanel)
        trades.update_results(self._results)

    def action_export_results(self) -> None:
        """Export backtest results."""
        if not self._results:
            self.notify("No results to export. Run a backtest first.", severity="warning")
            return

        try:
            from ..backtest_v2 import BacktestExporter

            exporter = BacktestExporter()
            export_path = exporter.export(self._results)
            self.notify(f"Exported to {export_path}", severity="information")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def action_clear_results(self) -> None:
        """Clear current results."""
        self._results = None

        # Reset all panels
        metrics = self.query_one("#metrics-panel", MetricsSummaryPanel)
        metrics._results = None
        metrics.refresh()

        equity = self.query_one("#equity-panel", EquityCurvePanel)
        equity._results = None
        equity.refresh()

        daily_table = self.query_one("#daily-pnl-table", DataTable)
        daily_table.clear()

        trades_table = self.query_one("#trades-table", DataTable)
        trades_table.clear()

        progress = self.query_one("#progress-panel", BacktestProgressPanel)
        progress.status_text = "Ready"
        progress.progress_pct = 0

        self.notify("Results cleared", severity="information")

    def set_results(self, results: BacktestResults) -> None:
        """Set results from external source."""
        self._results = results
        self._update_displays()
