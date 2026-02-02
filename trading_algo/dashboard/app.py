"""
Main Trading Dashboard Application.

A comprehensive terminal-based trading dashboard built with Textual.

Features:
- Real-time P&L tracking
- Open positions with stop/target levels
- Trade history (today/yesterday)
- Signal log from algorithm
- Activity log
- Keyboard controls (pause, quit, etc.)

Usage:
    from trading_algo.dashboard.app import TradingDashboard

    # Create and run dashboard
    dashboard = TradingDashboard()
    dashboard.run()

    # Or integrate with your algorithm
    async def main():
        dashboard = TradingDashboard()
        await dashboard.run_async()
"""

from typing import Any, Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Footer, Header
from textual.screen import Screen

from .store import DashboardStore, get_store
from .events import EventBus, get_event_bus, AlgorithmStatusEvent
from .models import AlgorithmStatus
from .widgets import (
    HeaderWidget,
    PnLWidget,
    PositionsWidget,
    TradesWidget,
    SignalsWidget,
    LogWidget,
)
from .backtest_screen import BacktestScreen


class HelpScreen(Screen):
    """Help screen showing keyboard shortcuts."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Close"),
        Binding("q", "pop_screen", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static("[bold]Trading Dashboard Help[/bold]\n", id="help-title"),
            Static(
                """
[bold]Keyboard Shortcuts:[/bold]

  [cyan]q[/cyan] / [cyan]Ctrl+C[/cyan]  - Quit dashboard
  [cyan]p[/cyan]            - Pause/Resume algorithm
  [cyan]r[/cyan]            - Refresh display
  [cyan]c[/cyan]            - Clear activity log
  [cyan]b[/cyan]            - Open backtest dashboard
  [cyan]h[/cyan] / [cyan]?[/cyan]        - Show this help
  [cyan]Escape[/cyan]       - Close help/dialogs

[bold]Navigation:[/bold]

  [cyan]Tab[/cyan]          - Move between widgets
  [cyan]↑[/cyan] / [cyan]↓[/cyan]        - Scroll within tables
  [cyan]Page Up/Down[/cyan] - Fast scroll

[bold]Trade History Tabs:[/bold]

  The trade history widget has tabs for Today and Yesterday.
  Click or use Tab to switch between them.

[bold]Status Indicators:[/bold]

  [green]● Running[/green]  - Algorithm is actively trading
  [yellow]● Paused[/yellow]   - Algorithm is paused
  [red]● Stopped[/red]  - Algorithm is not running

[bold]P&L Colors:[/bold]

  [green]+$XX.XX[/green]  - Profit
  [red]-$XX.XX[/red]  - Loss

Press [cyan]Escape[/cyan] or [cyan]q[/cyan] to close this help.
                """,
                id="help-content",
            ),
            id="help-container",
        )

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-container {
        width: 70;
        height: auto;
        max-height: 35;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #help-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #help-content {
        height: auto;
    }
    """


class TradingDashboard(App):
    """
    The main trading dashboard application.

    Provides a comprehensive view of:
    - Algorithm status
    - Real-time P&L
    - Open positions
    - Trade history
    - Signals
    - Activity log
    """

    TITLE = "Trading Dashboard"
    SUB_TITLE = "Multi-Edge Ensemble Trading System"

    CSS = """
    Screen {
        background: $surface-darken-1;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    #top-section {
        height: auto;
    }

    #middle-section {
        height: 1fr;
        min-height: 15;
    }

    #left-panel {
        width: 1fr;
    }

    #right-panel {
        width: 1fr;
    }

    #bottom-section {
        height: 1fr;
        min-height: 10;
    }

    HeaderWidget {
        dock: top;
    }

    Footer {
        dock: bottom;
    }

    .panel {
        padding: 0;
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
        Binding("p", "toggle_pause", "Pause"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "clear_log", "Clear Log"),
        Binding("b", "show_backtest", "Backtest"),
        Binding("h", "show_help", "Help"),
        Binding("question_mark", "show_help", "Help", show=False),
        Binding("escape", "escape", "Close", show=False),
    ]

    def __init__(
        self,
        store: DashboardStore | None = None,
        event_bus: EventBus | None = None,
        algorithm_name: str = "Trading Algorithm",
        backtest_callback: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.event_bus = event_bus or get_event_bus()
        self.algorithm_name = algorithm_name
        self._is_paused = False
        self._backtest_callback = backtest_callback

        # Set initial algorithm status
        initial_status = AlgorithmStatus(
            name=algorithm_name,
            is_running=True,
            is_paused=False,
        )
        self.store.set_algorithm_status(initial_status)

    def compose(self) -> ComposeResult:
        """Create dashboard layout."""
        yield HeaderWidget(store=self.store)
        yield Container(
            # Top section: P&L
            Horizontal(
                PnLWidget(store=self.store, classes="panel"),
                id="top-section",
            ),
            # Middle section: Positions and Trades
            Horizontal(
                Vertical(
                    PositionsWidget(store=self.store, classes="panel"),
                    id="left-panel",
                ),
                Vertical(
                    TradesWidget(store=self.store, classes="panel"),
                    id="right-panel",
                ),
                id="middle-section",
            ),
            # Bottom section: Signals and Log
            Horizontal(
                Vertical(
                    SignalsWidget(store=self.store, classes="panel"),
                    id="left-panel",
                ),
                Vertical(
                    LogWidget(store=self.store, classes="panel"),
                    id="right-panel",
                ),
                id="bottom-section",
            ),
            id="main-container",
        )
        yield Footer()

    def action_quit(self) -> None:
        """Quit the dashboard."""
        self.exit()

    def action_toggle_pause(self) -> None:
        """Toggle algorithm pause state."""
        self._is_paused = not self._is_paused

        status = self.store.algorithm_status
        if status:
            status.is_paused = self._is_paused
            self.store.set_algorithm_status(status)
            self.event_bus.emit(AlgorithmStatusEvent(
                status=status,
                source="dashboard",
            ))

        state_str = "paused" if self._is_paused else "resumed"
        self.notify(f"Algorithm {state_str}", severity="warning" if self._is_paused else "information")

    def action_refresh(self) -> None:
        """Force refresh all widgets."""
        self.refresh()
        self.notify("Display refreshed")

    def action_clear_log(self) -> None:
        """Clear the activity log."""
        try:
            log_widget = self.query_one(LogWidget)
            log_widget.clear_log()
            self.notify("Log cleared")
        except Exception:
            pass

    def action_show_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())

    def action_show_backtest(self) -> None:
        """Show backtest screen."""
        screen = BacktestScreen(run_backtest_callback=self._backtest_callback)
        self.push_screen(screen)

    def action_escape(self) -> None:
        """Handle escape key."""
        # If we're on a screen, pop it
        if len(self.screen_stack) > 1:
            self.pop_screen()


def run_dashboard(
    algorithm_name: str = "Trading Algorithm",
    store: DashboardStore | None = None,
    event_bus: EventBus | None = None,
    backtest_callback: Optional[Callable] = None,
) -> None:
    """
    Run the trading dashboard.

    Args:
        algorithm_name: Name to display in the header
        store: Optional custom store instance
        event_bus: Optional custom event bus instance
        backtest_callback: Optional callback for running backtests
    """
    app = TradingDashboard(
        store=store,
        event_bus=event_bus,
        algorithm_name=algorithm_name,
        backtest_callback=backtest_callback,
    )
    app.run()


async def run_dashboard_async(
    algorithm_name: str = "Trading Algorithm",
    store: DashboardStore | None = None,
    event_bus: EventBus | None = None,
    backtest_callback: Optional[Callable] = None,
) -> None:
    """
    Run the trading dashboard asynchronously.

    Use this when integrating with an async trading algorithm.
    """
    app = TradingDashboard(
        store=store,
        event_bus=event_bus,
        algorithm_name=algorithm_name,
        backtest_callback=backtest_callback,
    )
    await app.run_async()
