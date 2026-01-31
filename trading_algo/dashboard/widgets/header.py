"""
Header widget showing algorithm status and time.
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.widgets import Static
from textual.reactive import reactive

from ..store import DashboardStore, get_store


class HeaderWidget(Static):
    """
    Header bar showing:
    - Algorithm name and status
    - Current time
    - Quick stats
    """

    algorithm_name: reactive[str] = reactive("Trading Dashboard")
    status: reactive[str] = reactive("Stopped")
    status_color: reactive[str] = reactive("red")

    def __init__(self, store: DashboardStore | None = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.store.add_listener(self._on_store_update)

    def compose(self) -> ComposeResult:
        yield Static(id="header-content")

    def on_mount(self) -> None:
        """Set up periodic updates."""
        self.set_interval(1.0, self._update_time)
        self._update_content()

    def _on_store_update(self) -> None:
        """Handle store updates."""
        self.call_from_thread(self._update_content)

    def _update_time(self) -> None:
        """Update time display."""
        self._update_content()

    def _update_content(self) -> None:
        """Update header content."""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")

        algo_status = self.store.algorithm_status
        if algo_status:
            self.algorithm_name = algo_status.name
            self.status = algo_status.status_text
            if algo_status.is_running and not algo_status.is_paused:
                self.status_color = "green"
            elif algo_status.is_paused:
                self.status_color = "yellow"
            else:
                self.status_color = "red"
        else:
            self.status = "Connecting..."
            self.status_color = "yellow"

        # Build header content
        positions = len(self.store.positions)
        trades_today = self.store.trade_count_today
        total_pnl = self.store.total_pnl_today
        pnl_color = "green" if total_pnl >= 0 else "red"
        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"

        header_content = self.query_one("#header-content", Static)
        header_content.update(
            f"[bold]{self.algorithm_name}[/bold] │ "
            f"[{self.status_color}]● {self.status}[/{self.status_color}] │ "
            f"{date_str} {time_str} │ "
            f"Positions: [cyan]{positions}[/cyan] │ "
            f"Trades: [cyan]{trades_today}[/cyan] │ "
            f"P&L: [{pnl_color}]{pnl_str}[/{pnl_color}]"
        )

    DEFAULT_CSS = """
    HeaderWidget {
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 1;
    }

    #header-content {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    """
