"""
Trades widget showing trade history.
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.widgets import Static, DataTable, TabbedContent, TabPane
from textual.containers import Vertical

from ..store import DashboardStore, get_store
from ..models import TradeDirection, Trade


class TradesWidget(Static):
    """
    Trade history widget with tabs for:
    - Today's trades
    - Yesterday's trades

    Shows full trade details including entry/exit, stops, P&L.
    """

    def __init__(self, store: DashboardStore | None = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.store.add_listener(self._on_store_update)

    def compose(self) -> ComposeResult:
        with Vertical(
            Static("[bold]Trade History[/bold]", id="trades-title"),
            id="trades-container",
        ):
            with TabbedContent(id="trades-tabs"):
                yield TabPane("Today", id="today-tab")
                yield TabPane("Yesterday", id="yesterday-tab")

    def on_mount(self) -> None:
        """Set up the tables."""
        # Create tables for each tab
        today_tab = self.query_one("#today-tab", TabPane)
        yesterday_tab = self.query_one("#yesterday-tab", TabPane)

        today_table = DataTable(id="today-table")
        yesterday_table = DataTable(id="yesterday-table")

        self._setup_table(today_table)
        self._setup_table(yesterday_table)

        today_tab.mount(today_table)
        yesterday_tab.mount(yesterday_table)

        self._update_content()

    def _setup_table(self, table: DataTable) -> None:
        """Set up columns for a trade table."""
        table.add_columns(
            "Time",
            "Symbol",
            "Dir",
            "Entry",
            "Exit",
            "Stop",
            "Target",
            "P&L",
            "P&L %",
            "Duration",
            "Reason",
        )
        table.cursor_type = "row"
        table.zebra_stripes = True

    def _on_store_update(self) -> None:
        """Handle store updates."""
        self.call_from_thread(self._update_content)

    def _update_content(self) -> None:
        """Update trades tables."""
        try:
            today_table = self.query_one("#today-table", DataTable)
            yesterday_table = self.query_one("#yesterday-table", DataTable)

            # Update today's trades
            today_trades = self.store.get_trades_today()
            self._populate_table(today_table, today_trades)

            # Update yesterday's trades
            yesterday_trades = self.store.get_trades_yesterday()
            self._populate_table(yesterday_table, yesterday_trades)
        except Exception:
            pass  # Tables may not be mounted yet

    def _populate_table(self, table: DataTable, trades: list[Trade]) -> None:
        """Populate a table with trades."""
        table.clear()

        if not trades:
            table.add_row("─", "─", "─", "─", "─", "─", "─", "─", "─", "─", "─")
            return

        # Sort by exit time, most recent first
        sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min, reverse=True)

        for trade in sorted_trades:
            # Time
            time_str = trade.exit_time.strftime("%H:%M:%S") if trade.exit_time else "─"

            # Direction with color
            if trade.direction == TradeDirection.LONG:
                dir_str = "[green]LONG[/green]"
            else:
                dir_str = "[red]SHORT[/red]"

            # Prices
            entry_str = f"${trade.entry_price:.2f}"
            exit_str = f"${trade.exit_price:.2f}" if trade.exit_price else "─"
            stop_str = f"${trade.initial_stop:.2f}" if trade.initial_stop else "─"
            target_str = f"${trade.initial_target:.2f}" if trade.initial_target else "─"

            # P&L with color
            if trade.realized_pnl >= 0:
                pnl_str = f"[green]+${trade.realized_pnl:.2f}[/green]"
                pnl_pct_str = f"[green]+{trade.realized_pnl_pct:.2f}%[/green]"
            else:
                pnl_str = f"[red]-${abs(trade.realized_pnl):.2f}[/red]"
                pnl_pct_str = f"[red]{trade.realized_pnl_pct:.2f}%[/red]"

            # Duration
            if trade.duration_seconds < 60:
                duration_str = f"{trade.duration_seconds}s"
            elif trade.duration_seconds < 3600:
                duration_str = f"{trade.duration_seconds // 60}m"
            else:
                duration_str = f"{trade.duration_seconds // 3600}h {(trade.duration_seconds % 3600) // 60}m"

            # Reason (truncate if too long)
            reason = trade.exit_reason[:20] + "..." if len(trade.exit_reason) > 23 else trade.exit_reason

            table.add_row(
                time_str,
                trade.symbol,
                dir_str,
                entry_str,
                exit_str,
                stop_str,
                target_str,
                pnl_str,
                pnl_pct_str,
                duration_str,
                reason,
            )

    DEFAULT_CSS = """
    TradesWidget {
        height: auto;
        min-height: 12;
        border: solid $primary;
    }

    #trades-title {
        text-align: center;
        text-style: bold;
        padding: 0 1;
        background: $surface;
    }

    #trades-tabs {
        height: 1fr;
    }

    #today-table, #yesterday-table {
        height: 1fr;
    }
    """
