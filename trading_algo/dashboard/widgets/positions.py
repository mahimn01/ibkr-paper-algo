"""
Positions widget showing open positions.
"""

from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from ..store import DashboardStore, get_store
from ..models import TradeDirection


class PositionsWidget(Static):
    """
    Open positions table showing:
    - Symbol
    - Direction (LONG/SHORT)
    - Entry price
    - Current price
    - Stop loss
    - Take profit
    - Unrealized P&L
    - P&L %
    - Duration
    """

    def __init__(self, store: DashboardStore | None = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.store.add_listener(self._on_store_update)

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold]Open Positions[/bold]", id="positions-title"),
            DataTable(id="positions-table"),
            id="positions-container",
        )

    def on_mount(self) -> None:
        """Set up the table."""
        table = self.query_one("#positions-table", DataTable)
        table.add_columns(
            "Symbol",
            "Dir",
            "Qty",
            "Entry",
            "Current",
            "Stop",
            "Target",
            "P&L",
            "P&L %",
            "R:R",
        )
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._update_content()

    def _on_store_update(self) -> None:
        """Handle store updates."""
        self.call_from_thread(self._update_content)

    def _update_content(self) -> None:
        """Update positions table."""
        table = self.query_one("#positions-table", DataTable)
        table.clear()

        positions = self.store.positions_list
        if not positions:
            # Show placeholder
            table.add_row("─", "─", "─", "─", "─", "─", "─", "─", "─", "─")
            return

        for pos in positions:
            # Direction with color
            if pos.direction == TradeDirection.LONG:
                dir_str = "[green]LONG[/green]"
            else:
                dir_str = "[red]SHORT[/red]"

            # Prices
            entry_str = f"${pos.entry_price:.2f}"
            current_str = f"${pos.current_price:.2f}"
            stop_str = f"${pos.stop_loss:.2f}" if pos.stop_loss else "─"
            target_str = f"${pos.take_profit:.2f}" if pos.take_profit else "─"

            # P&L with color
            if pos.unrealized_pnl >= 0:
                pnl_str = f"[green]+${pos.unrealized_pnl:.2f}[/green]"
                pnl_pct_str = f"[green]+{pos.unrealized_pnl_pct:.2f}%[/green]"
            else:
                pnl_str = f"[red]-${abs(pos.unrealized_pnl):.2f}[/red]"
                pnl_pct_str = f"[red]{pos.unrealized_pnl_pct:.2f}%[/red]"

            # Risk:Reward
            rr = pos.risk_reward_current
            rr_str = f"{rr:.1f}:1" if rr else "─"

            table.add_row(
                pos.symbol,
                dir_str,
                str(pos.quantity),
                entry_str,
                current_str,
                stop_str,
                target_str,
                pnl_str,
                pnl_pct_str,
                rr_str,
            )

    DEFAULT_CSS = """
    PositionsWidget {
        height: auto;
        min-height: 8;
        max-height: 15;
        border: solid $primary;
    }

    #positions-title {
        text-align: center;
        text-style: bold;
        padding: 0 1;
        background: $surface;
    }

    #positions-table {
        height: 1fr;
    }
    """
