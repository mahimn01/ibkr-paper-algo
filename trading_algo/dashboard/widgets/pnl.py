"""
P&L widget showing profit/loss summary.
"""

from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Vertical, Horizontal

from ..store import DashboardStore, get_store


class PnLWidget(Static):
    """
    P&L summary widget showing:
    - Today's realized P&L
    - Today's unrealized P&L
    - Total P&L
    - Win rate
    - Trade count
    """

    def __init__(self, store: DashboardStore | None = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.store.add_listener(self._on_store_update)

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold]P&L Summary[/bold]", id="pnl-title"),
            Horizontal(
                Vertical(
                    Static("Realized", classes="pnl-label"),
                    Static("$0.00", id="realized-pnl", classes="pnl-value"),
                    classes="pnl-box",
                ),
                Vertical(
                    Static("Unrealized", classes="pnl-label"),
                    Static("$0.00", id="unrealized-pnl", classes="pnl-value"),
                    classes="pnl-box",
                ),
                Vertical(
                    Static("Total", classes="pnl-label"),
                    Static("$0.00", id="total-pnl", classes="pnl-value-large"),
                    classes="pnl-box total-box",
                ),
                id="pnl-row",
            ),
            Horizontal(
                Vertical(
                    Static("Win Rate", classes="stat-label"),
                    Static("0%", id="win-rate", classes="stat-value"),
                    classes="stat-box",
                ),
                Vertical(
                    Static("Trades", classes="stat-label"),
                    Static("0", id="trade-count", classes="stat-value"),
                    classes="stat-box",
                ),
                Vertical(
                    Static("Winners", classes="stat-label"),
                    Static("0", id="winners", classes="stat-value"),
                    classes="stat-box",
                ),
                Vertical(
                    Static("Losers", classes="stat-label"),
                    Static("0", id="losers", classes="stat-value"),
                    classes="stat-box",
                ),
                id="stats-row",
            ),
            id="pnl-container",
        )

    def on_mount(self) -> None:
        """Initial update."""
        self._update_content()

    def _on_store_update(self) -> None:
        """Handle store updates."""
        self.call_from_thread(self._update_content)

    def _update_content(self) -> None:
        """Update P&L content."""
        realized = self.store.total_realized_pnl_today
        unrealized = self.store.total_unrealized_pnl
        total = realized + unrealized
        win_rate = self.store.win_rate_today
        trades_today = self.store.get_trades_today()
        trade_count = len(trades_today)
        winners = sum(1 for t in trades_today if t.realized_pnl > 0)
        losers = sum(1 for t in trades_today if t.realized_pnl < 0)

        # Update realized P&L
        realized_widget = self.query_one("#realized-pnl", Static)
        if realized >= 0:
            realized_widget.update(f"[green]+${realized:.2f}[/green]")
        else:
            realized_widget.update(f"[red]-${abs(realized):.2f}[/red]")

        # Update unrealized P&L
        unrealized_widget = self.query_one("#unrealized-pnl", Static)
        if unrealized >= 0:
            unrealized_widget.update(f"[green]+${unrealized:.2f}[/green]")
        else:
            unrealized_widget.update(f"[red]-${abs(unrealized):.2f}[/red]")

        # Update total P&L
        total_widget = self.query_one("#total-pnl", Static)
        if total >= 0:
            total_widget.update(f"[bold green]+${total:.2f}[/bold green]")
        else:
            total_widget.update(f"[bold red]-${abs(total):.2f}[/bold red]")

        # Update stats
        self.query_one("#win-rate", Static).update(f"[cyan]{win_rate:.1f}%[/cyan]")
        self.query_one("#trade-count", Static).update(f"[cyan]{trade_count}[/cyan]")
        self.query_one("#winners", Static).update(f"[green]{winners}[/green]")
        self.query_one("#losers", Static).update(f"[red]{losers}[/red]")

    DEFAULT_CSS = """
    PnLWidget {
        height: auto;
        min-height: 8;
        border: solid $primary;
        padding: 0 1;
    }

    #pnl-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #pnl-row {
        height: 3;
        align: center middle;
    }

    .pnl-box {
        width: 1fr;
        height: 100%;
        align: center middle;
        border-right: solid $surface-lighten-1;
    }

    .total-box {
        border-right: none;
        background: $surface-lighten-1;
    }

    .pnl-label {
        text-align: center;
        color: $text-muted;
    }

    .pnl-value {
        text-align: center;
    }

    .pnl-value-large {
        text-align: center;
        text-style: bold;
    }

    #stats-row {
        height: 3;
        margin-top: 1;
        align: center middle;
    }

    .stat-box {
        width: 1fr;
        height: 100%;
        align: center middle;
    }

    .stat-label {
        text-align: center;
        color: $text-muted;
    }

    .stat-value {
        text-align: center;
    }
    """
