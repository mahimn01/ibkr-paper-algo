"""
Signals widget showing recent algorithm signals.
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from ..store import DashboardStore, get_store
from ..models import SignalType, SignalStrength


class SignalsWidget(Static):
    """
    Recent signals widget showing:
    - Time
    - Symbol
    - Signal type
    - Strength/confidence
    - Price
    - Reason
    - Component votes (for ensemble systems)
    """

    def __init__(self, store: DashboardStore | None = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.store.add_listener(self._on_store_update)

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold]Recent Signals[/bold]", id="signals-title"),
            DataTable(id="signals-table"),
            id="signals-container",
        )

    def on_mount(self) -> None:
        """Set up the table."""
        table = self.query_one("#signals-table", DataTable)
        table.add_columns(
            "Time",
            "Symbol",
            "Signal",
            "Strength",
            "Conf",
            "Price",
            "Stop",
            "Target",
            "Reason",
        )
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._update_content()

    def _on_store_update(self) -> None:
        """Handle store updates."""
        self.call_from_thread(self._update_content)

    def _update_content(self) -> None:
        """Update signals table."""
        table = self.query_one("#signals-table", DataTable)
        table.clear()

        signals = self.store.signals
        if not signals:
            table.add_row("─", "─", "─", "─", "─", "─", "─", "─", "─")
            return

        # Show most recent first
        for signal in reversed(signals[-50:]):
            # Time
            time_str = signal.timestamp.strftime("%H:%M:%S")

            # Signal type with color
            signal_colors = {
                SignalType.ENTRY_LONG: "green",
                SignalType.ENTRY_SHORT: "red",
                SignalType.EXIT: "yellow",
                SignalType.HOLD: "dim",
                SignalType.SCALE_IN: "cyan",
                SignalType.SCALE_OUT: "magenta",
            }
            color = signal_colors.get(signal.signal_type, "white")
            signal_str = f"[{color}]{signal.signal_type.value}[/{color}]"

            # Strength
            strength_colors = {
                SignalStrength.WEAK: "dim",
                SignalStrength.MODERATE: "yellow",
                SignalStrength.STRONG: "green",
                SignalStrength.VERY_STRONG: "bold green",
            }
            strength_color = strength_colors.get(signal.strength, "white")
            strength_str = f"[{strength_color}]{signal.strength.name}[/{strength_color}]"

            # Confidence
            conf_color = "green" if signal.confidence >= 0.7 else "yellow" if signal.confidence >= 0.5 else "red"
            conf_str = f"[{conf_color}]{signal.confidence:.0%}[/{conf_color}]"

            # Prices
            price_str = f"${signal.price:.2f}"
            stop_str = f"${signal.suggested_stop:.2f}" if signal.suggested_stop else "─"
            target_str = f"${signal.suggested_target:.2f}" if signal.suggested_target else "─"

            # Reason (truncate if needed)
            reason = signal.reason
            if len(reason) > 40:
                reason = reason[:37] + "..."

            table.add_row(
                time_str,
                signal.symbol,
                signal_str,
                strength_str,
                conf_str,
                price_str,
                stop_str,
                target_str,
                reason,
            )

    DEFAULT_CSS = """
    SignalsWidget {
        height: auto;
        min-height: 10;
        max-height: 20;
        border: solid $primary;
    }

    #signals-title {
        text-align: center;
        text-style: bold;
        padding: 0 1;
        background: $surface;
    }

    #signals-table {
        height: 1fr;
    }
    """
