"""
Logs widget showing activity log.
"""

from textual.app import ComposeResult
from textual.widgets import Static, RichLog
from textual.containers import Vertical

from ..store import DashboardStore, get_store


class LogWidget(Static):
    """
    Activity log widget showing:
    - Timestamp
    - Level (INFO, WARNING, ERROR)
    - Source
    - Message

    Auto-scrolls to latest entry.
    """

    def __init__(self, store: DashboardStore | None = None, **kwargs):
        super().__init__(**kwargs)
        self.store = store or get_store()
        self.store.add_listener(self._on_store_update)
        self._last_log_count = 0

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold]Activity Log[/bold]", id="log-title"),
            RichLog(id="log-output", highlight=True, markup=True),
            id="log-container",
        )

    def on_mount(self) -> None:
        """Initial update."""
        self._update_content()

    def _on_store_update(self) -> None:
        """Handle store updates."""
        self.call_from_thread(self._update_content)

    def _update_content(self) -> None:
        """Update log content."""
        log_output = self.query_one("#log-output", RichLog)
        logs = self.store.logs

        # Only add new entries
        new_logs = logs[self._last_log_count:]
        self._last_log_count = len(logs)

        for entry in new_logs:
            time_str = entry.timestamp.strftime("%H:%M:%S")

            # Level colors
            level_colors = {
                "DEBUG": "dim",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
            }
            level_color = level_colors.get(entry.level, "white")

            # Format entry
            source_str = f"[dim][{entry.source}][/dim] " if entry.source else ""
            log_output.write(
                f"[dim]{time_str}[/dim] [{level_color}]{entry.level:7}[/{level_color}] "
                f"{source_str}{entry.message}"
            )

    def clear_log(self) -> None:
        """Clear the log output."""
        log_output = self.query_one("#log-output", RichLog)
        log_output.clear()
        self._last_log_count = 0

    DEFAULT_CSS = """
    LogWidget {
        height: 1fr;
        min-height: 8;
        border: solid $primary;
    }

    #log-title {
        text-align: center;
        text-style: bold;
        padding: 0 1;
        background: $surface;
    }

    #log-output {
        height: 1fr;
        scrollbar-gutter: stable;
    }
    """
