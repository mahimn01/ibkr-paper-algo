from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

from trading_algo.llm.chat_protocol import ToolCall


class PromptToolkitMissing(RuntimeError):
    pass


@dataclass
class UIEvent:
    kind: str
    text: str | None = None
    tool: ToolCall | None = None
    ok: bool | None = None
    result: Any | None = None


def run_tui(
    *,
    run_turn: Callable[[str, Callable[[str], None], Callable[[ToolCall, bool, Any], None], Callable[[str], None]], str],
    unlock: Callable[[str], str] | None = None,
    lock: Callable[[], str] | None = None,
) -> None:
    """
    Prompt-toolkit based UI runner.

    `run_turn(user_text, stream_cb, tool_cb, status_cb) -> assistant_message` should perform one model turn
    (including tool execution), calling callbacks as streaming tokens arrive, tools complete, and statuses change.
    """
    try:  # pragma: no cover
        from prompt_toolkit.application import Application, run_in_terminal
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.dimension import D
        from prompt_toolkit.layout.margins import ScrollbarMargin
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, TextArea
    except Exception as exc:  # pragma: no cover
        raise PromptToolkitMissing("prompt_toolkit is required for --ui tui") from exc

    transcript_fragments: list[tuple[str, str]] = []
    transcript_plain = {"value": ""}
    transcript_ctl = FormattedTextControl(text=transcript_fragments, focusable=False, show_cursor=False)
    transcript_win = Window(
        content=transcript_ctl,
        wrap_lines=True,
        always_hide_cursor=True,
        right_margins=[ScrollbarMargin(display_arrows=True)],
    )
    status_text = {"value": "Ready"}
    status_ctl = FormattedTextControl(text=[("class:status", status_text["value"])])
    ui_state = {
        "busy": False,
        "status": "Ready",
        "spinner_i": 0,
        "active_tool": None,
        "md_in_code_fence": False,
    }
    tool_verbose = os.getenv("TUI_TOOL_VERBOSE", "").strip().lower() in {"1", "true", "yes", "y", "on"}

    input_box = TextArea(
        height=3,
        prompt="you> ",
        multiline=True,
        wrap_lines=True,
    )

    # prompt_toolkit doesn't expose the asyncio loop on Application in all versions.
    # Capture it from the UI thread when available and use it for thread-safe scheduling.
    ui_loop: asyncio.AbstractEventLoop | None = None

    def _call_in_loop(fn: Callable[[], None]) -> None:
        """
        Thread-safe UI updates: schedule on the application's asyncio loop.
        """
        loop = ui_loop
        if loop is None:
            # Best-effort: invalidate triggers redraw, but we can't safely mutate widgets
            # without the UI loop. Once a keypress happens, ui_loop will be captured.
            try:
                app.invalidate()
            except Exception:
                pass
            return

        def _wrapped() -> None:
            try:
                fn()
            finally:
                # Force a redraw after any UI mutation.
                try:
                    app.invalidate()
                except Exception:
                    pass

        loop.call_soon_threadsafe(_wrapped)

    def _set_status(text: str) -> None:
        ui_state["status"] = str(text)
        ui_state["busy"] = ui_state["status"] not in {"Ready", "Idle"}
        status_text["value"] = ui_state["status"]
        status_ctl.text = [("class:status", status_text["value"])]

    def _append_transcript(prefix: str, text: str) -> None:
        _append_rich_text(f"{prefix}{text}")
        # Scroll to bottom.
        try:
            transcript_win.vertical_scroll = 10**9  # type: ignore[attr-defined]
        except Exception:
            pass

    def _append_rich_text(text: str) -> None:
        """
        Append text to transcript with lightweight markdown + tool styling.

        This is intentionally minimal and incremental (stream-friendly).
        """
        fragments = _render_incremental(text, ui_state)
        if fragments:
            transcript_plain["value"] += str(text)
            transcript_fragments.extend(fragments)
            transcript_ctl.text = transcript_fragments

    def _summarize_tool_result(name: str, ok: bool, result: Any) -> str:
        if not isinstance(result, dict):
            return f"{'OK' if ok else 'ERR'} {name}"
        if not ok:
            err = result.get("error")
            return f"ERR {name} → {err if err else 'error'}"
        # Common OMS shapes.
        if "order_id" in result and "status" in result:
            return f"OK  {name} → order_id={result.get('order_id')} status={result.get('status')}"
        if name == "get_snapshot":
            inst = result.get("instrument") or {}
            sym = inst.get("symbol") if isinstance(inst, dict) else None
            last = result.get("last") or result.get("close")
            return f"OK  {name} → {sym} last={last}"
        return f"OK  {name}"

    def _append_tool_event(call: ToolCall, ok: bool, result: Any) -> None:
        name = str(call.name)
        ui_state["active_tool"] = None
        _append_transcript("\n\n[tool] ", _summarize_tool_result(name, ok, result))
        _append_transcript("\n", "")
        if tool_verbose:
            _append_transcript("\n       args: ", str(call.args))
            _append_transcript("\n     result: ", str(result))

    style = Style.from_dict(
        {
            "frame.label": "bold",
            "status": "fg:#00afff bold",
            "statusbar": "fg:#00afff bold",
            "tool_tag": "fg:#af5fff bold",
            "tool_ok": "fg:#5fff87 bold",
            "tool_err": "fg:#ff5f5f bold",
            "tool_dim": "fg:#808080",
            "md_heading": "fg:#00afff bold",
            "md_code": "fg:#ffd75f",
            "md_bold": "bold",
        }
    )

    kb = KeyBindings()

    @kb.add("c-c")  # pragma: no cover
    @kb.add("c-d")  # pragma: no cover
    def _exit(event) -> None:
        event.app.exit()

    def _submit_user_text(app: Application) -> None:
        nonlocal ui_loop
        # We are in the UI thread here; capture the running loop for background scheduling.
        try:
            ui_loop = asyncio.get_running_loop()
        except RuntimeError:
            # prompt_toolkit uses asyncio internally; this should not happen during app.run(),
            # but keep a safe fallback.
            ui_loop = None

        user_text = input_box.text.strip()
        input_box.text = ""
        if not user_text:
            return

        if user_text.strip() == "/unlock" and unlock is not None:
            def _do_unlock() -> None:
                import getpass

                token = getpass.getpass("TRADING_ORDER_TOKEN: ")
                msg = unlock(token)
                _append_transcript("\n\nyou> ", "/unlock")
                _append_transcript("\n\nGemini> ", msg)
                _append_transcript("\n", "")
                _set_status("Ready")
                input_box.read_only = False

            input_box.read_only = True
            _set_status("Unlocking…")
            # `Application.run_in_terminal` is not present in all prompt_toolkit versions;
            # use the module-level helper which works across versions.
            run_in_terminal(_do_unlock)
            app.invalidate()
            return

        if user_text.strip() == "/lock" and lock is not None:
            msg = lock()
            _append_transcript("\n\nyou> ", "/lock")
            _append_transcript("\n\nGemini> ", msg)
            _append_transcript("\n", "")
            _set_status("Ready")
            input_box.read_only = False
            app.invalidate()
            return

        _append_transcript("\n\nyou> ", user_text)
        _append_transcript("\n\nGemini> ", "")
        _set_status("Thinking…")
        input_box.read_only = True
        app.invalidate()

        # Thread-safe UI updates: always schedule work on the app event loop.
        # IMPORTANT: Coalesce stream chunks to avoid flooding the UI loop (which can delay "Ready").
        stream_lock = threading.Lock()
        stream_buf: list[str] = []
        flush_scheduled = {"value": False}

        def _flush_stream() -> None:
            with stream_lock:
                flush_scheduled["value"] = False
                if not stream_buf:
                    return
                text = "".join(stream_buf)
                stream_buf.clear()
            _append_transcript("", text)

        def _stream_cb(tok: str) -> None:
            t = str(tok)
            if not t:
                return
            with stream_lock:
                stream_buf.append(t)
                if flush_scheduled["value"]:
                    return
                flush_scheduled["value"] = True
            _call_in_loop(_flush_stream)

        def _tool_cb(call: ToolCall, ok: bool, result: Any) -> None:
            _call_in_loop(lambda: _append_tool_event(call, bool(ok), result))

        def _status_cb(text: str) -> None:
            t = str(text)
            def _do() -> None:
                # Compact tool-start indicator in the transcript.
                if t.startswith("Executing "):
                    tool_name = t[len("Executing ") :].strip().rstrip(".").rstrip("…").strip()
                    if tool_name and ui_state.get("active_tool") != tool_name:
                        ui_state["active_tool"] = tool_name
                        _append_transcript("\n\n[tool] ", f"… {tool_name}")
                _set_status(t)
            _call_in_loop(_do)

        def _worker() -> None:
            try:
                _status_cb("Thinking…")
                run_turn(user_text, _stream_cb, _tool_cb, _status_cb)
                def _finish() -> None:
                    # Flush any remaining streamed text before unlocking input.
                    _flush_stream()
                    if transcript_plain["value"] and not transcript_plain["value"].endswith("\n"):
                        _append_transcript("", "\n")
                    _append_transcript("\n", "")
                    _set_status("Ready")
                    input_box.read_only = False
                _call_in_loop(_finish)
            except Exception as exc:
                def _err() -> None:
                    _flush_stream()
                    _append_transcript("\n\nerror> ", str(exc))
                    _append_transcript("\n", "")
                    _set_status("Ready")
                    input_box.read_only = False
                _call_in_loop(_err)

        threading.Thread(target=_worker, daemon=True).start()

    # Key bindings:
    # - Enter: submit
    # - Alt+Enter: newline (portable across terminals)
    @kb.add("escape", "enter")
    def _on_alt_enter(event) -> None:  # pragma: no cover
        input_box.buffer.insert_text("\n")

    @kb.add("enter")
    def _on_enter(event) -> None:  # pragma: no cover
        _submit_user_text(event.app)

    root = HSplit(
        [
            Frame(transcript_win, title="Chat", height=D(weight=3)),
            Window(content=status_ctl, height=1, style="class:statusbar"),
            Frame(input_box, title="Input (Enter=send, Alt+Enter=newline, Ctrl-C=exit)"),
        ]
    )

    def _after_render(_) -> None:  # pragma: no cover
        nonlocal ui_loop
        if ui_loop is None:
            try:
                ui_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def _before_render(_) -> None:  # pragma: no cover
        # Lightweight spinner inside the status bar.
        if not ui_state["busy"]:
            return
        frames = ["|", "/", "-", "\\"]
        ui_state["spinner_i"] = (int(ui_state["spinner_i"]) + 1) % len(frames)
        status_ctl.text = [("class:status", f"{ui_state['status']} {frames[int(ui_state['spinner_i'])]}")]

    app = Application(
        layout=Layout(root, focused_element=input_box),
        key_bindings=kb,
        full_screen=True,
        style=style,
        refresh_interval=0.1,
        before_render=_before_render,
        after_render=_after_render,
    )
    app.run()


def _render_incremental(text: str, ui_state: dict[str, object]) -> list[tuple[str, str]]:
    """
    Extremely small markdown-ish renderer for prompt_toolkit formatted text.

    Handles:
    - Code fences ``` toggling a persistent state
    - Headings (# ...)
    - Inline code `...`
    - Bold **...**
    - Tool lines prefixed with "[tool]"
    - 'Gemini>' prefix in blue
    """
    out: list[tuple[str, str]] = []
    if not text:
        return out

    in_code = bool(ui_state.get("md_in_code_fence"))

    i = 0
    while i < len(text):
        if text.startswith("```", i):
            in_code = not in_code
            ui_state["md_in_code_fence"] = in_code
            out.append(("class:md_code", "```"))
            i += 3
            continue

        ch = text[i]

        # Start-of-line heuristics (headings and tool tag).
        # We only apply if we are not in a code fence.
        if not in_code:
            # Find the line start.
            line_start = text.rfind("\n", 0, i) + 1
            if i == line_start:
                if text.startswith("Gemini> ", i):
                    out.append(("class:md_heading", "Gemini>"))
                    out.append(("", " "))
                    i += len("Gemini> ")
                    continue
                if text.startswith("[tool]", i):
                    out.append(("class:tool_tag", "[tool]"))
                    i += len("[tool]")
                    continue
                if text.startswith("#", i):
                    # Consume the whole heading marker run ("###").
                    j = i
                    while j < len(text) and text[j] == "#":
                        j += 1
                    out.append(("class:md_heading", text[i:j]))
                    i = j
                    continue

        # Inline parsing.
        if in_code:
            out.append(("class:md_code", ch))
            i += 1
            continue

        if text.startswith("**", i):
            # Toggle bold: we don't keep state; just render the markers dim and text bold until next **.
            end = text.find("**", i + 2)
            if end != -1:
                out.append(("class:tool_dim", "**"))
                out.append(("class:md_bold", text[i + 2 : end]))
                out.append(("class:tool_dim", "**"))
                i = end + 2
                continue

        if ch == "`":
            end = text.find("`", i + 1)
            if end != -1:
                out.append(("class:tool_dim", "`"))
                out.append(("class:md_code", text[i + 1 : end]))
                out.append(("class:tool_dim", "`"))
                i = end + 1
                continue

        # Tool status words (OK/ERR) just after [tool] in our own emitted lines.
        if text.startswith("OK  ", i):
            out.append(("class:tool_ok", "OK  "))
            i += 4
            continue
        if text.startswith("ERR ", i):
            out.append(("class:tool_err", "ERR "))
            i += 4
            continue

        out.append(("", ch))
        i += 1

    return out
