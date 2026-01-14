from __future__ import annotations

import asyncio
import os
import subprocess
import shutil
import threading
from dataclasses import dataclass
from typing import Any, Callable

from trading_algo.llm.chat_protocol import ToolCall


class PromptToolkitMissing(RuntimeError):
    pass


class UserCancelled(RuntimeError):
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
        from prompt_toolkit.lexers import PygmentsLexer
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.dimension import D
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, TextArea
        from pygments.lexers.markup import MarkdownLexer
    except Exception as exc:  # pragma: no cover
        raise PromptToolkitMissing("prompt_toolkit is required for --ui tui") from exc

    transcript_plain = {"value": ""}
    transcript_area = TextArea(
        text="",
        multiline=True,
        wrap_lines=True,
        read_only=True,
        scrollbar=True,
        focusable=False,  # keep focus in the input box (avoid getting "stuck" in transcript)
        focus_on_click=False,
        lexer=PygmentsLexer(MarkdownLexer),
        name="transcript",
    )
    status_text = {"value": "Ready"}
    status_ctl = FormattedTextControl(text=[("class:status", status_text["value"])])
    ui_state = {
        "busy": False,
        "status": "Ready",
        "spinner_i": 0,
        "active_tool": None,
        "auto_scroll": True,
    }
    tool_verbose = os.getenv("TUI_TOOL_VERBOSE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    max_chars_env = os.getenv("TUI_MAX_TRANSCRIPT_CHARS", "").strip()
    max_transcript_chars = int(max_chars_env) if max_chars_env.isdigit() else 750_000
    mouse_support = os.getenv("TUI_MOUSE", "").strip().lower() in {"1", "true", "yes", "y", "on"}

    input_box = TextArea(
        height=3,
        prompt="you> ",
        multiline=True,
        wrap_lines=True,
    )

    # prompt_toolkit doesn't expose the asyncio loop on Application in all versions.
    # Capture it from the UI thread when available and use it for thread-safe scheduling.
    ui_loop: asyncio.AbstractEventLoop | None = None
    cancel_event = threading.Event()

    def _copy_to_clipboard(text: str) -> bool:
        """
        Best-effort system clipboard copy.

        Prefer platform utilities so users can paste outside the terminal.
        """
        if not text:
            return False

        if shutil.which("pbcopy"):
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            assert p.stdin is not None
            p.stdin.write(text.encode("utf-8"))
            p.stdin.close()
            return p.wait(timeout=5) == 0

        if shutil.which("wl-copy"):
            p = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
            assert p.stdin is not None
            p.stdin.write(text.encode("utf-8"))
            p.stdin.close()
            return p.wait(timeout=5) == 0

        if shutil.which("xclip"):
            p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            assert p.stdin is not None
            p.stdin.write(text.encode("utf-8"))
            p.stdin.close()
            return p.wait(timeout=5) == 0

        return False

    def _visible_height_estimate() -> int:
        """
        Estimate how many rows the transcript can show.

        prompt_toolkit doesn't easily expose the exact write_position height of a
        container without running a render pass. This heuristic is good enough
        to keep scrolling stable (and avoid "blank screen" when scroll is too
        large).
        """
        try:
            rows = int(app.output.get_size().rows)  # type: ignore[attr-defined]
        except Exception:
            rows = 40
        # status bar (1) + input frame (~5) + borders/padding (~2)
        return max(5, rows - 8)

    def _content_rows_estimate() -> int:
        # Conservative (no wrapping). Wrapping increases the true height, which
        # makes this estimate safe for clamping.
        return max(1, transcript_plain["value"].count("\n") + 1)

    def _max_scroll_estimate() -> int:
        return max(0, _content_rows_estimate() - _visible_height_estimate())

    def _scroll_to_bottom() -> None:
        try:
            transcript_area.window.vertical_scroll = _max_scroll_estimate()
            transcript_area.buffer.cursor_position = len(transcript_area.buffer.text)
        except Exception:
            pass

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
        # Keep the UI responsive for long sessions by trimming oldest transcript.
        # (The full chat history for the model is managed separately.)
        if max_transcript_chars > 0:
            plain = transcript_plain["value"]
            if len(plain) > max_transcript_chars:
                # Drop ~10% at a time, cutting at a newline boundary when possible.
                drop = max(1, int(max_transcript_chars * 0.1))
                cut = plain.find("\n", drop)
                if cut == -1:
                    cut = drop
                transcript_plain["value"] = plain[cut:]
                transcript_area.text = transcript_plain["value"]

        combined = f"{prefix}{text}"
        transcript_plain["value"] += combined
        # Efficient append into the buffer so we don't re-render the whole transcript for streaming.
        try:
            transcript_area.buffer.cursor_position = len(transcript_area.buffer.text)
            transcript_area.buffer.insert_text(combined, move_cursor=True)
        except Exception:
            # Fallback: reset full text.
            transcript_area.text = transcript_plain["value"]

        # Scroll to bottom.
        if ui_state.get("auto_scroll"):
            _scroll_to_bottom()

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
            _append_transcript("\n\n```json\n", "")
            _append_transcript("", f"args: {call.args}\n")
            _append_transcript("", f"result: {result}\n")
            _append_transcript("```\n", "")

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

    @kb.add("escape")
    def _cancel(event) -> None:  # pragma: no cover
        # ESC cancels the current in-flight model turn (best-effort).
        if not ui_state.get("busy"):
            return
        cancel_event.set()
        _set_status("Cancelling…")
        event.app.invalidate()

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

        if user_text.strip().startswith("/copy"):
            # Copy transcript to system clipboard.
            parts = user_text.strip().split()
            mode = parts[1].strip().lower() if len(parts) > 1 else "200"
            text = transcript_plain["value"]
            if mode not in {"all"}:
                try:
                    last_n = max(1, int(mode))
                except Exception:
                    last_n = 200
                lines = text.splitlines()
                text = "\n".join(lines[-last_n:]) + ("\n" if lines else "")
            ok = _copy_to_clipboard(text)
            _append_transcript("\n\nyou> ", user_text)
            _append_transcript("\n\nGemini> ", "Copied to clipboard." if ok else "Clipboard copy failed (no pbcopy/wl-copy/xclip).")
            _append_transcript("\n", "")
            _set_status("Ready")
            input_box.read_only = False
            app.invalidate()
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
        cancel_event.clear()
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
            if cancel_event.is_set():
                raise UserCancelled()
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
            if cancel_event.is_set():
                raise UserCancelled()
            _call_in_loop(lambda: _append_tool_event(call, bool(ok), result))

        def _status_cb(text: str) -> None:
            if cancel_event.is_set():
                raise UserCancelled()
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
            except UserCancelled:
                def _cancelled() -> None:
                    _flush_stream()
                    _append_transcript("\n\n[tool] ", "… cancelled")
                    _append_transcript("\n", "")
                    _set_status("Ready")
                    input_box.read_only = False
                _call_in_loop(_cancelled)
            except Exception as exc:
                err_text = str(exc)
                def _err() -> None:
                    _flush_stream()
                    _append_transcript("\n\nerror> ", err_text)
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

    def _scroll_transcript(delta_lines: int) -> None:
        """
        Scroll the transcript window by `delta_lines`.

        Negative values scroll up (older messages), positive values scroll down.
        """
        try:
            cur = int(getattr(transcript_area.window, "vertical_scroll", 0))
        except Exception:
            cur = 0
        new = cur + int(delta_lines)
        if new < 0:
            new = 0
        # Clamp to prevent scrolling past the bottom and ending up with a blank view.
        max_scroll = _max_scroll_estimate()
        if new > max_scroll:
            new = max_scroll
        try:
            transcript_area.window.vertical_scroll = new
        except Exception:
            return
        if delta_lines < 0:
            ui_state["auto_scroll"] = False
        app.invalidate()

    @kb.add("<scroll-up>")  # mouse wheel up
    def _on_scroll_up(event) -> None:  # pragma: no cover
        _scroll_transcript(-5)

    @kb.add("<scroll-down>")  # mouse wheel down
    def _on_scroll_down(event) -> None:  # pragma: no cover
        _scroll_transcript(+5)

    @kb.add("pageup")
    def _on_pageup(event) -> None:  # pragma: no cover
        _scroll_transcript(-20)

    @kb.add("pagedown")
    def _on_pagedown(event) -> None:  # pragma: no cover
        _scroll_transcript(+20)

    @kb.add("home")
    def _on_home(event) -> None:  # pragma: no cover
        ui_state["auto_scroll"] = False
        try:
            transcript_area.window.vertical_scroll = 0
        except Exception:
            pass
        event.app.invalidate()

    @kb.add("end")
    def _on_end(event) -> None:  # pragma: no cover
        ui_state["auto_scroll"] = True
        _scroll_to_bottom()
        event.app.invalidate()

    root = HSplit(
        [
            Frame(transcript_area, title="Chat (PgUp/PgDn/Home/End to scroll, /copy [N|all])", height=D(weight=3)),
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
        mouse_support=mouse_support,
        before_render=_before_render,
        after_render=_after_render,
    )
    app.run()
