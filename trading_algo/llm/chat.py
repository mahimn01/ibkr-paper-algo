from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from trading_algo.broker.base import Broker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.llm.chat_protocol import ChatModelReply, ToolCall
from trading_algo.llm.config import LLMConfig
from trading_algo.llm.gemini import GeminiClient, LLMClient
from trading_algo.llm.tools import ToolError, dispatch_tool, gemini_function_declarations, list_tools
from trading_algo.llm.tui import PromptToolkitMissing, run_tui
from trading_algo.oms import OrderManager
from trading_algo.risk import RiskLimits, RiskManager


_COLOR_ENABLED = True
_LOG = logging.getLogger(__name__)


def _c(code: str) -> str:
    if not _COLOR_ENABLED:
        return ""
    return f"\033[{code}m"


def _reset() -> str:
    return _c("0")


def _banner(title: str) -> str:
    line = "=" * max(10, len(title) + 6)
    return f"{_c('1;36')}{line}\n== {title} ==\n{line}{_reset()}"

def _dim(text: str) -> str:
    return f"{_c('2')}{text}{_reset()}"


def _pp(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(obj)


@dataclass
class ChatSession:
    broker: Broker
    trading: TradingConfig
    llm: LLMConfig
    client: LLMClient
    risk: RiskManager
    confirm_token: str | None = None
    stream: bool = True
    show_raw: bool = False
    max_tool_rounds: int = 5

    def __post_init__(self) -> None:
        # Gemini `contents` format, stored verbatim for thought-signature correctness.
        self._contents: list[dict[str, object]] = []

    def add_user_message(self, text: str) -> None:
        self._contents.append({"role": "user", "parts": [{"text": str(text)}]})

    def run_turn(
        self,
        *,
        on_stream_token: Callable[[str], None] | None = None,
        on_tool_executed: Callable[[ToolCall, bool, Any], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> ChatModelReply:
        """
        Executes one assistant turn using Gemini function calling.

        The model can call OMS tools. We execute the tools and send functionResponse parts back
        until the model returns a final text answer (no more function calls).
        """
        if not self.llm.enabled or self.llm.provider != "gemini":
            raise RuntimeError("Chat requires LLM_ENABLED=true and LLM_PROVIDER=gemini")
        if not self.llm.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY must be set")

        oms = OrderManager(self.broker, self.trading, confirm_token=self.confirm_token)
        try:
            tool_decl = {"functionDeclarations": gemini_function_declarations()}
            tools = [tool_decl]
            use_search = bool(self.llm.gemini_use_google_search)

            assistant_acc: list[str] = []

            for _ in range(int(self.max_tool_rounds)):
                if on_status is not None:
                    on_status("Thinking…")

                try:
                    model_content, emitted = self._call_model_with_tools(
                        tools=tools,
                        use_google_search=use_search,
                        on_stream_token=on_stream_token if self.stream else None,
                    )
                except Exception as exc:
                    # If Google Search is enabled, retry once without it (some API configs disallow mixing tools).
                    if use_search:
                        try:
                            model_content, emitted = self._call_model_with_tools(
                                tools=tools,
                                use_google_search=False,
                                on_stream_token=on_stream_token if self.stream else None,
                            )
                        except Exception as exc2:
                            msg = f"LLM request failed: {exc2}"
                            if on_status is not None:
                                on_status("Ready")
                            return ChatModelReply(assistant_message=msg, tool_calls=[])
                    else:
                        msg = f"LLM request failed: {exc}"
                        if on_status is not None:
                            on_status("Ready")
                        return ChatModelReply(assistant_message=msg, tool_calls=[])

                if emitted:
                    assistant_acc.append(emitted)

                # Persist model content verbatim for thought-signature correctness.
                self._contents.append(model_content)

                calls = _extract_function_calls(model_content)
                if not calls:
                    if on_status is not None:
                        on_status("Ready")
                    return ChatModelReply(assistant_message="".join(assistant_acc).strip(), tool_calls=[])

                function_response_parts: list[dict[str, object]] = []
                tool_calls: list[ToolCall] = []
                for call in calls:
                    tool_call = ToolCall(name=call.name, args=call.args, call_id=None)
                    tool_calls.append(tool_call)
                    if on_status is not None:
                        on_status(f"Executing {tool_call.name}…")
                    ok, result = self._execute_tool(tool_call, oms)
                    if on_tool_executed is not None:
                        on_tool_executed(tool_call, ok, result)
                    function_response_parts.append(
                        {
                            "functionResponse": {
                                "name": tool_call.name,
                                "response": {"ok": bool(ok), "result": result},
                            }
                        }
                    )

                # Per Gemini docs: group parallel function responses together in one content block.
                self._contents.append({"role": "user", "parts": function_response_parts})

            if on_status is not None:
                on_status("Ready")
            return ChatModelReply(assistant_message="".join(assistant_acc).strip() + "\n(Stopped after max_tool_rounds.)", tool_calls=[])
        finally:
            oms.close()

    def _call_model_with_tools(
        self,
        *,
        tools: list[dict[str, object]],
        use_google_search: bool,
        on_stream_token: Callable[[str], None] | None,
    ) -> tuple[dict[str, object], str]:
        """
        Call Gemini with function calling enabled.

        Returns:
          (model_content, aggregated_text_emitted)
        """
        if not self.stream or on_stream_token is None:
            data = self.client.generate_content(
                contents=list(self._contents),
                system=_SYSTEM_PROMPT,
                tools=tools,
                use_google_search=use_google_search,
            )
            content = _extract_first_candidate_content(data)
            return content, _extract_text_from_content(content)

        parts_acc: list[dict[str, object]] = []
        text_acc: list[str] = []
        for evt in self.client.stream_generate_content(
            contents=list(self._contents),
            system=_SYSTEM_PROMPT,
            tools=tools,
            use_google_search=use_google_search,
        ):
            content = _maybe_extract_candidate_content(evt)
            if not content:
                continue
            for part in list(content.get("parts") or []):
                if not isinstance(part, dict):
                    continue
                parts_acc.append(part)
                t = part.get("text")
                if t:
                    s = str(t)
                    text_acc.append(s)
                    on_stream_token(s)

        model_content: dict[str, object] = {"role": "model", "parts": parts_acc}
        return model_content, "".join(text_acc)

    def _execute_tool(self, call: ToolCall, oms: OrderManager) -> tuple[bool, Any]:
        try:
            result = dispatch_tool(
                call_name=call.name,
                call_args=call.args,
                broker=self.broker,
                oms=oms,
                allowed_kinds=self.llm.allowed_kinds(),
                allowed_symbols=self.llm.allowed_symbols(),
            )
            return True, result
        except ToolError as exc:
            return False, {"error": str(exc)}
        except Exception as exc:
            return False, {"error": str(exc)}


_SYSTEM_PROMPT = (
    "You are a trading agent operating a PAPER-trading only OMS.\n"
    "You may call provided functions (tools) to fetch data and to place/modify/cancel orders.\n"
    "Rules:\n"
    "- Never claim an order is placed unless the tool returns success.\n"
    "- If you are unsure, call get_snapshot/get_positions/get_account/list_open_orders.\n"
    "- Prefer limit orders when appropriate; be explicit about params.\n"
    "- Use concise, readable Markdown in normal text.\n"
)


def _extract_first_candidate_content(data: dict[str, object]) -> dict[str, object]:
    try:
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            c0 = candidates[0]
            if isinstance(c0, dict):
                content = c0.get("content")
                if isinstance(content, dict):
                    return content  # type: ignore[return-value]
    except Exception:
        pass
    raise RuntimeError(f"Unexpected Gemini response shape: {data}")


def _maybe_extract_candidate_content(evt: object) -> dict[str, object] | None:
    try:
        if not isinstance(evt, dict):
            return None
        candidates = evt.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        c0 = candidates[0]
        if not isinstance(c0, dict):
            return None
        content = c0.get("content")
        if not isinstance(content, dict):
            return None
        return content  # type: ignore[return-value]
    except Exception:
        return None


def _extract_text_from_content(content: dict[str, object]) -> str:
    out: list[str] = []
    for part in list(content.get("parts") or []):
        if isinstance(part, dict) and part.get("text"):
            out.append(str(part.get("text")))
    return "".join(out)


@dataclass(frozen=True)
class _FnCall:
    name: str
    args: dict[str, Any]


def _extract_function_calls(content: dict[str, object]) -> list[_FnCall]:
    calls: list[_FnCall] = []
    for part in list(content.get("parts") or []):
        if not isinstance(part, dict):
            continue
        fc = part.get("functionCall")
        if not isinstance(fc, dict):
            continue
        name = str(fc.get("name", "")).strip()
        args = fc.get("args")
        if not name:
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append(_FnCall(name=name, args=dict(args)))
    return calls


def _load_dotenv_if_present() -> None:
    if not os.path.exists(".env"):
        return
    with open(".env", "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if os.getenv(k) is None or os.getenv(k) == "":
                os.environ[k] = v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading_algo.llm.chat", description="Interactive terminal chat (Gemini + OMS tools)")
    p.add_argument("--broker", choices=["ibkr", "sim"], default=None, help="Override TRADING_BROKER")
    p.add_argument(
        "--confirm-token",
        default=None,
        help="Must match TRADING_ORDER_TOKEN if TRADING_CONFIRM_TOKEN_REQUIRED=true",
    )
    p.add_argument("--ibkr-host", default=None)
    p.add_argument("--ibkr-port", default=None)
    p.add_argument("--ibkr-client-id", default=None)
    p.add_argument("--no-stream", action="store_true", help="Disable Gemini streaming")
    p.add_argument("--show-raw", action="store_true", help="Also store/display raw model JSON")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--quiet-ibkr-logs", action="store_true", help="Reduce noisy ib_insync portfolio logs")
    p.add_argument("--ui", choices=["auto", "plain", "rich", "tui"], default="auto", help="Terminal UI mode")
    return p


def main(argv: list[str] | None = None) -> int:
    global _COLOR_ENABLED
    _load_dotenv_if_present()
    cfg = TradingConfig.from_env()
    llm_cfg = LLMConfig.from_env()
    args = build_parser().parse_args(argv)

    if args.no_color:
        _COLOR_ENABLED = False

    # Enable basic line editing / history if available (no external deps).
    try:  # pragma: no cover
        import readline  # noqa: F401
    except Exception:
        pass

    ibkr = IBKRConfig(
        host=args.ibkr_host or cfg.ibkr.host,
        port=int(args.ibkr_port or cfg.ibkr.port),
        client_id=int(args.ibkr_client_id or cfg.ibkr.client_id),
    )
    cfg = TradingConfig(
        broker=args.broker or cfg.broker,
        live_enabled=cfg.live_enabled,
        require_paper=True,
        dry_run=cfg.dry_run,
        order_token=cfg.order_token,
        confirm_token_required=cfg.confirm_token_required,
        db_path=cfg.db_path,
        poll_seconds=cfg.poll_seconds,
        ibkr=ibkr,
    )

    if not str(llm_cfg.gemini_model).startswith("gemini-3"):
        raise SystemExit(
            f"Refusing to run with GEMINI_MODEL={llm_cfg.gemini_model!r}; set GEMINI_MODEL to a Gemini 3 model id "
            "(e.g. gemini-3-pro-preview or gemini-3-flash-preview)."
        )
    if not llm_cfg.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is empty; set it in .env or your shell to use chat.")

    if cfg.broker == "sim":
        from trading_algo.broker.sim import SimBroker

        broker: Broker = SimBroker()
        # Provide one default quote so "get_snapshot" works immediately.
        from trading_algo.instruments import InstrumentSpec

        broker.connect()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)  # type: ignore[attr-defined]
        broker.disconnect()
    else:
        from trading_algo.broker.ibkr import IBKRBroker

        broker = IBKRBroker(cfg.ibkr, require_paper=True)

    effective_model = llm_cfg.normalized_gemini_model()
    client = GeminiClient(api_key=llm_cfg.gemini_api_key or "", model=effective_model)

    if args.quiet_ibkr_logs:
        logging.getLogger("ib_insync").setLevel(logging.WARNING)
        logging.getLogger("ib_insync.client").setLevel(logging.WARNING)
        logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)

    use_rich = _should_use_rich(args.ui)
    if str(args.ui).lower() in {"tui", "pt", "prompt"}:
        use_rich = False
        ui = _PlainUI()
    else:
        ui = _RichUI() if use_rich else _PlainUI()

    ui.header(
        broker=cfg.broker,
        dry_run=cfg.dry_run,
        live_enabled=cfg.live_enabled,
        db_path=(cfg.db_path or "off"),
        allowed_symbols=sorted(llm_cfg.allowed_symbols()),
        model=effective_model,
        stream=(not args.no_stream),
        google_search=bool(llm_cfg.gemini_use_google_search),
        gemini_key=(llm_cfg.gemini_api_key or ""),
        model_normalized=(effective_model != llm_cfg.gemini_model),
        original_model=llm_cfg.gemini_model,
    )

    session = ChatSession(
        broker=broker,
        trading=cfg,
        llm=llm_cfg,
        client=client,
        risk=RiskManager(RiskLimits()),
        confirm_token=args.confirm_token,
        stream=not bool(args.no_stream),
        show_raw=bool(args.show_raw),
    )

    # IMPORTANT:
    # IBKR + ib_insync are not safe to call across threads. The TUI runs model/tool
    # turns in a background thread; to avoid cross-thread ib_insync usage we connect
    # and disconnect the broker inside that worker thread per turn.
    tui_mode = str(args.ui).lower() in {"tui", "pt", "prompt"}

    if not tui_mode:
        broker.connect()
    try:
        if tui_mode:
            def _unlock(token: str) -> str:
                if not cfg.confirm_token_required:
                    return "Token gate is disabled (TRADING_CONFIRM_TOKEN_REQUIRED=false); /unlock is not required."
                expected = cfg.order_token
                if not expected:
                    return "TRADING_ORDER_TOKEN is not set; set it first to enable order sending."
                if str(token) != str(expected):
                    return "Unlock failed: token mismatch. Order sending remains locked."
                session.confirm_token = str(token)
                return "Unlocked for this session. Order sending via OMS is now enabled (still paper-only)."

            def _lock() -> str:
                if not cfg.confirm_token_required:
                    return "Token gate is disabled (TRADING_CONFIRM_TOKEN_REQUIRED=false); /lock has no effect."
                session.confirm_token = None
                return "Locked. Order sending via OMS is disabled."

            def _run_turn(user_text, stream_cb, tool_cb, status_cb) -> str:
                broker.connect()
                try:
                    session.add_user_message(str(user_text))
                    reply = session.run_turn(
                        on_stream_token=stream_cb,
                        on_tool_executed=tool_cb,
                        on_status=status_cb,
                    )
                    return reply.assistant_message
                finally:
                    broker.disconnect()

            run_tui(run_turn=_run_turn, unlock=_unlock, lock=_lock)  # type: ignore[arg-type]
            return 0

        while True:
            try:
                user = ui.prompt()
            except (EOFError, KeyboardInterrupt):
                ui.println("")
                return 0
            if not user:
                continue
            if user in {"/quit", "/exit"}:
                return 0
            if user.strip() == "/unlock":
                import getpass

                if not cfg.confirm_token_required:
                    ui.println("Token gate is disabled (TRADING_CONFIRM_TOKEN_REQUIRED=false); /unlock is not required.")
                    continue
                if not cfg.order_token:
                    ui.error("TRADING_ORDER_TOKEN is not set; set it first to enable order sending.")
                    continue
                token = getpass.getpass("TRADING_ORDER_TOKEN: ")
                if token != cfg.order_token:
                    ui.error("Unlock failed: token mismatch.")
                    continue
                session.confirm_token = token
                ui.println("Unlocked for this session.")
                continue
            if user.strip() == "/lock":
                if not cfg.confirm_token_required:
                    ui.println("Token gate is disabled (TRADING_CONFIRM_TOKEN_REQUIRED=false); /lock has no effect.")
                    continue
                session.confirm_token = None
                ui.println("Locked.")
                continue
            if user == "/help":
                ui.show_help(list_tools())
                continue
            if user == "/tools":
                ui.show_tools(list_tools())
                continue
            if user == "/config":
                ui.show_config(cfg, llm_cfg, effective_model, confirm_token=bool(args.confirm_token))
                continue
            if user == "/clear":  # pragma: no cover
                ui.clear()
                continue

            session.add_user_message(user)

            ui.assistant_prefix()

            def _on_token(tok: str) -> None:
                ui.stream(tok)

            executed: list[tuple[ToolCall, bool, Any]] = []

            def _on_tool(call: ToolCall, ok: bool, result: Any) -> None:
                executed.append((call, ok, result))

            try:
                reply = session.run_turn(on_stream_token=_on_token if session.stream else None, on_tool_executed=_on_tool)
            except Exception as exc:
                if session.stream:
                    ui.println("")
                ui.error(str(exc))
                continue
            if session.stream:
                ui.println("")
            if not session.stream:
                ui.println(reply.assistant_message.strip() or "(no assistant_message returned)")
            if executed:
                ui.show_tool_calls(executed)
    finally:
        if not tui_mode:
            broker.disconnect()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


def _should_use_rich(mode: str) -> bool:
    mode = str(mode or "auto").strip().lower()
    if mode == "plain":
        return False
    if mode == "rich":
        return True
    try:  # pragma: no cover
        import rich  # noqa: F401

        return True
    except Exception:
        return False


class _UIBase:
    def header(
        self,
        *,
        broker: str,
        dry_run: bool,
        live_enabled: bool,
        db_path: str,
        allowed_symbols: list[str],
        model: str,
        stream: bool,
        google_search: bool,
        gemini_key: str,
        model_normalized: bool,
        original_model: str,
    ) -> None:
        raise NotImplementedError

    def prompt(self) -> str:
        raise NotImplementedError

    def assistant_prefix(self) -> None:
        raise NotImplementedError

    def stream(self, text: str) -> None:
        raise NotImplementedError

    def println(self, text: str) -> None:
        raise NotImplementedError

    def error(self, text: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def show_help(self, tools: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def show_tools(self, tools: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def show_config(self, cfg: TradingConfig, llm_cfg: LLMConfig, model: str, *, confirm_token: bool) -> None:
        raise NotImplementedError

    def show_tool_calls(self, executed: list[tuple[ToolCall, bool, Any]]) -> None:
        raise NotImplementedError


class _PlainUI(_UIBase):
    def header(
        self,
        *,
        broker: str,
        dry_run: bool,
        live_enabled: bool,
        db_path: str,
        allowed_symbols: list[str],
        model: str,
        stream: bool,
        google_search: bool,
        gemini_key: str,
        model_normalized: bool,
        original_model: str,
    ) -> None:
        print(_banner("IBKR Paper OMS Chat"))
        print(f"broker={broker} dry_run={dry_run} live_enabled={live_enabled} db={db_path}")
        key = gemini_key or ""
        key_hint = f"present(len={len(key)} prefix={key[:4]})" if key else "missing"
        flags = []
        if "," in key:
            flags.append("has_comma")
        if any(c.isspace() for c in key):
            flags.append("has_whitespace")
        if key.startswith(("\"", "'")) or key.endswith(("\"", "'")):
            flags.append("has_quotes")
        if flags:
            key_hint += f" flags={','.join(flags)}"
        sym = ",".join(allowed_symbols) if allowed_symbols else "ALL"
        print(f"allowed_symbols={sym} model={model} stream={stream} google_search={google_search} gemini_key={key_hint}")
        if model_normalized:
            print(f"note: normalized GEMINI_MODEL from {original_model!r} -> {model!r}")
        print(_dim("Commands: /help /tools /config /clear /exit"))

    def prompt(self) -> str:
        return input(f"{_c('1;32')}you>{_reset()} ").strip()

    def assistant_prefix(self) -> None:
        print(f"{_c('1;34')}Gemini>{_reset()} ", end="", flush=True)

    def stream(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def println(self, text: str) -> None:
        print(text)

    def error(self, text: str) -> None:
        print(f"{_c('1;31')}error:{_reset()} {text}")

    def clear(self) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print(_banner("IBKR Paper OMS Chat"))

    def show_help(self, tools: list[dict[str, Any]]) -> None:
        print(_banner("Commands"))
        print("/help  /tools  /config  /clear  /exit")
        print("")
        print("Trading tools the model can call:")
        print(", ".join([t["name"] for t in tools]))

    def show_tools(self, tools: list[dict[str, Any]]) -> None:
        print(_banner("Tools"))
        for t in tools:
            print(f"- {t['name']} args={_pp(t['args'])}")

    def show_config(self, cfg: TradingConfig, llm_cfg: LLMConfig, model: str, *, confirm_token: bool) -> None:
        print(_banner("Config"))
        print(f"broker={cfg.broker}")
        print(f"dry_run={cfg.dry_run} (TRADING_DRY_RUN)")
        print(f"live_enabled={cfg.live_enabled} (TRADING_LIVE_ENABLED)")
        print(f"order_token_set={bool(cfg.order_token)} (TRADING_ORDER_TOKEN)")
        print(f"confirm_token_required={bool(cfg.confirm_token_required)} (TRADING_CONFIRM_TOKEN_REQUIRED)")
        print(f"confirm_token_provided={confirm_token} (--confirm-token)")
        allowed = sorted(llm_cfg.allowed_symbols())
        print(f"allowed_symbols={','.join(allowed) if allowed else 'ALL'}")
        print(f"google_search={llm_cfg.gemini_use_google_search} (LLM_USE_GOOGLE_SEARCH)")
        print(f"model={model} (GEMINI_MODEL)")

    def show_tool_calls(self, executed: list[tuple[ToolCall, bool, Any]]) -> None:
        print(_banner("Tool Calls"))
        for call, ok, result in executed:
            status = "ok" if ok else "error"
            color = "1;32" if ok else "1;31"
            print(f"{_c('1;33')}{call.name}{_reset()} args={_pp(call.args)}")
            print(f"  -> {_c(color)}{status}{_reset()} result={_pp(result)}")


class _RichUI(_UIBase):
    def __init__(self) -> None:
        from rich.console import Console

        self._console = Console()

    def header(
        self,
        *,
        broker: str,
        dry_run: bool,
        live_enabled: bool,
        db_path: str,
        allowed_symbols: list[str],
        model: str,
        stream: bool,
        google_search: bool,
        gemini_key: str,
        model_normalized: bool,
        original_model: str,
    ) -> None:
        from rich.panel import Panel
        from rich.table import Table

        key = gemini_key or ""
        key_hint = f"len={len(key)} prefix={key[:4]}" if key else "missing"
        t = Table.grid(expand=True)
        t.add_column()
        t.add_column(justify="right")
        t.add_row("broker", broker)
        t.add_row("dry_run", str(dry_run))
        t.add_row("live_enabled", str(live_enabled))
        t.add_row("db", db_path)
        t.add_row("allowed_symbols", ",".join(allowed_symbols) if allowed_symbols else "ALL")
        t.add_row("model", model)
        t.add_row("stream", str(stream))
        t.add_row("google_search", str(google_search))
        t.add_row("gemini_key", key_hint)
        if model_normalized:
            t.add_row("model_note", f"{original_model!r} -> {model!r}")
        self._console.print(Panel(t, title="IBKR Paper OMS Chat", border_style="cyan"))
        self._console.print("[dim]Commands: /help /tools /config /clear /exit[/dim]")

    def prompt(self) -> str:
        from rich.prompt import Prompt

        return str(Prompt.ask("[bold green]you[/bold green]")).strip()

    def assistant_prefix(self) -> None:
        self._console.print("[bold blue]Gemini>[/bold blue] ", end="")

    def stream(self, text: str) -> None:
        self._console.print(text, end="", soft_wrap=True)

    def println(self, text: str) -> None:
        self._console.print(text)

    def error(self, text: str) -> None:
        self._console.print(f"[bold red]error:[/bold red] {text}")

    def clear(self) -> None:
        self._console.clear()

    def show_help(self, tools: list[dict[str, Any]]) -> None:
        from rich.panel import Panel

        tool_names = ", ".join([t["name"] for t in tools])
        self._console.print(
            Panel(
                "[bold]/help[/bold]  [bold]/tools[/bold]  [bold]/config[/bold]  [bold]/clear[/bold]  [bold]/exit[/bold]\n\n"
                f"[dim]Tools:[/dim] {tool_names}",
                title="Help",
                border_style="cyan",
            )
        )

    def show_tools(self, tools: list[dict[str, Any]]) -> None:
        from rich.table import Table

        table = Table(title="Tools", show_lines=True)
        table.add_column("name", style="bold")
        table.add_column("args", overflow="fold")
        for t in tools:
            table.add_row(str(t["name"]), _pp(t.get("args")))
        self._console.print(table)

    def show_config(self, cfg: TradingConfig, llm_cfg: LLMConfig, model: str, *, confirm_token: bool) -> None:
        from rich.table import Table

        table = Table(title="Config", show_lines=True)
        table.add_column("key", style="bold")
        table.add_column("value", overflow="fold")
        table.add_row("broker", str(cfg.broker))
        table.add_row("dry_run (TRADING_DRY_RUN)", str(cfg.dry_run))
        table.add_row("live_enabled (TRADING_LIVE_ENABLED)", str(cfg.live_enabled))
        table.add_row("order_token_set (TRADING_ORDER_TOKEN)", str(bool(cfg.order_token)))
        table.add_row("confirm_token_required (TRADING_CONFIRM_TOKEN_REQUIRED)", str(bool(cfg.confirm_token_required)))
        table.add_row("confirm_token_provided (--confirm-token)", str(confirm_token))
        allowed = sorted(llm_cfg.allowed_symbols())
        table.add_row("allowed_symbols", ",".join(allowed) if allowed else "ALL")
        table.add_row("google_search (LLM_USE_GOOGLE_SEARCH)", str(llm_cfg.gemini_use_google_search))
        table.add_row("model (GEMINI_MODEL)", str(model))
        self._console.print(table)

    def show_tool_calls(self, executed: list[tuple[ToolCall, bool, Any]]) -> None:
        from rich.table import Table

        table = Table(title="Tool Calls", show_lines=True)
        table.add_column("tool", style="bold yellow")
        table.add_column("status")
        table.add_column("args", overflow="fold")
        table.add_column("result", overflow="fold")
        for call, ok, result in executed:
            status = "[green]ok[/green]" if ok else "[red]error[/red]"
            table.add_row(call.name, status, _pp(call.args), _pp(result))
        self._console.print(table)
