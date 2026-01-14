from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable
import inspect

from trading_algo.broker.base import Broker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.llm.chat_protocol import ChatModelReply, ToolCall
from trading_algo.llm.config import LLMConfig
from trading_algo.llm.context_manager import maybe_compact_history
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

def _call_generate_content(client: LLMClient, **kwargs: object) -> dict[str, object]:
    """
    Call `client.generate_content` while remaining compatible with older test doubles.

    Some unit tests use lightweight fake clients that don't accept newly-added kwargs.
    We filter kwargs based on the callable signature.
    """
    fn = getattr(client, "generate_content")
    try:
        params = set(inspect.signature(fn).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return fn(**filtered)  # type: ignore[misc]
    except Exception:
        # Last-resort: call with the historical baseline args.
        baseline_keys = {"contents", "system", "tools", "use_google_search", "cached_content"}
        filtered = {k: v for k, v in kwargs.items() if k in baseline_keys}
        return fn(**filtered)  # type: ignore[misc]


def _call_stream_generate_content(client: LLMClient, **kwargs: object):
    fn = getattr(client, "stream_generate_content")
    try:
        params = set(inspect.signature(fn).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return fn(**filtered)  # type: ignore[misc]
    except Exception:
        baseline_keys = {"contents", "system", "tools", "use_google_search", "cached_content"}
        filtered = {k: v for k, v in kwargs.items() if k in baseline_keys}
        return fn(**filtered)  # type: ignore[misc]


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
        self._last_usage_metadata: dict[str, object] | None = None
        self._last_grounding_metadata: dict[str, object] | None = None

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
        cache_name: str | None = None
        try:
            tool_decl = {"functionDeclarations": gemini_function_declarations()}
            tools = [tool_decl]
            use_search = bool(self.llm.gemini_use_google_search)

            assistant_acc: list[str] = []

            # Split history into a cacheable prefix and a dynamic suffix (the current user turn).
            prefix: list[dict[str, object]] = []
            suffix: list[dict[str, object]] = []
            if self._contents and isinstance(self._contents[-1], dict) and self._contents[-1].get("role") == "user":
                prefix = list(self._contents[:-1])
                suffix = [self._contents[-1]]
            else:
                prefix = list(self._contents)

            # If enabled, prefetch a grounded research brief via Google Search.
            # NOTE: Gemini built-in tools are not always compatible with custom function calling tool declarations.
            # We run this as a separate request (no functionDeclarations) and feed the result into the tool loop.
            research_brief = None
            prefetch_attempted = False
            if use_search and suffix and isinstance(suffix[0], dict):
                user_text = _extract_text_from_content(suffix[0]) if suffix[0].get("parts") else ""
                if user_text and _should_prefetch_google_search_brief(user_text):
                    prefetch_attempted = True
                    if on_status is not None:
                        on_status("Researching (Google Search)…")
                    research_brief = self._prefetch_google_search_brief(user_text)
                    if research_brief:
                        # Insert as context right before the current user message (suffix).
                        prefix = list(prefix) + [
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": (
                                            "<google_search_brief>\n"
                                            f"{research_brief}\n"
                                            "</google_search_brief>\n"
                                            "Use this as factual, cited context.\n"
                                        )
                                    }
                                ],
                            }
                        ]
            search_unavailable = bool(use_search) and bool(prefetch_attempted) and not bool(research_brief)

            # Token-aware compaction on the stable prefix only (never touch current-turn tool loop content).
            compacted, stats, _summary = maybe_compact_history(
                client=self.client,
                llm_cfg=self.llm,
                system_prompt=_SYSTEM_PROMPT,
                contents=prefix,
                tools=tools,
                use_google_search=False,
            )
            if compacted != prefix:
                prefix = compacted
                # Replace stored history (keep current user message at the end).
                self._contents = list(prefix) + list(suffix)
            else:
                # Ensure our in-memory contents reflect any inserted research brief.
                self._contents = list(prefix) + list(suffix)

            # Optional explicit caching to reduce costs for multi-call tool loops.
            dynamic_contents: list[dict[str, object]] | None = None
            if (
                self.llm.gemini_explicit_caching
                and hasattr(self.client, "create_cache")
                and hasattr(self.client, "delete_cache")
                and prefix
                and int((stats.exact_tokens or stats.approx_tokens)) >= int(self.llm.gemini_cache_min_tokens)
            ):
                cache_name = getattr(self.client, "create_cache")(
                    contents=list(prefix),
                    system=_SYSTEM_PROMPT,
                    ttl_seconds=int(self.llm.gemini_cache_ttl_seconds),
                    display_name="oms-chat-turn",
                )
                dynamic_contents = list(suffix)

            for _ in range(int(self.max_tool_rounds)):
                if on_status is not None:
                    on_status("Thinking…")

                try:
                    model_content, emitted = self._call_model_with_tools(
                        contents=(dynamic_contents if dynamic_contents is not None else list(self._contents)),
                        system=(None if dynamic_contents is not None else _SYSTEM_PROMPT),
                        tools=tools,
                        # Built-in tools + function calling are not reliably supported together.
                        # Google Search grounding is handled via the research prefetch step above.
                        use_google_search=False,
                        on_stream_token=on_stream_token if self.stream else None,
                        cached_content=cache_name,
                    )
                except Exception as exc:
                    msg = f"LLM request failed: {exc}"
                    if on_status is not None:
                        on_status("Ready")
                    return ChatModelReply(assistant_message=msg, tool_calls=[])

                if emitted:
                    assistant_acc.append(emitted)

                # Persist model content verbatim for thought-signature correctness.
                self._contents.append(model_content)
                if dynamic_contents is not None:
                    dynamic_contents.append(model_content)

                calls = _extract_function_calls(model_content)
                if not calls:
                    msg = "".join(assistant_acc).strip()
                    if search_unavailable:
                        msg = ("_(note: Google Search grounding is enabled, but no search brief was available for this turn.)_\n\n" + msg).strip()
                    msg = _maybe_add_citations(msg, self._last_grounding_metadata)
                    if on_status is not None:
                        on_status(_format_ready_status(self._last_usage_metadata))
                    return ChatModelReply(assistant_message=msg, tool_calls=[])

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
                fr_block = {"role": "user", "parts": function_response_parts}
                self._contents.append(fr_block)
                if dynamic_contents is not None:
                    dynamic_contents.append(fr_block)

            if on_status is not None:
                on_status(_format_ready_status(self._last_usage_metadata))
            return ChatModelReply(assistant_message="".join(assistant_acc).strip() + "\n(Stopped after max_tool_rounds.)", tool_calls=[])
        finally:
            oms.close()
            if cache_name and hasattr(self.client, "delete_cache") and self.llm.gemini_cache_delete_after_turn:
                try:
                    getattr(self.client, "delete_cache")(cache_name)
                except Exception:
                    pass

    def _call_model_with_tools(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None,
        tools: list[dict[str, object]],
        use_google_search: bool,
        on_stream_token: Callable[[str], None] | None,
        cached_content: str | None,
    ) -> tuple[dict[str, object], str]:
        """
        Call Gemini with function calling enabled.

        Returns:
          (model_content, aggregated_text_emitted)
        """
        if not self.stream or on_stream_token is None:
            data = _call_generate_content(
                self.client,
                contents=list(contents),
                system=system,
                tools=tools,
                use_google_search=use_google_search,
                use_url_context=False,
                use_code_execution=False,
                include_thoughts=bool(getattr(self.llm, "gemini_include_thoughts", False)),
                cached_content=cached_content,
            )
            self._last_usage_metadata = _extract_usage_metadata(data)
            self._last_grounding_metadata = _extract_grounding_metadata(data)
            content = _extract_first_candidate_content(data)
            # Do not treat "thought" parts as user-visible assistant text.
            return content, _extract_text_from_content(content, include_thoughts=False)

        parts_acc: list[dict[str, object]] = []
        text_acc: list[str] = []
        for evt in _call_stream_generate_content(
            self.client,
            contents=list(contents),
            system=system,
            tools=tools,
            use_google_search=use_google_search,
            use_url_context=False,
            use_code_execution=False,
            include_thoughts=bool(getattr(self.llm, "gemini_include_thoughts", False)),
            cached_content=cached_content,
        ):
            if isinstance(evt, dict):
                # Streaming usageMetadata is typically present in a final chunk.
                um = evt.get("usageMetadata")
                if isinstance(um, dict):
                    self._last_usage_metadata = dict(um)
            content = _maybe_extract_candidate_content(evt)
            if not content:
                continue
            for part in list(content.get("parts") or []):
                if not isinstance(part, dict):
                    continue
                parts_acc.append(part)
                # Stream thought summaries if present.
                if bool(getattr(self.llm, "gemini_include_thoughts", False)) and part.get("thought") and part.get("text"):
                    on_stream_token(f"\n\n[thought] {str(part.get('text')).strip()}\n")
                    # Thought summaries are not part of the user-visible assistant message.
                    continue
                t = part.get("text")
                if t:
                    s = str(t)
                    text_acc.append(s)
                    on_stream_token(s)
            gm = _maybe_extract_grounding_metadata_from_event(evt)
            if gm is not None:
                self._last_grounding_metadata = gm

        model_content: dict[str, object] = {"role": "model", "parts": parts_acc}
        return model_content, "".join(text_acc)

    def _prefetch_google_search_brief(self, user_text: str) -> str | None:
        """
        Fetch a short grounded research brief using Gemini's built-in Google Search tool.

        This runs without custom function tools and returns text with citations inserted.
        """
        try:
            data = _call_generate_content(
                self.client,
                contents=[{"role": "user", "parts": [{"text": str(user_text)}]}],
                system=(
                    "You are an expert trader doing rapid, factual research.\n"
                    "If using Google Search grounding, summarize the most relevant recent information.\n"
                    "Return:\n"
                    "- 3-6 bullet points\n"
                    "- include sources via citations\n"
                    "Keep it short.\n"
                ),
                tools=None,
                use_google_search=True,
                use_url_context=False,
                use_code_execution=False,
                include_thoughts=bool(getattr(self.llm, "gemini_include_thoughts", False)),
                cached_content=None,
            )
            gm = _extract_grounding_metadata(data)
            content = _extract_first_candidate_content(data)
            text = _extract_text_from_content(content).strip()
            if not text:
                return None
            return _maybe_add_citations(text, gm)
        except Exception:
            return None

    def _execute_tool(self, call: ToolCall, oms: OrderManager) -> tuple[bool, Any]:
        try:
                result = dispatch_tool(
                    call_name=call.name,
                    call_args=call.args,
                    broker=self.broker,
                    oms=oms,
                    allowed_kinds=self.llm.allowed_kinds(),
                    allowed_symbols=self.llm.allowed_symbols(),
                    enforce_allowlist=bool(getattr(self.llm, "enforce_allowlist", False)),
                    llm_client=self.client,
                )
                return True, result
        except ToolError as exc:
            return False, {"error": str(exc)}
        except Exception as exc:
            return False, {"error": str(exc)}


_SYSTEM_PROMPT = (
    "You are **Gemini**, an autonomous trading agent operating a PAPER-trading only OMS.\n"
    "Your job is to research, reason, and execute trades (paper only) with high precision.\n"
    "\n"
    "You have these capabilities:\n"
    "- OMS tools (function calling): market snapshots, positions/account, open orders, place/modify/cancel orders, OMS reconcile/track.\n"
    "- IBKR news tools: list_news_providers, get_historical_news, get_news_article.\n"
    "- Web research: you can call `research_web` to get a short, cited web-grounded brief (optionally with URL context and code execution).\n"
    "- Optional Google Search grounding: the system may provide a `<google_search_brief>` context block with citations.\n"
    "\n"
    "Rules:\n"
    "- PAPER ONLY: never attempt live trading.\n"
    "- Truthfulness: never claim an order was placed/modified/cancelled unless the tool result returns ok=true.\n"
    "- If uncertain, fetch data via tools (get_snapshot/get_positions/get_account/list_open_orders/news tools).\n"
    "- If a `<google_search_brief>` is present, treat it as grounded context and preserve its citations in your reasoning.\n"
    "- Prefer limit orders when appropriate; be explicit about parameters and assumptions.\n"
    "- Output: concise, readable Markdown.\n"
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


def _extract_text_from_content(content: dict[str, object], *, include_thoughts: bool = False) -> str:
    out: list[str] = []
    for part in list(content.get("parts") or []):
        if not isinstance(part, dict) or not part.get("text"):
            continue
        if not include_thoughts and part.get("thought"):
            continue
        if part.get("text"):
            out.append(str(part.get("text")))
    return "".join(out)


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "hola",
    "howdy",
    "good morning",
    "good afternoon",
    "good evening",
}


def _should_prefetch_google_search_brief(user_text: str) -> bool:
    """
    Avoid burning a Google Search request on non-search-worthy inputs (e.g. greetings).
    Keep web search enabled by default, but only prefetch when it is likely useful.
    """
    text = user_text.strip()
    if not text:
        return False
    lower = text.lower().strip()
    if lower in _GREETINGS:
        return False
    if len(lower) <= 12 and lower.replace("!", "").replace(".", "") in _GREETINGS:
        return False

    # If it looks like a question or explicitly requests research, prefetch.
    research_keywords = (
        "news",
        "headline",
        "latest",
        "today",
        "yesterday",
        "this week",
        "what happened",
        "why",
        "how",
        "when",
        "search",
        "web",
        "research",
        "article",
        "source",
        "cite",
        "citation",
        "earnings",
        "macro",
        "cpi",
        "fed",
        "rate",
        "inflation",
    )
    if "?" in lower:
        return True
    if any(k in lower for k in research_keywords):
        return True

    # URLs or tickers/symbol-like tokens usually benefit from grounding.
    if "http://" in lower or "https://" in lower or "www." in lower:
        return True

    tokens = [t.strip(".,:;()[]{}<>\"'") for t in text.split()]
    if any(t.isdigit() for t in tokens):
        return True
    if any(t.isupper() and 1 <= len(t) <= 6 and t.isalpha() for t in tokens):
        return True

    # For longer prompts, a brief can still help.
    return len(lower) >= 80


def _extract_usage_metadata(data: dict[str, object]) -> dict[str, object] | None:
    um = data.get("usageMetadata")
    return dict(um) if isinstance(um, dict) else None


def _extract_grounding_metadata(data: dict[str, object]) -> dict[str, object] | None:
    try:
        c0 = (data.get("candidates") or [])[0]
        if not isinstance(c0, dict):
            return None
        gm = c0.get("groundingMetadata")
        return dict(gm) if isinstance(gm, dict) else None
    except Exception:
        return None


def _maybe_extract_grounding_metadata_from_event(evt: object) -> dict[str, object] | None:
    try:
        if not isinstance(evt, dict):
            return None
        candidates = evt.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        c0 = candidates[0]
        if not isinstance(c0, dict):
            return None
        gm = c0.get("groundingMetadata")
        return dict(gm) if isinstance(gm, dict) else None
    except Exception:
        return None


def _format_ready_status(usage: dict[str, object] | None) -> str:
    if not usage:
        return "Ready"
    try:
        prompt = usage.get("promptTokenCount")
        out = usage.get("candidatesTokenCount")
        total = usage.get("totalTokenCount")
        cached = usage.get("cachedContentTokenCount")
        bits = []
        if isinstance(prompt, int):
            bits.append(f"p={prompt}")
        if isinstance(out, int):
            bits.append(f"o={out}")
        if isinstance(total, int):
            bits.append(f"t={total}")
        if isinstance(cached, int) and cached > 0:
            bits.append(f"cached={cached}")
        if bits:
            return "Ready · " + " ".join(bits)
    except Exception:
        pass
    return "Ready"


def _maybe_add_citations(text: str, grounding: dict[str, object] | None) -> str:
    if not grounding or not text:
        return text
    supports = grounding.get("groundingSupports")
    chunks = grounding.get("groundingChunks")
    if not isinstance(supports, list) or not isinstance(chunks, list):
        return text

    # Build index->uri map.
    uris: list[str | None] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            uris.append(None)
            continue
        web = ch.get("web")
        if isinstance(web, dict) and isinstance(web.get("uri"), str):
            uris.append(str(web.get("uri")))
            continue
        uris.append(None)

    def _end_index(support: dict[str, object]) -> int:
        seg = support.get("segment")
        if isinstance(seg, dict) and isinstance(seg.get("endIndex"), int):
            return int(seg.get("endIndex"))
        return 0

    sorted_supports = sorted([s for s in supports if isinstance(s, dict)], key=_end_index, reverse=True)
    out = text
    for s in sorted_supports:
        seg = s.get("segment")
        if not isinstance(seg, dict):
            continue
        end = seg.get("endIndex")
        idxs = s.get("groundingChunkIndices")
        if not isinstance(end, int) or not isinstance(idxs, list) or end <= 0:
            continue
        links: list[str] = []
        for i in idxs:
            if not isinstance(i, int):
                continue
            uri = uris[i] if 0 <= i < len(uris) else None
            if uri:
                links.append(f"[{i+1}]({uri})")
        if not links:
            continue
        citation = " " + ", ".join(links)
        out = out[:end] + citation + out[end:]
    return out


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

    if not llm_cfg.enabled or llm_cfg.provider != "gemini":
        raise SystemExit(
            "Chat requires Gemini to be enabled.\n"
            "Set in `.env` (or your shell):\n"
            "  LLM_ENABLED=true\n"
            "  LLM_PROVIDER=gemini\n"
            "And ensure `GEMINI_API_KEY` is set."
        )

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
    client: LLMClient
    if llm_cfg.gemini_prefer_sdk:
        try:
            from trading_algo.llm.gemini_sdk import GeminiSDKClient

            client = GeminiSDKClient(api_key=llm_cfg.gemini_api_key or "", model=effective_model)
        except Exception:
            client = GeminiClient(api_key=llm_cfg.gemini_api_key or "", model=effective_model)
    else:
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
        print(f"enforce_allowlist={bool(getattr(llm_cfg, 'enforce_allowlist', False))} (LLM_ENFORCE_ALLOWLIST)")
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
