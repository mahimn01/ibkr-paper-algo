from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from trading_algo.llm.config import LLMConfig
from trading_algo.llm.gemini import GeminiClient, LLMClient


@dataclass
class ContextStats:
    approx_tokens: int
    exact_tokens: int | None = None


def approx_token_count_for_contents(*, contents: list[dict[str, object]], system: str | None = None) -> int:
    """
    Very rough heuristic: ~4 chars per token for English.

    This is a safety check to avoid calling `countTokens` on every turn.
    """
    chars = 0
    if system:
        chars += len(str(system))
    for c in contents:
        if not isinstance(c, dict):
            continue
        parts = c.get("parts")
        if not isinstance(parts, list):
            continue
        for p in parts:
            if not isinstance(p, dict):
                continue
            if "text" in p and p.get("text") is not None:
                chars += len(str(p.get("text")))
            if "functionCall" in p and isinstance(p.get("functionCall"), dict):
                chars += len(json.dumps(p.get("functionCall"), default=str))
            if "functionResponse" in p and isinstance(p.get("functionResponse"), dict):
                chars += len(json.dumps(p.get("functionResponse"), default=str))
    return max(1, int(chars / 4))


def _render_for_summary(contents: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role", "")).strip().lower() or "unknown"
        parts = c.get("parts")
        if not isinstance(parts, list):
            continue
        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("text"):
                lines.append(f"{role.upper()}: {str(p.get('text')).strip()}")
                continue
            fc = p.get("functionCall")
            if isinstance(fc, dict):
                name = fc.get("name")
                args = fc.get("args")
                lines.append(f"{role.upper()}: TOOL_CALL {name} args={json.dumps(args, default=str)}")
                continue
            fr = p.get("functionResponse")
            if isinstance(fr, dict):
                name = fr.get("name")
                resp = fr.get("response")
                lines.append(f"{role.upper()}: TOOL_RESULT {name} {json.dumps(resp, default=str)}")
                continue
    return "\n".join(lines).strip()


def _summarize_via_model(
    *,
    client: LLMClient,
    llm_cfg: LLMConfig,
    system_prompt: str,
    summarize_model: str,
    text: str,
) -> str:
    """
    Summarize older conversation into a compact 'memory' message.

    Uses a separate (potentially cheaper) model when the client is GeminiClient.
    """
    # Keep summarization deterministic-ish without fighting Gemini 3 defaults.
    summarizer_system = (
        "You are a conversation memory compressor for a PAPER-trading OMS.\n"
        "Create a compact, lossless-as-possible memory of the conversation.\n"
        "Include:\n"
        "- User objectives and constraints (paper trading only, safety preferences).\n"
        "- Trading constraints, risk limits, and allowed markets/symbols.\n"
        "- Any open orders, key positions/account facts mentioned, and unresolved questions.\n"
        "- Tool outcomes and important errors.\n"
        "Output strictly in Markdown.\n"
        "Keep it concise: <= 2000 characters.\n"
    )

    prompt = (
        "<context>\n"
        f"{text}\n"
        "</context>\n\n"
        "<task>\n"
        "Summarize the conversation above into a durable memory for future turns.\n"
        "</task>\n"
    )

    use_client: LLMClient = client
    if isinstance(client, GeminiClient) and summarize_model and summarize_model != client.model:
        use_client = GeminiClient(api_key=client.api_key, model=summarize_model, timeout_s=client.timeout_s)

    data = use_client.generate_content(
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        system=summarizer_system + "\n\n" + system_prompt,
        tools=None,
        use_google_search=False,
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text_out = "".join(str(p.get("text", "")) for p in parts).strip()
        return text_out or "(summary unavailable)"
    except Exception:
        return "(summary unavailable)"


def maybe_compact_history(
    *,
    client: LLMClient,
    llm_cfg: LLMConfig,
    system_prompt: str,
    contents: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    use_google_search: bool = False,
) -> tuple[list[dict[str, object]], ContextStats, str | None]:
    """
    Compact older history into a single memory message when token budget is exceeded.

    Returns:
      (new_contents, stats, summary_text_or_none)
    """
    approx = approx_token_count_for_contents(contents=contents, system=system_prompt)
    stats = ContextStats(approx_tokens=approx, exact_tokens=None)

    # Cheap early exit: under trigger ratio, skip exact counting/summarization.
    trigger_tokens = int(llm_cfg.max_context_tokens * float(llm_cfg.summarize_trigger_ratio))
    if approx < max(1, trigger_tokens):
        return contents, stats, None

    exact: int | None = None
    try:
        exact = int(
            client.count_tokens(
                contents=list(contents),
                system=system_prompt,
                tools=tools,
                use_google_search=use_google_search,
            )
        )
    except Exception:
        exact = None
    stats.exact_tokens = exact

    over = (exact is not None and exact > llm_cfg.max_context_tokens) or (exact is None and approx > llm_cfg.max_context_tokens)
    if not over:
        return contents, stats, None

    keep_n = max(1, int(llm_cfg.keep_recent_contents))
    if len(contents) <= keep_n + 1:
        return contents, stats, None

    old = contents[:-keep_n]
    recent = contents[-keep_n:]
    rendered = _render_for_summary(old)
    if not rendered:
        return contents, stats, None

    summarize_model = llm_cfg.summarize_model.strip() or llm_cfg.normalized_gemini_model()
    summary = _summarize_via_model(
        client=client,
        llm_cfg=llm_cfg,
        system_prompt=system_prompt,
        summarize_model=summarize_model,
        text=rendered,
    )

    memory = (
        "[conversation_memory]\n"
        f"updated_at_epoch_s={time.time():.0f}\n\n"
        f"{summary}\n"
    )
    new_contents = [{"role": "user", "parts": [{"text": memory}]}] + recent
    return new_contents, stats, summary

