from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from trading_algo.llm.gemini import LLMClient


class GeminiSDKMissing(RuntimeError):
    pass


def _format_connect_error(exc: Exception) -> str:
    msg = str(exc)
    if "nodename nor servname provided" in msg or "Name or service not known" in msg:
        return (
            "Network/DNS error: unable to resolve 'generativelanguage.googleapis.com'. "
            "Check your internet connection and DNS/VPN/firewall settings, then retry."
        )
    return msg


def _import_sdk():  # pragma: no cover
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        return genai, types
    except Exception as exc:
        raise GeminiSDKMissing(
            "google-genai SDK is not installed. Install with: pip install google-genai"
        ) from exc


@dataclass
class GeminiSDKClient(LLMClient):
    """
    Gemini client backed by the official google-genai SDK.

    This is optional: the codebase still includes a REST client fallback.
    """

    api_key: str
    model: str
    api_version: str = "v1beta"

    def _client(self):  # pragma: no cover
        genai, _ = _import_sdk()
        return genai.Client(api_key=self.api_key, http_options={"api_version": self.api_version})

    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        data = self.generate_content(
            contents=[{"role": "user", "parts": [{"text": str(prompt)}]}],
            system=system,
            tools=None,
            use_google_search=use_google_search,
        )
        # Match REST behavior: candidates[0].content.parts[].text
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(str(p.get("text", "")) for p in parts)
        except Exception:
            return str(data)

    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False):
        for evt in self.stream_generate_content(
            contents=[{"role": "user", "parts": [{"text": str(prompt)}]}],
            system=system,
            tools=None,
            use_google_search=use_google_search,
        ):
            yield evt

    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        use_url_context: bool = False,
        use_code_execution: bool = False,
        include_thoughts: bool = False,
        response_mime_type: str | None = None,
        response_json_schema: dict[str, object] | None = None,
        cached_content: str | None = None,
    ) -> dict[str, object]:
        genai, types = _import_sdk()
        client = self._client()

        tool_list: list[dict[str, Any]] = []
        if tools:
            tool_list.extend([dict(t) for t in tools])
        if use_google_search:
            tool_list.append({"google_search": {}})
        if use_url_context:
            tool_list.append({"url_context": {}})
        if use_code_execution:
            tool_list.append({"code_execution": {}})

        cfg: dict[str, Any] = {"temperature": 1.0, "thinking_config": {"thinking_level": "high"}}
        if include_thoughts:
            cfg["thinking_config"]["include_thoughts"] = True
        if response_mime_type:
            cfg["response_mime_type"] = response_mime_type
        if response_json_schema:
            cfg["response_json_schema"] = response_json_schema
        if tool_list:
            cfg["tools"] = tool_list
        if cached_content:
            cfg["cached_content"] = cached_content

        if system:
            cfg["system_instruction"] = system

        # The SDK returns typed objects; convert to dict for compatibility with the rest of the codebase.
        try:
            resp = client.models.generate_content(model=self.model, contents=contents, config=cfg)
            return json.loads(resp.model_dump_json())
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(_format_connect_error(exc)) from exc

    def stream_generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        use_url_context: bool = False,
        use_code_execution: bool = False,
        include_thoughts: bool = False,
        response_mime_type: str | None = None,
        response_json_schema: dict[str, object] | None = None,
        cached_content: str | None = None,
    ):
        genai, types = _import_sdk()
        client = self._client()

        tool_list: list[dict[str, Any]] = []
        if tools:
            tool_list.extend([dict(t) for t in tools])
        if use_google_search:
            tool_list.append({"google_search": {}})
        if use_url_context:
            tool_list.append({"url_context": {}})
        if use_code_execution:
            tool_list.append({"code_execution": {}})

        cfg: dict[str, Any] = {"temperature": 1.0, "thinking_config": {"thinking_level": "high"}}
        if include_thoughts:
            cfg["thinking_config"]["include_thoughts"] = True
        if response_mime_type:
            cfg["response_mime_type"] = response_mime_type
        if response_json_schema:
            cfg["response_json_schema"] = response_json_schema
        if tool_list:
            cfg["tools"] = tool_list
        if cached_content:
            cfg["cached_content"] = cached_content
        if system:
            cfg["system_instruction"] = system

        try:
            stream = client.models.generate_content_stream(model=self.model, contents=contents, config=cfg)
            for chunk in stream:
                yield json.loads(chunk.model_dump_json())
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(_format_connect_error(exc)) from exc

    def count_tokens(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
        use_url_context: bool = False,
        use_code_execution: bool = False,
        cached_content: str | None = None,
    ) -> int:
        # The SDK offers count_tokens. Keep compatibility with our REST client signature.
        genai, _ = _import_sdk()
        client = self._client()

        tool_list: list[dict[str, Any]] = []
        if tools:
            tool_list.extend([dict(t) for t in tools])
        if use_google_search:
            tool_list.append({"google_search": {}})
        if use_url_context:
            tool_list.append({"url_context": {}})
        if use_code_execution:
            tool_list.append({"code_execution": {}})

        cfg: dict[str, Any] = {}
        if system:
            cfg["system_instruction"] = system
        if tool_list:
            cfg["tools"] = tool_list
        if cached_content:
            cfg["cached_content"] = cached_content

        try:
            resp = client.models.count_tokens(model=self.model, contents=contents, config=cfg or None)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(_format_connect_error(exc)) from exc
        try:
            return int(resp.total_tokens)
        except Exception:
            # Best-effort fallback.
            try:
                d = json.loads(resp.model_dump_json())
                return int(d.get("totalTokens", 0))
            except Exception:
                return 0
