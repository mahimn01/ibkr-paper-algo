from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError


class LLMClient(Protocol):
    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str: ...
    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False): ...
    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ) -> dict[str, object]: ...
    def stream_generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ): ...


@dataclass(frozen=True)
class GeminiClient(LLMClient):
    """
    Minimal Gemini REST client (no extra deps).

    Docs endpoint pattern (v1beta):
      https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent?key=API_KEY

    If `use_google_search=True`, adds tools=[{googleSearch:{}}] for grounding.
    """

    api_key: str
    model: str = "gemini-3"
    timeout_s: float = 30.0

    def generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False) -> str:
        _validate_api_key(self.api_key)
        if not self.model:
            raise RuntimeError("GEMINI_MODEL is required")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:generateContent"
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            # Gemini 3 docs recommend temperature default 1.0.
            "generationConfig": {"temperature": 1.0, "thinkingConfig": {"thinkingLevel": "high"}},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": str(system)}]}
        if use_google_search:
            payload["tools"] = [{"googleSearch": {}}]

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc
        try:
            # candidates[0].content.parts[].text
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(str(p.get("text", "")) for p in parts)
        except Exception as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc

    def stream_generate(self, *, prompt: str, system: str | None = None, use_google_search: bool = False):
        """
        Streaming version of `generate`.

        Uses `:streamGenerateContent` endpoint and yields text chunks as they arrive.
        The caller is responsible for buffering if it needs the full response.
        """
        _validate_api_key(self.api_key)
        if not self.model:
            raise RuntimeError("GEMINI_MODEL is required")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:streamGenerateContent?alt=sse"
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {"temperature": 1.0, "thinkingConfig": {"thinkingLevel": "high"}},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": str(system)}]}
        if use_google_search:
            payload["tools"] = [{"googleSearch": {}}]

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
                # Gemini streaming responses are SSE. Parse multi-line events robustly.
                for obj in _iter_sse_json_objects(resp):
                    chunk = _extract_text(obj)
                    if chunk:
                        yield chunk
        except HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc

    def generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ) -> dict[str, object]:
        """
        Low-level content generation with optional tool/function calling.
        Returns the parsed JSON response.
        """
        _validate_api_key(self.api_key)
        if not self.model:
            raise RuntimeError("GEMINI_MODEL is required")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:generateContent"
        payload: dict[str, object] = {
            "contents": list(contents),
            "generationConfig": {"temperature": 1.0, "thinkingConfig": {"thinkingLevel": "high"}},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": str(system)}]}

        tools_list: list[dict[str, object]] = []
        if tools:
            tools_list.extend(list(tools))
        if use_google_search:
            tools_list.append({"googleSearch": {}})
        if tools_list:
            payload["tools"] = tools_list

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc
        return data

    def stream_generate_content(
        self,
        *,
        contents: list[dict[str, object]],
        system: str | None = None,
        tools: list[dict[str, object]] | None = None,
        use_google_search: bool = False,
    ):
        """
        Streaming content generation (SSE). Yields parsed JSON event objects.

        NOTE: The caller should consume the entire stream; function call metadata and/or
        thought signatures may arrive in chunks with empty text.
        """
        _validate_api_key(self.api_key)
        if not self.model:
            raise RuntimeError("GEMINI_MODEL is required")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:streamGenerateContent?alt=sse"
        payload: dict[str, object] = {
            "contents": list(contents),
            "generationConfig": {"temperature": 1.0, "thinkingConfig": {"thinkingLevel": "high"}},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": str(system)}]}

        tools_list: list[dict[str, object]] = []
        if tools:
            tools_list.extend(list(tools))
        if use_google_search:
            tools_list.append({"googleSearch": {}})
        if tools_list:
            payload["tools"] = tools_list

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
                for obj in _iter_sse_json_objects(resp):
                    yield obj
        except HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc


def _extract_text(data: object) -> str:
    try:
        obj = data  # type: ignore[assignment]
        parts = obj["candidates"][0]["content"]["parts"]
        return "".join(str(p.get("text", "")) for p in parts)
    except Exception:
        return ""


def _iter_sse_json_objects(resp) -> "list[object] | Any":
    """
    Parse SSE stream into JSON objects.

    SSE events are separated by a blank line. Each event may contain multiple `data:` lines.
    """
    data_lines: list[str] = []
    for raw in resp:
        line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
        if line == "":
            if not data_lines:
                continue
            payload = "\n".join(data_lines).strip()
            data_lines = []
            if not payload or payload == "[DONE]":
                continue
            try:
                yield json.loads(payload)
            except Exception:
                continue
            continue

        # Ignore comments / event types.
        if line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue
        # Some implementations may send raw JSON without `data:` prefix; support that.
        if line.startswith("{") or line.startswith("["):
            data_lines.append(line)

    # Flush if stream ends without trailing blank line.
    if data_lines:
        payload = "\n".join(data_lines).strip()
        if payload and payload != "[DONE]":
            try:
                yield json.loads(payload)
            except Exception:
                return


def _format_http_error(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    detail = body.strip()
    try:
        j = json.loads(detail)
        # Common shape: {"error":{"code":400,"message":"...","status":"INVALID_ARGUMENT"}}
        if isinstance(j, dict) and "error" in j and isinstance(j["error"], dict):
            err = j["error"]
            msg = err.get("message") or detail
            status = err.get("status") or ""
            code = err.get("code") or exc.code
            return f"Gemini HTTP {code} {status}: {msg}"
    except Exception:
        pass
    return f"Gemini HTTP {exc.code}: {detail or exc.reason}"


def _validate_api_key(api_key: str) -> None:
    """
    Fail fast with actionable errors so we don't end up with opaque HTTP 400s.
    """
    key = str(api_key or "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required")
    # Catch the most common copy/paste issues.
    for bad in [" ", "\t", "\n", "\r"]:
        if bad in key:
            raise RuntimeError("GEMINI_API_KEY contains whitespace; remove spaces/newlines and try again.")
    if "," in key:
        raise RuntimeError("GEMINI_API_KEY contains ',' (did you paste with a trailing comma?)")
    if key.startswith(("\"", "'")) or key.endswith(("\"", "'")):
        raise RuntimeError("GEMINI_API_KEY appears to include quotes; remove them in .env.")
    # Basic sanity: Google API keys generally start with 'AIza'.
    if not key.startswith("AIza"):
        raise RuntimeError("GEMINI_API_KEY format looks wrong (expected to start with 'AIza...').")
