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
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        if not self.model:
            raise RuntimeError("GEMINI_MODEL is required")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
        )
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": str(system)}]}
        if use_google_search:
            payload["tools"] = [{"googleSearch": {}}]

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
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
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        if not self.model:
            raise RuntimeError("GEMINI_MODEL is required")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model)}:streamGenerateContent?alt=sse&key={urllib.parse.quote(self.api_key)}"
        )
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": str(system)}]}
        if use_google_search:
            payload["tools"] = [{"googleSearch": {}}]

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
                # Gemini streaming responses are typically SSE (lines prefixed with "data: ").
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    chunk = _extract_text(obj)
                    if chunk:
                        yield chunk
        except HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc


def _extract_text(data: object) -> str:
    try:
        obj = data  # type: ignore[assignment]
        parts = obj["candidates"][0]["content"]["parts"]
        return "".join(str(p.get("text", "")) for p in parts)
    except Exception:
        return ""


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
