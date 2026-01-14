# Changelog

## Unreleased

### Gemini chat + tools
- Fix duplicated output by separating streamed thought summaries (`[thought]`) from user-visible assistant text.
- Add Google Search prefetch heuristics to avoid unnecessary grounding calls on greetings, while still prefetching for trade/research-like prompts.
- Expand Gemini REST client support for built-in tools (`google_search`, `url_context`, `code_execution`), structured outputs (JSON schema), and streamed thought summaries.
- Add optional official `google-genai` SDK client (`GEMINI_PREFER_SDK=true`) with improved error messaging for network/DNS issues.
- Add token counting + history compaction utilities (token-aware trimming/summarization) to keep long chats stable.
- Add explicit caching support for multi-round tool loops (configurable TTL/min tokens).

### TUI
- Rework TUI transcript rendering for better stability and copy/scroll ergonomics; add best-effort cancel (ESC) and transcript copy helpers.

### Safety/config
- Default allowlist enforcement off (`LLM_ENFORCE_ALLOWLIST=false`) while preserving paper-only trading constraints elsewhere.

### Tests
- Add unit tests covering prefetch behavior, caching/compaction utilities, REST tool payloads, and news tool wiring.

