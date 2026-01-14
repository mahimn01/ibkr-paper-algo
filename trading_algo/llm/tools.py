from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from trading_algo.broker.base import Broker, OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument
from trading_algo.market_data import MarketDataClient, MarketDataConfig
from trading_algo.llm.gemini import GeminiClient, LLMClient
from trading_algo.oms import OrderManager


class ToolError(RuntimeError):
    pass


def list_tools() -> list[dict[str, Any]]:
    """
    For display and Gemini function calling declarations.
    """
    return [
        {"name": "get_snapshot", "args": {"kind": "STK|FUT|FX|OPT", "symbol": "str", "exchange": "str?", "currency": "str?", "expiry": "str?", "right": "C|P?", "strike": "float?", "multiplier": "str?"}},
        {"name": "get_positions", "args": {}},
        {"name": "get_account", "args": {}},
        {"name": "list_open_orders", "args": {}},
        {"name": "list_news_providers", "args": {}},
        {"name": "get_historical_news", "args": {"kind": "STK|FUT|FX|OPT", "symbol": "str", "provider_codes": "list[str]?", "start_datetime": "str?", "end_datetime": "str?", "max_results": "int?"}},
        {"name": "get_news_article", "args": {"provider_code": "str", "article_id": "str", "format": "TEXT|HTML?"}},
        {"name": "research_web", "args": {"query": "str", "urls": "list[str]?", "use_code_execution": "bool?", "max_bullets": "int?"}},
        {"name": "place_order", "args": {"order": {"instrument": {"kind": "STK|FUT|FX|OPT", "symbol": "str"}, "side": "BUY|SELL", "qty": "float", "type": "MKT|LMT|STP|STPLMT"}}},
        {"name": "modify_order", "args": {"order_id": "str", "order": {"...": "same as place_order"}}},
        {"name": "cancel_order", "args": {"order_id": "str"}},
        {"name": "oms_reconcile", "args": {}},
        {"name": "oms_track", "args": {"poll_seconds": "float?", "timeout_seconds": "float?"}},
    ]


def gemini_function_declarations() -> list[dict[str, Any]]:
    """
    Gemini v1beta function calling declarations.

    NOTE: This is a minimal OpenAPI-subset schema intended to be robust; all inputs are validated
    again server-side (instrument + order validation) before hitting the broker.
    """
    return [
        {
            "name": "get_snapshot",
            "description": "Fetch a market data snapshot (best-effort; may be delayed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["STK", "FUT", "FX", "OPT"], "description": "Instrument kind."},
                    "symbol": {"type": "string", "description": "Ticker/symbol, e.g. AAPL or EURUSD."},
                    "exchange": {"type": "string", "description": "Optional exchange, e.g. SMART."},
                    "currency": {"type": "string", "description": "Optional currency, e.g. USD."},
                    "expiry": {"type": "string", "description": "Futures expiry YYYYMM or YYYYMMDD."},
                    "right": {"type": "string", "enum": ["C", "P"], "description": "Option right (C/P)."},
                    "strike": {"type": "number", "description": "Option strike price."},
                    "multiplier": {"type": "string", "description": "Option multiplier (often 100)."},
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_positions",
            "description": "List current account positions.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_account",
            "description": "Fetch current account values (e.g. NetLiquidation).",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "list_open_orders",
            "description": "List currently open orders/trades.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "list_news_providers",
            "description": "List available IBKR news providers for this account.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_historical_news",
            "description": "Fetch historical news headlines for a symbol/instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["STK", "FUT", "FX", "OPT"]},
                    "symbol": {"type": "string"},
                    "exchange": {"type": "string"},
                    "currency": {"type": "string"},
                    "expiry": {"type": "string"},
                    "right": {"type": "string", "enum": ["C", "P"]},
                    "strike": {"type": "number"},
                    "multiplier": {"type": "string"},
                    "provider_codes": {"type": "array", "items": {"type": "string"}},
                    "start_datetime": {"type": "string", "description": "Optional start datetime (IB format)."},
                    "end_datetime": {"type": "string", "description": "Optional end datetime (IB format)."},
                    "max_results": {"type": "integer"},
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_news_article",
            "description": "Fetch the full text of a news article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_code": {"type": "string"},
                    "article_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["TEXT", "HTML"]},
                },
                "required": ["provider_code", "article_id"],
            },
        },
        {
            "name": "research_web",
            "description": "Do web-grounded research for a query and return a short cited brief.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The research question/query to search for."},
                    "urls": {"type": "array", "items": {"type": "string"}, "description": "Optional URLs to ground using URL Context."},
                    "use_code_execution": {"type": "boolean", "description": "If true, allow code execution for calculations."},
                    "max_bullets": {"type": "integer", "description": "Max bullet points to return (default 6)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "place_order",
            "description": "Place a new order (paper-only enforcement happens at broker connect).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order": {
                        "type": "object",
                        "properties": {
                            "instrument": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": ["STK", "FUT", "FX", "OPT"]},
                                    "symbol": {"type": "string"},
                                    "exchange": {"type": "string"},
                                    "currency": {"type": "string"},
                                    "expiry": {"type": "string"},
                                    "right": {"type": "string", "enum": ["C", "P"]},
                                    "strike": {"type": "number"},
                                    "multiplier": {"type": "string"},
                                },
                                "required": ["kind", "symbol"],
                            },
                            "side": {"type": "string", "enum": ["BUY", "SELL"]},
                            "qty": {"type": "number"},
                            "type": {"type": "string", "enum": ["MKT", "LMT", "STP", "STPLMT"]},
                            "limit_price": {"type": "number"},
                            "stop_price": {"type": "number"},
                            "tif": {"type": "string", "description": "Time-in-force (e.g. DAY)."},
                            "outside_rth": {"type": "boolean"},
                        },
                        "required": ["instrument", "side", "qty", "type"],
                    }
                },
                "required": ["order"],
            },
        },
        {
            "name": "modify_order",
            "description": "Modify/replace an existing order by order_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "order": {
                        "type": "object",
                        "properties": {
                            "instrument": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": ["STK", "FUT", "FX", "OPT"]},
                                    "symbol": {"type": "string"},
                                    "exchange": {"type": "string"},
                                    "currency": {"type": "string"},
                                    "expiry": {"type": "string"},
                                    "right": {"type": "string", "enum": ["C", "P"]},
                                    "strike": {"type": "number"},
                                    "multiplier": {"type": "string"},
                                },
                                "required": ["kind", "symbol"],
                            },
                            "side": {"type": "string", "enum": ["BUY", "SELL"]},
                            "qty": {"type": "number"},
                            "type": {"type": "string", "enum": ["MKT", "LMT", "STP", "STPLMT"]},
                            "limit_price": {"type": "number"},
                            "stop_price": {"type": "number"},
                            "tif": {"type": "string"},
                            "outside_rth": {"type": "boolean"},
                        },
                        "required": ["instrument", "side", "qty", "type"],
                    },
                },
                "required": ["order_id", "order"],
            },
        },
        {
            "name": "cancel_order",
            "description": "Cancel an existing order by order_id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
        {
            "name": "oms_reconcile",
            "description": "Reconcile OMS DB state with broker open orders.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "oms_track",
            "description": "Poll open orders and record lifecycle transitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "poll_seconds": {"type": "number"},
                    "timeout_seconds": {"type": "number"},
                },
            },
        },
    ]


def dispatch_tool(
    *,
    call_name: str,
    call_args: dict[str, Any],
    broker: Broker,
    oms: OrderManager,
    allowed_kinds: set[str],
    allowed_symbols: set[str],
    enforce_allowlist: bool = False,
    llm_client: LLMClient | None = None,
) -> Any:
    name = str(call_name).strip()
    args = dict(call_args or {})
    if name == "get_snapshot":
        inst = _parse_instrument(args)
        if enforce_allowlist:
            _enforce_allowlist(inst, allowed_kinds, allowed_symbols)
        md = MarketDataClient(broker, MarketDataConfig())
        snap = md.get_snapshot(inst)
        return asdict(snap)

    if name == "get_positions":
        positions = broker.get_positions()
        return [asdict(p) for p in positions]

    if name == "get_account":
        acct = broker.get_account_snapshot()
        return asdict(acct)

    if name == "list_open_orders":
        st = broker.list_open_order_statuses()
        return [asdict(s) for s in st]

    if name == "list_news_providers":
        providers = broker.list_news_providers()
        return [asdict(p) for p in providers]

    if name == "get_historical_news":
        inst = _parse_instrument(args)
        if enforce_allowlist:
            _enforce_allowlist(inst, allowed_kinds, allowed_symbols)
        provider_codes = args.get("provider_codes")
        if provider_codes is not None and not isinstance(provider_codes, list):
            raise ToolError("provider_codes must be a list of strings")
        start_dt = args.get("start_datetime")
        end_dt = args.get("end_datetime")
        max_results = int(args.get("max_results", 25))
        items = broker.get_historical_news(
            inst,
            provider_codes=[str(x) for x in (provider_codes or [])],
            start_datetime=(str(start_dt) if start_dt is not None else None),
            end_datetime=(str(end_dt) if end_dt is not None else None),
            max_results=max_results,
        )
        return [asdict(it) for it in items]

    if name == "get_news_article":
        provider_code = str(args.get("provider_code", "")).strip()
        article_id = str(args.get("article_id", "")).strip()
        fmt = str(args.get("format", "TEXT")).strip().upper()
        if not provider_code or not article_id:
            raise ToolError("get_news_article requires provider_code and article_id")
        article = broker.get_news_article(provider_code=provider_code, article_id=article_id, format=fmt)
        return asdict(article)

    if name == "research_web":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("research_web requires query")
        urls = args.get("urls")
        if urls is not None and not isinstance(urls, list):
            raise ToolError("research_web urls must be a list of strings")
        url_list = [str(u).strip() for u in (urls or []) if str(u).strip()]
        use_code_execution = bool(args.get("use_code_execution", False))
        max_bullets = int(args.get("max_bullets", 6))
        max_bullets = max(1, min(max_bullets, 12))
        if llm_client is None:
            raise ToolError("research_web requires a Gemini client (GEMINI_API_KEY) to be configured")

        system = (
            "You are an expert trading research analyst.\n"
            "Use Google Search grounding and URL context tools if available to answer the query.\n"
            f"Return {max_bullets} bullet points max, each with citations.\n"
            "Be factual and avoid speculation.\n"
        )
        schema = {
            "type": "object",
            "properties": {
                "bullets": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["bullets"],
        }
        prompt = query
        if url_list:
            prompt += "\n\nUse these URLs as additional context:\n" + "\n".join(url_list)
        data = llm_client.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            system=system,
            tools=None,
            use_google_search=True,
            use_url_context=bool(url_list),
            use_code_execution=bool(use_code_execution),
            include_thoughts=False,
            response_mime_type="application/json",
            response_json_schema=schema,
            cached_content=None,
        )
        try:
            candidates = data.get("candidates")
            c0 = candidates[0] if isinstance(candidates, list) and candidates else None
            gm = c0.get("groundingMetadata") if isinstance(c0, dict) else None
            content = c0.get("content") if isinstance(c0, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            text = "".join(str(p.get("text", "")) for p in (parts or []) if isinstance(p, dict)).strip()
        except Exception as exc:
            raise ToolError(f"research_web unexpected response: {exc}") from exc

        # Attach citations if grounding metadata exists.
        if isinstance(gm, dict) and text:
            supports = gm.get("groundingSupports")
            chunks = gm.get("groundingChunks")
            if isinstance(supports, list) and isinstance(chunks, list):
                uris: list[str | None] = []
                for ch in chunks:
                    if not isinstance(ch, dict):
                        uris.append(None)
                        continue
                    web = ch.get("web")
                    if isinstance(web, dict) and isinstance(web.get("uri"), str):
                        uris.append(str(web.get("uri")))
                    else:
                        uris.append(None)
                def _end_index(s):
                    seg = s.get("segment") if isinstance(s, dict) else None
                    return int(seg.get("endIndex", 0)) if isinstance(seg, dict) else 0
                for sup in sorted([s for s in supports if isinstance(s, dict)], key=_end_index, reverse=True):
                    seg = sup.get("segment")
                    idxs = sup.get("groundingChunkIndices")
                    if not isinstance(seg, dict) or not isinstance(idxs, list):
                        continue
                    end = seg.get("endIndex")
                    if not isinstance(end, int) or end <= 0:
                        continue
                    links: list[str] = []
                    for i in idxs:
                        if not isinstance(i, int):
                            continue
                        uri = uris[i] if 0 <= i < len(uris) else None
                        if uri:
                            links.append(f"[{i+1}]({uri})")
                    if links:
                        text = text[:end] + " " + ", ".join(links) + text[end:]

        # Parse JSON text if possible; else return raw text.
        bullets: list[str] = []
        try:
            obj = json.loads(text) if text else {}
            if isinstance(obj, dict) and isinstance(obj.get("bullets"), list):
                bullets = [str(x) for x in obj.get("bullets") if str(x).strip()]
        except Exception:
            bullets = []

        rendered = ""
        if bullets:
            rendered = "\n".join([f"- {b}" for b in bullets]).strip()
        else:
            rendered = text

        return {"query": query, "urls": url_list, "text": rendered, "grounded": bool(gm)}

    if name == "place_order":
        req = _parse_order_request(args.get("order"))
        if enforce_allowlist:
            _enforce_allowlist(req.instrument, allowed_kinds, allowed_symbols)
        res = oms.submit(req)
        return {"order_id": res.order_id, "status": res.status}

    if name == "modify_order":
        order_id = str(args.get("order_id", "")).strip()
        if not order_id:
            raise ToolError("modify_order requires order_id")
        req = _parse_order_request(args.get("order"))
        if enforce_allowlist:
            _enforce_allowlist(req.instrument, allowed_kinds, allowed_symbols)
        res = oms.modify(order_id, req)
        return {"order_id": res.order_id, "status": res.status}

    if name == "cancel_order":
        order_id = str(args.get("order_id", "")).strip()
        if not order_id:
            raise ToolError("cancel_order requires order_id")
        oms.cancel(order_id)
        return {"order_id": order_id, "status": "CancelRequested", "ts": time.time()}

    if name == "oms_reconcile":
        return oms.reconcile()

    if name == "oms_track":
        poll_seconds = float(args.get("poll_seconds", 1.0))
        timeout = args.get("timeout_seconds")
        timeout_seconds = float(timeout) if timeout is not None else None
        oms.track_open_orders(poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
        return {"ok": True}

    raise ToolError(f"Unknown tool: {name}")


def _parse_instrument(obj: dict[str, Any]) -> InstrumentSpec:
    kind = str(obj.get("kind", "STK")).strip().upper()
    symbol = str(obj.get("symbol", "")).strip().upper()
    if not symbol:
        raise ToolError("instrument.symbol is required")
    inst = InstrumentSpec(
        kind=kind,
        symbol=symbol,
        exchange=(str(obj["exchange"]).strip() if obj.get("exchange") else None),
        currency=(str(obj["currency"]).strip().upper() if obj.get("currency") else None),
        expiry=(str(obj["expiry"]).strip() if obj.get("expiry") else None),
        right=(str(obj["right"]).strip().upper() if obj.get("right") else None),
        strike=(float(obj["strike"]) if obj.get("strike") is not None else None),
        multiplier=(str(obj["multiplier"]).strip() if obj.get("multiplier") else None),
    )
    return validate_instrument(inst)


def _parse_order_request(order_obj: Any) -> OrderRequest:
    if not isinstance(order_obj, dict):
        raise ToolError("order must be an object")
    inst_obj = order_obj.get("instrument")
    if not isinstance(inst_obj, dict):
        raise ToolError("order.instrument must be an object")
    inst = _parse_instrument(inst_obj)

    req = OrderRequest(
        instrument=inst,
        side=str(order_obj.get("side", "BUY")).strip().upper(),
        quantity=float(order_obj.get("qty", 0.0)),
        order_type=str(order_obj.get("type", "MKT")).strip().upper(),
        limit_price=(float(order_obj["limit_price"]) if order_obj.get("limit_price") is not None else None),
        stop_price=(float(order_obj["stop_price"]) if order_obj.get("stop_price") is not None else None),
        tif=str(order_obj.get("tif", "DAY")).strip().upper(),
        outside_rth=bool(order_obj.get("outside_rth", False)),
        good_till_date=(str(order_obj.get("good_till_date")).strip() if order_obj.get("good_till_date") else None),
        account=(str(order_obj.get("account")).strip() if order_obj.get("account") else None),
        order_ref=(str(order_obj.get("order_ref")).strip() if order_obj.get("order_ref") else None),
        oca_group=(str(order_obj.get("oca_group")).strip() if order_obj.get("oca_group") else None),
        transmit=bool(order_obj.get("transmit", True)),
    )
    return req.normalized()


def _enforce_allowlist(inst: InstrumentSpec, allowed_kinds: set[str], allowed_symbols: set[str]) -> None:
    if inst.kind.upper() not in {k.upper() for k in allowed_kinds}:
        raise ToolError(f"Instrument kind not allowed: {inst.kind}")
    if allowed_symbols and inst.symbol.upper() not in {s.upper() for s in allowed_symbols}:
        raise ToolError(f"Symbol not allowed: {inst.symbol}")
